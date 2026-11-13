// The Tsavorite backend for nanosim (GGML_TSAVORITE_TARGET=nanosim).
//
// The device is the Xtensa ISS of a nanosim release (TSI_TXE_DEVICE=xtensa_sim), reached through the runtime's
// session API (tsi-rt/shim/TsiSessionCAPI.h). The serving software never names a TXE count: a launch is one grid
// over the session's TXEs, and the runtime places it. Kernels are hand-written TXE programs built as Ten blobs.
// An operation runs here only when a kernel in tsi_nanosim_kernels() accepts it; everything else falls back to the
// CPU backend, which reads this backend's buffers in place: shared DRAM is host-mapped, so the buffer type is a
// host buffer type and no copies are made.
//
// Environment:
//   TSI_TXE_DEVICE=xtensa_sim           required (the runtime's POSIX default runs FFM .so blobs, not Ten blobs)
//   TSI_XTENSA_SIM_DEVICE               the release's device/bin/TXEXtensaSimDevice
//   USER_DRAM_SIZE                      shared DRAM in MiB (runtime default otherwise)
//   GGML_TSAVORITE_KERNEL_DIR           directory of the kernels' .blob files

#include "ggml-tsavorite.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "tsi-rt/shim/TsiSessionCAPI.h"
#include "tsi_ggml_kargs.h"   // ggml-tsi-kernel/nanosim/include: the kernels' argument block

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define TSI_NANOSIM_NAME "Tsavorite"
#define TSI_NANOSIM_ALIGNMENT 128   // tsi_alloc returns 128-byte aligned blocks (runtime v2-api-userguide.md 6)

// ---- runtime session ----

struct tsi_nanosim_runtime {
    std::mutex     mutex;
    tsi_session_t  session = 0;
    tsi_queue_t    queue   = 0;
    bool           failed  = false;
    std::unordered_map<std::string, tsi_blob_t> blobs;
};

static tsi_nanosim_runtime & tsi_nanosim_rt() {
    static tsi_nanosim_runtime rt;
    return rt;
}

// At exit: unload the blobs and close the queue and the session, so the runtime does not fault a queue left open.
// The runtime itself is released by its own process-exit cleanup (TsiSessionCAPI.h).
static void tsi_nanosim_stop() {
    tsi_nanosim_runtime & rt = tsi_nanosim_rt();
    std::lock_guard<std::mutex> lock(rt.mutex);
    if (!rt.session) {
        return;
    }
    for (auto & b : rt.blobs) {
        tsi_blob_unload(b.second);
    }
    rt.blobs.clear();
    tsi_queue_destroy(rt.queue);
    tsi_session_destroy(rt.session);
    rt.queue   = 0;
    rt.session = 0;
}

// Open the process's session and its one queue, once. Every caller that touches device memory or launches comes
// through here first.
static bool tsi_nanosim_start() {
    tsi_nanosim_runtime & rt = tsi_nanosim_rt();
    std::lock_guard<std::mutex> lock(rt.mutex);
    if (rt.session) {
        return true;
    }
    if (rt.failed) {
        return false;
    }
    const char * device = getenv("TSI_TXE_DEVICE");
    if (!device || strcmp(device, "xtensa_sim") != 0) {
        GGML_LOG_ERROR("%s: TSI_TXE_DEVICE must be xtensa_sim for the nanosim target (it is %s)\n", __func__,
                       device ? device : "unset");
        rt.failed = true;
        return false;
    }
    tsi_session_desc_t desc = {};
    desc.size = sizeof(desc);
    rt.session = tsi_session_create(&desc);
    if (!rt.session) {
        GGML_LOG_ERROR("%s: tsi_session_create failed, status %d\n", __func__, (int) tsi_last_status());
        rt.failed = true;
        return false;
    }
    rt.queue = tsi_queue_create(rt.session);
    if (!rt.queue) {
        GGML_LOG_ERROR("%s: tsi_queue_create failed, status %d\n", __func__, (int) tsi_last_status());
        tsi_session_destroy(rt.session);
        rt.session = 0;
        rt.failed  = true;
        return false;
    }
    atexit(tsi_nanosim_stop);
    return true;
}

// The blob of a kernel, loaded on first use from GGML_TSAVORITE_KERNEL_DIR.
static tsi_blob_t tsi_nanosim_blob(const char * name) {
    tsi_nanosim_runtime & rt = tsi_nanosim_rt();
    std::lock_guard<std::mutex> lock(rt.mutex);
    auto it = rt.blobs.find(name);
    if (it != rt.blobs.end()) {
        return it->second;
    }
    const char * dir = getenv("GGML_TSAVORITE_KERNEL_DIR");
    if (!dir) {
        GGML_LOG_ERROR("%s: GGML_TSAVORITE_KERNEL_DIR is not set\n", __func__);
        return 0;
    }
    const std::string path = std::string(dir) + "/" + name + ".blob";
    tsi_blob_t blob = tsi_blob_load(rt.session, name, path.c_str());
    if (!blob) {
        GGML_LOG_ERROR("%s: tsi_blob_load(%s) failed, status %d\n", __func__, path.c_str(), (int) tsi_last_status());
        return 0;
    }
    rt.blobs[name] = blob;
    return blob;
}

// One launch of a blob as a grid of `cells` cells over shared-DRAM arguments; returns when it has completed.
static bool tsi_nanosim_launch(tsi_blob_t blob, void * args, size_t args_size, uint32_t cells) {
    tsi_nanosim_runtime & rt = tsi_nanosim_rt();
    tsi_spmd_args_t grid = {};
    grid.numGridSizes = 1;
    grid.gridSize[0]  = cells;
    if (tsi_enqueue_blob_launch(rt.queue, blob, tsi_shmem_handle_from_ptr(args), args_size, &grid) == TSI_CMD_INVALID) {
        GGML_LOG_ERROR("%s: tsi_enqueue_blob_launch failed, status %d\n", __func__, (int) tsi_last_status());
        return false;
    }
    TSI_Status st = tsi_wait_queue(rt.queue, TSI_FOREVER);
    if (st != 0) {
        GGML_LOG_ERROR("%s: tsi_wait_queue failed, status %d\n", __func__, (int) st);
        return false;
    }
    return true;
}

// ---- kernel arguments ----

static_assert(sizeof(tsi_ggml_kargs) % 8 == 0, "the grid tail follows the argument block");
static_assert(GGML_MAX_OP_PARAMS == TSI_GGML_OP_PARAMS * sizeof(int32_t), "op_params size");

// One argument block in shared DRAM, reused: launches are synchronous and graph_compute runs on one thread.
static tsi_ggml_kargs * tsi_nanosim_args() {
    static tsi_ggml_kargs * args = nullptr;
    if (!args) {
        args = (tsi_ggml_kargs *) tsi_alloc(sizeof(tsi_ggml_kargs), tsi::MemorySpace::SHARED_DRAM);
    }
    return args;
}

static void tsi_nanosim_desc(tsi_ggml_tensor * d, const struct ggml_tensor * t) {
    d->data = tsi_shmem_handle_from_ptr(t->data);
    d->type = (int32_t) t->type;
    d->pad  = 0;
    for (int i = 0; i < 4; i++) {
        d->ne[i] = t->ne[i];
        d->nb[i] = (int64_t) t->nb[i];
    }
}

static tsi_ggml_kargs * tsi_nanosim_pack(const struct ggml_tensor * node, int n_src) {
    tsi_ggml_kargs * a = tsi_nanosim_args();
    if (!a) {
        return nullptr;
    }
    memset(a, 0, sizeof(*a));
    a->version = TSI_GGML_KARGS_VERSION;
    a->op      = (uint32_t) node->op;
    a->n_src   = (uint32_t) n_src;
    tsi_nanosim_desc(&a->dst, node);
    for (int i = 0; i < n_src; i++) {
        tsi_nanosim_desc(&a->src[i], node->src[i]);
    }
    memcpy(a->op_params, node->op_params, sizeof(a->op_params));
    return a;
}

// Every row start 128-byte aligned: the data offset (for a view; buffers are 128-byte aligned) and nb[1..3].
static bool tsi_nanosim_rows_aligned(const struct ggml_tensor * t) {
    if (t->view_src && t->view_offs % TSI_NANOSIM_ALIGNMENT) {
        return false;
    }
    return t->nb[1] % TSI_NANOSIM_ALIGNMENT == 0 && t->nb[2] % TSI_NANOSIM_ALIGNMENT == 0 &&
           t->nb[3] % TSI_NANOSIM_ALIGNMENT == 0;
}

// ---- kernels ----

// RMS_NORM, f32 (ggml-tsi-kernel nanosim/rms_norm_f32.c): one grid cell per row.
static bool tsi_nanosim_rms_norm_supports(const struct ggml_tensor * node) {
    const struct ggml_tensor * src = node->src[0];
    return src && src->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 && ggml_are_same_shape(src, node) &&
           src->nb[0] == sizeof(float) && node->nb[0] == sizeof(float) && src->ne[0] % 32 == 0 &&
           src->ne[0] < (1 << 24) && ggml_nrows(src) <= INT32_MAX && tsi_nanosim_rows_aligned(src) &&
           tsi_nanosim_rows_aligned(node);
}

static bool tsi_nanosim_rms_norm_launch(tsi_blob_t blob, struct ggml_tensor * node) {
    if ((uintptr_t) node->data % TSI_NANOSIM_ALIGNMENT || (uintptr_t) node->src[0]->data % TSI_NANOSIM_ALIGNMENT) {
        GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
        return false;
    }
    tsi_ggml_kargs * a = tsi_nanosim_pack(node, 1);
    return a && tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) ggml_nrows(node->src[0]));
}


// A hand-written kernel: which operations it accepts and how it is launched. A kernel enters the table after it
// passes its standalone check on tdu_xtmp (taos-mk docs/agent-harness/workflow-llamacpp.md).
struct tsi_nanosim_kernel {
    enum ggml_op op;
    const char * blob;                                  // file name under GGML_TSAVORITE_KERNEL_DIR, without .blob
    bool (*supports)(const struct ggml_tensor * node);  // every type, shape, stride and op_params it accepts
    bool (*launch)(tsi_blob_t blob, struct ggml_tensor * node);
};

static const std::vector<tsi_nanosim_kernel> & tsi_nanosim_kernels() {
    static const std::vector<tsi_nanosim_kernel> kernels = {
        { GGML_OP_RMS_NORM, "rms_norm_f32", tsi_nanosim_rms_norm_supports, tsi_nanosim_rms_norm_launch },
    };
    return kernels;
}

static const tsi_nanosim_kernel * tsi_nanosim_find_kernel(const struct ggml_tensor * node) {
    for (const tsi_nanosim_kernel & k : tsi_nanosim_kernels()) {
        if (k.op == node->op && k.supports(node)) {
            return &k;
        }
    }
    return nullptr;
}

// ---- buffers ----

struct tsi_nanosim_buffer_context {
    void * base;
    size_t size;
};

static void tsi_nanosim_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (tsi_nanosim_buffer_context *) buffer->context;
    tsi_dealloc(ctx->base, tsi::MemorySpace::SHARED_DRAM);
    delete ctx;
}

static void * tsi_nanosim_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((tsi_nanosim_buffer_context *) buffer->context)->base;
}

static void tsi_nanosim_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value,
                                             size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
    GGML_UNUSED(buffer);
}

static void tsi_nanosim_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
                                          size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
    GGML_UNUSED(buffer);
}

static void tsi_nanosim_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
                                          size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
    GGML_UNUSED(buffer);
}

static bool tsi_nanosim_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src,
                                          struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
    GGML_UNUSED(buffer);
}

static void tsi_nanosim_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (tsi_nanosim_buffer_context *) buffer->context;
    memset(ctx->base, value, ctx->size);
}

static const struct ggml_backend_buffer_i tsi_nanosim_buffer_i = {
    /* .free_buffer   = */ tsi_nanosim_buffer_free,
    /* .get_base      = */ tsi_nanosim_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ tsi_nanosim_buffer_memset_tensor,
    /* .set_tensor    = */ tsi_nanosim_buffer_set_tensor,
    /* .get_tensor    = */ tsi_nanosim_buffer_get_tensor,
    /* .cpy_tensor    = */ tsi_nanosim_buffer_cpy_tensor,
    /* .clear         = */ tsi_nanosim_buffer_clear,
    /* .reset         = */ nullptr,
};

// ---- buffer type ----

static const char * tsi_nanosim_buft_get_name(ggml_backend_buffer_type_t buft) {
    return TSI_NANOSIM_NAME;
    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t tsi_nanosim_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (!tsi_nanosim_start()) {
        return nullptr;
    }
    size = size ? size : TSI_NANOSIM_ALIGNMENT;   // tsi_alloc does not reject 0 in release builds
    void * base = nullptr;
    try {
        base = tsi_alloc((int64_t) size, tsi::MemorySpace::SHARED_DRAM);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: tsi_alloc(%zu) failed: %s\n", __func__, size, e.what());
        return nullptr;
    }
    if (!base) {
        GGML_LOG_ERROR("%s: tsi_alloc(%zu) failed\n", __func__, size);
        return nullptr;
    }
    auto * ctx = new tsi_nanosim_buffer_context{ base, size };
    return ggml_backend_buffer_init(buft, tsi_nanosim_buffer_i, ctx, size);
}

static size_t tsi_nanosim_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return TSI_NANOSIM_ALIGNMENT;
    GGML_UNUSED(buft);
}

static bool tsi_nanosim_buft_is_host(ggml_backend_buffer_type_t buft) {
    return true;   // shared DRAM is host-mapped: the CPU backend reads and writes it in place
    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_tsavorite_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name       = */ tsi_nanosim_buft_get_name,
            /* .alloc_buffer   = */ tsi_nanosim_buft_alloc_buffer,
            /* .get_alignment  = */ tsi_nanosim_buft_get_alignment,
            /* .get_max_size   = */ nullptr,
            /* .get_alloc_size = */ nullptr,
            /* .is_host        = */ tsi_nanosim_buft_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_tsavorite_reg(), 0),
        /* .context = */ nullptr,
    };
    return &buft;
}

// ---- backend ----

static const char * tsi_nanosim_backend_get_name(ggml_backend_t backend) {
    return TSI_NANOSIM_NAME;
    GGML_UNUSED(backend);
}

static void tsi_nanosim_backend_free(ggml_backend_t backend) {
    delete backend;   // the session stays open for the process: buffers outlive backends
}

static bool tsi_nanosim_is_noop(const struct ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return ggml_is_empty(node);
    }
}

static enum ggml_status tsi_nanosim_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (tsi_nanosim_is_noop(node)) {
            continue;
        }
        const tsi_nanosim_kernel * k = tsi_nanosim_find_kernel(node);
        if (!k) {
            GGML_LOG_ERROR("%s: no kernel for %s (%s)\n", __func__, ggml_op_desc(node), node->name);
            return GGML_STATUS_FAILED;
        }
        tsi_blob_t blob = tsi_nanosim_blob(k->blob);
        if (!blob || !k->launch(blob, node)) {
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

static ggml_guid_t tsi_nanosim_guid(void) {
    static ggml_guid guid = { 0x54, 0x53, 0x49, 0x2d, 0x6e, 0x61, 0x6e, 0x6f, 0x73, 0x69, 0x6d, 0x2d, 0x54, 0x58, 0x45, 0x01 };
    return &guid;
}

static const struct ggml_backend_i tsi_nanosim_backend_i = {
    /* .get_name           = */ tsi_nanosim_backend_get_name,
    /* .free               = */ tsi_nanosim_backend_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,   // every launch waits for its completion
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ tsi_nanosim_backend_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
    /* .graph_reserve      = */ nullptr,
    /* .buffer_size        = */ nullptr,
    /* .reset              = */ nullptr,
    /* .profile            = */ nullptr,
};

ggml_backend_t ggml_backend_tsavorite_init(void) {
    if (!tsi_nanosim_start()) {
        return nullptr;
    }
    return new ggml_backend{
        /* .guid    = */ tsi_nanosim_guid(),
        /* .iface   = */ tsi_nanosim_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_tsavorite_reg(), 0),
        /* .context = */ nullptr,
    };
}

bool ggml_backend_is_tsavorite(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, tsi_nanosim_guid());
}

void ggml_backend_tsavorite_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback,
                                               void * user_data) {
    GGML_UNUSED(backend);
    GGML_UNUSED(abort_callback);
    GGML_UNUSED(user_data);
}

void ggml_backend_tsavorite_capture_next_compute(ggml_backend_t backend) {
    GGML_UNUSED(backend);
}

// ---- device ----

static const char * tsi_nanosim_device_get_name(ggml_backend_dev_t dev) {
    return TSI_NANOSIM_NAME;
    GGML_UNUSED(dev);
}

static const char * tsi_nanosim_device_get_description(ggml_backend_dev_t dev) {
    return "TXE on nanosim (Xtensa ISS)";
    GGML_UNUSED(dev);
}

// Shared DRAM as USER_DRAM_SIZE (MiB) asks for; the runtime does not report free memory, so free = total.
static void tsi_nanosim_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    const char * mib = getenv("USER_DRAM_SIZE");
    *total = mib ? (size_t) strtoull(mib, nullptr, 10) << 20 : 0;
    *free  = *total;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type tsi_nanosim_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
    GGML_UNUSED(dev);
}

static void tsi_nanosim_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = tsi_nanosim_device_get_name(dev);
    props->description = tsi_nanosim_device_get_description(dev);
    props->type        = tsi_nanosim_device_get_type(dev);
    tsi_nanosim_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
    };
}

static ggml_backend_t tsi_nanosim_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_tsavorite_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t tsi_nanosim_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_tsavorite_buffer_type();
    GGML_UNUSED(dev);
}

// No-op operations (views, reshapes, leaves) cost nothing and graph_compute skips them; the rest need a kernel.
static bool tsi_nanosim_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    return tsi_nanosim_is_noop(op) || tsi_nanosim_find_kernel(op) != nullptr;
    GGML_UNUSED(dev);
}

// Kernels read and write shared DRAM only, so only this backend's buffers qualify.
static bool tsi_nanosim_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == tsi_nanosim_buft_get_name;
    GGML_UNUSED(dev);
}

static bool tsi_nanosim_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    return false;
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
}

static const struct ggml_backend_device_i tsi_nanosim_device_i = {
    /* .get_name             = */ tsi_nanosim_device_get_name,
    /* .get_description      = */ tsi_nanosim_device_get_description,
    /* .get_memory           = */ tsi_nanosim_device_get_memory,
    /* .get_type             = */ tsi_nanosim_device_get_type,
    /* .get_props            = */ tsi_nanosim_device_get_props,
    /* .init_backend         = */ tsi_nanosim_device_init_backend,
    /* .get_buffer_type      = */ tsi_nanosim_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ tsi_nanosim_device_supports_op,
    /* .supports_buft        = */ tsi_nanosim_device_supports_buft,
    /* .offload_op           = */ tsi_nanosim_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ---- registry ----

static const char * tsi_nanosim_reg_get_name(ggml_backend_reg_t reg) {
    return TSI_NANOSIM_NAME;
    GGML_UNUSED(reg);
}

static size_t tsi_nanosim_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t tsi_nanosim_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    static struct ggml_backend_device device = {
        /* .iface   = */ tsi_nanosim_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    GGML_ASSERT(index == 0);
    return &device;
}

static const struct ggml_backend_reg_i tsi_nanosim_reg_i = {
    /* .get_name         = */ tsi_nanosim_reg_get_name,
    /* .get_device_count = */ tsi_nanosim_reg_get_device_count,
    /* .get_device       = */ tsi_nanosim_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_tsavorite_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ tsi_nanosim_reg_i,
        /* .context     = */ nullptr,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_tsavorite_reg)
