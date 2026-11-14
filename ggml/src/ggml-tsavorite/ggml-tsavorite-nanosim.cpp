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

#include <cmath>
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

// Row-wise launch: one grid cell per dst row, after checking the actual data addresses.
static bool tsi_nanosim_launch_rows(tsi_blob_t blob, struct ggml_tensor * node, int n_src) {
    for (int i = -1; i < n_src; i++) {
        const struct ggml_tensor * t = i < 0 ? node : node->src[i];
        if ((uintptr_t) t->data % TSI_NANOSIM_ALIGNMENT) {
            GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
            return false;
        }
    }
    tsi_ggml_kargs * a = tsi_nanosim_pack(node, n_src);
    return a && tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) ggml_nrows(node));
}

// f32, rows contiguous, ne0 a multiple of 32: the row-wise kernels' common conditions on one tensor. A dimension of
// size 1 may have any stride (it is never stepped).
static bool tsi_nanosim_f32_rows(const struct ggml_tensor * t, int64_t ne0) {
    if (!t || t->type != GGML_TYPE_F32 || t->nb[0] != sizeof(float) || t->ne[0] != ne0 || ne0 % 32 ||
        ne0 >= (1 << 24) || ggml_nrows(t) > INT32_MAX) {
        return false;
    }
    if (t->view_src && t->view_offs % TSI_NANOSIM_ALIGNMENT) {
        return false;
    }
    for (int d = 1; d < 4; d++) {
        if (t->ne[d] > 1 && t->nb[d] % TSI_NANOSIM_ALIGNMENT) {
            return false;
        }
    }
    return true;
}

// ADD, MUL, SCALE, GLU (SWIGLU, two inputs), f32 (ggml-tsi-kernel nanosim/eltwise_f32.c). The second input of ADD
// and MUL may broadcast along dimensions 1..3.
static bool tsi_nanosim_eltwise_supports(const struct ggml_tensor * node) {
    const int64_t ne0 = node->ne[0];
    const struct ggml_tensor * a = node->src[0], * b = node->src[1];
    if (!tsi_nanosim_f32_rows(node, ne0) || !tsi_nanosim_f32_rows(a, ne0) || !ggml_are_same_shape(a, node)) {
        return false;
    }
    switch (node->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            if (!tsi_nanosim_f32_rows(b, ne0)) {
                return false;
            }
            for (int d = 1; d < 4; d++) {
                if (b->ne[d] != node->ne[d] && b->ne[d] != 1) {
                    return false;
                }
            }
            return true;
        case GGML_OP_SCALE:
            return true;
        case GGML_OP_GLU:
            return ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU && tsi_nanosim_f32_rows(b, ne0) &&
                   ggml_are_same_shape(b, node);
        default:
            return false;
    }
}

static bool tsi_nanosim_eltwise_launch(tsi_blob_t blob, struct ggml_tensor * node) {
    return tsi_nanosim_launch_rows(blob, node, node->op == GGML_OP_SCALE ? 1 : 2);
}

// ---- small tensors the TXE's scalar core reads (indices, positions) ----

// The core sees DRAM offset X at core address X only inside [0x500000, 0xff800000); lower offsets alias its private
// RAM and the blob segment (tsisw/taos-mk CLAUDE.md, memory map). A tensor the core reads is copied into a staging
// buffer whose handle lies inside that window (found as the harness's alloc_low does: allocations outside it are
// held until one lands inside, then freed).
#define TSI_NANOSIM_CORE_LO 0x500000u
#define TSI_NANOSIM_CORE_HI 0xff800000u

static void * tsi_nanosim_core_staging(size_t bytes) {
    static void * buf = nullptr;
    static size_t size = 0;
    if (buf && size >= bytes) {
        return buf;
    }
    if (buf) {
        tsi_dealloc(buf, tsi::MemorySpace::SHARED_DRAM);
        buf  = nullptr;
        size = 0;
    }
    // The runtime serves small blocks from the top of DRAM, above the core's window (handles near 0xfff00000 with
    // USER_DRAM_SIZE 4096, measured 2026-11-13), and larger ones below it: ask for at least 1 MiB.
    const size_t want = bytes < (1u << 20) ? (1u << 20) : bytes;
    std::vector<void *> held;
    for (int i = 0; i < 16 && !buf; i++) {
        void * p = nullptr;
        try {
            p = tsi_alloc((int64_t) want, tsi::MemorySpace::SHARED_DRAM);
        } catch (const std::exception &) {
            break;
        }
        const uint64_t h = (uint64_t) tsi_shmem_handle_from_ptr(p);
        if (h >= TSI_NANOSIM_CORE_LO && h + want <= TSI_NANOSIM_CORE_HI) {
            buf  = p;
            size = want;
        } else {
            held.push_back(p);
        }
    }
    for (void * p : held) {
        tsi_dealloc(p, tsi::MemorySpace::SHARED_DRAM);
    }
    if (!buf) {
        GGML_LOG_ERROR("%s: no %zu-byte block inside the core's window\n", __func__, want);
    }
    return buf;
}

// Copy t (indices or positions) to the staging buffer and point descriptor d at it.
static bool tsi_nanosim_stage_for_core(tsi_ggml_tensor * d, const struct ggml_tensor * t) {
    void * buf = tsi_nanosim_core_staging(ggml_nbytes(t));
    if (!buf) {
        return false;
    }
    memcpy(buf, t->data, ggml_nbytes(t));
    d->data = tsi_shmem_handle_from_ptr(buf);
    if (getenv("GGML_TSAVORITE_DEBUG")) {
        const int32_t * v = (const int32_t *) buf;
        fprintf(stderr, "%s: %s: %zu bytes staged at handle %#llx: %d %d %d %d\n", __func__, t->name, ggml_nbytes(t),
                      (unsigned long long) d->data, v[0], ggml_nbytes(t) > 4 ? v[1] : 0, ggml_nbytes(t) > 8 ? v[2] : 0,
                      ggml_nbytes(t) > 12 ? v[3] : 0);
    }
    return true;
}

// ---- row copies (ggml-tsi-kernel nanosim/copy_rows.c) ----

// A destination the copy kernel writes whole rows of: f32 (128-byte aligned rows) or f16 (64-byte aligned rows).
static bool tsi_nanosim_copy_dst(const struct ggml_tensor * t, int64_t ne0) {
    if (!t || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) || t->ne[0] != ne0 ||
        t->nb[0] != ggml_type_size(t->type)) {
        return false;
    }
    const size_t align = t->type == GGML_TYPE_F32 ? TSI_NANOSIM_ALIGNMENT : TSI_NANOSIM_ALIGNMENT / 2;
    if (t->view_src && t->view_offs % align) {
        return false;
    }
    for (int d = 1; d < 4; d++) {
        if (t->ne[d] > 1 && t->nb[d] % align) {
            return false;
        }
    }
    return true;
}

// CONT, CPY (f32 rows to a contiguous f32 or f16 tensor), GET_ROWS (f32, i32 indices), SET_ROWS (f32 rows into
// f32 or f16, i64 or i32 indices).
static bool tsi_nanosim_copy_supports(const struct ggml_tensor * node) {
    const struct ggml_tensor * s = node->src[0], * ix = node->src[1];
    if (!s || !tsi_nanosim_f32_rows(s, s->ne[0])) {
        return false;
    }
    switch (node->op) {
        case GGML_OP_CONT:
        case GGML_OP_CPY:   // source rows back to back into a contiguous f32 or f16 destination
            return (node->type == GGML_TYPE_F32 || node->type == GGML_TYPE_F16) && ggml_is_contiguous(node) &&
                   ggml_nelements(node) == ggml_nelements(s) && !(node->view_src && node->view_offs % 64);
        case GGML_OP_GET_ROWS:
            return node->type == GGML_TYPE_F32 && ix && ix->type == GGML_TYPE_I32 && ggml_is_contiguous(ix) &&
                   s->ne[2] == 1 && s->ne[3] == 1 && ix->ne[1] == 1 && ix->ne[2] == 1 && ix->ne[3] == 1 &&
                   tsi_nanosim_copy_dst(node, s->ne[0]);
        case GGML_OP_SET_ROWS:
            return ix && (ix->type == GGML_TYPE_I64 || ix->type == GGML_TYPE_I32) && ggml_is_contiguous(ix) &&
                   s->ne[2] == 1 && s->ne[3] == 1 && node->ne[2] == 1 && node->ne[3] == 1 && ix->ne[0] == s->ne[1] &&
                   ix->ne[1] == 1 && ix->ne[2] == 1 && ix->ne[3] == 1 && tsi_nanosim_copy_dst(node, s->ne[0]);
        default:
            return false;
    }
}

static bool tsi_nanosim_copy_launch(tsi_blob_t blob, struct ggml_tensor * node) {
    const struct ggml_tensor * s = node->src[0], * ix = node->src[1];
    if (getenv("GGML_TSAVORITE_DEBUG")) {
        fprintf(stderr, "copy_launch %s %s: src %#llx (mod128 %u) dst %#llx (mod128 %u) ix %p\n", ggml_op_name(node->op),
                node->name, (unsigned long long) tsi_shmem_handle_from_ptr(s->data), (unsigned) ((uintptr_t) s->data % 128),
                (unsigned long long) tsi_shmem_handle_from_ptr(node->data), (unsigned) ((uintptr_t) node->data % 128),
                ix ? ix->data : nullptr);
    }
    const bool indexed = node->op == GGML_OP_GET_ROWS || node->op == GGML_OP_SET_ROWS;

    if ((uintptr_t) s->data % TSI_NANOSIM_ALIGNMENT ||
        (uintptr_t) node->data % (node->type == GGML_TYPE_F16 ? 64 : TSI_NANOSIM_ALIGNMENT)) {
        GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
        return false;
    }
    tsi_ggml_kargs * a = tsi_nanosim_pack(node, indexed ? 2 : 1);
    if (a && indexed && !tsi_nanosim_stage_for_core(&a->src[1], ix)) {
        return false;
    }
    const int64_t cells = node->op == GGML_OP_GET_ROWS ? ix->ne[0] : node->op == GGML_OP_SET_ROWS ? s->ne[1] : ggml_nrows(s);
    return a && tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) cells);
}

// ---- SOFT_MAX (ggml-tsi-kernel nanosim/soft_max_f32.c) ----

// soft_max_ext with an optional f16 or f32 mask broadcast along dims 2..3, max_bias 0 and no sinks.
static bool tsi_nanosim_soft_max_supports(const struct ggml_tensor * node) {
    const struct ggml_tensor * x = node->src[0], * m = node->src[1];
    float max_bias;
    memcpy(&max_bias, (const float *) node->op_params + 1, sizeof(float));
    if (max_bias != 0.0f || node->src[2] || !tsi_nanosim_f32_rows(x, x->ne[0]) ||
        !tsi_nanosim_f32_rows(node, x->ne[0]) || !ggml_are_same_shape(x, node)) {
        return false;
    }
    if (!m) {
        return true;
    }
    if (m->type == GGML_TYPE_F32) {
        return tsi_nanosim_f32_rows(m, x->ne[0]) && m->ne[1] >= x->ne[1];
    }
    return tsi_nanosim_copy_dst(m, x->ne[0]) && m->type == GGML_TYPE_F16 && m->ne[1] >= x->ne[1];
}

static bool tsi_nanosim_soft_max_launch(tsi_blob_t blob, struct ggml_tensor * node) {
    const struct ggml_tensor * m = node->src[1];
    if ((uintptr_t) node->data % TSI_NANOSIM_ALIGNMENT || (uintptr_t) node->src[0]->data % TSI_NANOSIM_ALIGNMENT ||
        (m && (uintptr_t) m->data % (m->type == GGML_TYPE_F16 ? 64 : TSI_NANOSIM_ALIGNMENT))) {
        GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
        return false;
    }
    tsi_ggml_kargs * a = tsi_nanosim_pack(node, m ? 2 : 1);
    return a && tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) ggml_nrows(node));
}

// ---- MUL_MAT, f16 x f32 (ggml-tsi-kernel nanosim/mul_mat_f16.c): attention's KQ and KQV ----

static bool tsi_nanosim_mul_mat_f16_supports(const struct ggml_tensor * node) {
    const struct ggml_tensor * w = node->src[0], * x = node->src[1];
    if (!w || !x || w->type != GGML_TYPE_F16 || w->nb[0] != sizeof(ggml_fp16_t) || w->ne[0] % 16 ||
        w->ne[1] % 32 || w->ne[0] >= (1 << 24) || w->ne[1] >= (1 << 24) || x->ne[0] != w->ne[0] ||
        x->ne[2] % w->ne[2] || x->ne[3] % w->ne[3] || node->ne[0] != w->ne[1] || node->ne[1] != x->ne[1]) {
        return false;
    }
    if (w->view_src && w->view_offs % TSI_NANOSIM_ALIGNMENT) {
        return false;
    }
    for (int d = 1; d < 4; d++) {
        if (w->ne[d] > 1 && w->nb[d] % TSI_NANOSIM_ALIGNMENT) {
            return false;
        }
    }
    return tsi_nanosim_f32_rows(x, x->ne[0]) && tsi_nanosim_f32_rows(node, node->ne[0]);
}

static bool tsi_nanosim_mul_mat_f16_launch(tsi_blob_t blob, struct ggml_tensor * node) {
    if ((uintptr_t) node->data % TSI_NANOSIM_ALIGNMENT || (uintptr_t) node->src[0]->data % TSI_NANOSIM_ALIGNMENT ||
        (uintptr_t) node->src[1]->data % TSI_NANOSIM_ALIGNMENT) {
        GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
        return false;
    }
    tsi_ggml_kargs * a = tsi_nanosim_pack(node, 2);
    const int64_t cells = node->ne[2] * node->ne[3] * ((node->ne[0] + 127) / 128);
    return a && tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) cells);
}

// ---- ROPE (ggml-tsi-kernel nanosim/rope_f32.c) ----

// NeoX mode, f32, n_dims = ne0 of 64 or 128, ext_factor 0, no frequency factors, forward.
static bool tsi_nanosim_rope_supports(const struct ggml_tensor * node) {
    const struct ggml_tensor * x = node->src[0], * p = node->src[1];
    const int32_t n_dims = node->op_params[1], mode = node->op_params[2];
    float ext_factor;
    memcpy(&ext_factor, (const int32_t *) node->op_params + 7, sizeof(float));
    return x && p && node->src[2] == nullptr && mode == GGML_ROPE_TYPE_NEOX && ext_factor == 0.0f &&
           n_dims == x->ne[0] && (n_dims == 64 || n_dims == 128) && p->type == GGML_TYPE_I32 &&
           ggml_is_contiguous(p) && p->ne[0] == x->ne[2] && tsi_nanosim_f32_rows(x, x->ne[0]) &&
           tsi_nanosim_f32_rows(node, x->ne[0]) && ggml_are_same_shape(x, node);
}

static bool tsi_nanosim_rope_launch(tsi_blob_t blob, struct ggml_tensor * node) {
    const struct ggml_tensor * p = node->src[1];
    if ((uintptr_t) node->data % TSI_NANOSIM_ALIGNMENT || (uintptr_t) node->src[0]->data % TSI_NANOSIM_ALIGNMENT) {
        GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
        return false;
    }
    tsi_ggml_kargs * a = tsi_nanosim_pack(node, 2);
    if (!a || !tsi_nanosim_stage_for_core(&a->src[1], p)) {
        return false;
    }
    float freq_base;
    memcpy(&freq_base, (const int32_t *) node->op_params + 5, sizeof(float));
    const float theta_scale = powf(freq_base, -2.0f/node->op_params[1]);   // as ggml_compute_forward_rope_flt
    memcpy(&a->op_params[15], &theta_scale, sizeof(float));
    return tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) (node->ne[2] * node->ne[3]));
}

// ---- MUL_MAT, Q8_0 weights ----

// A raw descriptor for a buffer the backend owns (repacked weights, activation scratch).
static void tsi_nanosim_raw_desc(tsi_ggml_tensor * d, void * ptr, int32_t type, std::initializer_list<int64_t> ne,
                                 std::initializer_list<int64_t> nb) {
    d->data = tsi_shmem_handle_from_ptr(ptr);
    d->type = type;
    d->pad  = 0;
    int i = 0;
    for (int64_t v : ne) d->ne[i++] = v;
    for (; i < 4; i++) d->ne[i] = 1;
    i = 0;
    for (int64_t v : nb) d->nb[i++] = v;
    for (; i < 4; i++) d->nb[i] = d->nb[i - 1] * d->ne[i - 1];
}

// The kernel's layout of a Q8_0 weight [K, N] (ggml-tsi-kernel nanosim/mul_mat_q8_0.c): int8 rows of K bytes and the
// block scales as f32 [K/32][N]. Made on the host from the GGUF blocks (34 bytes: fp16 d, 32 int8), once per weight
// tensor of a weights buffer; the GGUF bytes stay in place for the CPU's use (decided 2026-11-13: repack, option 1).
struct tsi_nanosim_q8_weights {
    void * wq = nullptr;
    void * dw = nullptr;
    void * ds = nullptr;   // row-major scales [N][K/32] in 128-byte rows, made when GET_ROWS first reads the tensor
};

static bool tsi_nanosim_q8_repack(const struct ggml_tensor * w, tsi_nanosim_q8_weights & r) {
    const int64_t k = w->ne[0], n = w->ne[1], nbl = k / 32;
    try {
        r.wq = tsi_alloc(n * k, tsi::MemorySpace::SHARED_DRAM);
        r.dw = tsi_alloc(nbl * n * (int64_t) sizeof(float), tsi::MemorySpace::SHARED_DRAM);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: %s: tsi_alloc failed: %s\n", __func__, w->name, e.what());
        return false;
    }
    int8_t * wq = (int8_t *) r.wq;
    float * dw = (float *) r.dw;
    for (int64_t i = 0; i < n; i++) {
        const uint8_t * row = (const uint8_t *) w->data + i * w->nb[1];
        for (int64_t b = 0; b < nbl; b++) {
            ggml_fp16_t d;
            memcpy(&d, row + b * 34, sizeof(d));
            dw[b * n + i] = ggml_fp16_to_fp32(d);
            memcpy(wq + i * k + b * 32, row + b * 34 + 2, 32);
        }
    }
    return true;
}

static void tsi_nanosim_q8_free(tsi_nanosim_q8_weights & r) {
    if (r.wq) tsi_dealloc(r.wq, tsi::MemorySpace::SHARED_DRAM);
    if (r.dw) tsi_dealloc(r.dw, tsi::MemorySpace::SHARED_DRAM);
    if (r.ds) tsi_dealloc(r.ds, tsi::MemorySpace::SHARED_DRAM);
    r = {};
}

// Repacks of weights-buffer tensors, by data address; other tensors are repacked for each launch (their contents
// may change between launches).
static std::unordered_map<const void *, tsi_nanosim_q8_weights> & tsi_nanosim_q8_cache() {
    static std::unordered_map<const void *, tsi_nanosim_q8_weights> cache;
    return cache;
}

// Shared-DRAM scratch that grows as needed: the quantized activations and their scales.
static void * tsi_nanosim_scratch(int slot, size_t bytes) {
    static void * ptr[2] = { nullptr, nullptr };
    static size_t size[2] = { 0, 0 };
    if (size[slot] < bytes) {
        if (ptr[slot]) tsi_dealloc(ptr[slot], tsi::MemorySpace::SHARED_DRAM);
        ptr[slot]  = tsi_alloc((int64_t) bytes, tsi::MemorySpace::SHARED_DRAM);
        size[slot] = bytes;
    }
    return ptr[slot];
}

// Q8_0 weights [K, N] (2-D, rows contiguous), f32 activations [K, M] (2-D, rows contiguous and 128-byte aligned), f32
// dst [N, M]; K and N multiples of 128.
static bool tsi_nanosim_mul_mat_q8_0_supports(const struct ggml_tensor * node) {
    const struct ggml_tensor * w = node->src[0], * x = node->src[1];
    return w && x && w->type == GGML_TYPE_Q8_0 && x->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
           w->ne[2] == 1 && w->ne[3] == 1 && x->ne[2] == 1 && x->ne[3] == 1 && w->ne[0] % 128 == 0 &&
           w->ne[1] % 128 == 0 && w->ne[0] < (1 << 24) && w->ne[1] < (1 << 24) && x->ne[0] == w->ne[0] &&
           ggml_is_contiguous_rows(w) && tsi_nanosim_f32_rows(x, x->ne[0]) && tsi_nanosim_f32_rows(node, w->ne[1]) &&
           x->ne[1] < (1 << 20);
}

static bool tsi_nanosim_mul_mat_q8_0_launch(tsi_blob_t blob, struct ggml_tensor * node) {
    const struct ggml_tensor * w = node->src[0], * x = node->src[1];
    const int64_t k = w->ne[0], n = w->ne[1], m = x->ne[1], mp = (m + 15) / 16 * 16, nbl = k / 32;
    if ((uintptr_t) x->data % TSI_NANOSIM_ALIGNMENT || (uintptr_t) node->data % TSI_NANOSIM_ALIGNMENT) {
        GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
        return false;
    }
    tsi_blob_t qblob = tsi_nanosim_blob("quantize_q8_0_f32");
    if (!qblob) {
        return false;
    }
    // weights: cached for a weights buffer, otherwise repacked now and freed after the launch
    tsi_nanosim_q8_weights wr, tmp;
    const bool cached = w->buffer && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    if (cached) {
        auto it = tsi_nanosim_q8_cache().find(w->data);
        if (it == tsi_nanosim_q8_cache().end()) {
            if (!tsi_nanosim_q8_repack(w, wr)) return false;
            tsi_nanosim_q8_cache()[w->data] = wr;
        } else {
            wr = it->second;
        }
    } else {
        if (!tsi_nanosim_q8_repack(w, tmp)) return false;
        wr = tmp;
    }
    void * xq = tsi_nanosim_scratch(0, (size_t) (mp * k));
    void * xd = tsi_nanosim_scratch(1, (size_t) (mp * nbl * 128));
    // 1. the activations to Q8_0: one grid cell per row
    tsi_ggml_kargs * a = tsi_nanosim_args();
    memset(a, 0, sizeof(*a));
    a->version = TSI_GGML_KARGS_VERSION;
    a->op      = (uint32_t) GGML_OP_NONE;
    a->n_src   = 2;
    tsi_nanosim_raw_desc(&a->dst, xq, GGML_TYPE_I8, { k, mp }, { 1, k });
    tsi_nanosim_desc(&a->src[0], x);
    tsi_nanosim_raw_desc(&a->src[1], xd, GGML_TYPE_F32, { nbl * 32, mp }, { 4, nbl * 128 });
    bool ok = tsi_nanosim_launch(qblob, a, sizeof(*a), (uint32_t) m);
    // 2. the matmul: one grid cell per 128-column tile
    if (ok) {
        memset(a, 0, sizeof(*a));
        a->version = TSI_GGML_KARGS_VERSION;
        a->op      = (uint32_t) GGML_OP_MUL_MAT;
        a->n_src   = 4;
        tsi_nanosim_desc(&a->dst, node);
        tsi_nanosim_raw_desc(&a->src[0], wr.wq, GGML_TYPE_I8, { k, n }, { 1, k });
        tsi_nanosim_raw_desc(&a->src[1], wr.dw, GGML_TYPE_F32, { n, nbl }, { 4, n * 4 });
        tsi_nanosim_raw_desc(&a->src[2], xq, GGML_TYPE_I8, { k, mp }, { 1, k });
        tsi_nanosim_raw_desc(&a->src[3], xd, GGML_TYPE_F32, { nbl * 32, mp }, { 4, nbl * 128 });
        ok = tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) (n / 128));
    }
    if (!cached) {
        tsi_nanosim_q8_free(tmp);
    }
    return ok;
}

// ---- GET_ROWS from Q8_0 (ggml-tsi-kernel nanosim/get_rows_q8_0.c): the token embedding ----

static int64_t tsi_nanosim_q8_srow(int64_t k) {
    return ((k / 32) * (int64_t) sizeof(float) + TSI_NANOSIM_ALIGNMENT - 1) / TSI_NANOSIM_ALIGNMENT * TSI_NANOSIM_ALIGNMENT;
}

static bool tsi_nanosim_get_rows_q8_0_supports(const struct ggml_tensor * node) {
    const struct ggml_tensor * w = node->src[0], * ix = node->src[1];
    return w && ix && w->type == GGML_TYPE_Q8_0 && node->type == GGML_TYPE_F32 && w->ne[2] == 1 && w->ne[3] == 1 &&
           w->ne[0] % 128 == 0 && (uint64_t) w->ne[0] * (uint64_t) w->ne[1] < (1ull << 32) && w->ne[1] < (1 << 24) &&
           ggml_is_contiguous_rows(w) && ix->type == GGML_TYPE_I32 && ggml_is_contiguous(ix) && ix->ne[1] == 1 &&
           ix->ne[2] == 1 && ix->ne[3] == 1 && tsi_nanosim_f32_rows(node, w->ne[0]);
}

static bool tsi_nanosim_get_rows_q8_0_launch(tsi_blob_t blob, struct ggml_tensor * node) {
    const struct ggml_tensor * w = node->src[0], * ix = node->src[1];
    const int64_t k = w->ne[0], n = w->ne[1], srow = tsi_nanosim_q8_srow(k);
    if ((uintptr_t) node->data % TSI_NANOSIM_ALIGNMENT) {
        GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
        return false;
    }
    tsi_nanosim_q8_weights wr, tmp;
    const bool cached = w->buffer && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    tsi_nanosim_q8_weights * e = &tmp;
    if (cached) {
        e = &tsi_nanosim_q8_cache()[w->data];
    }
    if (!e->wq && !tsi_nanosim_q8_repack(w, *e)) {
        return false;
    }
    if (!e->ds) {
        try {
            e->ds = tsi_alloc(n * srow, tsi::MemorySpace::SHARED_DRAM);
        } catch (const std::exception &) {
            return false;
        }
        const float * dw = (const float *) e->dw;
        for (int64_t r = 0; r < n; r++) {
            float * row = (float *) ((char *) e->ds + r * srow);
            for (int64_t b = 0; b < k / 32; b++) {
                row[b] = dw[b * n + r];
            }
        }
    }
    wr = *e;
    tsi_ggml_kargs * a = tsi_nanosim_args();
    memset(a, 0, sizeof(*a));
    a->version = TSI_GGML_KARGS_VERSION;
    a->op      = (uint32_t) GGML_OP_GET_ROWS;
    a->n_src   = 3;
    tsi_nanosim_desc(&a->dst, node);
    tsi_nanosim_raw_desc(&a->src[0], wr.wq, GGML_TYPE_I8, { k, n }, { 1, k });
    tsi_nanosim_desc(&a->src[1], ix);
    if (!tsi_nanosim_stage_for_core(&a->src[1], ix)) {
        if (!cached) tsi_nanosim_q8_free(tmp);
        return false;
    }
    tsi_nanosim_raw_desc(&a->src[2], wr.ds, GGML_TYPE_F32, { k / 32, n }, { 4, srow });
    const bool ok = tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) ix->ne[0]);
    if (!cached) {
        tsi_nanosim_q8_free(tmp);
    }
    return ok;
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
        { GGML_OP_ADD,      "eltwise_f32",  tsi_nanosim_eltwise_supports,  tsi_nanosim_eltwise_launch },
        { GGML_OP_MUL,      "eltwise_f32",  tsi_nanosim_eltwise_supports,  tsi_nanosim_eltwise_launch },
        { GGML_OP_SCALE,    "eltwise_f32",  tsi_nanosim_eltwise_supports,  tsi_nanosim_eltwise_launch },
        { GGML_OP_GLU,      "eltwise_f32",  tsi_nanosim_eltwise_supports,  tsi_nanosim_eltwise_launch },
        { GGML_OP_MUL_MAT,  "mul_mat_q8_0", tsi_nanosim_mul_mat_q8_0_supports, tsi_nanosim_mul_mat_q8_0_launch },
        { GGML_OP_MUL_MAT,  "mul_mat_f16",  tsi_nanosim_mul_mat_f16_supports,  tsi_nanosim_mul_mat_f16_launch },
        { GGML_OP_CONT,     "copy_rows",    tsi_nanosim_copy_supports,     tsi_nanosim_copy_launch },
        { GGML_OP_CPY,      "copy_rows",    tsi_nanosim_copy_supports,     tsi_nanosim_copy_launch },
        { GGML_OP_GET_ROWS, "get_rows_q8_0", tsi_nanosim_get_rows_q8_0_supports, tsi_nanosim_get_rows_q8_0_launch },
        { GGML_OP_GET_ROWS, "copy_rows",    tsi_nanosim_copy_supports,     tsi_nanosim_copy_launch },
        { GGML_OP_SET_ROWS, "copy_rows",    tsi_nanosim_copy_supports,     tsi_nanosim_copy_launch },
        { GGML_OP_SOFT_MAX, "soft_max_f32", tsi_nanosim_soft_max_supports, tsi_nanosim_soft_max_launch },
        { GGML_OP_ROPE,     "rope_f32",     tsi_nanosim_rope_supports,     tsi_nanosim_rope_launch },
    };
    return kernels;
}

// GGML_TSAVORITE_SKIP_OPS: comma-separated ggml op names (ggml_op_name, e.g. GLU,MUL) kept on the CPU, for triage.
static bool tsi_nanosim_skipped(enum ggml_op op) {
    static const std::string skip = [] {
        const char * v = getenv("GGML_TSAVORITE_SKIP_OPS");
        return v ? "," + std::string(v) + "," : std::string();
    }();
    return !skip.empty() && skip.find("," + std::string(ggml_op_name(op)) + ",") != std::string::npos;
}

static const tsi_nanosim_kernel * tsi_nanosim_find_kernel(const struct ggml_tensor * node) {
    if (tsi_nanosim_skipped(node->op)) {
        return nullptr;
    }
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
