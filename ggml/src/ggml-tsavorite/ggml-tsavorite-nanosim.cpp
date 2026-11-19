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

// This tree's ggml_log_internal (ggml/src/ggml.c) passes only GGML_LOG_LEVEL_TSAVORITE messages to the log callback
// and drops the rest, so the backend's errors (allocation failures, launch failures) go to stderr directly.
#undef GGML_LOG_ERROR
#define GGML_LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
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

// What each launch was, by its session command ID, so a fault (reported by command ID on the event callback) can be
// tied to its node: a ring of the most recent launches.
struct tsi_nanosim_cmd_record {
    tsi_cmd_id_t cmd = TSI_CMD_INVALID;
    const char * blob = "";
    char         node[64] = {};
};
static tsi_nanosim_cmd_record tsi_nanosim_cmds[8192];
static const char * tsi_nanosim_cur_node = "";
static const char * tsi_nanosim_cur_op   = "";

// GGML_TSAVORITE_STATS=1: launches by kernel and ggml op, printed at exit.
static std::map<std::string, uint64_t> & tsi_nanosim_stats() {
    static std::map<std::string, uint64_t> m;
    return m;
}

static void tsi_nanosim_print_stats() {
    uint64_t total = 0;
    for (const auto & kv : tsi_nanosim_stats()) {
        total += kv.second;
    }
    fprintf(stderr, "ggml-tsavorite: %llu launches\n", (unsigned long long) total);
    for (const auto & kv : tsi_nanosim_stats()) {
        fprintf(stderr, "ggml-tsavorite:   %8llu  %s\n", (unsigned long long) kv.second, kv.first.c_str());
    }
}

// The event callback runs on a runtime thread, and a launch can fault before tsi_nanosim_launch has recorded it: the
// records are published and read under this mutex. A fault on a command not yet recorded while a launch is being
// enqueued is kept until that launch is recorded (and printed then either way); other commands (blob loads) are not
// recorded and their faults print at once.
static std::mutex   tsi_nanosim_cmds_mutex;
static bool         tsi_nanosim_enqueuing = false;
static tsi_cmd_id_t tsi_nanosim_early_fault = TSI_CMD_INVALID;
static int          tsi_nanosim_early_status = 0;

static void tsi_nanosim_print_fault(int status, tsi_cmd_id_t cmd, const tsi_nanosim_cmd_record * r) {
    fprintf(stderr, "ggml-tsavorite: queue fault, status %d, command %llu: %s\n", status, (unsigned long long) cmd,
            r ? r->node : "(not a recorded launch)");
    if (r) {
        fprintf(stderr, "ggml-tsavorite: faulting launch: blob %s\n", r->blob);
    }
}

static void tsi_nanosim_on_event(const tsi_event_t * ev, void *) {
    if (!ev || ev->status == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(tsi_nanosim_cmds_mutex);
    const tsi_nanosim_cmd_record & r = tsi_nanosim_cmds[ev->cmd_id % 8192];
    if (tsi_nanosim_enqueuing && ev->cmd_id != TSI_CMD_INVALID && r.cmd != ev->cmd_id &&
        tsi_nanosim_early_fault == TSI_CMD_INVALID) {   // possibly the launch being enqueued: recorded shortly
        tsi_nanosim_early_fault  = ev->cmd_id;
        tsi_nanosim_early_status = (int) ev->status;
        return;
    }
    const bool recorded = ev->cmd_id != TSI_CMD_INVALID && r.cmd == ev->cmd_id;
    tsi_nanosim_print_fault((int) ev->status, ev->cmd_id, recorded ? &r : nullptr);
}

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
    desc.size     = sizeof(desc);
    desc.on_event = tsi_nanosim_on_event;
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
static bool tsi_nanosim_wait();

// The blob of a kernel, loaded on first use. A blob load is a device command of its own: issued while a launch is in
// flight it took the place of that launch's response on device 0 and the runtime failed the launch without a message
// (Qwen3-0.6B with GGML_TSAVORITE_ENQUEUE_GRAPH, 2026-11-14), so the queue is drained first.
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
    if (!tsi_nanosim_wait()) {
        return 0;
    }
    tsi_blob_t blob = tsi_blob_load(rt.session, name, path.c_str());
    if (!blob) {
        GGML_LOG_ERROR("%s: tsi_blob_load(%s) failed, status %d\n", __func__, path.c_str(), (int) tsi_last_status());
        return 0;
    }
    rt.blobs[name] = blob;
    return blob;
}

// GGML_TSAVORITE_ENQUEUE_GRAPH=1: launches are enqueued and not waited for; graph_compute enqueues a whole graph and
// waits once at its end. This needs the launches of the one queue to run in queue order, as they do on one TXE (the
// FPGA configuration: one TXE, four host CPUs): a launch may then read what the launches before it wrote, and a
// scratch buffer may be reused by the next launch without a wait. The host waits (tsi_nanosim_wait)
// before it frees, reallocates or reads memory a pending launch may use, before an argument slot comes round, and
// before it loads a blob (tsi_nanosim_blob). The runtime keeps a queue's commands in order across TXEs too (it
// dispatches a command once the previous one has responded: runtime TsavRTContext.cpp). Unset (the default): every
// launch is waited for.
static uint32_t tsi_nanosim_pending = 0;      // launches enqueued since the last wait
static uint32_t tsi_nanosim_slots_used = 0;   // argument slots handed out since the last wait

// The counts are cleared only once the wait succeeds: after a failed wait the launches are unresolved, so every later
// wait fails too and no caller reuses or frees memory they may use.
static bool tsi_nanosim_wait() {
    if (tsi_nanosim_pending) {
        TSI_Status st = tsi_wait_queue(tsi_nanosim_rt().queue, TSI_FOREVER);
        if (st != 0) {
            GGML_LOG_ERROR("%s: tsi_wait_queue failed, status %d\n", __func__, (int) st);
            return false;
        }
    }
    tsi_nanosim_pending    = 0;
    tsi_nanosim_slots_used = 0;
    return true;
}

// One launch of a blob as a grid of `cells` cells over shared-DRAM arguments (an argument slot, tsi_nanosim_args).
static bool tsi_nanosim_launch(tsi_blob_t blob, void * args, size_t args_size, uint32_t cells) {
    static const bool sync = !(getenv("GGML_TSAVORITE_ENQUEUE_GRAPH") && atoi(getenv("GGML_TSAVORITE_ENQUEUE_GRAPH")) != 0);
    tsi_nanosim_runtime & rt = tsi_nanosim_rt();
    tsi_spmd_args_t grid = {};
    grid.numGridSizes = 1;
    grid.gridSize[0]  = cells;
    {
        std::lock_guard<std::mutex> lock(tsi_nanosim_cmds_mutex);
        tsi_nanosim_enqueuing = true;
    }
    const tsi_cmd_id_t cmd = tsi_enqueue_blob_launch(rt.queue, blob, tsi_shmem_handle_from_ptr(args), args_size, &grid);
    if (cmd == TSI_CMD_INVALID) {
        std::lock_guard<std::mutex> lock(tsi_nanosim_cmds_mutex);
        tsi_nanosim_enqueuing = false;
        if (tsi_nanosim_early_fault != TSI_CMD_INVALID) {
            tsi_nanosim_print_fault(tsi_nanosim_early_status, tsi_nanosim_early_fault, nullptr);
            tsi_nanosim_early_fault = TSI_CMD_INVALID;
        }
        GGML_LOG_ERROR("%s: tsi_enqueue_blob_launch failed, status %d\n", __func__, (int) tsi_last_status());
        return false;
    }
    const char * blob_name = "?";
    for (const auto & kv : rt.blobs) {
        if (kv.second == blob) {
            blob_name = kv.first.c_str();
        }
    }
    {
        std::lock_guard<std::mutex> lock(tsi_nanosim_cmds_mutex);
        tsi_nanosim_cmd_record & r = tsi_nanosim_cmds[cmd % 8192];
        r.cmd  = cmd;
        r.blob = blob_name;
        snprintf(r.node, sizeof(r.node), "%s (%u cells)", tsi_nanosim_cur_node, cells);
        tsi_nanosim_enqueuing = false;
        if (tsi_nanosim_early_fault != TSI_CMD_INVALID) {
            tsi_nanosim_print_fault(tsi_nanosim_early_status, tsi_nanosim_early_fault,
                                    tsi_nanosim_early_fault == cmd ? &r : nullptr);
            tsi_nanosim_early_fault = TSI_CMD_INVALID;
        }
    }
    static const bool stats = getenv("GGML_TSAVORITE_STATS") && atoi(getenv("GGML_TSAVORITE_STATS")) != 0;
    if (stats) {
        static bool registered = false;
        if (!registered) {   // the map first: it must outlive the handler
            registered = true;
            tsi_nanosim_stats();
            atexit(tsi_nanosim_print_stats);
        }
        tsi_nanosim_stats()[std::string(blob_name) + " " + tsi_nanosim_cur_op]++;
    }
    tsi_nanosim_pending++;
    return !sync || tsi_nanosim_wait();
}

// ---- kernel arguments ----

static_assert(sizeof(tsi_ggml_kargs) % 8 == 0, "the grid tail follows the argument block");
static_assert(GGML_MAX_OP_PARAMS == TSI_GGML_OP_PARAMS * sizeof(int32_t), "op_params size");

// Argument blocks in shared DRAM, one per launch: a ring of slots (a pending launch reads its block when it runs).
// When the ring comes round to a slot whose launch may still be pending, wait. graph_compute runs on one thread.
#define TSI_NANOSIM_ARG_SLOTS 4096u
#define TSI_NANOSIM_ARG_STRIDE 640u   // sizeof(tsi_ggml_kargs) with room for the grid tail, 128-byte multiple
static_assert(sizeof(tsi_ggml_kargs) + 32 <= TSI_NANOSIM_ARG_STRIDE, "argument slot size");

static tsi_ggml_kargs * tsi_nanosim_args() {
    static char *   ring = nullptr;
    static uint32_t next = 0;
    if (!ring) {
        try {
            ring = (char *) tsi_alloc((int64_t) TSI_NANOSIM_ARG_SLOTS * TSI_NANOSIM_ARG_STRIDE, tsi::MemorySpace::SHARED_DRAM);
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: tsi_alloc failed: %s\n", __func__, e.what());
            return nullptr;
        }
    }
    if (tsi_nanosim_slots_used == TSI_NANOSIM_ARG_SLOTS && !tsi_nanosim_wait()) {   // the ring came round
        return nullptr;
    }
    tsi_nanosim_slots_used++;
    tsi_ggml_kargs * a = (tsi_ggml_kargs *) (ring + (size_t) (next++ % TSI_NANOSIM_ARG_SLOTS) * TSI_NANOSIM_ARG_STRIDE);
    return a;
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

// The data address a launch will see is a multiple of align: buffers are TSI_NANOSIM_ALIGNMENT aligned, so only a
// view's offset (from its root tensor) can break it. supports_op must accept nothing that a launch rejects.
static bool tsi_nanosim_offs_aligned(const struct ggml_tensor * t, size_t align = TSI_NANOSIM_ALIGNMENT) {
    return !(t->view_src && t->view_offs % align);
}

// Every row start 128-byte aligned: the data offset (for a view; buffers are 128-byte aligned) and nb[1..3].
static bool tsi_nanosim_rows_aligned(const struct ggml_tensor * t) {
    if (!tsi_nanosim_offs_aligned(t)) {
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
           src->ne[0] < (1 << 23) && ggml_nrows(src) <= INT32_MAX && tsi_nanosim_rows_aligned(src) &&
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
    if (!tsi_nanosim_offs_aligned(t)) {
        return false;
    }
    for (int d = 1; d < 4; d++) {
        if (t->ne[d] > 1 && t->nb[d] % TSI_NANOSIM_ALIGNMENT) {
            return false;
        }
    }
    return true;
}

// f32 with byte strides a multiple of 4 (the element gathers; MUL's b with ne0 = 1).
static bool tsi_nanosim_f32_any(const struct ggml_tensor * t) {
    if (!t || t->type != GGML_TYPE_F32 || ggml_nbytes(t) >= (1ull << 31)) {
        return false;
    }
    for (int d = 0; d < 4; d++) {
        if (t->nb[d] % sizeof(float)) {
            return false;
        }
    }
    return true;
}

// ADD, MUL, SCALE, GLU (SWIGLU, two inputs), UNARY SIGMOID, f32 (ggml-tsi-kernel nanosim/eltwise_f32.c). The second
// input of ADD and MUL may broadcast along dimensions 1..3, and MUL's along dimension 0 too.
static bool tsi_nanosim_eltwise_supports(const struct ggml_tensor * node) {
    const int64_t ne0 = node->ne[0];
    const struct ggml_tensor * a = node->src[0], * b = node->src[1];
    if (!tsi_nanosim_f32_rows(node, ne0) || !tsi_nanosim_f32_rows(a, ne0) || !ggml_are_same_shape(a, node)) {
        return false;
    }
    switch (node->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            // MUL also takes b with ne0 = 1 at any 4-byte aligned address (the MoE expert weights)
            if (!(node->op == GGML_OP_MUL && b && b->ne[0] == 1 && tsi_nanosim_f32_any(b) && ne0 > 1) &&
                !tsi_nanosim_f32_rows(b, ne0)) {
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
        case GGML_OP_UNARY:
            return ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID;
        case GGML_OP_GLU:
            return ggml_get_glu_op(node) == GGML_GLU_OP_SWIGLU && tsi_nanosim_f32_rows(b, ne0) &&
                   ggml_are_same_shape(b, node);
        default:
            return false;
    }
}

static bool tsi_nanosim_eltwise_launch(tsi_blob_t blob, struct ggml_tensor * node) {
    const int n_src = node->op == GGML_OP_SCALE || node->op == GGML_OP_UNARY ? 1 : 2;
    if (n_src == 2 && node->src[1]->ne[0] == 1 && node->ne[0] > 1) {   // b's element is read from its line
        if ((uintptr_t) node->data % TSI_NANOSIM_ALIGNMENT || (uintptr_t) node->src[0]->data % TSI_NANOSIM_ALIGNMENT) {
            GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
            return false;
        }
        tsi_ggml_kargs * a = tsi_nanosim_pack(node, 2);
        return a && tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) ggml_nrows(node));
    }
    return tsi_nanosim_launch_rows(blob, node, n_src);
}

// ---- row copies (ggml-tsi-kernel nanosim/copy_rows.c) ----

// The copy kernel's (and soft_max's mask) alignment: f32 rows in whole 128-byte lines, f16 rows in half lines.
static size_t tsi_nanosim_copy_align(enum ggml_type type) {
    return type == GGML_TYPE_F16 ? TSI_NANOSIM_ALIGNMENT / 2 : TSI_NANOSIM_ALIGNMENT;
}

// A destination the copy kernel writes whole rows of: f32 (128-byte aligned rows) or f16 (64-byte aligned rows).
static bool tsi_nanosim_copy_dst(const struct ggml_tensor * t, int64_t ne0) {
    if (!t || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) || t->ne[0] != ne0 ||
        t->nb[0] != ggml_type_size(t->type)) {
        return false;
    }
    const size_t align = tsi_nanosim_copy_align(t->type);
    if (!tsi_nanosim_offs_aligned(t, align)) {
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
                   ggml_nelements(node) == ggml_nelements(s) &&
                   tsi_nanosim_offs_aligned(node, tsi_nanosim_copy_align(node->type));
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
        fprintf(stderr, "copy_launch %s %s: src %#llx (mod %u: %u) dst %#llx (mod %u: %u) ix %p\n",
                ggml_op_name(node->op), node->name, (unsigned long long) tsi_shmem_handle_from_ptr(s->data),
                TSI_NANOSIM_ALIGNMENT, (unsigned) ((uintptr_t) s->data % TSI_NANOSIM_ALIGNMENT),
                (unsigned long long) tsi_shmem_handle_from_ptr(node->data), TSI_NANOSIM_ALIGNMENT,
                (unsigned) ((uintptr_t) node->data % TSI_NANOSIM_ALIGNMENT), ix ? ix->data : nullptr);
    }
    const bool indexed = node->op == GGML_OP_GET_ROWS || node->op == GGML_OP_SET_ROWS;

    if ((uintptr_t) s->data % TSI_NANOSIM_ALIGNMENT || (uintptr_t) node->data % tsi_nanosim_copy_align(node->type)) {
        GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
        return false;
    }
    tsi_ggml_kargs * a = tsi_nanosim_pack(node, indexed ? 2 : 1);
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
        (m && (uintptr_t) m->data % tsi_nanosim_copy_align(m->type))) {
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
    if (!tsi_nanosim_offs_aligned(w)) {
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
    if ((uintptr_t) node->data % TSI_NANOSIM_ALIGNMENT || (uintptr_t) node->src[0]->data % TSI_NANOSIM_ALIGNMENT) {
        GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
        return false;
    }
    tsi_ggml_kargs * a = tsi_nanosim_pack(node, 2);   // positions: read by the TDU and converted on the TVU, not staged
    if (!a) {
        return false;
    }
    float freq_base;
    memcpy(&freq_base, (const int32_t *) node->op_params + 5, sizeof(float));
    const float theta_scale = powf(freq_base, -2.0f/node->op_params[1]);   // as ggml_compute_forward_rope_flt
    memcpy(&a->op_params[15], &theta_scale, sizeof(float));
    return tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) (node->ne[2] * node->ne[3]));
}

// ---- element gathers (ggml-tsi-kernel nanosim/gather_f32.c): LFM2's short convolution, the MoE router ----

// CONCAT along dim 0 (3-D), SSM_CONV, and CPY/CONT of any f32 source into a contiguous f32 dst; dst element count a
// multiple of 32 and dst 128-byte aligned.
static bool tsi_nanosim_gather_supports(const struct ggml_tensor * node) {
    const struct ggml_tensor * s0 = node->src[0], * s1 = node->src[1];
    if (!tsi_nanosim_f32_any(node) || !ggml_is_contiguous(node) || !tsi_nanosim_offs_aligned(node) ||
        ggml_nelements(node) % 32 || node->ne[3] != 1) {
        return false;
    }
    switch (node->op) {
        case GGML_OP_CONCAT:
            return ggml_get_op_params_i32(node, 0) == 0 && tsi_nanosim_f32_any(s0) && tsi_nanosim_f32_any(s1) &&
                   s0->ne[3] == 1 && s1->ne[3] == 1;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return tsi_nanosim_f32_any(s0) && s0->ne[3] == 1 && ggml_nelements(s0) == ggml_nelements(node);
        case GGML_OP_SSM_CONV:
            return tsi_nanosim_f32_any(s0) && tsi_nanosim_f32_any(s1) && node->ne[0] % 32 == 0 && s1->ne[0] <= 8 &&
                   s0->ne[3] == 1 && node->ne[2] * node->ne[1] * (node->ne[0] / 32) < (1 << 30);
        default:
            return false;
    }
}

// The MoE router's small ops (any element count; a partial last line is merged): GET_ROWS of f32 by i32 indices
// with any strides, SUM_ROWS, CLAMP, DIV (b broadcast where its ne is 1); dst 128-byte aligned (CLAMP is in place:
// its view keeps its source's offset).
static bool tsi_nanosim_gather_small_supports(const struct ggml_tensor * node) {
    const struct ggml_tensor * s0 = node->src[0], * s1 = node->src[1];
    if (!tsi_nanosim_f32_any(node) || !ggml_is_contiguous(node) || !tsi_nanosim_offs_aligned(node) ||
        node->ne[3] != 1 || !tsi_nanosim_f32_any(s0) || s0->ne[3] != 1 || ggml_nelements(node) >= (1 << 30)) {
        return false;
    }
    switch (node->op) {
        case GGML_OP_GET_ROWS:
            return s1 && s1->type == GGML_TYPE_I32 && s1->ne[2] == 1 && s1->ne[3] == 1 && s1->nb[0] % 4 == 0 &&
                   s1->nb[1] % 4 == 0 && node->ne[0] == s0->ne[0] && node->ne[1] == s1->ne[0] &&
                   node->ne[2] == s1->ne[1] && s0->ne[2] == s1->ne[1] && ggml_nbytes(s1) < (1 << 20);
        case GGML_OP_SUM_ROWS:
            return node->ne[0] == 1 && node->ne[1] == s0->ne[1] && node->ne[2] == s0->ne[2] && s0->ne[0] <= 4096;
        case GGML_OP_CLAMP:
            return ggml_are_same_shape(node, s0);
        case GGML_OP_DIV:
            if (!ggml_are_same_shape(node, s0) || !tsi_nanosim_f32_any(s1) || s1->ne[3] != 1) {
                return false;
            }
            for (int d = 0; d < 3; d++) {
                if (s1->ne[d] != node->ne[d] && s1->ne[d] != 1) {
                    return false;
                }
            }
            return true;
        default:
            return false;
    }
}

static bool tsi_nanosim_gather_small_launch(tsi_blob_t blob, struct ggml_tensor * node) {
    if ((uintptr_t) node->data % TSI_NANOSIM_ALIGNMENT) {
        GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
        return false;
    }
    const int n_src = node->op == GGML_OP_GET_ROWS || node->op == GGML_OP_DIV ? 2 : 1;
    tsi_ggml_kargs * a = tsi_nanosim_pack(node, n_src);
    return a && tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) ((ggml_nelements(node) + 31) / 32));
}

// ARGSORT, descending, of f32 rows of 32 (ggml-tsi-kernel nanosim/argsort_f32.c; the MoE router's top-k): one grid
// cell per row; rows contiguous and 128-byte aligned in src and dst.
static bool tsi_nanosim_argsort_supports(const struct ggml_tensor * node) {
    const struct ggml_tensor * s = node->src[0];
    return s && ggml_get_op_params_i32(node, 0) == GGML_SORT_ORDER_DESC && node->type == GGML_TYPE_I32 &&
           ggml_is_contiguous(node) && node->ne[0] == 32 && tsi_nanosim_f32_rows(s, 32) && ggml_is_contiguous(s) &&
           ggml_are_same_shape(s, node) && ggml_nrows(node) < (1 << 24) && tsi_nanosim_offs_aligned(node);
}

static bool tsi_nanosim_argsort_launch(tsi_blob_t blob, struct ggml_tensor * node) {
    return tsi_nanosim_launch_rows(blob, node, 1);
}

static bool tsi_nanosim_gather_launch(tsi_blob_t blob, struct ggml_tensor * node) {
    if ((uintptr_t) node->data % TSI_NANOSIM_ALIGNMENT) {
        GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
        return false;
    }
    tsi_ggml_kargs * a = tsi_nanosim_pack(node, node->src[1] ? 2 : 1);
    return a && tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) (ggml_nelements(node) / 32));
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
        if (r.wq) tsi_dealloc(r.wq, tsi::MemorySpace::SHARED_DRAM);   // nothing launched reads it yet
        r = {};
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

// Pending launches may read the repack: if the wait for them fails, it is kept (leaked), not freed under them.
static void tsi_nanosim_q8_free(tsi_nanosim_q8_weights & r) {
    if ((r.wq || r.dw || r.ds) && !tsi_nanosim_wait()) {
        GGML_LOG_ERROR("%s: the queue did not drain; the Q8_0 repack is not freed\n", __func__);
        r = {};
        return;
    }
    if (r.wq) tsi_dealloc(r.wq, tsi::MemorySpace::SHARED_DRAM);
    if (r.dw) tsi_dealloc(r.dw, tsi::MemorySpace::SHARED_DRAM);
    if (r.ds) tsi_dealloc(r.ds, tsi::MemorySpace::SHARED_DRAM);
    r = {};
}

// Repacks of weights-buffer tensors, by data address, freed with their weights buffer (tsi_nanosim_buffer_free);
// other tensors are repacked for each launch (their contents may change between launches).
static std::unordered_map<const void *, tsi_nanosim_q8_weights> & tsi_nanosim_q8_cache() {
    static std::unordered_map<const void *, tsi_nanosim_q8_weights> cache;
    return cache;
}

// Shared-DRAM scratch that grows as needed: the quantized activations and their scales.
// The activations last quantized into scratch slots 0 and 1, so the next Q8_0 MUL_MAT of the same tensor reuses them
// (Q, K and V share their input, as do gate and up): reset per graph and whenever the slots are written otherwise.
static const struct ggml_tensor * tsi_nanosim_quantized = nullptr;
static bool tsi_nanosim_fuse_on(const char * name);

static void * tsi_nanosim_scratch(int slot, size_t bytes) {
    static void * ptr[2] = { nullptr, nullptr };
    static size_t size[2] = { 0, 0 };
    if (size[slot] < bytes) {
        tsi_nanosim_quantized = nullptr;
        if (ptr[slot] && !tsi_nanosim_wait()) {   // pending launches may still use the old buffer
            return nullptr;
        }
        if (ptr[slot]) tsi_dealloc(ptr[slot], tsi::MemorySpace::SHARED_DRAM);
        ptr[slot]  = nullptr;
        size[slot] = 0;
        try {
            ptr[slot] = tsi_alloc((int64_t) bytes, tsi::MemorySpace::SHARED_DRAM);
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: tsi_alloc(%zu) failed: %s\n", __func__, bytes, e.what());
            return nullptr;
        }
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
    } else {   // the host reads the tensor: wait for the launches that may write it
        if (!tsi_nanosim_wait() || !tsi_nanosim_q8_repack(w, tmp)) return false;
        wr = tmp;
    }
    void * xq = tsi_nanosim_scratch(0, (size_t) (mp * k));
    void * xd = tsi_nanosim_scratch(1, (size_t) (mp * nbl * 128));
    if (!xq || !xd) {
        if (!cached) tsi_nanosim_q8_free(tmp);
        return false;
    }
    // 1. the activations to Q8_0: one grid cell per row, unless the scratch already holds this tensor's
    tsi_ggml_kargs * a = nullptr;
    bool ok = true;
    if (tsi_nanosim_quantized != x || !tsi_nanosim_fuse_on("quant")) {
        a = tsi_nanosim_args();
        if (!a) {
            if (!cached) tsi_nanosim_q8_free(tmp);
            return false;
        }
        memset(a, 0, sizeof(*a));
        a->version = TSI_GGML_KARGS_VERSION;
        a->op      = (uint32_t) GGML_OP_NONE;
        a->n_src   = 2;
        tsi_nanosim_raw_desc(&a->dst, xq, GGML_TYPE_I8, { k, mp }, { 1, k });
        tsi_nanosim_desc(&a->src[0], x);
        tsi_nanosim_raw_desc(&a->src[1], xd, GGML_TYPE_F32, { nbl * 32, mp }, { 4, nbl * 128 });
        ok = tsi_nanosim_launch(qblob, a, sizeof(*a), (uint32_t) m);
        tsi_nanosim_quantized = ok ? x : nullptr;
    }
    // 2. the matmul: one grid cell per 128-column tile, with its own argument slot
    if (ok && (ok = (a = tsi_nanosim_args()) != nullptr)) {
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
    } else if (!tsi_nanosim_wait()) {   // the host reads the tensor: wait for the launches that may write it
        return false;
    }
    if (!e->wq && !tsi_nanosim_q8_repack(w, *e)) {
        return false;
    }
    if (!e->ds) {
        try {
            e->ds = tsi_alloc(n * srow, tsi::MemorySpace::SHARED_DRAM);
        } catch (const std::exception &) {
            e->ds = nullptr;
            if (!cached) tsi_nanosim_q8_free(tmp);
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
    if (!a) {
        if (!cached) tsi_nanosim_q8_free(tmp);
        return false;
    }
    memset(a, 0, sizeof(*a));
    a->version = TSI_GGML_KARGS_VERSION;
    a->op      = (uint32_t) GGML_OP_GET_ROWS;
    a->n_src   = 3;
    tsi_nanosim_desc(&a->dst, node);
    tsi_nanosim_raw_desc(&a->src[0], wr.wq, GGML_TYPE_I8, { k, n }, { 1, k });
    tsi_nanosim_desc(&a->src[1], ix);
    tsi_nanosim_raw_desc(&a->src[2], wr.ds, GGML_TYPE_F32, { k / 32, n }, { 4, srow });
    const bool ok = tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) ix->ne[0]);
    if (!cached) {
        tsi_nanosim_q8_free(tmp);
    }
    return ok;
}

// ---- MUL_MAT_ID, Q8_0 experts (ggml-tsi-kernel nanosim/mul_mat_q8_0.c, MUL_MAT_ID path) ----

// experts [K, N, n_expert] Q8_0 (rows contiguous, experts consecutive), activations f32 [K, ne11, n_tok] with ne11 1
// or n_used, ids i32 [n_used, n_tok], dst f32 [N, n_used, n_tok]; K and N multiples of 128.
static bool tsi_nanosim_mul_mat_id_supports(const struct ggml_tensor * node) {
    const struct ggml_tensor * w = node->src[0], * x = node->src[1], * ids = node->src[2];
    return w && x && ids && w->type == GGML_TYPE_Q8_0 && x->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 &&
           ids->type == GGML_TYPE_I32 && w->ne[3] == 1 && x->ne[3] == 1 && w->ne[0] % 128 == 0 && w->ne[1] % 128 == 0 &&
           x->ne[0] == w->ne[0] && ggml_is_contiguous(w) && (x->ne[1] == 1 || x->ne[1] == ids->ne[0]) &&
           x->ne[2] == ids->ne[1] && node->ne[0] == w->ne[1] && node->ne[1] == ids->ne[0] && node->ne[2] == ids->ne[1] &&
           (uint64_t) w->ne[0] * (uint64_t) w->ne[1] * (uint64_t) w->ne[2] < (1ull << 32) &&
           tsi_nanosim_f32_rows(x, x->ne[0]) && tsi_nanosim_f32_rows(node, node->ne[0]);
}

static bool tsi_nanosim_mul_mat_id_launch(tsi_blob_t blob, struct ggml_tensor * node) {
    const struct ggml_tensor * w = node->src[0], * x = node->src[1], * ids = node->src[2];
    const int64_t k = w->ne[0], ncols = w->ne[1] * w->ne[2], rows = x->ne[1] * x->ne[2], rp = rows + 16, nbl = k / 32;
    if ((uintptr_t) x->data % TSI_NANOSIM_ALIGNMENT || (uintptr_t) node->data % TSI_NANOSIM_ALIGNMENT) {
        GGML_LOG_ERROR("%s: %s: unaligned data\n", __func__, node->name);
        return false;
    }
    tsi_blob_t qblob = tsi_nanosim_blob("quantize_q8_0_f32");
    if (!qblob) {
        return false;
    }
    // the experts as one [K, N * n_expert] matrix: cached for a weights buffer
    struct ggml_tensor w2 = *w;
    w2.ne[1] = ncols;
    w2.ne[2] = 1;
    w2.nb[2] = w2.nb[3] = w->nb[1] * ncols;
    tsi_nanosim_q8_weights wr, tmp;
    const bool cached = w->buffer && ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    tsi_nanosim_q8_weights * e = cached ? &tsi_nanosim_q8_cache()[w->data] : &tmp;
    if (!cached && !tsi_nanosim_wait()) {   // the host reads the tensor: wait for the launches that may write it
        return false;
    }
    if (!e->wq && !tsi_nanosim_q8_repack(&w2, *e)) {
        return false;
    }
    wr = *e;
    void * xq = tsi_nanosim_scratch(0, (size_t) (rp * k));
    void * xd = tsi_nanosim_scratch(1, (size_t) (rp * nbl * 128));
    tsi_ggml_kargs * a = nullptr;
    bool ok = xq && xd && (a = tsi_nanosim_args()) != nullptr;
    tsi_nanosim_quantized = nullptr;   // the slots now hold MUL_MAT_ID's rows
    if (ok) {   // 1. the activation rows to Q8_0
        memset(a, 0, sizeof(*a));
        a->version = TSI_GGML_KARGS_VERSION;
        a->n_src   = 2;
        tsi_nanosim_raw_desc(&a->dst, xq, GGML_TYPE_I8, { k, rp }, { 1, k });
        tsi_nanosim_desc(&a->src[0], x);
        tsi_nanosim_raw_desc(&a->src[1], xd, GGML_TYPE_F32, { nbl * 32, rp }, { 4, nbl * 128 });
        ok = tsi_nanosim_launch(qblob, a, sizeof(*a), (uint32_t) rows);
    }
    if (ok && (ok = (a = tsi_nanosim_args()) != nullptr)) {   // 2. one grid cell per (slot, token, 128-column tile)
        memset(a, 0, sizeof(*a));
        a->version = TSI_GGML_KARGS_VERSION;
        a->op      = (uint32_t) GGML_OP_MUL_MAT_ID;
        a->n_src   = 5;
        tsi_nanosim_desc(&a->dst, node);
        tsi_nanosim_raw_desc(&a->src[0], wr.wq, GGML_TYPE_I8, { k, ncols }, { 1, k });
        tsi_nanosim_raw_desc(&a->src[1], wr.dw, GGML_TYPE_F32, { ncols, nbl }, { 4, ncols * 4 });
        tsi_nanosim_raw_desc(&a->src[2], xq, GGML_TYPE_I8, { k, rp }, { 1, k });
        tsi_nanosim_raw_desc(&a->src[3], xd, GGML_TYPE_F32, { nbl * 32, rp }, { 4, nbl * 128 });
        tsi_nanosim_desc(&a->src[4], ids);
        a->op_params[0] = (int32_t) x->ne[1];
        ok = tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) (node->ne[1] * node->ne[2] * (node->ne[0] / 128)));
    } else if (!(xq && xd)) {
        GGML_LOG_ERROR("%s: %s: no activation scratch (%lld rows of %lld)\n", __func__, node->name, (long long) rp,
                       (long long) k);
    }
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
        { GGML_OP_UNARY,    "eltwise_f32",  tsi_nanosim_eltwise_supports,  tsi_nanosim_eltwise_launch },
        { GGML_OP_MUL_MAT,  "mul_mat_q8_0", tsi_nanosim_mul_mat_q8_0_supports, tsi_nanosim_mul_mat_q8_0_launch },
        { GGML_OP_MUL_MAT,  "mul_mat_f16",  tsi_nanosim_mul_mat_f16_supports,  tsi_nanosim_mul_mat_f16_launch },
        { GGML_OP_MUL_MAT_ID, "mul_mat_q8_0", tsi_nanosim_mul_mat_id_supports, tsi_nanosim_mul_mat_id_launch },
        { GGML_OP_CONT,     "copy_rows",    tsi_nanosim_copy_supports,     tsi_nanosim_copy_launch },
        { GGML_OP_CPY,      "copy_rows",    tsi_nanosim_copy_supports,     tsi_nanosim_copy_launch },
        { GGML_OP_GET_ROWS, "get_rows_q8_0", tsi_nanosim_get_rows_q8_0_supports, tsi_nanosim_get_rows_q8_0_launch },
        { GGML_OP_GET_ROWS, "copy_rows",    tsi_nanosim_copy_supports,     tsi_nanosim_copy_launch },
        { GGML_OP_SET_ROWS, "copy_rows",    tsi_nanosim_copy_supports,     tsi_nanosim_copy_launch },
        { GGML_OP_SOFT_MAX, "soft_max_f32", tsi_nanosim_soft_max_supports, tsi_nanosim_soft_max_launch },
        { GGML_OP_ROPE,     "rope_f32",     tsi_nanosim_rope_supports,     tsi_nanosim_rope_launch },
        { GGML_OP_CONCAT,   "gather_f32",   tsi_nanosim_gather_supports,   tsi_nanosim_gather_launch },
        { GGML_OP_SSM_CONV, "gather_f32",   tsi_nanosim_gather_supports,   tsi_nanosim_gather_launch },
        { GGML_OP_CPY,      "gather_f32",   tsi_nanosim_gather_supports,   tsi_nanosim_gather_launch },   // after copy_rows
        { GGML_OP_CONT,     "gather_f32",   tsi_nanosim_gather_supports,   tsi_nanosim_gather_launch },
        { GGML_OP_GET_ROWS, "gather_f32",   tsi_nanosim_gather_small_supports, tsi_nanosim_gather_small_launch },   // after copy_rows
        { GGML_OP_SUM_ROWS, "gather_f32",   tsi_nanosim_gather_small_supports, tsi_nanosim_gather_small_launch },
        { GGML_OP_CLAMP,    "gather_f32",   tsi_nanosim_gather_small_supports, tsi_nanosim_gather_small_launch },
        { GGML_OP_DIV,      "gather_f32",   tsi_nanosim_gather_small_supports, tsi_nanosim_gather_small_launch },
        { GGML_OP_ARGSORT,  "argsort_f32",  tsi_nanosim_argsort_supports,  tsi_nanosim_argsort_launch },
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

// Pending launches may use the buffer, and the Q8_0 repacks of its weights: wait for them, then free both. If the
// wait fails they are kept (leaked) rather than freed under a launch.
static void tsi_nanosim_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (tsi_nanosim_buffer_context *) buffer->context;
    const char * b0 = (const char *) ctx->base, * b1 = b0 + ctx->size;
    if (!tsi_nanosim_wait()) {
        GGML_LOG_ERROR("%s: the queue did not drain; the buffer is not freed\n", __func__);
        delete ctx;
        return;
    }
    auto & cache = tsi_nanosim_q8_cache();
    for (auto it = cache.begin(); it != cache.end();) {
        if ((const char *) it->first >= b0 && (const char *) it->first < b1) {
            tsi_nanosim_q8_free(it->second);
            it = cache.erase(it);
        } else {
            ++it;
        }
    }
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

// ---- fusion: several ggml nodes in one launch (GGML_TSAVORITE_NO_FUSE=1 turns it off) ----

// GGML_TSAVORITE_FUSE_OFF=rms,add,kqv,quant,normquant,gluquant,normrope turns single fusions off (triage); GGML_TSAVORITE_NO_FUSE=1 all of them.
static bool tsi_nanosim_fuse_on(const char * name) {
    static const std::string off = [] {
        const char * v = getenv("GGML_TSAVORITE_FUSE_OFF");
        return v ? "," + std::string(v) + "," : std::string();
    }();
    static const bool none = getenv("GGML_TSAVORITE_NO_FUSE") && atoi(getenv("GGML_TSAVORITE_NO_FUSE")) != 0;
    return !none && off.find("," + std::string(name) + ",") == std::string::npos;
}

// A fused launch reads its inputs while it writes outputs that, unfused, are written only after those inputs are dead.
// The allocator may then have placed an output over an input's memory at another offset (it reuses dead memory), and
// writing output row r overwrites input rows still to be read (LFM2-8B-A1B with a compute buffer sized for 64 tokens,
// 2026-11-16; the KQV fusion has the same hazard). Exact aliasing (same data, same shape and strides) is safe for the
// row-wise kernels here: each line is read before it is written. Any other overlap, among outputs too, is not.
static bool tsi_nanosim_fused_disjoint(std::initializer_list<const struct ggml_tensor *> outs,
                                       std::initializer_list<const struct ggml_tensor *> ins) {
    const auto overlap = [](const struct ggml_tensor * a, const struct ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * a1 = a0 + ggml_nbytes(a);
        const char * b0 = (const char *) b->data, * b1 = b0 + ggml_nbytes(b);
        if (!(a0 < b1 && b0 < a1)) {
            return false;
        }
        return !(a->data == b->data && ggml_are_same_shape(a, b) && ggml_are_same_stride(a, b));
    };
    for (auto o = outs.begin(); o != outs.end(); ++o) {
        for (const struct ggml_tensor * in : ins) {
            if (overlap(*o, in)) {
                if (getenv("GGML_TSAVORITE_DEBUG")) {
                    fprintf(stderr, "ggml-tsavorite: not fused: %s overlaps input %s\n", (*o)->name, in->name);
                }
                return false;
            }
        }
        for (auto o2 = o + 1; o2 != outs.end(); ++o2) {
            if (overlap(*o, *o2)) {
                if (getenv("GGML_TSAVORITE_DEBUG")) {
                    fprintf(stderr, "ggml-tsavorite: not fused: %s overlaps output %s\n", (*o)->name, (*o2)->name);
                }
                return false;
            }
        }
    }
    return true;
}

// The norm's output y feeds a Q8_0 MUL_MAT soon after (Q/K/V, gate/up): the fused norm launch also writes y's Q8_0
// activations into the matmul's scratch (rms_norm_f32 quantizes each output line as quantize_q8_0_f32 does), and the
// MUL_MAT finds them there (tsi_nanosim_quantized) and skips its quantize launch.
static bool tsi_nanosim_norm_quant_args(struct ggml_cgraph * cgraph, int next, const struct ggml_tensor * y,
                                        tsi_ggml_kargs * a, const char * fusion = "normquant") {
    if (!tsi_nanosim_fuse_on(fusion) || y->ne[2] != 1 || y->ne[3] != 1 || !ggml_is_contiguous(y) ||
        y->ne[0] % 128) {
        return false;
    }
    bool used = false;
    for (int j = next; j < cgraph->n_nodes && j < next + 16 && !used; j++) {
        const struct ggml_tensor * n = cgraph->nodes[j];
        used = n->op == GGML_OP_MUL_MAT && n->src[1] == y && n->src[0]->type == GGML_TYPE_Q8_0 &&
               tsi_nanosim_mul_mat_q8_0_supports(n);
    }
    if (!used) {
        return false;
    }
    const int64_t k = y->ne[0], m = y->ne[1], mp = (m + 15) / 16 * 16, nbl = k / 32;
    void * xq = tsi_nanosim_scratch(0, (size_t) (mp * k));   // the sizes the MUL_MAT asks for: no reallocation there
    void * xd = tsi_nanosim_scratch(1, (size_t) (mp * nbl * 128));
    if (!xq || !xd) {
        return false;
    }
    tsi_nanosim_raw_desc(&a->src[4], xq, GGML_TYPE_I8, { k, mp }, { 1, k });
    const uint64_t h = (uint64_t) tsi_shmem_handle_from_ptr(xd);
    a->n_src         = 5;
    a->op_params[8]  = 1;
    a->op_params[10] = (int32_t) (uint32_t) h;
    a->op_params[11] = (int32_t) (uint32_t) (h >> 32);
    a->op_params[12] = (int32_t) (nbl * 128);
    return true;
}

// RMS_NORM followed by MUL by the norm weight (every norm of Qwen3 and LFM2): one rms_norm_f32 launch with the weight
// as src[1]; the kernel rounds as the CPU's pair does. Returns the number of nodes it covered (0: not fused).
static int tsi_nanosim_fuse_rms_norm_mul(struct ggml_cgraph * cgraph, int i) {
    static const enum ggml_op ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL };
    if (!ggml_can_fuse(cgraph, i, ops, 2) || tsi_nanosim_skipped(GGML_OP_RMS_NORM) || tsi_nanosim_skipped(GGML_OP_MUL)) {
        return 0;
    }
    struct ggml_tensor * norm = cgraph->nodes[i], * mul = cgraph->nodes[i + 1];
    const struct ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    if (!tsi_nanosim_rms_norm_supports(norm) || !tsi_nanosim_f32_rows(mul, mul->ne[0]) ||
        !tsi_nanosim_f32_rows(w, norm->ne[0])) {
        return 0;
    }
    for (int d = 1; d < 4; d++) {
        if (w->ne[d] != norm->ne[d] && w->ne[d] != 1) {
            return 0;
        }
    }
    if ((uintptr_t) mul->data % TSI_NANOSIM_ALIGNMENT || (uintptr_t) norm->src[0]->data % TSI_NANOSIM_ALIGNMENT ||
        (uintptr_t) w->data % TSI_NANOSIM_ALIGNMENT || !tsi_nanosim_fused_disjoint({ mul }, { norm->src[0], w })) {
        return 0;
    }
    tsi_blob_t blob = tsi_nanosim_blob("rms_norm_f32");
    tsi_ggml_kargs * a = blob ? tsi_nanosim_args() : nullptr;
    if (!a) {
        return -1;
    }
    memset(a, 0, sizeof(*a));
    a->version = TSI_GGML_KARGS_VERSION;
    a->op      = (uint32_t) GGML_OP_RMS_NORM;
    a->n_src   = 2;
    tsi_nanosim_desc(&a->dst, mul);
    tsi_nanosim_desc(&a->src[0], norm->src[0]);
    tsi_nanosim_desc(&a->src[1], w);
    memcpy(a->op_params, norm->op_params, sizeof(a->op_params));
    const bool quant = tsi_nanosim_norm_quant_args(cgraph, i + 2, mul, a);
    if (!tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) ggml_nrows(norm))) {
        return -1;
    }
    if (quant) {
        tsi_nanosim_quantized = mul;
    }
    return 2;
}

// ADD (the residual), then RMS_NORM and MUL of its sum (every layer's second and later norms): one rms_norm_f32
// launch that also stores the sum, which the next residual ADD reads. The ADD may have other users; the norm only the
// MUL.
static int tsi_nanosim_fuse_add_rms_norm_mul(struct ggml_cgraph * cgraph, int i) {
    if (i + 2 >= cgraph->n_nodes || tsi_nanosim_skipped(GGML_OP_ADD) || tsi_nanosim_skipped(GGML_OP_RMS_NORM) ||
        tsi_nanosim_skipped(GGML_OP_MUL)) {
        return 0;
    }
    struct ggml_tensor * add = cgraph->nodes[i], * norm = cgraph->nodes[i + 1], * mul = cgraph->nodes[i + 2];
    if (add->op != GGML_OP_ADD || norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || norm->src[0] != add ||
        (mul->src[0] != norm && mul->src[1] != norm) || !ggml_node_has_n_uses(cgraph, i + 1, 1) || add->view_src) {
        return 0;
    }
    const struct ggml_tensor * x = add->src[0], * b = add->src[1];
    const struct ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    if (!tsi_nanosim_rms_norm_supports(norm) || !ggml_are_same_shape(x, add) || !ggml_are_same_shape(b, add) ||
        !ggml_are_same_shape(mul, add) || !tsi_nanosim_f32_rows(x, add->ne[0]) || !tsi_nanosim_f32_rows(b, add->ne[0]) ||
        !tsi_nanosim_f32_rows(add, add->ne[0]) || !tsi_nanosim_f32_rows(mul, add->ne[0]) ||
        !tsi_nanosim_f32_rows(w, add->ne[0])) {
        return 0;
    }
    for (int d = 1; d < 4; d++) {
        if (w->ne[d] != add->ne[d] && w->ne[d] != 1) {
            return 0;
        }
    }
    for (const struct ggml_tensor * t : { x, b, (const struct ggml_tensor *) add, (const struct ggml_tensor *) mul, w }) {
        if ((uintptr_t) t->data % TSI_NANOSIM_ALIGNMENT) {
            return 0;
        }
    }
    if (!tsi_nanosim_fused_disjoint({ add, mul }, { x, b, w })) {
        return 0;
    }
    tsi_blob_t blob = tsi_nanosim_blob("rms_norm_f32");
    tsi_ggml_kargs * a = blob ? tsi_nanosim_args() : nullptr;
    if (!a) {
        return -1;
    }
    memset(a, 0, sizeof(*a));
    a->version = TSI_GGML_KARGS_VERSION;
    a->op      = (uint32_t) GGML_OP_RMS_NORM;
    a->n_src   = 4;
    tsi_nanosim_desc(&a->dst, mul);
    tsi_nanosim_desc(&a->src[0], x);
    tsi_nanosim_desc(&a->src[1], w);
    tsi_nanosim_desc(&a->src[2], b);
    tsi_nanosim_desc(&a->src[3], add);
    memcpy(a->op_params, norm->op_params, sizeof(a->op_params));
    const bool quant = tsi_nanosim_norm_quant_args(cgraph, i + 3, mul, a);
    if (!tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) ggml_nrows(add))) {
        return -1;
    }
    if (quant) {
        tsi_nanosim_quantized = mul;
    }
    return 3;
}

// MUL_MAT f16 (attention KQV) whose result is PERMUTEd and made CONTiguous (kqv_out): mul_mat_f16 stores each row at
// its destination's strides, so it writes the CONT's tensor directly through the permutation and the copy goes. The
// MUL_MAT's result is then never written, so neither it nor the PERMUTE view may have another user or be an output.
static int tsi_nanosim_fuse_mul_mat_f16_cont(struct ggml_cgraph * cgraph, int i) {
    if (i + 2 >= cgraph->n_nodes || tsi_nanosim_skipped(GGML_OP_MUL_MAT) || tsi_nanosim_skipped(GGML_OP_CONT)) {
        return 0;
    }
    struct ggml_tensor * mm = cgraph->nodes[i], * perm = cgraph->nodes[i + 1], * cont = cgraph->nodes[i + 2];
    if (mm->op != GGML_OP_MUL_MAT || perm->op != GGML_OP_PERMUTE || cont->op != GGML_OP_CONT || perm->src[0] != mm ||
        cont->src[0] != perm || !ggml_node_has_n_uses(cgraph, i, 1) || ggml_node_get_use_count(cgraph, i + 1) != 1 ||
        (perm->flags & GGML_TENSOR_FLAG_OUTPUT) || mm->type != GGML_TYPE_F32 ||
        cont->type != GGML_TYPE_F32 || !ggml_is_contiguous(cont) || !tsi_nanosim_mul_mat_f16_supports(mm)) {
        return 0;
    }
    // ggml_permute: dimension d of mm becomes dimension op_params[d] of the view. The CONT stores the view's elements
    // contiguously in the view's order, whatever shape it is given (llama uses ggml_cont_2d), so the strides come from
    // the view's sizes.
    if (ggml_nelements(cont) != ggml_nelements(perm)) {
        return 0;
    }
    size_t vs[4];
    vs[0] = sizeof(float);
    for (int k = 1; k < 4; k++) {
        vs[k] = vs[k - 1] * (size_t) perm->ne[k - 1];
    }
    struct ggml_tensor m2 = *mm;
    m2.data     = cont->data;
    m2.view_src = nullptr;
    for (int d = 0; d < 4; d++) {
        const int axis = ggml_get_op_params_i32(perm, d);
        if (axis < 0 || axis > 3) {
            return 0;
        }
        m2.nb[d] = vs[axis];
    }
    if (m2.nb[0] != sizeof(float) || !tsi_nanosim_f32_rows(&m2, m2.ne[0]) || (uintptr_t) cont->data % TSI_NANOSIM_ALIGNMENT) {
        return 0;
    }
    // The allocator may give the CONT the memory of the KQ softmax (dead after the MUL_MAT, unfused): writing the
    // CONT's tensor while the MUL_MAT still reads its inputs would corrupt them (LFM2-1.2B, 2026-11-14).
    const char * c0 = (const char *) cont->data, * c1 = c0 + ggml_nbytes(cont);
    for (int k = 0; k < 2; k++) {
        const struct ggml_tensor * src = mm->src[k];
        const char * s0 = (const char *) src->data, * s1 = s0 + ggml_nbytes(src);
        if (c0 < s1 && s0 < c1) {
            return 0;
        }
    }
    tsi_blob_t blob = tsi_nanosim_blob("mul_mat_f16");
    return blob && tsi_nanosim_mul_mat_f16_launch(blob, &m2) ? 3 : -1;
}

// RMS_NORM, MUL by its weight, then ROPE (Qwen3's and LFM2's Q/K norms): one rope_f32 launch that normalizes each head
// row before rotating it (op_params[13] = 1, [14] = eps, src[2] = w); cos/sin stay once per token.
static int tsi_nanosim_fuse_norm_rope(struct ggml_cgraph * cgraph, int i) {
    if (!tsi_nanosim_fuse_on("normrope") || i + 2 >= cgraph->n_nodes || tsi_nanosim_skipped(GGML_OP_RMS_NORM) ||
        tsi_nanosim_skipped(GGML_OP_MUL) || tsi_nanosim_skipped(GGML_OP_ROPE)) {
        return 0;
    }
    struct ggml_tensor * norm = cgraph->nodes[i], * mul = cgraph->nodes[i + 1], * rope = cgraph->nodes[i + 2];
    if (norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE ||
        (mul->src[0] != norm && mul->src[1] != norm) || rope->src[0] != mul || !ggml_node_has_n_uses(cgraph, i, 1) ||
        !ggml_node_has_n_uses(cgraph, i + 1, 1) || !tsi_nanosim_rms_norm_supports(norm) ||
        !tsi_nanosim_rope_supports(rope) || norm->src[0]->ne[0] != rope->op_params[1]) {
        return 0;
    }
    const struct ggml_tensor * x = norm->src[0], * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    if (!tsi_nanosim_f32_rows(w, x->ne[0]) || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1 ||
        !ggml_are_same_shape(x, rope) || !tsi_nanosim_f32_rows(x, x->ne[0]) || (uintptr_t) w->data % TSI_NANOSIM_ALIGNMENT) {
        return 0;
    }
    // the ROPE launch with x in place of its input
    struct ggml_tensor r2 = *rope;
    r2.src[0] = const_cast<struct ggml_tensor *>(x);
    r2.src[2] = const_cast<struct ggml_tensor *>(w);
    tsi_blob_t blob = tsi_nanosim_blob("rope_f32");
    if (!blob || (uintptr_t) r2.data % TSI_NANOSIM_ALIGNMENT || (uintptr_t) x->data % TSI_NANOSIM_ALIGNMENT ||
        !tsi_nanosim_fused_disjoint({ rope }, { x, w })) {
        return blob ? 0 : -1;
    }
    tsi_ggml_kargs * a = tsi_nanosim_pack(&r2, 3);
    if (!a) {
        return -1;
    }
    float freq_base;
    memcpy(&freq_base, (const int32_t *) rope->op_params + 5, sizeof(float));
    const float theta_scale = powf(freq_base, -2.0f/rope->op_params[1]);   // as ggml_compute_forward_rope_flt
    memcpy(&a->op_params[15], &theta_scale, sizeof(float));
    a->op_params[13] = 1;
    a->op_params[14] = norm->op_params[0];   // eps
    return tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) (rope->ne[2] * rope->ne[3])) ? 3 : -1;
}

// SWIGLU whose output feeds a Q8_0 MUL_MAT (the FFN down projection): eltwise_f32 also writes the matmul's Q8_0
// activations, as the fused norms do.
static int tsi_nanosim_fuse_swiglu_quant(struct ggml_cgraph * cgraph, int i) {
    struct ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_GLU || tsi_nanosim_skipped(GGML_OP_GLU) || !tsi_nanosim_eltwise_supports(node) ||
        ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU) {
        return 0;
    }
    const struct ggml_tensor * ts[3] = { node, node->src[0], node->src[1] };
    for (const struct ggml_tensor * t : ts) {
        if ((uintptr_t) t->data % TSI_NANOSIM_ALIGNMENT) {
            return 0;
        }
    }
    tsi_blob_t blob = tsi_nanosim_blob("eltwise_f32");
    tsi_ggml_kargs * a = blob ? tsi_nanosim_pack(node, 2) : nullptr;
    if (!a) {
        return -1;
    }
    if (!tsi_nanosim_norm_quant_args(cgraph, i + 1, node, a, "gluquant")) {
        return 0;   // the slot is left unused; the node goes the ordinary way
    }
    if (!tsi_nanosim_launch(blob, a, sizeof(*a), (uint32_t) ggml_nrows(node))) {
        return -1;
    }
    tsi_nanosim_quantized = node;
    return 1;
}

// Every node is enqueued, then the graph is waited for once (a no-op unless GGML_TSAVORITE_ENQUEUE_GRAPH, see
// tsi_nanosim_launch): the CPU backend reads this backend's results in place (host buffers) once graph_compute returns.
static enum ggml_status tsi_nanosim_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    tsi_nanosim_quantized = nullptr;   // tensors of an earlier graph may share addresses with this one's
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (tsi_nanosim_is_noop(node)) {
            continue;
        }
        tsi_nanosim_cur_node     = node->name;
        tsi_nanosim_cur_op       = ggml_op_desc(node);
        if (tsi_nanosim_fuse_on("kqv") && node->op == GGML_OP_MUL_MAT && node->src[0] && node->src[0]->type == GGML_TYPE_F16) {
            tsi_nanosim_cur_op = "MUL_MAT+CONT";
            const int fused = tsi_nanosim_fuse_mul_mat_f16_cont(cgraph, i);
            if (fused < 0) {
                fprintf(stderr, "ggml-tsavorite: mul_mat_f16 MUL_MAT+CONT (%s) failed\n", node->name);
                tsi_nanosim_wait();
                return GGML_STATUS_FAILED;
            }
            if (fused > 0) {
                i += fused - 1;
                continue;
            }
            tsi_nanosim_cur_op = ggml_op_desc(node);
        }
        if (node->op == GGML_OP_GLU) {
            tsi_nanosim_cur_op = "SWIGLU+quant";
            const int fused = tsi_nanosim_fuse_swiglu_quant(cgraph, i);
            if (fused < 0) {
                fprintf(stderr, "ggml-tsavorite: eltwise_f32 SWIGLU+quant (%s) failed\n", node->name);
                tsi_nanosim_wait();
                return GGML_STATUS_FAILED;
            }
            if (fused > 0) {
                continue;
            }
            tsi_nanosim_cur_op = ggml_op_desc(node);
        }
        if (node->op == GGML_OP_RMS_NORM) {
            tsi_nanosim_cur_op = "RMS_NORM+MUL+ROPE";
            const int fused = tsi_nanosim_fuse_norm_rope(cgraph, i);
            if (fused < 0) {
                fprintf(stderr, "ggml-tsavorite: rope_f32 RMS_NORM+MUL+ROPE (%s) failed\n", node->name);
                tsi_nanosim_wait();
                return GGML_STATUS_FAILED;
            }
            if (fused > 0) {
                i += fused - 1;
                continue;
            }
            tsi_nanosim_cur_op = ggml_op_desc(node);
        }
        if ((node->op == GGML_OP_RMS_NORM && tsi_nanosim_fuse_on("rms")) || (node->op == GGML_OP_ADD && tsi_nanosim_fuse_on("add"))) {
            const bool add = node->op == GGML_OP_ADD;
            tsi_nanosim_cur_op = add ? "ADD+RMS_NORM+MUL" : "RMS_NORM+MUL";
            const int fused = add ? tsi_nanosim_fuse_add_rms_norm_mul(cgraph, i) : tsi_nanosim_fuse_rms_norm_mul(cgraph, i);
            if (fused < 0) {
                fprintf(stderr, "ggml-tsavorite: rms_norm_f32 %s (%s) failed\n", tsi_nanosim_cur_op, node->name);
                tsi_nanosim_wait();
                return GGML_STATUS_FAILED;
            }
            if (fused > 0) {
                i += fused - 1;
                continue;
            }
            tsi_nanosim_cur_op = ggml_op_desc(node);
        }
        const tsi_nanosim_kernel * k = tsi_nanosim_find_kernel(node);
        if (!k) {
            GGML_LOG_ERROR("%s: no kernel for %s (%s)\n", __func__, ggml_op_desc(node), node->name);
            return GGML_STATUS_FAILED;
        }
        tsi_blob_t blob = tsi_nanosim_blob(k->blob);
        if (!blob || !k->launch(blob, node)) {
            fprintf(stderr, "ggml-tsavorite: %s %s (%s) failed: ne [%lld, %lld, %lld, %lld]\n", k->blob, ggml_op_desc(node),
                    node->name, (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2],
                    (long long) node->ne[3]);
            tsi_nanosim_wait();
            return GGML_STATUS_FAILED;
        }
    }
    if (!tsi_nanosim_wait()) {
        fprintf(stderr, "ggml-tsavorite: a launch of this graph (%d nodes) failed; without GGML_TSAVORITE_ENQUEUE_GRAPH the failing node is named\n",
                cgraph->n_nodes);
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

static void tsi_nanosim_backend_synchronize(ggml_backend_t backend) {
    tsi_nanosim_wait();
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
    /* .synchronize        = */ tsi_nanosim_backend_synchronize,
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

// Shared DRAM as USER_DRAM_SIZE (MiB) asks for, else the runtime's default for the Xtensa ISS device (256 MiB:
// runtime include/device/posix/PosixSimDeviceConfig.h kDefaultPosixSimUserDramBytes, which XtensaSimDeviceConfig
// inherits); the runtime does not report free memory, so free = total.
static void tsi_nanosim_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    const char * mib = getenv("USER_DRAM_SIZE");
    *total = (size_t) (mib ? strtoull(mib, nullptr, 10) : 256) << 20;
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
    const bool ok = tsi_nanosim_is_noop(op) || tsi_nanosim_find_kernel(op) != nullptr;
    if (!ok && getenv("GGML_TSAVORITE_DEBUG")) {   // what stays on the CPU, and why it does not fit
        fprintf(stderr, "ggml-tsavorite: not supported: %s %s %s [%lld,%lld,%lld,%lld]", ggml_op_desc(op), op->name,
                ggml_type_name(op->type), (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2],
                (long long) op->ne[3]);
        for (int i = 0; i < GGML_MAX_SRC && op->src[i]; i++) {
            const struct ggml_tensor * t = op->src[i];
            fprintf(stderr, " src%d %s [%lld,%lld,%lld,%lld] nb [%zu,%zu,%zu,%zu]%s", i, ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], t->nb[0],
                    t->nb[1], t->nb[2], t->nb[3], t->view_src ? " view" : "");
        }
        fprintf(stderr, "\n");
    }
    return ok;
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
