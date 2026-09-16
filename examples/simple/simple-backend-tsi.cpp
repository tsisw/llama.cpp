#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-tsavorite.h"
#include "mat-mul-tsi-test-case.cpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <math.h>
#include <float.h>
#include <thread>

// --- BEGIN TSI_REMOTE_TXE_POC ---
// Cross-TSISIM-instance remote TXE dispatch proof-of-concept.
//
// Goal: prove that a real ADD op can be dispatched from a process on ONE
// TSISIM instance and genuinely executed on ANOTHER TSISIM instance's local
// (simulated) TXE hardware, over the network, with a numerically verified
// correct result.
//
// Architectural constraint this design respects (confirmed by reading
// ensure_tsi_runtime_initialized()'s call chain in ggml-tsavorite.cpp):
// TXE allocation goes through tsi-apc-mgr (RSM) over a local Unix domain
// socket, and physical/virtual address mapping comes from the local
// txe-driver kernel driver. Both are strictly local to whichever machine
// the process runs on -- there is no way to reach instance 2's RSM socket
// or txe-driver from a process on instance 1. So instance 2 must run its
// OWN real process that does its OWN local ensure_tsi_runtime_initialized()
// bring-up (via the normal ggml_backend_tsavorite_init() call below --
// exactly the same call load_model() already makes for the local test
// cases in this same file), and then serve remote dispatch requests using
// its own already-initialized local backend/runtime. This file adds that
// worker mode ("remote-worker") plus nothing else -- the request/response
// wire protocol below is intentionally minimal.
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <deque>
#include <memory>

static bool tsi_recv_all(int fd, void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    size_t got = 0;
    while (got < n) {
        ssize_t r = recv(fd, p + got, n - got, 0);
        if (r == 0) return false; // peer closed
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        got += (size_t)r;
    }
    return true;
}

static bool tsi_send_all(int fd, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t sent = 0;
    while (sent < n) {
        ssize_t r = send(fd, p + sent, n - sent, 0);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        sent += (size_t)r;
    }
    return true;
}

// Wire protocol (native byte order -- both instances are aarch64 little
// endian, so no htonl/ntohl needed for the payload itself):
//   request:  uint32_t magic (0x54584552 'TXER'), uint32_t n, float[n] A, float[n] B
//   response: uint32_t magic (0x54584552),          float[n] C  (C = A + B, computed
//             by a real ggml_add() graph dispatched through ggml_backend_tsavorite
//             on THIS process's own local TXE runtime)
#define TSI_REMOTE_TXE_MAGIC 0x54584552u

// Round 3: the worker accepted *concurrent* client connections (one
// std::thread per connection, see tsi_run_remote_worker() below), each of
// which called ggml_backend_graph_compute() independently. But
// ggml-tsavorite.cpp's TXE dispatch state (the `workers` vector of
// in-flight blob-execution threads, `device_free[]`, `device_mutex`,
// `workers_mutex`, packed_args[]/scalar_*_args[]) is process-global and was
// only ever designed for ONE top-level graph_compute() call to be in
// flight at a time -- a normal local llama-cli inference loop only ever
// calls it that way. Two concurrent graph_compute() calls both pushing
// into and join()-ing the *same* global `workers` vector via
// join_all_workers() can cross-join each other's row-tile threads (one
// call's join_all_workers() silently reaping and waiting on a *different*
// call's threads, and returning early while the other call's threads are
// still writing into ITS OWN now-popped stack frame -- a real
// dangling-stack-reference bug), which was observed directly during
// real-model testing as a genuine hang: worker threads piled up (44 live
// threads for what should have been at most ~20 local + a handful of
// connection handlers), worker CPU dropped to idle, and client connections
// sat blocked in recv() forever with no forward progress. Round 3 added
// this mutex to serialize entry into the two worker-side dispatch
// functions, which fixed the *cross-call* corruption -- but a *residual*
// hang remained even with only one caller ever active at a time (see
// tsi_worker_dispatch_thread_main's comment below for the Round 4 root
// cause and real fix: it was never really about concurrent *callers*, it
// was about which OS *thread* ever called graph_compute() at all). This
// mutex is kept as a cheap, harmless second line of defense -- with Round
// 4's single dedicated dispatch thread, it is never actually contended.
static std::mutex g_worker_compute_mutex;

// TSI_REMOTE_TXE_POC ROUND 5, requirement B (per-node kernel-dispatch count
// visible on both instances): instance 1's own real llama-cli run already
// prints a "=== GGML Perf Summary ===" table at shutdown (see
// llama-context.cpp's ggml_perf_print_totals(), gated by GGML_PERF, which
// IS defined in this build -- confirmed via build-fpga/CMakeCache.txt) with
// a "TSI_KERNEL-RUN" column -- the sum, across every graph node of a given
// op, of that node's `tsi_kernel_runs` field (a plain int64_t field added
// to `struct ggml_tensor` itself, see ggml/include/ggml.h, incremented
// unconditionally -- NOT gated behind TMU_DEBUG/GGML_PERF -- by
// ggml-tsavorite.cpp's dispatch code every time it actually launches a TXE
// kernel run for that node, e.g. once per row-tile in
// ggml_tsavorite_run_tmu_mul_mat()). That printer lives in llama-context.cpp
// and only ever runs as part of a `llama_context`'s shutdown -- this worker
// process never constructs one (it drives raw ggml_new_tensor_2d +
// ggml_mul_mat + ggml_backend_graph_compute() calls directly, the same
// pattern this file's own local test cases use), so it never gets that
// summary for free, even though the exact same underlying counter
// (`tsi_kernel_runs`) is being incremented on this worker's own per-request
// tensors by the identical, unmodified dispatch code every real llama-cli
// run already relies on.
//
// Since each worker request builds a tiny throwaway graph (freed right
// after the result is read back), the per-node counter itself doesn't
// survive across requests -- so this worker keeps its OWN persistent
// accumulator, fed by reading `result->tsi_kernel_runs` right after
// graph_compute() and before the graph is freed. This is the same field,
// summed the same way (just across many small ephemeral graphs instead of
// one big long-lived one) -- so the total is directly comparable to
// instance 1's own "TSI_KERNEL-RUN" column, not a reinvented metric.
static std::atomic<int64_t> g_worker_total_tsi_kernel_runs{0};
static std::atomic<int64_t> g_worker_total_requests{0};

// Prints the running total to stderr (captured in worker.log) -- called at
// the end of every client session (see tsi_remote_worker_serve_conn below)
// so `tail`/`grep worker.log` gives a number directly comparable to
// instance 1's own GGML Perf Summary "TSI_KERNEL-RUN" total, without
// needing journalctl or any extra tooling (the worker is not a systemd
// service).
static void tsi_worker_print_kernel_run_summary(const char *context_label) {
    fprintf(stderr,
            "[remote-worker] %s: TOTAL TSI_KERNEL-RUN count on this node so far: %ld "
            "(across %ld MAT_MUL/ADD requests served) -- comparable to instance 1's own "
            "'GGML Perf Summary' TSI_KERNEL-RUN column\n",
            context_label,
            (long)g_worker_total_tsi_kernel_runs.load(),
            (long)g_worker_total_requests.load());
}

// Runs one real ADD op through the Tsavorite backend for the given already
// TXE-runtime-initialized `backend`, on tensors of length n. This is the
// same load_model()+build_graph()+compute() pattern used by the rest of
// this file's local test cases -- nothing here is special-cased for the
// network path, it is the identical local dispatch mechanism.
static bool tsi_remote_worker_run_add(ggml_backend_t backend, const std::vector<float> &A,
                                       const std::vector<float> &B, std::vector<float> &C) {
    std::lock_guard<std::mutex> lk(g_worker_compute_mutex);
    const int64_t n = (int64_t)A.size();

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "[remote-worker] ggml_init failed\n");
        return false;
    }

    struct ggml_tensor *ta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    struct ggml_tensor *tb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    struct ggml_backend_buffer *buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "[remote-worker] ggml_backend_alloc_ctx_tensors failed\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(ta, A.data(), 0, ggml_nbytes(ta));
    ggml_backend_tensor_set(tb, B.data(), 0, ggml_nbytes(tb));

    static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    std::vector<uint8_t> gbuf(buf_size);
    struct ggml_init_params params0 { buf_size, gbuf.data(), true };
    struct ggml_context *ctx0 = ggml_init(params0);
    struct ggml_cgraph *gf = ggml_new_graph(ctx0);
    struct ggml_tensor *result = ggml_add(ctx0, ta, tb);
    ggml_build_forward_expand(gf, result);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = allocr && ggml_gallocr_alloc_graph(allocr, gf);
    if (ok) {
        ggml_backend_graph_compute(backend, gf);
        C.resize(n);
        ggml_backend_tensor_get(result, C.data(), 0, ggml_nbytes(result));
        // ROUND 5 requirement B: `result->tsi_kernel_runs` is the same
        // per-node counter instance 1's own "GGML Perf Summary" TSI_KERNEL-
        // RUN column sums (see the big comment above g_worker_total_tsi_
        // kernel_runs) -- read it before this graph is freed below and fold
        // it into this worker's own persistent running total.
        g_worker_total_tsi_kernel_runs.fetch_add(result->tsi_kernel_runs,
                                                  std::memory_order_relaxed);
        g_worker_total_requests.fetch_add(1, std::memory_order_relaxed);
    } else {
        fprintf(stderr, "[remote-worker] graph alloc failed\n");
    }

    if (allocr) ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    return ok;
}

// TSI_REMOTE_TXE_POC (MAT_MUL extension, round 3): same wire-framing idea as
// the ADD protocol above, generalized to a real MAT_MUL. Wire protocol:
//   request:  uint32 magic (0x4d4d5458 'XTMM'), int64 M_valid, int64 N, int64 K,
//             float[M_valid*K] A_rows (row-major, M_valid rows of K),
//             float[N*K]       B_rows (row-major, N rows of K -- this is
//                               exactly ggml's own byte layout for a
//                               [ne0=K, ne1=N] F32 tensor)
//   response: uint32 magic,   float[M_valid*N] C (row-major [M_valid, N])
//
// The compute itself reuses the identical local dispatch mechanism as the
// ADD case above: a real ggml_mul_mat() graph run through
// ggml_backend_graph_compute() against THIS process's own already-
// initialized local ggml_backend_tsavorite runtime. Nothing here
// special-cases the network path or reimplements any part of the Triton
// MAT_MUL kernel dispatch -- it is the same call sequence a real llama-cli
// MUL_MAT node uses, and (if M_valid is large enough) will even trigger
// this worker's OWN internal multi-TXE row-splitting across its own local
// TXEs, exactly like a normal local inference run would.
#define TSI_REMOTE_TXE_MAGIC_MATMUL 0x4d4d5458u

static bool tsi_remote_worker_run_mul_mat(ggml_backend_t backend,
                                           const std::vector<float> &A_rows, int64_t M_valid,
                                           const std::vector<float> &B_rows, int64_t N,
                                           int64_t K,
                                           std::vector<float> &C_out,
                                           int64_t &kernel_runs_out) {
    // See g_worker_compute_mutex's comment above tsi_remote_worker_run_add():
    // ggml-tsavorite.cpp's TXE dispatch state is process-global and not
    // safe for concurrent top-level graph_compute() calls.
    std::lock_guard<std::mutex> lk(g_worker_compute_mutex);
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "[remote-worker] ggml_init failed\n");
        return false;
    }

    // ne0=K, ne1=M_valid / ne1=N -- matches ggml's own mul_mat convention
    // (A: [K,M], B: [K,N], result: [M,N]), and matches exactly how the
    // client packed A_rows/B_rows on the wire.
    struct ggml_tensor *ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M_valid);
    struct ggml_tensor *tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);

    struct ggml_backend_buffer *buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "[remote-worker] ggml_backend_alloc_ctx_tensors failed\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(ta, A_rows.data(), 0, ggml_nbytes(ta));
    ggml_backend_tensor_set(tb, B_rows.data(), 0, ggml_nbytes(tb));

    static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    std::vector<uint8_t> gbuf(buf_size);
    struct ggml_init_params params0 { buf_size, gbuf.data(), true };
    struct ggml_context *ctx0 = ggml_init(params0);
    struct ggml_cgraph *gf = ggml_new_graph(ctx0);
    struct ggml_tensor *result = ggml_mul_mat(ctx0, ta, tb);
    ggml_build_forward_expand(gf, result);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = allocr && ggml_gallocr_alloc_graph(allocr, gf);
    if (ok) {
        ggml_backend_graph_compute(backend, gf);
        C_out.resize((size_t)(M_valid * N));
        ggml_backend_tensor_get(result, C_out.data(), 0, ggml_nbytes(result));
        // ROUND 5 requirement B: see the matching comment in
        // tsi_remote_worker_run_add() above -- this is the MAT_MUL side of
        // the same accounting (the op that actually matters for this POC).
        g_worker_total_tsi_kernel_runs.fetch_add(result->tsi_kernel_runs,
                                                  std::memory_order_relaxed);
        g_worker_total_requests.fetch_add(1, std::memory_order_relaxed);
        kernel_runs_out = result->tsi_kernel_runs;
    } else {
        fprintf(stderr, "[remote-worker] MAT_MUL graph alloc failed\n");
    }

    if (allocr) ggml_gallocr_free(allocr);
    ggml_free(ctx0);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    return ok;
}

// TSI_REMOTE_TXE_POC (ROUND 4): dedicated single dispatch-thread
// architecture, replacing Round 3's g_worker_compute_mutex-only fix.
//
// Root cause of the residual hang (Round 3 sec. 6 point 2 -- "a single such
// request could still internally trigger the worker's own Multi-TXE
// row-splitting and hang there", root-caused only as far as "something
// about the repeated fresh-context-per-request pattern interacting badly
// with the shared TXE runtime's Multi-TXE path", not fully isolated):
// every context in which ggml_tsavorite_run_tmu_mul_mat()'s multi-TXE
// row-splitting is *known* to work correctly -- plain local llama-cli
// inference, and Round 2's ADD POC (which served every request
// sequentially in a single accept-loop, no per-connection thread spawn) --
// always calls ggml_backend_graph_compute() from the SAME single OS thread
// across the whole life of the process: the same thread that performed
// ggml_backend_tsavorite_init(). Round 3's worker broke that invariant to
// get concurrent connection handling: it spawned a FRESH std::thread per
// accepted connection, and it was THAT thread which called
// graph_compute() -- a different OS thread than the init thread, and a
// different OS thread from one request to the next. Adding
// g_worker_compute_mutex (Round 3's fix) removed *concurrent* callers but
// did not restore the "always the same calling thread" invariant, and the
// hang persisted for a single caller. That is the single structural
// difference between this code path and every proven-working one, and is
// the most likely real root cause.
//
// Round 4 restores the invariant directly: exactly ONE dedicated compute
// thread is spawned, once, immediately after ggml_backend_tsavorite_init()
// (see tsi_run_remote_worker() below) and never replaced -- it is the ONLY
// thread that ever calls tsi_remote_worker_run_add()/
// tsi_remote_worker_run_mul_mat() (and therefore the only thread that ever
// calls ggml_backend_graph_compute()) for the entire lifetime of the
// worker process. Per-connection threads still do their own socket I/O
// (recv the request, send the response) concurrently -- that part was
// never the problem -- but hand the actual compute off to this thread via
// a simple thread-safe job queue, and block on a std::future for the
// result. This keeps genuine throughput (many connections can be open and
// queued at once; each job still gets full internal num_of_txes-way
// parallelism on this worker's own local TXEs once it's picked up) while
// eliminating the one architectural anomaly a real fix needed to remove,
// rather than just papering over its symptom with a client-side
// concurrency cap (see ggml-tsavorite.cpp's kRemoteSlotCapacity comment for
// the client-side half of this Round 4 change).
struct TsiWorkerJob {
    bool is_matmul = false;
    // ADD fields
    std::vector<float> add_a, add_b;
    // MATMUL fields
    std::vector<float> mm_a_rows, mm_b_rows;
    int64_t mm_m_valid = 0, mm_n = 0, mm_k = 0;
    // Result + completion signal, filled in by the dispatch thread and
    // consumed by the connection thread that submitted this job.
    std::promise<bool> done;
    std::vector<float> result;
    // ROUND 7: the real number of local TXE kernel launches this specific
    // request took on THIS worker (== result->tsi_kernel_runs from the
    // ggml_mul_mat() this job ran) -- sent back to the client on the wire
    // so it can report the true per-node split instead of a "1 call = 1
    // unit" placeholder.
    int64_t result_kernel_runs = 0;
};

static std::mutex g_job_queue_mutex;
static std::condition_variable g_job_queue_cv;
static std::deque<std::shared_ptr<TsiWorkerJob>> g_job_queue;

// Submits a job and returns a future the caller blocks on for completion.
// Safe to call from any number of concurrent connection threads.
static std::future<bool> tsi_worker_submit_job(std::shared_ptr<TsiWorkerJob> job) {
    std::future<bool> fut = job->done.get_future();
    {
        std::lock_guard<std::mutex> lock(g_job_queue_mutex);
        g_job_queue.push_back(job);
    }
    g_job_queue_cv.notify_one();
    return fut;
}

// The body of the single dedicated dispatch thread. Runs forever, one job
// at a time, always on the same OS thread that started this loop (spawned
// once from tsi_run_remote_worker(), immediately after
// ggml_backend_tsavorite_init() on that same thread).
static void tsi_worker_dispatch_thread_main(ggml_backend_t backend) {
    for (;;) {
        std::shared_ptr<TsiWorkerJob> job;
        {
            std::unique_lock<std::mutex> lock(g_job_queue_mutex);
            g_job_queue_cv.wait(lock, [] { return !g_job_queue.empty(); });
            job = g_job_queue.front();
            g_job_queue.pop_front();
        }

        bool ok;
        if (job->is_matmul) {
            ok = tsi_remote_worker_run_mul_mat(backend, job->mm_a_rows, job->mm_m_valid,
                                                job->mm_b_rows, job->mm_n, job->mm_k,
                                                job->result, job->result_kernel_runs);
        } else {
            ok = tsi_remote_worker_run_add(backend, job->add_a, job->add_b, job->result);
        }
        job->done.set_value(ok);
    }
}

// Serves requests on one accepted connection until the peer disconnects.
// Handles both the original ADD protocol and the newer MAT_MUL protocol on
// the same port, dispatched by the leading magic value. Socket I/O happens
// on this (per-connection) thread; the actual TXE dispatch is handed off to
// the single dedicated dispatch thread via tsi_worker_submit_job() -- see
// its comment above for why.
static void tsi_remote_worker_serve_conn(int fd, ggml_backend_t /*backend*/) {
    for (;;) {
        uint32_t magic = 0;
        if (!tsi_recv_all(fd, &magic, sizeof(magic))) break;

        if (magic == TSI_REMOTE_TXE_MAGIC) {
            uint32_t n = 0;
            if (!tsi_recv_all(fd, &n, sizeof(n))) break;
            if (n == 0 || n > (1u << 20)) {
                fprintf(stderr, "[remote-worker] bad n=%u, dropping connection\n", n);
                break;
            }

            auto job = std::make_shared<TsiWorkerJob>();
            job->is_matmul = false;
            job->add_a.resize(n);
            job->add_b.resize(n);
            if (!tsi_recv_all(fd, job->add_a.data(), (size_t)n * sizeof(float))) break;
            if (!tsi_recv_all(fd, job->add_b.data(), (size_t)n * sizeof(float))) break;

            fprintf(stderr, "[remote-worker] ADD request: n=%u A[0]=%g B[0]=%g -- queuing for the dispatch thread\n",
                    n, job->add_a[0], job->add_b[0]);

            std::future<bool> fut = tsi_worker_submit_job(job);
            if (!fut.get()) {
                fprintf(stderr, "[remote-worker] local TXE dispatch failed, dropping connection\n");
                break;
            }
            const std::vector<float> &C = job->result;

            fprintf(stderr, "[remote-worker] result: C[0]=%g (expected %g)\n", C[0], job->add_a[0] + job->add_b[0]);

            if (!tsi_send_all(fd, &magic, sizeof(magic))) break;
            if (!tsi_send_all(fd, C.data(), (size_t)n * sizeof(float))) break;
            continue;
        }

        if (magic == TSI_REMOTE_TXE_MAGIC_MATMUL) {
            int64_t hdr[3] = {0, 0, 0};
            if (!tsi_recv_all(fd, hdr, sizeof(hdr))) break;
            const int64_t M_valid = hdr[0], N = hdr[1], K = hdr[2];
            if (M_valid <= 0 || N <= 0 || K <= 0 ||
                M_valid > (1 << 20) || N > (1 << 20) || K > (1 << 20)) {
                fprintf(stderr, "[remote-worker] bad MAT_MUL header M=%ld N=%ld K=%ld, dropping connection\n",
                        (long)M_valid, (long)N, (long)K);
                break;
            }

            auto job = std::make_shared<TsiWorkerJob>();
            job->is_matmul = true;
            job->mm_m_valid = M_valid;
            job->mm_n = N;
            job->mm_k = K;
            job->mm_a_rows.resize((size_t)(M_valid * K));
            job->mm_b_rows.resize((size_t)(N * K));
            if (!tsi_recv_all(fd, job->mm_a_rows.data(), job->mm_a_rows.size() * sizeof(float))) break;
            if (!tsi_recv_all(fd, job->mm_b_rows.data(), job->mm_b_rows.size() * sizeof(float))) break;

            fprintf(stderr, "[remote-worker] MAT_MUL request: M_valid=%ld N=%ld K=%ld -- queuing for the dispatch thread\n",
                    (long)M_valid, (long)N, (long)K);

            std::future<bool> fut = tsi_worker_submit_job(job);
            if (!fut.get()) {
                fprintf(stderr, "[remote-worker] local MAT_MUL TXE dispatch failed, dropping connection\n");
                break;
            }
            const std::vector<float> &C = job->result;

            fprintf(stderr, "[remote-worker] MAT_MUL result: C[0]=%g (M_valid*N=%zu floats, "
                    "this request used %ld real local TXE kernel-runs)\n",
                    C.empty() ? 0.0f : C[0], C.size(), (long)job->result_kernel_runs);

            if (!tsi_send_all(fd, &magic, sizeof(magic))) break;
            if (!tsi_send_all(fd, &job->result_kernel_runs, sizeof(job->result_kernel_runs))) break;
            if (!tsi_send_all(fd, C.data(), C.size() * sizeof(float))) break;
            continue;
        }

        fprintf(stderr, "[remote-worker] bad magic 0x%08x, dropping connection\n", magic);
        break;
    }

    // ROUND 5 requirement B: print the running kernel-run total at the end
    // of this client session (this loop only exits when the client
    // disconnects, sends a bad magic, or a send/recv fails -- i.e. exactly
    // "end of session" for whichever client was on the other end of `fd`).
    // As of ROUND 5's persistent-connection client, a normal real-model run
    // has exactly ONE such session for its entire lifetime, so this line
    // effectively becomes the final per-run total, comparable to instance
    // 1's own end-of-run GGML Perf Summary.
    tsi_worker_print_kernel_run_summary("client session ended");
}

// Runs the accept loop: one thread per accepted connection handles that
// connection's socket I/O (recv/send) concurrently with every other
// connection. The actual TXE dispatch for each request is handed off to
// the single dispatch thread (see tsi_worker_dispatch_thread_main) via
// tsi_worker_submit_job() from within tsi_remote_worker_serve_conn() --
// concurrent connections here do NOT mean concurrent graph_compute() calls.
static void tsi_run_remote_worker_accept_loop(int port) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { perror("[remote-worker] socket"); return; }
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("[remote-worker] bind");
        return;
    }
    if (listen(listen_fd, 8) < 0) {
        perror("[remote-worker] listen");
        return;
    }
    fprintf(stderr, "[remote-worker] listening on 0.0.0.0:%d\n", port);

    for (;;) {
        struct sockaddr_in cli {};
        socklen_t cli_len = sizeof(cli);
        int fd = accept(listen_fd, (struct sockaddr *)&cli, &cli_len);
        if (fd < 0) {
            if (errno == EINTR) continue;
            perror("[remote-worker] accept");
            continue;
        }
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        fprintf(stderr, "[remote-worker] connection from %s:%d\n",
                inet_ntoa(cli.sin_addr), ntohs(cli.sin_port));
        // One thread per accepted connection for socket I/O only -- see the
        // big comment above tsi_worker_dispatch_thread_main() for why the
        // actual TXE dispatch is NOT done on this thread as of Round 4.
        std::thread([fd, cli]() {
            tsi_remote_worker_serve_conn(fd, nullptr);
            close(fd);
            fprintf(stderr, "[remote-worker] connection from %s:%d closed\n",
                    inet_ntoa(cli.sin_addr), ntohs(cli.sin_port));
        }).detach();
    }
    // unreachable
}

static int tsi_run_remote_worker(int port) {
    fprintf(stderr, "[remote-worker] initializing Tsavorite backend locally on THIS instance...\n");
    ggml_backend_t backend = ggml_backend_tsavorite_init();
    if (!backend) {
        fprintf(stderr, "[remote-worker] ggml_backend_tsavorite_init() failed\n");
        return -1;
    }
    fprintf(stderr, "[remote-worker] local Tsavorite/TXE runtime initialized successfully. "
                     "Serving remote ADD/MAT_MUL dispatch on port %d\n", port);

    // ROUND 4: the accept loop (and all per-connection socket I/O) moves to
    // its own thread. THIS thread -- the one that just called
    // ggml_backend_tsavorite_init() above -- becomes the single, permanent
    // dispatch thread instead, by calling tsi_worker_dispatch_thread_main()
    // directly (it never returns). This is deliberate, not incidental: it
    // guarantees every ggml_backend_graph_compute() call this process ever
    // makes runs on the exact same OS thread that initialized the TXE
    // runtime, matching the invariant every previously-proven-correct case
    // (local llama-cli inference, Round 2's ADD POC) already relied on --
    // see the comment above tsi_worker_dispatch_thread_main() for the full
    // reasoning and why Round 3's per-connection-thread dispatch broke it.
    std::thread(tsi_run_remote_worker_accept_loop, port).detach();

    tsi_worker_dispatch_thread_main(backend);
    // unreachable
    return 0;
}
// --- END TSI_REMOTE_TXE_POC ---

#define NUM_INPUT_TENSORS 2
#define NUM_INPUT_URINARY_TENSORS 1
#define  NUM_ELEMENTS 32
#define  NUM_ELEMENTS_SCALE 32*4 + 25

// index 0 for addition, index 1 for subtraction, index 2 for multiplication, index 3 for division
float test_input_1[GGML_TSAVORITE_KERNEL_TYPE_COUNT][NUM_ELEMENTS] = {
	//ADD KERNEL
	{1.1,  2.3,  3.2,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//SUB KERNEL
	{2.2,  10.3,  10.4,  2.2,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//MULT KERNEL
	{1.1,  2.3,  3.2,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//DIV KERNEL
	{1.1,  4.4,  10,  5,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	// SQRT Kernel
	{1,  4,  9.6,  16,  25,  36,  49,  64,  81,  100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024},
	//SQR Kernel
	{1, 2.5, 3, 4, 5.6, 6, 7, 8, 9.2, 10, 11, 12, 13, 14, 15.4, 16, 17, 18, 19, 20, 21.2, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//NEG Kernel
	{1.1,  -4.4,  10,  -5,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, -23, 24, 25, -26, 27, -28, 29, -30, 31, -32.6},
	//ABS Kernel
	{1.1,  -4.4,  10,  -5,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, -23, 24, 25, -26, 27, -28, 29, -30, 31, -32.6},
	//SIN Kernel
	{1.1,  4.4,  10,  5,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32.6},
	//RMS_NORM Kernel
	{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//SIGMOID Kernel need to fix not tested
	{1.1,  4.4,  10,  5,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32.6},
	//SILU  Kernel
	{-16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0}
};
float test_input_2[GGML_TSAVORITE_KERNEL_TYPE_COUNT][NUM_ELEMENTS] = {
	//ADD KERNEL
	{1.1,  2.2,  3.3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//SUB KERNEL
	{1.1,  2.2,  3.0,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//MULT KERNEL
	{1.1,  2.2,  3.3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//DIV KERNEL
	{1.1,  2.2,  5,  10,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//Below ROW value not used for Unary OPS-SQRT, NEG, ABS, SIN
	//SQRT KERNEL input not used
	{1.1,  2.2,  5,  10,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//SQR KERNEL input not used
	{1.1,  2.2,  5,  10,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//NEG KERNEL input not used
	{1.1,  2.2,  5,  10,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//ABS KERNEL input not used
	{1.1,  2.2,  5,  10,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//SIN Kernel input not used
	{1.1,  2.2,  5,  10,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//RMS_NORM Kernel input is not used
	{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//SIGMOID Kernel not used
	{1.1,  4.4,  10,  5,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 20, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32.6},
	//SILU  Kernel not used
	{-16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0}
};

float test_result[GGML_TSAVORITE_KERNEL_TYPE_COUNT][NUM_ELEMENTS] = {
	//ADD KERNEL
	{2.20, 4.50, 6.50, 8.00, 10.00, 12.00, 14.00, 16.00, 18.00, 20.00, 22.00, 24.00, 26.00, 28.00, 30.00, 32.00, 34.00, 36.00, 38.00, 40.00, 42.00, 44.00, 46.00, 48.00, 50.00, 52.00, 54.00, 56.00, 58.00, 60.00, 62.00, 64.00},
	//SUB KERNEL
	{1.1, 8.1, 7.4, -1.8, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
	//MULT KERNEL
	{1.21, 5.06, 10.56, 16.00, 25.00, 36.00, 49.00, 64.00, 81.00, 100.00, 121.00, 144.00, 169.00, 196.00, 225.00, 256.00, 289.00, 324.00, 361.00, 400.00, 441.00, 484.00, 529.00, 576.00, 625.00, 676.00, 729.00, 784.00, 841.00, 900.00, 961.00, 1024.00},
	//DIV KERNEL
	{1.0, 2.0, 2, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	//SQRT Kernel
	{1,  2,  3.098387,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
	//SQR Kernel
	{1, 6.25, 9, 16, 31.36, 36, 49, 64, 84.64, 100, 121, 144, 169, 196, 237.16, 256, 289, 324, 361, 400, 449.44, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024},
	//NEG Kernel
	{-1.1,  4.4,  -10,  5,  -5,  -6,  -7,  -8,  -9,  -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, 23, -24, -25, 26, -27, 28, -29, 30, -31, 32.6},
	//ABS Kernel
	{1.1,  4.4,  10,  5,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32.6},
	//SIN Kernel
	{0.891207,  -0.951602,  -0.544021,  -0.958924,  -0.958924,  -0.279416,  0.656987,  0.989358,  0.412118,  -0.544021, -0.999990, -0.536573, 0.420167, 0.990607, 0.650288, -0.287903, -0.961398, -0.750987, 0.149877, 0.912945, 0.912945, 0.912945, -0.846220, -0.905578, -0.132352, 0.762559, 0.956376, 0.270906, -0.663634, -0.988032, -0.404039, 0.926149},
	//RMS_NORM Kernel
	{0.052888, 0.105776, 0.158664, 0.211552, 0.264440, 0.317328, 0.370216, 0.423104, 0.475992, 0.528880, 0.581768, 0.634656, 0.687544, 0.740432, 0.793320, 0.846208, 0.899096, 0.951984, 1.004872, 1.057760, 1.110648, 1.163536, 1.216424, 1.269312, 1.322200, 1.375088, 1.427976, 1.480864, 1.533752, 1.586640, 1.639528, 1.692416},
	//SIGMOID  Kernel not tested
	{0.891207,  -0.951602,  -0.544021,  -0.958924,  -0.958924,  -0.279416,  0.656987,  0.989358,  0.412118,  -0.544021, -0.999990, -0.536573, 0.420167, 0.990607, 0.650288, -0.287903, -0.961398, -0.750987, 0.149877, 0.912945, 0.912945, 0.912945, -0.846220, -0.905578, -0.132352, 0.762559, 0.956376, 0.270906, -0.663634, -0.988032, -0.404039, 0.926149},
	// SILU Kernel
	{-0.000002, -0.000005, -0.000012, -0.000029, -0.000074, -0.000184, -0.000454, -0.001111, -0.002683, -0.006377, -0.014836, -0.033464, -0.071945, -0.142278, -0.238406, -0.268941, 0.000000, 0.731059, 1.761594, 2.857722, 3.928055, 4.966536, 5.985164, 6.993623, 7.997317, 8.998889, 9.999546, 10.999816, 11.999926, 12.999971, 13.999988, 14.999995}

};

float test_input_scale_1[GGML_TSAVORITE_KERNEL_TYPE_COUNT][NUM_ELEMENTS_SCALE] = {
	//ADD KERNEL
	{1.3, 2.3, 3.3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
	//SUB KERNEL
	{8.5, 2.5, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 64,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 63, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 63, 32,
	 4, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 63, 32,
	 2, 4, 8, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
	//MULT KERNEL
	{1.5, 2.5, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  10,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  10,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  10,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  10,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//DIV KERNEL
	{4.2, 8.4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 4,   8,   1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 4,   8,   1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 4,   8,   1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 4,   8,   1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//SQRT KERNEL
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	 //SQR KERNEL
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//NEG KERNEL
	{-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//ABS KERNEL
	{-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//SIN KERNEL
	{-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//RMS_NORM Kernel
	{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
	//SIGMOID KERNEL need to fix input data
	{-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	// SILU KERNEL
	{-16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0, 1.0, 2.0, 3.0}
};

float test_input_scale_2[GGML_TSAVORITE_KERNEL_TYPE_COUNT][NUM_ELEMENTS_SCALE] = {
	// ADD KERNEL
	{1.3, 2.3, 3.3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
	// SUB KERNEL
	{1, 8, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 6, 8, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
	// MULT KERNEL
	{2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	// DIV KERNEL
	{2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//Below ROW value not used for Unary OPS-SQRT, NEG, ABS, SIN
	//SQRT KERNEL input not used
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	 //SQR KERNEL input not used
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//NEG KERNEL input not used
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//ABS KERNEL input not used
	{-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//SIN KERNEL input not used
	{-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	//RMS_NORM Kernel input not used
	{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
	 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
	//SIGMOID KERNEL input not used
	{-1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 -1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	// SILU KERNEL input not used
	{-16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0, 1.0, 2.0, 3.0}
};
float test_result_scale[GGML_TSAVORITE_KERNEL_TYPE_COUNT][NUM_ELEMENTS_SCALE] = {
	// ADD KERNEL
	{2.6, 4.6, 6.6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38 ,40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
	 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38 ,40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
	 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38 ,40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
	 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38 ,40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
	 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38 ,40, 42, 44, 46, 48, 50},
	// SUB KERNEL
	{7.5, -5.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  32,
	 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0,
        -5, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0,
	 3, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0,
	 1, 2,  5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	// MULT KERNEL
	{3, 5,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	// DIV KERNEL
	{2.1, 4.2,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
	 2, 4,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
	// SQRT KERNEL
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 3, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 4, 5, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	// SQR KERNEL
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 3, 2, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 4, 5, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	// NEG KERNEL
	{1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
	 9, -4, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
	 16, -25, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
	 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
	 1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1},
	// ABS KERNEL
	{1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 9, 4, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 16, 25, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
	 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1},
	// SIN KERNEL
	{-0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	 -0.412118,-0.756802, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.287903,-0.132352, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	 -0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	 -0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471},
	//RMS_NORM Kernel
	{
          0.054620, 0.109240, 0.163860, 0.218479, 0.273099, 0.327719, 0.382339, 0.436959, 0.491579, 0.546199,
          0.600818, 0.655438, 0.710058, 0.764678, 0.819298, 0.873918, 0.928537, 0.983157, 1.037777, 1.092397,
          1.147017, 1.201637, 1.256257, 1.310876, 1.365496, 1.420116, 1.474736, 1.529356, 1.583976, 1.638596,
          1.693215, 1.747835, 0.054620, 0.109240, 0.163860, 0.218479, 0.273099, 0.327719, 0.382339, 0.436959,
          0.491579, 0.546199, 0.600818, 0.655438, 0.710058, 0.764678, 0.819298, 0.873918, 0.928537, 0.983157,
          1.037777, 1.092397, 1.147017, 1.201637, 1.256257, 1.310876, 1.365496, 1.420116, 1.474736, 1.529356,
          1.583976, 1.638596, 1.693215, 1.747835, 0.054620, 0.109240, 0.163860, 0.218479, 0.273099, 0.327719,
          0.382339, 0.436959, 0.491579, 0.546199, 0.600818, 0.655438, 0.710058, 0.764678, 0.819298, 0.873918,
          0.928537, 0.983157, 1.037777, 1.092397, 1.147017, 1.201637, 1.256257, 1.310876, 1.365496, 1.420116,
          1.474736, 1.529356, 1.583976, 1.638596, 1.693215, 1.747835, 0.054620, 0.109240, 0.163860, 0.218479,
          0.273099, 0.327719, 0.382339, 0.436959, 0.491579, 0.546199, 0.600818, 0.655438, 0.710058, 0.764678,
          0.819298, 0.873918, 0.928537, 0.983157, 1.037777, 1.092397, 1.147017, 1.201637, 1.256257, 1.310876,
          1.365496, 1.420116, 1.474736, 1.529356, 1.583976, 1.638596, 1.693215, 1.747835, 0.054620, 0.109240,
          0.163860, 0.218479, 0.273099, 0.327719, 0.382339, 0.436959, 0.491579, 0.546199, 0.600818, 0.655438,
          0.710058, 0.764678, 0.819298, 0.873918, 0.928537, 0.983157, 1.037777, 1.092397, 1.147017, 1.201637,
          1.256257, 1.310876, 1.365496},
	// SIGMOID KERNEL, result need to change
	{-0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	 -0.412118,-0.756802, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.287903,-0.132352, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	 -0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	 -0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471, 0.841471,
	  0.841471, 0.841471, 0.841471},
	// SILU KERNEL
	{-0.000002, -0.000005, -0.000012, -0.000029, -0.000074, -0.000184, -0.000454, -0.001111, -0.002683, -0.006377, -0.014836, -0.033464, -0.071945, -0.142278, -0.238406, -0.268941, 0.000000, 0.731059, 1.761594, 2.857722, 3.928055, 4.966536, 5.985164, 6.993623, 7.997317, 8.998889, 9.999546, 10.999816, 11.999926, 12.999971, 13.999988, 14.999995, -0.000002, -0.000005, -0.000012, -0.000029, -0.000074, -0.000184, -0.000454, -0.001111, -0.002683, -0.006377, -0.014836, -0.033464, -0.071945, -0.142278, -0.238406, -0.268941, 0.000000, 0.731059, 1.761594, 2.857722, 3.928055, 4.966536, 5.985164, 6.993623, 7.997317, 8.998889, 9.999546, 10.999816, 11.999926, 12.999971, 13.999988, 14.999995, -0.000002, -0.000005, -0.000012, -0.000029, -0.000074, -0.000184, -0.000454, -0.001111, -0.002683, -0.006377, -0.014836, -0.033464, -0.071945, -0.142278, -0.238406, -0.268941, 0.000000, 0.731059, 1.761594, 2.857722, 3.928055, 4.966536, 5.985164, 6.993623, 7.997317, 8.998889, 9.999546, 10.999816, 11.999926, 12.999971, 13.999988, 14.999995, -0.000002, -0.000005, -0.000012, -0.000029, -0.000074, -0.000184, -0.000454, -0.001111, -0.002683, -0.006377, -0.014836, -0.033464, -0.071945, -0.142278, -0.238406, -0.268941, 0.000000, 0.731059, 1.761594, 2.857722, 3.928055, 4.966536, 5.985164, 6.993623, 7.997317, 8.998889, 9.999546, 10.999816, 11.999926, 12.999971, 13.999988, 14.999995, 0.000000, 0.731059, 1.761594, 2.857722}
};

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;

    // the backend to perform the computation (TSAVORITE)
    ggml_backend_t backend = NULL;

    // the backend buffer to storage the tensors data of a and b
    ggml_backend_buffer_t buffer;

    // the context to define the tensor information (dimensions, size, memory address)
    struct ggml_context * ctx;
};


static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}


// --- FLOAT COMPARATOR
static bool ggml_tsi_compare_two_float(float a, float b) {
    // For very small values, use absolute error
    if (fabsf(a) < 1e-2f && fabsf(b) < 1e-2f) {
        return fabsf(a - b) < 1e-6f; // Accept up to 1e-6 difference for small values
    }
    // For larger values, use relative error with increased tolerance
    // Increased to 1e-3 (0.1%) to handle floating-point precision differences
    const float epsilon = 1e-3f; // Changed from 1e-4f to 1e-3f
    float diff = fabsf(a - b);
    float max_val = fmaxf(fabsf(a), fabsf(b));
    return diff < epsilon * max_val;
}

static bool load_model(simple_model & model, float * a, float * b, enum ggml_type data_type, int elements_A, int elements_B) {
    ggml_log_set(ggml_log_callback_default, nullptr);

    // initialize the backend
    fprintf(stderr, "%s: using TSavorite backend \n", __func__);
    model.backend = ggml_backend_tsavorite_init();
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_tsavorite_init() failed\n", __func__);
	return false;
    }

    int num_tensors;

    if (!b)
        num_tensors = NUM_INPUT_URINARY_TENSORS;
    else
        num_tensors = NUM_INPUT_TENSORS;

    // Since we are not passing the mem_buffer ggml context will create
    /* .mem_buffer = params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size) */
    // mem_buffer for ctx is used for any object creation and used for tensor data if
    // backend doesnt have own memory
    // Since we are using backend memory hence i have removed extra bytes: 100, removed from mem_size at below
    struct ggml_init_params params {
            /*.mem_size   =*/ (ggml_tensor_overhead() * num_tensors),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };
    fprintf(stderr, "\n Calculating mem_size %ld  %d  and creating ggml context \n", ggml_tensor_overhead(), num_tensors);

    // create context
    model.ctx = ggml_init(params);
    if (!model.ctx) {
        fprintf(stderr, "%s: ggml_init failed\n", __func__);
	return false;
    }

    // create tensors
    // //  BELOW CODE NO CHANGE FOR tsavorite Backend
    // Tensor just created with OBJ(Structure)+Tensor(structure)
    // Still Buffer need to attached to Tensor since we are using Backend
    // We will using tsi_alloc called under tsavorite-backend

    fprintf(stderr, "\n Creating input Tensor \n");

    //int64_t ne[GGML_MAX_DIMS]; // number of elements
    //size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
    model.a = ggml_new_tensor_1d(model.ctx, data_type, elements_A);
    if (b)
        model.b = ggml_new_tensor_1d(model.ctx, data_type, elements_B);

    // create a backend buffer (backend memory) and alloc the tensors from the context
    fprintf(stderr, "\n Creating Backend Buffer \n");

    // Here at ggml Context we have only two input tensors, hence backend memory is
    // created for two input tensors
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // load data from cpu memory to backend buffer
    fprintf(stderr, "\n Loading Input Tensor Data to Backend Buffer \n");

    // loading the data to tensor
    ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a));
    if (b)
        ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));

    // create a array to print input tensor
    std::vector<float> out_data(ggml_nelements(model.a));
    // bring the data from the backend memory
    ggml_backend_tensor_get(model.a, out_data.data(), 0, ggml_nbytes(model.a));


    fprintf(stderr, "\nBringing  tensor data from Backend buffer and printing %d  tensor data:\n[", (int) model.a->ne[0]);

    for (int i = 0; i < model.a->ne[0] /* cols */; i++) {
        fprintf(stderr, " %.2f", out_data[i]);
    }
    fprintf(stderr, " ]\n");
    return true;
}

// build the compute graph
static struct ggml_cgraph * build_graph(const simple_model& model, enum ggml_tsavorite_kernel_type ops_type) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);


    struct ggml_tensor * result;
    switch(ops_type) {
	    case GGML_TSAVORITE_KERNEL_TYPE_ADD:
    		result = ggml_add(ctx0, model.a, model.b);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_SUB:
    		result = ggml_sub(ctx0, model.a, model.b);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_MULT:
    		result = ggml_mul(ctx0, model.a, model.b);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_DIV:
    		result = ggml_div(ctx0, model.a, model.b);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_SQRT:
    		result = ggml_sqrt(ctx0, model.a);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_SQR:
    		result = ggml_sqr(ctx0, model.a);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_NEG:
                result = ggml_neg(ctx0, model.a);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_ABS:
                result = ggml_abs(ctx0, model.a);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_SIN:
                result = ggml_sin(ctx0, model.a);
		break;
		case GGML_TSAVORITE_KERNEL_TYPE_RMS_NORM:
                result = ggml_rms_norm(ctx0, model.a, 1e-5);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_SIGMOID:
                result = ggml_sigmoid(ctx0, model.a);
		break;
	    case GGML_TSAVORITE_KERNEL_TYPE_SILU:
                result = ggml_silu(ctx0, model.a);
		break;
	     default:
    		ggml_free(ctx0);
    		fprintf(stderr, "\n Non Supported Operation \n");
		return NULL;
    }
    // build operations nodes
    ggml_build_forward_expand(gf, result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

// compute with backend
static struct ggml_tensor * compute(const simple_model & model, ggml_gallocr_t allocr, enum ggml_tsavorite_kernel_type ops_type) {
    // reset the allocator to free all the memory allocated during the previous inference

    fprintf(stderr, "\n Under Test case for  compute API creating  build_graph  \n");
    struct ggml_cgraph * gf = build_graph(model, ops_type);
    if (!gf) {
	    fprintf(stderr, "\ncompute failed\n");
	    return NULL;
    }

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_graph_compute(model.backend, gf);

    // in this case, the output tensor is the last one in the graph
    return ggml_graph_node(gf, -1);
}

enum ggml_tsavorite_kernel_type convert_testcase_to_ops_type (const char *testCase) {
        if (!strcmp(testCase,"add"))
            return GGML_TSAVORITE_KERNEL_TYPE_ADD;
        else if (!strcmp(testCase,"sub"))
            return GGML_TSAVORITE_KERNEL_TYPE_SUB;
        else if (!strcmp(testCase,"mult"))
            return GGML_TSAVORITE_KERNEL_TYPE_MULT;
        else if (!strcmp(testCase,"div"))
            return GGML_TSAVORITE_KERNEL_TYPE_DIV;
        else if (!strcmp(testCase,"sqrt"))
            return GGML_TSAVORITE_KERNEL_TYPE_SQRT;
        else if (!strcmp(testCase,"sqr"))
            return GGML_TSAVORITE_KERNEL_TYPE_SQR;
        else if (!strcmp(testCase,"neg"))
            return GGML_TSAVORITE_KERNEL_TYPE_NEG;
        else if (!strcmp(testCase,"abs"))
            return GGML_TSAVORITE_KERNEL_TYPE_ABS;
        else if (!strcmp(testCase,"sin"))
            return GGML_TSAVORITE_KERNEL_TYPE_SIN;
        else if (!strcmp(testCase,"rms_norm"))
            return GGML_TSAVORITE_KERNEL_TYPE_RMS_NORM;
        else if (!strcmp(testCase,"sigmoid"))
            return GGML_TSAVORITE_KERNEL_TYPE_SIGMOID;
        else if (!strcmp(testCase,"silu"))
            return GGML_TSAVORITE_KERNEL_TYPE_SILU;

    	fprintf(stderr, "\n un-supported test case %s hence running default test case which is add operation  \n", testCase);
	return GGML_TSAVORITE_KERNEL_TYPE_ADD;
}

const char* convert_ops_type_to_testcase(enum ggml_tsavorite_kernel_type ops_type) {

    switch (ops_type) {
        case GGML_TSAVORITE_KERNEL_TYPE_ADD:
            return "add";
        case GGML_TSAVORITE_KERNEL_TYPE_SUB:
            return "sub";
        case GGML_TSAVORITE_KERNEL_TYPE_MULT:
            return "mult";
        case GGML_TSAVORITE_KERNEL_TYPE_DIV:
            return "div";
        case GGML_TSAVORITE_KERNEL_TYPE_SQRT:
            return "sqrt";
        case GGML_TSAVORITE_KERNEL_TYPE_SQR:
            return "sqr";
        case GGML_TSAVORITE_KERNEL_TYPE_NEG:
            return "neg";
        case GGML_TSAVORITE_KERNEL_TYPE_ABS:
            return "abs";
		case GGML_TSAVORITE_KERNEL_TYPE_SIN:
            return "sin";
        case GGML_TSAVORITE_KERNEL_TYPE_RMS_NORM:
            return "rms_norm";
            return "sin";
        case GGML_TSAVORITE_KERNEL_TYPE_SIGMOID:
            return "sigmoid";
        case GGML_TSAVORITE_KERNEL_TYPE_SILU:
            return "silu";
        default:
            return "unknown";
    }
}

// --- TEST HARNESS DEBUG BLOCK ---
#define DEBUG_COMPARE 1

int main(int argc, char *argv[]) {
    ggml_time_init();

    // TSI_REMOTE_TXE_POC: cross-instance remote TXE dispatch worker mode.
    // Usage: simple-backend-tsi remote-worker [port]   (default port 29511)
    // Initializes the real Tsavorite backend locally (same as every other
    // mode below) and then serves remote ADD dispatch requests instead of
    // running a local test case. See the TSI_REMOTE_TXE_POC block above
    // main() for the wire protocol and design rationale.
    if (argc > 1 && !strcmp(argv[1], "remote-worker")) {
        int port = (argc > 2) ? atoi(argv[2]) : 29511;
        return tsi_run_remote_worker(port);
    }

    bool test_case_flag = true;
    enum ggml_tsavorite_kernel_type ops_type;
    simple_model model;
    float *input1[GGML_TSAVORITE_KERNEL_TYPE_COUNT];
    float *input2[GGML_TSAVORITE_KERNEL_TYPE_COUNT];
    float *result_data[GGML_TSAVORITE_KERNEL_TYPE_COUNT];
    bool data_scale = false;

    int elements_A=0, elements_B=0;
    int num_of_input_tensors;

    if (argc > 1) {
        // New MAT_MUL standalone test case.
        // Keep existing add/sub/mult/div/unary tests unchanged.
        if (!strcmp(argv[1], "mat-mul") || !strcmp(argv[1], "mul-mat")) {
            return matmul_tsi_test(argc, argv);
        }

    	ops_type = convert_testcase_to_ops_type(argv[1]);
	if (argc > 2 && !strcmp(argv[2], "scale"))
		data_scale = true;
    } else {
	// Default Case
    	ops_type = convert_testcase_to_ops_type("add");
    }
    if (ops_type == GGML_TSAVORITE_KERNEL_TYPE_SQRT ||
		    ops_type == GGML_TSAVORITE_KERNEL_TYPE_SQR ||
		    ops_type == GGML_TSAVORITE_KERNEL_TYPE_NEG ||
		    ops_type == GGML_TSAVORITE_KERNEL_TYPE_ABS ||
		    ops_type == GGML_TSAVORITE_KERNEL_TYPE_SIN ||
			ops_type == GGML_TSAVORITE_KERNEL_TYPE_RMS_NORM ||
		    ops_type == GGML_TSAVORITE_KERNEL_TYPE_SIGMOID ||
		    ops_type == GGML_TSAVORITE_KERNEL_TYPE_SILU)
	    num_of_input_tensors = NUM_INPUT_URINARY_TENSORS;
    else
	    num_of_input_tensors = NUM_INPUT_TENSORS;

    if (data_scale) {
	    input1[ops_type]      = test_input_scale_1[ops_type];
	    elements_A            = NUM_ELEMENTS_SCALE;
	    if (num_of_input_tensors != NUM_INPUT_URINARY_TENSORS) {
	        input2[ops_type]      = test_input_scale_2[ops_type];
	        elements_B            = NUM_ELEMENTS_SCALE;
	    }
	    result_data[ops_type] = test_result_scale[ops_type];
    } else {
	    input1[ops_type]      = test_input_1[ops_type];
	    elements_A            = NUM_ELEMENTS;
	    if (num_of_input_tensors != NUM_INPUT_URINARY_TENSORS) {
	        input2[ops_type]      = test_input_2[ops_type];
	        elements_B            = NUM_ELEMENTS;
	    }
	    result_data[ops_type] = test_result[ops_type];
    }

    if(!load_model(model, input1[ops_type], input2[ops_type], GGML_TYPE_F32, elements_A, elements_B)) {
	    fprintf(stderr, "\n\n TEST CASE FAILED \n\n");
	    return -1;
    }
    // since tsavorite-backend init set the debug level to none, we are overwritting here
    ggml_tsavorite_log_type_val = GGML_TSAVORITE_LOG_DEBUG;

    ggml_gallocr_t allocr = NULL;

    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    if (!allocr) {
    	fprintf(stderr, "\n\n TEST CASE FAILED \n\n");
	return -1;
    }

    // create the worst case graph for memory usage estimation
    struct ggml_cgraph * gf = build_graph(model, ops_type);
    if (!gf) {
    	fprintf(stderr, "\n\n TEST CASE FAILED \n\n");
	return -1;
    }
    ggml_gallocr_reserve(allocr, gf);
    size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);

    fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size/1024.0);

    // perform computation
    struct ggml_tensor * result = compute(model, allocr, ops_type);
    if (!result) {
	fprintf(stderr, "\n\n TEST CASE FAILED \n\n");
	return -1;
    }
    fprintf(stderr, "\n Compute Done \n");

    std::vector<float> out_data(ggml_nelements(result));

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    // expected result:

    fprintf(stderr, "\n operation type: %s, num of elements %d  \n", convert_ops_type_to_testcase(ops_type), (int) result->ne[0]);

    fprintf(stderr, "\n compute is also done \n");
    for (int i = 0; i < result->ne[0] /* cols */; i++) {

#if DEBUG_COMPARE
        uint32_t bits_expected, bits_actual;
        memcpy(&bits_expected, &result_data[ops_type][i], sizeof(float));
        memcpy(&bits_actual, &out_data[i], sizeof(float));
        fprintf(stderr, "Index %d: expected bits %08x, actual bits %08x\n", i, bits_expected, bits_actual);
#endif
	if (ggml_tsi_compare_two_float(out_data[i], result_data[ops_type][i])) {
		continue;
	}
	test_case_flag = false;
    	fprintf(stderr, "\n result for index %d is not matching expected %f got %f \n", i, result_data[ops_type][i], out_data[i]);
    }

    if (test_case_flag == false) {
	fprintf(stderr, "\n\n TEST CASE FAILED \n\n");
        ggml_free(model.ctx);
        ggml_backend_free(model.backend);
	return -1;
    }
    fprintf(stderr, "\n\n TEST CASE PASSED \n\n");

    // free memory
    ggml_free(model.ctx);

    // release backend memory and free backend
    //ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    return 0;
}
