#!/usr/bin/env bash
# wholegraph.sh - one-shot whole-graph integration step for the tsisim/FPGA box.
#
# Runs the ENTIRE flow in a single execution:
#   1. capture   TSI_WHOLEGRAPH=capture llama-cli  -> <dir>/forward.mlir (+ forward.manifest)
#   2. compile   compile_graph_fpga.py --target fpga -> <dir>/out_fpga/{host/host.so, blobs/*.blob}
#   3. verify|run TSI_WHOLEGRAPH=<mode> llama-cli   -> next token(s) from the compiled whole graph,
#                                                      checked against llama.cpp's own per-op output
#
# The SDK / Xtensa / toolbox environment is set up here (edit the DEFAULT_* below if paths move).
# config-setup.sh is NOT auto-sourced (it runs git-submodule + venv side effects); pass --setup
# <path/to/config-setup.sh> if you want the full SDK env sourced instead of these direct exports.
#
# Modes (--mode):
#   verify  run the compiled forward once, diff its next token vs the per-op reference (default)
#   run     like verify, but also let llama sample the compiled token
#   dump    write graph.txt only (no compile)
#   gen     generate N tokens, each from the compiled forward (prefill-only: re-capture + recompile
#           over the growing sequence per token; O(n^2), no KV cache reuse -- see the doc's §12).
#           Add --verify-each to also diff compiled vs per-op on every token (slower, prints MATCH).
#
# Usage:
#   ./wholegraph.sh [-m MODEL] [-p PROMPT] [-n N] [-d DIR] [-c CONFIG] [--mode verify|run|dump|gen]
#                   [--verify-each] [--host tsisim|x86] [--sdk SDK_ROOT] [--venv VENV] [--cli LLAMA_CLI]
#                   [--force] [--setup FILE]
#
# --host tsisim (default): compile natively on the aarch64 tsisim box  -> txe_arm.json
#        x86             : compile on an x86 build box (aarch64 cross)  -> txe_compiler_config.json
#
# Examples:
#   ./wholegraph.sh -p "hello world"                         # verify, 1 token, default model
#   ./wholegraph.sh -m /root/tinyllama-v0-f32.gguf -p "what is 1+1" -n 1 --mode verify
#   ./wholegraph.sh -p "hello world" --mode gen -n 16        # generate 16 tokens on the compiled forward
#   ./wholegraph.sh -p "hi" --mode dump                      # just dump graph.txt (no compile)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------- SDK / toolchain env defaults
DEFAULT_SDK="/tsi/tsi-sw/sdk/sdk-r.0.4.17"
DEFAULT_XT_TOOLS="/proj/vendors/cadence/xtensa/XtDevTools/install/tools/RJ-2025.5-linux-arm/XtensaTools"
DEFAULT_XT_SYSTEM="/proj/rel/cadence/TENcore/XtensaRegistry/RJ-2025.5-linux-arm"
DEFAULT_LICENSE="27012@tlicmgr.tsavoritesi.net"

# ---------------------------------------------------------------- run defaults
MODEL="/root/tinyllama-v0-f32.gguf"
PROMPT="hello world"
N=1
DIR="/root/wg"
CONFIG=""            # fpga txe_compiler_config.json; default resolved below
MODE="verify"        # verify | run | dump
HOST="tsisim"        # tsisim = native aarch64 compile (txe_arm.json); x86 = cross compile (stock config)
SDK=""               # MLIR_SDK_VERSION; default depends on --host
VENV=""              # python venv dir (uses <venv>/bin/python3); default: python3 on PATH
LLAMA_CLI=""         # default: sibling ./llama-cli, else on PATH
FORCE=0
VERIFY_EACH=0        # gen mode: also diff compiled vs per-op each token (slower)
SETUP=""             # optional config-setup.sh to source
LLAMA_FLAGS="${LLAMA_FLAGS:--ctk f32 -ctv f32 -fa off}"
TSI_WG_SKIP="${TSI_WG_SKIP:-0}"

die() { echo "error: $*" >&2; exit 1; }

# ---------------------------------------------------------------- arg parsing
while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model)   MODEL="$2"; shift 2 ;;
        -p|--prompt)  PROMPT="$2"; shift 2 ;;
        -n|--ntokens) N="$2"; shift 2 ;;
        -d|--dir)     DIR="$2"; shift 2 ;;
        -c|--config)  CONFIG="$2"; shift 2 ;;
        --mode)       MODE="$2"; shift 2 ;;
        --host)       HOST="$2"; shift 2 ;;
        --sdk)        SDK="$2"; shift 2 ;;
        --venv)       VENV="$2"; shift 2 ;;
        --cli)        LLAMA_CLI="$2"; shift 2 ;;
        --force)      FORCE=1; shift ;;
        --verify-each) VERIFY_EACH=1; shift ;;
        --setup)      SETUP="$2"; shift 2 ;;
        -h|--help)    grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1  (use -h)" ;;
    esac
done
case "$MODE" in verify|run|dump|gen) ;; *) die "--mode must be verify|run|dump|gen (got '$MODE')" ;; esac
case "$HOST" in tsisim|x86) ;; *) die "--host must be tsisim|x86 (got '$HOST')" ;; esac

# ---------------------------------------------------------------- environment
if [ -n "$SETUP" ]; then
    [ -f "$SETUP" ] || die "config-setup not found: $SETUP"
    echo "== sourcing $SETUP =="
    # shellcheck disable=SC1090
    SDK_VERSION="${SDK_VERSION:-}" . "$SETUP"
fi
if [ "$HOST" = tsisim ]; then
    # Native aarch64 compile on the tsisim box: arm Xtensa tools + native g++/ar + box SDK.
    export MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-${SDK:-$DEFAULT_SDK}}"
    export XT_TOOLS_DIR="${XT_TOOLS_DIR:-$DEFAULT_XT_TOOLS}"
    export XT_SYSTEM_DIR="${XT_SYSTEM_DIR:-$DEFAULT_XT_SYSTEM}"
    export TSI_RT_LIB_DIR="${TSI_RT_LIB_DIR:-$MLIR_SDK_VERSION/tsisim/runtime/lib}"
else
    # x86 build box: the stock config hardcodes the x86 Xtensa + x86-hosted aarch64 cross toolchain,
    # so only MLIR_SDK_VERSION / TOOLBOX_DIR need to resolve. Source config-setup.sh (--setup) on the
    # x86 box, or pass --sdk / export MLIR_SDK_VERSION for the x86 SDK root.
    [ -n "${SDK:-}" ] && export MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-$SDK}"
    [ -n "${MLIR_SDK_VERSION:-}" ] || die "x86 host: set MLIR_SDK_VERSION (x86 SDK root) via --sdk, env, or --setup"
fi
export TOOLBOX_DIR="${TOOLBOX_DIR:-$MLIR_SDK_VERSION/toolbox/build/install-fpga}"
export TMU_MMA_SHAPE="${TMU_MMA_SHAPE:-8,1}"
export LM_LICENSE_FILE="${LM_LICENSE_FILE:-$DEFAULT_LICENSE}"
# so the compiler's own tools (mlir-translate/llc) and libs resolve without full config-setup:
export PATH="$MLIR_SDK_VERSION/compiler/bin:$PATH"
export LD_LIBRARY_PATH="$MLIR_SDK_VERSION/toolbox/build/install-posix/lib:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------- resolve helpers
[ -z "$LLAMA_CLI" ] && LLAMA_CLI="$( [ -x "$HERE/llama-cli" ] && echo "$HERE/llama-cli" || command -v llama-cli || echo "$HERE/llama-cli" )"
[ -x "$LLAMA_CLI" ] || die "llama-cli not found/executable: $LLAMA_CLI  (pass --cli)"
PY="$( [ -n "$VENV" ] && echo "$VENV/bin/python3" || command -v python3 || echo python3 )"
COMPILE_PY="$HERE/compile_graph_fpga.py"
[ -f "$COMPILE_PY" ] || die "compile_graph_fpga.py not next to this script: $COMPILE_PY"
if [ -z "$CONFIG" ]; then
    # tsisim -> txe_arm.json (native aarch64); x86 -> txe_compiler_config.json (stock cross build).
    if [ "$HOST" = tsisim ]; then
        __order="txe_arm.json txe_compiler_config.json"
    else
        __order="txe_compiler_config.json txe_arm.json"
    fi
    __cands="${TXE_FPGA_CONFIG:-}"
    for __name in $__order; do
        __cands="$__cands $HERE/$__name $HERE/../../ggml-tsi-kernel/fpga-kernel/$__name"
    done
    for __cfg in $__cands; do
        [ -n "$__cfg" ] && [ -f "$__cfg" ] && { CONFIG="$__cfg"; break; }
    done
fi
export TXE_FPGA_CONFIG="$CONFIG"
mkdir -p "$DIR"

echo "=== whole-graph integration ==="
printf '  %-16s %s\n' host "$HOST" model "$MODEL" prompt "$PROMPT" ntokens "$N" dir "$DIR" \
       mode "$MODE" config "$CONFIG" cli "$LLAMA_CLI" python "$PY" \
       MLIR_SDK_VERSION "$MLIR_SDK_VERSION" TOOLBOX_DIR "$TOOLBOX_DIR"
[ -f "$MODEL" ] || die "model not found: $MODEL"

# ---------------------------------------------------------------- dump mode: graph.txt only
if [ "$MODE" = dump ]; then
    echo "--- dump: writing $DIR/graph.txt ---"
    TSI_WHOLEGRAPH=dump TSI_WG_DIR="$DIR" TSI_WG_SKIP="$TSI_WG_SKIP" \
        "$LLAMA_CLI" -m "$MODEL" -p "$PROMPT" -n 1 --no-warmup $LLAMA_FLAGS
    echo "=== done (dump) -> $DIR/graph.txt ==="
    exit 0
fi

# ---------------------------------------------------------------- gen mode: prefill-only, N tokens
# Each token is produced by the compiled whole-graph forward. Because the compiled host.so is fixed
# to one sequence length, we re-capture + recompile over the growing sequence every step (no KV
# cache reuse), then sample the compiled token and append it. O(n^2) and recompiles per token, so
# it's a correctness/throughput demo, not the fast path (see the design doc's §12).
if [ "$MODE" = gen ]; then
    [ -f "$CONFIG" ] || die "fpga config not found: $CONFIG  (pass -c)"
    "$PY" -c "import os,sys; sys.exit(0 if '\${' not in os.path.expandvars(open(sys.argv[1]).read()) else 1)" "$CONFIG" \
        || die "unresolved \${...} in $CONFIG (pass --setup or export the SDK env)"
    echo "--- gen: $N tokens, prefill-only (recompile per step) ---"
    seq="$PROMPT"
    printf '%s' "$PROMPT"
    __first_ms=0; __rest_ms=0; __rest_cnt=0
    for ((i = 1; i <= N; i++)); do
        __t0=$(date +%s%N)
        echo "  [step $i/$N] capture+compile+run over $(printf %s "$seq" | wc -c | tr -d ' ') prompt chars" >&2
        TSI_WHOLEGRAPH=capture TSI_WG_DIR="$DIR" TSI_WG_SKIP="$TSI_WG_SKIP" \
            "$LLAMA_CLI" -m "$MODEL" -p "$seq" -n 1 --no-warmup $LLAMA_FLAGS >"$DIR/gen.capture.log" 2>&1 \
            || { tail -20 "$DIR/gen.capture.log" >&2; die "step $i: capture failed"; }
        grep -q 'func.func @forward' "$DIR/forward.mlir" || die "step $i: capture produced no @forward"
        "$PY" "$COMPILE_PY" --target fpga --config "$CONFIG" "$DIR/forward.mlir" "$DIR/out_fpga" \
            >"$DIR/gen.compile.log" 2>&1 || { tail -20 "$DIR/gen.compile.log" >&2; die "step $i: compile failed"; }
        # --verify-each: run in verify mode (compiled + per-op, prints the diff) instead of run.
        # verify doesn't overwrite the logits, so llama samples the per-op token -- identical to the
        # compiled token whenever the step MATCHes, so generation continues the same either way.
        step_mode=run
        [ "$VERIFY_EACH" = 1 ] && step_mode=verify
        raw=$(TSI_WHOLEGRAPH="$step_mode" TSI_WG_DIR="$DIR" TSI_WG_SKIP="$TSI_WG_SKIP" \
              TSI_WG_LIB="$DIR/out_fpga/host/host.so" \
              "$LLAMA_CLI" -m "$MODEL" -p "$seq" -n 1 --no-warmup --no-display-prompt $LLAMA_FLAGS 2>"$DIR/gen.run.err" || true)
        if [ "$VERIFY_EACH" = 1 ]; then
            v=$(grep -a 'VERIFY compiled vs per-op' "$DIR/gen.run.err" | tail -1)
            echo "  [step $i] ${v:-<no VERIFY line found>}" >&2
        fi
        # The tsavorite runtime prints a "TSI deploy ..." banner + profiler lines to STDOUT; strip
        # them WITH their leading whitespace/newlines so only llama's generated token remains and its
        # own single leading space (SentencePiece "_word") is preserved exactly (no stray double space).
        tok=$(printf '%s' "$raw" | tr -d '\n' | sed -E \
              -e 's/[[:space:]]*TSI deploy yaml=[^[:space:]]* txe_count=[0-9]+ multi_thread_enable=[0-9]+ advanced_matmul_shape_offload=[0-9]+//g' \
              -e 's/[[:space:]]*OPU Profiling Results://g' -e 's/[[:space:]]*Profiler disabled//g')
        case "$tok" in
            *"TSI deploy"*|*hal_lib_init*|*"HAL "*)
                die "step $i: runtime banner leaked into stdout, cannot isolate the generated token" ;;
        esac
        __dt=$(( ($(date +%s%N) - __t0) / 1000000 ))
        if [ "$i" -eq 1 ]; then __first_ms=$__dt; else __rest_ms=$((__rest_ms + __dt)); __rest_cnt=$((__rest_cnt + 1)); fi
        echo "  [step $i] ${__dt} ms (capture+compile+run)" >&2
        seq="$seq$tok"
        printf '%s' "$tok"
    done
    printf '\n'
    echo "[timing] first token: ${__first_ms} ms (capture+compile+run over the prompt)" >&2
    [ "$__rest_cnt" -gt 0 ] && echo "[timing] subsequent tokens: $((__rest_ms / __rest_cnt)) ms/token avg over $__rest_cnt (each = recapture+recompile+run)" >&2
    echo "=== done (gen) ==="
    exit 0
fi

# ---------------------------------------------------------------- 1) capture
if [ "$FORCE" = 1 ] || [ ! -f "$DIR/forward.mlir" ]; then
    echo "--- [1/3] capture ---"
    TSI_WHOLEGRAPH=capture TSI_WG_DIR="$DIR" TSI_WG_SKIP="$TSI_WG_SKIP" \
        "$LLAMA_CLI" -m "$MODEL" -p "$PROMPT" -n 1 --no-warmup $LLAMA_FLAGS
    grep -q 'func.func @forward' "$DIR/forward.mlir" || die "capture produced no @forward in $DIR/forward.mlir"
else
    echo "--- [1/3] capture: cached ($DIR/forward.mlir) ---"
fi

# ---------------------------------------------------------------- 2) compile
if [ "$FORCE" = 1 ] || [ ! -f "$DIR/out_fpga/host/host.so" ]; then
    echo "--- [2/3] compile ---"
    [ -f "$CONFIG" ] || die "fpga config not found: $CONFIG  (pass -c)"
    # fail early with a clear message if the config still has unresolved ${...}
    if "$PY" -c "import os,sys; sys.exit(0 if '\${' not in os.path.expandvars(open(sys.argv[1]).read()) else 1)" "$CONFIG"; then :; else
        echo "unresolved \${...} in $CONFIG - these env vars are needed:" >&2
        grep -oE '\$\{[A-Za-z_][A-Za-z0-9_]*\}' "$CONFIG" | sort -u >&2
        die "set the missing var(s) (or pass --setup <config-setup.sh>)"
    fi
    "$PY" "$COMPILE_PY" --target fpga --config "$CONFIG" "$DIR/forward.mlir" "$DIR/out_fpga"
    [ -f "$DIR/out_fpga/host/host.so" ] || die "compile did not produce $DIR/out_fpga/host/host.so"
else
    echo "--- [2/3] compile: cached ($DIR/out_fpga/host/host.so) ---"
fi

# ---------------------------------------------------------------- 3) verify | run
echo "--- [3/3] $MODE ---"
TSI_WHOLEGRAPH="$MODE" TSI_WG_DIR="$DIR" TSI_WG_SKIP="$TSI_WG_SKIP" \
TSI_WG_LIB="$DIR/out_fpga/host/host.so" \
    "$LLAMA_CLI" -m "$MODEL" -p "$PROMPT" -n "$N" --no-warmup $LLAMA_FLAGS
echo "=== done ($MODE) ==="
