#!/usr/bin/env bash
# decode.sh - fixed-L KV-cache decode flow for the tsisim/FPGA box.
#
# Runs the compiled decode graph autoregressively with a host-side KV cache:
#   1. build    cmake --build <build-dir> --target decode_run   (the stateful runner; box-only)
#   2. emit     decode_run --emit  -> <dir>/forward_decode.mlir  (one @forward: logits + k/v_new)
#   3. compile  compile_graph_fpga.py --target fpga -> <dir>/out_decode/host/host.so
#   4. run      decode_run --lib host.so  -> generate tokens; --verify diffs each vs a CPU prefill
#
# Unlike wholegraph.sh (prefill-only, recompiles per token), this compiles ONE decode graph and
# reuses it every step via the host KV cache - O(L) per token, no recompile.
#
# Token ids (not a prompt string) are the input; get them from ref_check: ref_check <gguf> "<prompt>".
#
# Input is a prompt string (-p, tokenized by llama) or raw token ids (--ids). The generated tokens
# are printed as ids and as detokenized text.
#
# --txes N tiles the compile across N TXEs (1..20) and brings the runtime up with the same count
# (both must match; the flag sets both). Changing N forces a recompile.
#
# Usage:
#   ./decode.sh -m MODEL {-p "prompt" | --ids "id0 id1 ..."} [--L N] [--gen N] [--verify] [--txes N]
#               [-d DIR] [--build-dir BD] [--lib host.so] [-c CONFIG]
#               [--host tsisim|x86] [--sdk SDK_ROOT] [--venv VENV]
#
# Examples:
#   ./decode.sh -m /root/tinyllama-v0-f32.gguf -p "hello world" --gen 16            # generate + print text
#   ./decode.sh -m /root/tinyllama-v0-f32.gguf -p "hello world" --gen 16 --verify   # + check each vs prefill
#   ./decode.sh -m /root/tinyllama-v0-f32.gguf -p "hello world" --gen 16 --txes 8   # tile across 8 TXEs
#   ./decode.sh -m /root/tinyllama-v0-f32.gguf --ids "1 2 3 4" --L 8 --verify       # validate on raw ids
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# ---------------------------------------------------------------- SDK / toolchain env defaults
DEFAULT_SDK="/tsi/tsi-sw/sdk/sdk-r.0.4.17"
DEFAULT_XT_TOOLS="/proj/vendors/cadence/xtensa/XtDevTools/install/tools/RJ-2025.5-linux-arm/XtensaTools"
DEFAULT_XT_SYSTEM="/proj/rel/cadence/TENcore/XtensaRegistry/RJ-2025.5-linux-arm"
DEFAULT_LICENSE="27012@tlicmgr.tsavoritesi.net"

# ---------------------------------------------------------------- run defaults
MODEL="/root/tinyllama-v0-f32.gguf"
IDS=""
PROMPT=""
L=""                 # fixed max cache length; default: prompt+gen+2 (runner chooses if empty)
GEN=0
VERIFY=0
DIR="/root/wg-decode"
BUILD_DIR=""         # TSI cmake build dir (has the decode_run target); required unless --lib given
DECODE_RUN=""        # path to a prebuilt decode_run binary (skips the build step)
LIB=""               # prebuilt decode host.so (skips emit+compile)
CONFIG=""
HOST="tsisim"
SDK=""
VENV=""
FORCE=0
TXES=""              # multi-TXE: compile num_txes + runtime txe_count (1..20); empty = leave as configured

die() { echo "error: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model)    MODEL="$2"; shift 2 ;;
        --ids)         IDS="$2"; shift 2 ;;
        -p|--prompt)   PROMPT="$2"; shift 2 ;;
        --L)           L="$2"; shift 2 ;;
        --gen)         GEN="$2"; shift 2 ;;
        --verify)      VERIFY=1; shift ;;
        -d|--dir)      DIR="$2"; shift 2 ;;
        --build-dir)   BUILD_DIR="$2"; shift 2 ;;
        --decode-run)  DECODE_RUN="$2"; shift 2 ;;
        --lib)         LIB="$2"; shift 2 ;;
        -c|--config)   CONFIG="$2"; shift 2 ;;
        --host)        HOST="$2"; shift 2 ;;
        --sdk)         SDK="$2"; shift 2 ;;
        --venv)        VENV="$2"; shift 2 ;;
        --txes)        TXES="$2"; shift 2 ;;
        --force)       FORCE=1; shift ;;
        -h|--help)     grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1  (use -h)" ;;
    esac
done
[ -n "$IDS" ] || [ -n "$PROMPT" ] || die "give -p \"prompt text\" or --ids \"id0 id1 ...\""
# input passed to decode_run: either a --prompt string or positional token ids
if [ -n "$PROMPT" ]; then INPUT=(--prompt "$PROMPT"); else INPUT=($IDS); fi
[ -f "$MODEL" ] || die "model not found: $MODEL"
case "$HOST" in tsisim|x86) ;; *) die "--host must be tsisim|x86 (got '$HOST')" ;; esac

# ---------------------------------------------------------------- environment (mirrors wholegraph.sh)
if [ "$HOST" = tsisim ]; then
    export MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-${SDK:-$DEFAULT_SDK}}"
    export XT_TOOLS_DIR="${XT_TOOLS_DIR:-$DEFAULT_XT_TOOLS}"
    export XT_SYSTEM_DIR="${XT_SYSTEM_DIR:-$DEFAULT_XT_SYSTEM}"
    export TSI_RT_LIB_DIR="${TSI_RT_LIB_DIR:-$MLIR_SDK_VERSION/tsisim/runtime/lib}"
else
    [ -n "${SDK:-}" ] && export MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-$SDK}"
    [ -n "${MLIR_SDK_VERSION:-}" ] || die "x86 host: set MLIR_SDK_VERSION via --sdk or env"
fi
export TOOLBOX_DIR="${TOOLBOX_DIR:-$MLIR_SDK_VERSION/toolbox/build/install-fpga}"
export TMU_MMA_SHAPE="${TMU_MMA_SHAPE:-8,1}"
export LM_LICENSE_FILE="${LM_LICENSE_FILE:-$DEFAULT_LICENSE}"
export PATH="$MLIR_SDK_VERSION/compiler/bin:$PATH"
export LD_LIBRARY_PATH="$MLIR_SDK_VERSION/toolbox/build/install-posix/lib:${TSI_RT_LIB_DIR:-}:${LD_LIBRARY_PATH:-}"

PY="$( [ -n "$VENV" ] && echo "$VENV/bin/python3" || command -v python3 || echo python3 )"
COMPILE_PY="$HERE/compile_graph_fpga.py"
[ -f "$COMPILE_PY" ] || die "compile_graph_fpga.py not next to this script: $COMPILE_PY"
if [ -z "$CONFIG" ]; then
    for __name in txe_arm.json txe_compiler_config.json; do
        for __c in "${TXE_FPGA_CONFIG:-}" "$HERE/$__name" "$HERE/../../ggml-tsi-kernel/fpga-kernel/$__name"; do
            [ -n "$__c" ] && [ -f "$__c" ] && { CONFIG="$__c"; break 2; }
        done
    done
fi
export TXE_FPGA_CONFIG="$CONFIG"
mkdir -p "$DIR"

# multi-TXE: tile the compile across N TXEs and bring the runtime up with the same count. The two
# MUST match, so we set both from one flag. A deployment yaml in $DIR (pointed at via the env the
# runtime checks first) overrides the bundled one. Changing N invalidates the cached host.so.
if [ -n "$TXES" ]; then
    case "$TXES" in ''|*[!0-9]*) die "--txes must be an integer 1..20 (got '$TXES')" ;; esac
    { [ "$TXES" -ge 1 ] && [ "$TXES" -le 20 ]; } || die "--txes out of range 1..20 (got '$TXES')"
    export TSI_NUM_TXES="$TXES"                                  # compile: tile across N TXEs
    if [ "$TXES" -gt 1 ]; then
        export TSI_ENABLE_MULTI_TXE="${TSI_ENABLE_MULTI_TXE:-1}"     # compiler gates num_txes>1
        # host.so needs LLVM libomp (__kmpc_*). Honor a preset TSI_OMP_LIB_DIR, else find libomp.
        if [ -z "${TSI_OMP_LIB_DIR:-}" ]; then
            for __d in /usr/lib/llvm-*/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib /usr/lib64; do
                if ls "$__d"/libomp.so* >/dev/null 2>&1; then TSI_OMP_LIB_DIR="$__d"; break; fi
            done
        fi
        if [ -n "${TSI_OMP_LIB_DIR:-}" ]; then export TSI_OMP_LIB_DIR; else
            echo "  warning: libomp not found; set TSI_OMP_LIB_DIR (LLVM lib dir with libomp.so) for --txes>1" >&2
        fi
    fi
    cat > "$DIR/tsavorite-model-deployment.yaml" <<EOF
txe_count: $TXES
multi_thread_enable: true
advanced_matmul_shape_offload: false
EOF
    export TSAVORITE_MODEL_DEPLOYMENT_YAML="$DIR/tsavorite-model-deployment.yaml"   # runtime: N TXEs
    FORCE=1                                                      # recompile (num_txes changes the blobs)
fi

echo "=== KV-cache decode ==="
printf '  %-14s %s\n' host "$HOST" model "$MODEL" input "${PROMPT:-$IDS}" L "${L:-auto}" gen "$GEN" \
       verify "$VERIFY" txes "${TXES:-default}" dir "$DIR" config "$CONFIG"

# ---------------------------------------------------------------- 1) build decode_run
if [ -z "$DECODE_RUN" ]; then
    if [ -n "$BUILD_DIR" ]; then
        echo "--- [1/4] build decode_run ---"
        cmake --build "$BUILD_DIR" --target decode_run
        DECODE_RUN="$(find "$BUILD_DIR" -name decode_run -type f -perm -u+x 2>/dev/null | head -1)"
    fi
    [ -n "$DECODE_RUN" ] || DECODE_RUN="$HERE/decode_run"
fi
[ -x "$DECODE_RUN" ] || die "decode_run not found/executable: $DECODE_RUN  (pass --build-dir or --decode-run)"

Largs=(); [ -n "$L" ] && Largs=(--L "$L")

# ---------------------------------------------------------------- 2) emit + 3) compile (unless --lib)
if [ -z "$LIB" ]; then
    [ -f "$CONFIG" ] || die "fpga config not found: $CONFIG  (pass -c)"
    echo "--- [2/4] emit decode graph ---"
    # pass --gen so emit and run pick the same default L (L = n_ids + gen + 2) when --L is omitted
    "$DECODE_RUN" "$MODEL" --emit "$DIR/forward_decode.mlir" "${Largs[@]}" --gen "$GEN" "${INPUT[@]}"
    grep -q 'func.func @forward' "$DIR/forward_decode.mlir" || die "emit produced no @forward"
    echo "--- [3/4] compile ---"
    if [ "$FORCE" = 1 ] || [ ! -f "$DIR/out_decode/host/host.so" ]; then
        "$PY" "$COMPILE_PY" --target fpga --config "$CONFIG" "$DIR/forward_decode.mlir" "$DIR/out_decode"
    fi
    LIB="$DIR/out_decode/host/host.so"
fi
[ -f "$LIB" ] || die "decode host.so not found: $LIB"

# ---------------------------------------------------------------- 4) run
echo "--- [4/4] run (gen=$GEN, verify=$VERIFY) ---"
vflag=(); [ "$VERIFY" = 1 ] && vflag=(--verify)
"$DECODE_RUN" "$MODEL" --lib "$LIB" "${Largs[@]}" --gen "$GEN" "${vflag[@]}" "${INPUT[@]}"
echo "=== done (decode) ==="
