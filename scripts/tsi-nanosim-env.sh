# Source this to run a GGML_TSAVORITE_TARGET=nanosim build on the Xtensa ISS of a nanosim release.
#   NANOSIM_ROOT=/path/to/nanosim-release . scripts/tsi-nanosim-env.sh
# USER_DRAM_SIZE (MiB) and GGML_TSAVORITE_KERNEL_DIR may be set before sourcing; defaults below.
if [ -z "$NANOSIM_ROOT" ] || [ ! -x "$NANOSIM_ROOT/device/bin/TXEXtensaSimDevice" ]; then
    echo "tsi-nanosim-env.sh: set NANOSIM_ROOT to a nanosim release (runtime/, device/bin/TXEXtensaSimDevice)" >&2
    return 1 2>/dev/null || exit 1
fi
export LD_LIBRARY_PATH="$NANOSIM_ROOT/runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TSI_RUNTIME_BINARY_DIR_OVERRIDE="$NANOSIM_ROOT/runtime"
export TSI_XTENSA_SIM_DEVICE="$NANOSIM_ROOT/device/bin/TXEXtensaSimDevice"
export TSI_TXE_DEVICE=xtensa_sim
export USER_DRAM_SIZE="${USER_DRAM_SIZE:-4096}"
