#!/usr/bin/env python3
"""Compile a whole exported graph MLIR module (one func @forward, captured by the whole-graph
interception) through the TSI mlir-compiler as a single unit, for the posix or fpga target.
After compiling it reports blob count and sizes.

Targets:
  posix (default): default TXECompilerConfig (host-native; validates the driver anywhere,
                   incl. macOS). Does NOT test FPGA.
  fpga           : TXECompilerConfig.from_json(<fpga txe_compiler_config.json>), txe_target="Ten",
                   Xtensa blob backend. Needs config-setup.sh env + Xtensa toolchain => SDK box only.
                   Point --config (or $TXE_FPGA_CONFIG) at ggml-tsi-kernel/fpga-kernel/txe_compiler_config.json.

Run with the venv that has the mlir_external_packages wheel (mlir-compiler/venv).

    compile_graph_fpga.py [--target posix|fpga] [--config <txe_compiler_config.json>] <forward.mlir> <out_dir>
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from tsavorite.compiler_config import TXECompilerConfig
    from tsi_raw_backend import RawGraphBackend
except ImportError as e:
    print(
        f"error: {e}\nRun with a venv that has the mlir_external_packages wheel "
        f"(e.g. mlir-compiler/venv/bin/python3 {__file__})",
        file=sys.stderr,
    )
    sys.exit(1)


def _default_fpga_config():
    # Try $TXE_FPGA_CONFIG, then a sibling llama.cpp checkout's fpga config.
    env = os.environ.get("TXE_FPGA_CONFIG")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for base in (here.parents[3], here.parents[4]):  # tsi_repo/, or one up
        cand = base / "llama.cpp" / "ggml-tsi-kernel" / "fpga-kernel" / "txe_compiler_config.json"
        if cand.exists():
            return cand
    return None


def _fpga_config(config_path):
    if config_path is None:
        config_path = _default_fpga_config()
    if config_path is None or not Path(config_path).exists():
        print("error: fpga txe_compiler_config.json not found. Pass --config <path> or set "
              "$TXE_FPGA_CONFIG (ggml-tsi-kernel/fpga-kernel/txe_compiler_config.json).", file=sys.stderr)
        sys.exit(1)
    expanded = os.path.expandvars(Path(config_path).read_text())
    if "${" in expanded:
        print("error: unresolved ${...} in fpga compiler config - source config-setup.sh first "
              "(need MLIR_SDK_VERSION, TOOLBOX_DIR, ...)", file=sys.stderr)
        sys.exit(1)
    tmp = Path(config_path).with_suffix(".expanded.json")
    tmp.write_text(expanded)
    cfg = TXECompilerConfig.from_json(tmp)

    # The whole-graph linalg->txe.tmu_matmul lowering (MatmulToTMUPattern) uses tmu_mma_shape
    # VERBATIM for f16/f32 matmuls, and the verifier requires product == getNumMMAsForBitWidth(32)
    # == 8. The DSL fpga config ships tmu_mma_shape=[1,1] (product 1 -> "invalid TXE shape 1x1 for
    # element type f32"), which is fine for its hand-written matmul kernel but not this path.
    # Override to [8,1] (the DSL TMU kernel's shape). $TMU_MMA_SHAPE="a,b" overrides for tuning.
    env_mma = os.environ.get("TMU_MMA_SHAPE")
    if env_mma:
        cfg = cfg._replace(tmu_mma_shape=[int(x) for x in env_mma.split(",")])
    else:
        cur = cfg.tmu_mma_shape
        if not cur or int(cur[0]) * int(cur[1]) != 8:
            cfg = cfg._replace(tmu_mma_shape=[8, 1])
    print(f"=== tmu_mma_shape = {list(cfg.tmu_mma_shape)} (f32 needs product 8) ===", file=sys.stderr)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir", help="whole-graph forward.mlir or forward.mlirbc")
    ap.add_argument("out_dir")
    ap.add_argument("--target", choices=["posix", "fpga"], default="posix")
    ap.add_argument("--config", default=None, help="fpga txe_compiler_config.json (fpga target)")
    args = ap.parse_args()

    # Read bytes, not text. Baked weight constants make the module far too big to carry as hex, so it
    # is emitted as MLIR bytecode instead, where blobs are raw binary. Module.parse accepts either
    # form, and text decodes cleanly to str for the pretty-print below.
    mlir_bytes = Path(args.mlir).read_bytes()
    is_bytecode = mlir_bytes[:4] == b"ML\xefR"
    model = mlir_bytes if is_bytecode else mlir_bytes.decode()
    out = Path(args.out_dir)

    config = _fpga_config(args.config) if args.target == "fpga" else TXECompilerConfig(log_mlir=True)

    kind = "bytecode" if is_bytecode else "text"
    print(f"=== compiling whole graph: target={args.target}  in={args.mlir} ({kind}, "
          f"{len(mlir_bytes) / 1048576:.1f} MiB)  out={out} ===")
    RawGraphBackend(config).compile(
        model=model, input_types=[], compilation_type="aot",
        output_dir=str(out), verbose=True,
    )

    host_obj = out / "host" / "host.o"
    blobs = sorted((out / "blobs").glob("*.blob")) if (out / "blobs").exists() else []
    print("\n=== STEP-0 RESULT ===")
    print(f"host.o: {'OK ' + str(host_obj) if host_obj.exists() else 'MISSING'}")
    print(f"blob count: {len(blobs)}")
    total = 0
    for b in blobs:
        sz = b.stat().st_size
        total += sz
        print(f"  {b.name}: {sz} bytes")
    print(f"total blob bytes: {total}")
    if not host_obj.exists():
        sys.exit(1)

    # Link host.o -> host.so so the backend can dlopen it (TSI_WHOLEGRAPH=run). We also generate a
    # tiny C shim `tsi_forward_argv(void**)` that unpacks a pointer array into the N-arg
    # _mlir_ciface_forward, so the backend can call the compiled forward WITHOUT libffi (the arg
    # count is fixed at compile time). N comes from the real ciface signature in host.ll.
    # Undefined tsi_* symbols in host.o resolve at dlopen(RTLD_GLOBAL) against the backend process's
    # runtime; pass $TSI_RT_LIB_DIR to link the runtime explicitly if your loader needs it.
    import subprocess, re
    host_so  = out / "host" / "host.so"
    host_ll  = out / "host" / "host.ll"
    cc       = os.environ.get("CC", "cc")

    nargs = None
    if host_ll.exists():
        m = re.search(r"define\s+void\s+@_mlir_ciface_forward\(([^)]*)\)", host_ll.read_text())
        if m:
            nargs = 0 if m.group(1).strip() == "" else len(m.group(1).split(","))
    if nargs is None:
        print("host.so: SKIPPED (could not read _mlir_ciface_forward arg count from host.ll)")
        return
    print(f"_mlir_ciface_forward takes {nargs} pointer args")

    wrapper_c = out / "host" / "tsi_forward_argv.c"
    proto     = ", ".join(["void *"] * nargs) if nargs else "void"
    argv_list = ", ".join(f"a[{i}]" for i in range(nargs))
    wrapper_c.write_text(
        "/* auto-generated: unpack void** into the fixed-arity _mlir_ciface_forward (no libffi). */\n"
        f"void _mlir_ciface_forward({proto});\n"
        "void tsi_forward_argv(void **a) { _mlir_ciface_forward(" + argv_list + "); }\n"
    )

    link_cmd = [cc, "-shared", "-fPIC", str(host_obj), str(wrapper_c), "-o", str(host_so)]
    rt_dir = os.environ.get("TSI_RT_LIB_DIR")
    if rt_dir:
        link_cmd += [f"-L{rt_dir}", "-lTsavRTShimCAPI", f"-Wl,-rpath,{rt_dir}"]
    rc = subprocess.call(link_cmd)
    print(f"host.so: {'OK ' + str(host_so) if (rc == 0 and host_so.exists()) else 'FAILED (link manually: ' + ' '.join(link_cmd) + ')'}")


if __name__ == "__main__":
    main()
