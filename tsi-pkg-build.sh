#!/usr/bin/env bash
# ==============================================================================
# tsi-pkg-build.sh (source-safe)
#
# USAGE (source is recommended)
# ============================
#
# SDK_VERSION IS MANDATORY (for EVERY invocation)
# -----------------------------------------------
# SDK_VERSION must be provided explicitly by the user as an environment variable.
#
# Correct usage:
#   SDK_VERSION=0.4.1 source tsi-pkg-build.sh [build-mode] [flags...] [MLIR_COMPILER_DIR] [TOOLBOX_DIR]
#
# Positional SDK_VERSION arguments are NOT supported:
#   source tsi-pkg-build.sh SDK_VERSION=0.4.1   # NOT supported
#
# If SDK_VERSION is not provided, the script will fail fast.
#
# ------------------------------------------------------------------------------
# Tsavorite Deployment Configuration (llama.cpp)
# ------------------------------------------------------------------------------
# llama.cpp supports an deployment configuration file:
#
# tsavorite-model-deployment.yaml
#
# This file controls how the Tsavorite backend uses TXEs and whether
# multi-threaded execution is enabled at runtime.
#
# ------------------------------------------------------------------------------
# Configuration Options
# ------------------------------------------------------------------------------
#
# 1) Number of TXEs
# -----------------
# - Specifies how many TXEs are available for execution.
# - Value can be 1 or more.
#
# 2) Multi-threading (enable / disable)
# -------------------------------------
#
# a) Multi-threading DISABLED:
# - llama.cpp uses host-generated code produced by mlir_compiler.
# - The generated host code always targets TXE0.
# - No dynamic TXE selection or scheduling is performed.
#
# b) Multi-threading ENABLED:
# - llama.cpp Tsavorite backend contains host-side scheduling logic.
# - At runtime, the backend:
#   * Scans for a free TXE
#   * Selects an available TXE dynamically
#   * Creates a host thread bound to the selected TXE
# - This enables concurrent execution across multiple TXEs.
#
# ------------------------------------------------------------------------------
# Deployment File Location and Usage
# ------------------------------------------------------------------------------
#
# FPGA:
# -----
# - During FPGA bundle creation, tsavorite-model-deployment.yaml is packaged
#   alongside the Tsavorite shared libraries.
# - Inside the final tarball, the file is placed in the same directory
#   as the Tsavorite .so files (tsi-ggml/).
# - At runtime (after untarring), the Tsavorite backend loads this file
#   from the same directory as the shared libraries.
#
# POSIX:
# ------
# - For POSIX builds, tsavorite-model-deployment.yaml is expected to be present
#   in the llama.cpp root directory (the working directory where llama.cpp
#   is built and executed).
#
# Example:
#   /proj/work/akapoor/llama-cpp-april-16/llama.cpp/tsavorite-model-deployment.yaml
#
# - If present, the Tsavorite backend loads this file at runtime to determine
#   TXE configuration and multi-threading behavior.
#
## ------------------------------------------------------------------------------
## FPGA Packaging – Deployment File
## ------------------------------------------------------------------------------
##
## FPGA packaging requires ./tsavorite-model-deployment.yaml to exist in the
## current working directory.
##
## tsi-pkg-build.sh does not generate deployment configuration files.
## The deployment configuration is maintained only in
## tsavorite-model-deployment.yaml.
##
## During packaging, the file is copied directly into the FPGA bundle
## alongside the Tsavorite shared libraries.
##
## ------------------------------------------------------------------------------
#
#
# Build modes (optional):
# release
# debug
# debug-tmu
# debug-tmu-detail
#
# - release : GGML_PERF_RELEASE
# - debug : POSIX => GGML_PERF_DETAIL
#           FPGA => GGML_PERF (GGML_PERF_DETAIL disabled to avoid expensive file logging)
# - debug-tmu : GGML_PERF_DETAIL + TMU_DEBUG
# - debug-tmu-detail : GGML_PERF_DETAIL + TMU_DEBUG + TMU_DEBUG_VALIDATE
#
# Submodules:
# - First run in a fresh repo checkout: auto "git submodule update --init --recursive"
# - Later runs: submodule update is OPTIONAL unless pass:
#   git-submodule-pull
# - If ggml-tsi-kernel is missing, script forces submodule init even without the flag.
#
# Blob build (OFF by default):
# build-fpga-blobs : build blobs in ggml-tsi-kernel/fpga-kernel only
# build-posix-blobs : build blobs in ggml-tsi-kernel/posix-kernel only
# build-all-blobs : build blobs for both fpga+posix kernels
#
# Auto blob safeguards (ON by default):
# - If deleted ggml-tsi-kernel (rm -rf) or host objects are missing:
#   * POSIX build auto-builds POSIX blobs if required for link
#   * FPGA build auto-builds FPGA blobs if required for link
# Disable both with:
#   no-auto-blobs
#
# Python virtual env (only used for blob generation):
# overwrite-venv : delete blob-creation venv and recreate it (installs deps)
# NOTE: this alone does NOT build blobs unless blob flag is also set.
# git-submodule-pull : ALSO forces overwrite-venv AND build-all-blobs (as requested)
#
# Build selection:
# Default (no build-selection flags): build-posix + build-fpga + package
#
# build-posix
#   * Build POSIX ggml/llama.cpp with TMU + TVU enabled
#   * Output directory: ./build-posix
#
# build-posix-tmu-only
#   * Build POSIX ggml/llama.cpp with TMU enabled and TVU disabled
#   * Output directory: ./build-posix-tmu-only
#
# build-posix-tmu-disable
#   * Build POSIX ggml/llama.cpp with TVU enabled and TMU disabled
#   * Output directory: ./build-posix-tmu-disable
#
# build-fpga
#   * Build FPGA ggml/llama.cpp with TMU + TVU enabled
#   * Output directory: ./build-fpga
#
# build-fpga-tmu-only
#   * Build FPGA ggml/llama.cpp with TMU enabled and TVU disabled
#   * Output directory: ./build-fpga-tmu-only
#
# build-fpga-tmu-disable
#   * Build FPGA ggml/llama.cpp with TVU enabled and TMU disabled
#   * Output directory: ./build-fpga-tmu-disable
#
# package
#   * Package FPGA bundle (requires an FPGA build dir already built)
#
# Incremental build:
# incremental : do not rm -rf build dirs (both llama.cpp + kernels)
#
# Cleanup:
# clean : rm -rf build-* (llama.cpp) and kernel build dirs in ggml-tsi-kernel
# clean-all : clean + remove python venv blob-creation
#
# Coverage:
# enable_coverage : adds -DENABLE_COVERAGE=ON
#
# Help:
# help \
# -h \
# --help \
# -help
#
# ==============================================================================
#
# EXAMPLES (SDK_VERSION REQUIRED)
# ===============================
#
# 1) Default (posix + fpga + package) with default build-type (debug):
#    SDK_VERSION=0.4.1 source tsi-pkg-build.sh
#
# 2) POSIX only:
#    SDK_VERSION=0.4.1 source tsi-pkg-build.sh debug build-posix
#
# 3) POSIX TMU-only:
#    SDK_VERSION=0.4.1 source tsi-pkg-build.sh debug build-posix-tmu-only
#
# 4) POSIX TMU disabled (TVU-only):
#    SDK_VERSION=0.4.1 source tsi-pkg-build.sh debug build-posix-tmu-disable
#
# 5) FPGA only (TMU+TVU):
#    SDK_VERSION=0.4.1 source tsi-pkg-build.sh debug build-fpga
#
# 6) FPGA TMU-only:
#    SDK_VERSION=0.4.1 source tsi-pkg-build.sh debug build-fpga-tmu-only
#
# 7) FPGA TMU disabled (TVU-only):
#    SDK_VERSION=0.4.1 source tsi-pkg-build.sh debug build-fpga-tmu-disable
#
# 8) Debug TMU:
#    SDK_VERSION=0.4.1 source tsi-pkg-build.sh debug-tmu build-fpga
#
# 9) Debug TMU detail (adds TMU_DEBUG_VALIDATE):
#    SDK_VERSION=0.4.1 source tsi-pkg-build.sh debug-tmu-detail build-posix build-fpga
#
# 10) Build blobs explicitly:
#     SDK_VERSION=0.4.1 source tsi-pkg-build.sh build-all-blobs
#     SDK_VERSION=0.4.1 source tsi-pkg-build.sh build-fpga-blobs
#     SDK_VERSION=0.4.1 source tsi-pkg-build.sh build-posix-blobs
# 10a) Triton MAT_MUL default build:
#     If no triton option is provided, TRITON_MAT_MUL is enabled by default.
#     This builds only the Triton MAT_MUL kernel and passes -DTRITON_MAT_MUL=1
#     to ggml-tsavorite.cpp.
#     SDK_VERSION=0.4.9 source tsi-pkg-build.sh build-posix
#
# 10b) Triton MAT_MUL explicit build:
#     Same as default, but explicitly selects Triton MAT_MUL.
#     SDK_VERSION=0.4.9 source tsi-pkg-build.sh triton mat_mul build-posix
#
# 10c) Triton ADD explicit build:
#     Selects Triton ADD and disables Triton MAT_MUL for this build.
#     SDK_VERSION=0.4.9 source tsi-pkg-build.sh triton add build-posix
#
# 10d) Triton ADD + MAT_MUL explicit build:
#     Selects both Triton ADD and Triton MAT_MUL when compiler support is available.
#     SDK_VERSION=0.4.9 source tsi-pkg-build.sh triton all build-posix
#
# 11) Incremental builds (do not delete build dirs):
#     SDK_VERSION=0.4.1 source tsi-pkg-build.sh incremental build-posix build-fpga
#
# 12) Provide explicit paths:
#     SDK_VERSION=0.4.1 source tsi-pkg-build.sh debug build-fpga /path/to/compiler /path/to/toolbox/install-fpga
#
# 13) Package only (existing FPGA build dir already built):
#     SDK_VERSION=0.4.1 source tsi-pkg-build.sh package
#
# 14) FPGA packaging:
#     Ensure ./tsavorite-model-deployment.yaml exists in the current
#     working directory before running package.
#
#       SDK_VERSION=0.4.1 source tsi-pkg-build.sh build-fpga package
# ==============================================================================

log_error(){ echo "ERROR: $*" >&2; }
log_info(){ echo "INFO: $*"; }


if [ -z "$SDK_VERSION" ]; then
  echo "ERROR: SDK_VERSION not set. Usage: SDK_VERSION=<version> source tsi-pkg-build.sh"
  return 1
fi

export SDK_VERSION


__TSI_SOURCED=0
(return 0 2>/dev/null) && __TSI_SOURCED=1
__TSI_OLD_SET="$(set +o)"
__TSI_SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
__TSI_SCRIPT_DIR="$(cd "$(dirname "${__TSI_SCRIPT_PATH}")" 2>/dev/null && pwd)"


# --- VENV TRACKING (FIX) ---
# If script activates blob-creation venv, restore previous env when sourced.
__OLD_VIRTUAL_ENV=""
__TSI_CHANGED_VENV=0

run() {
  "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    log_error "cmd failed ($rc): $*"
    return $rc
  fi
  return 0
}

absdir() { (cd "$1" 2>/dev/null && pwd); }

tolower(){ echo "$1" | tr '[:upper:]' '[:lower:]'; }

die() {
  log_error "$*"
  if [ "$__TSI_SOURCED" -eq 1 ]; then return 1; else exit 1; fi
}

cleanup() {
  # --- VENV RESTORE (FIX) ---
  if [ "${__TSI_CHANGED_VENV:-0}" -eq 1 ]; then
    if declare -F deactivate >/dev/null 2>&1; then
      deactivate >/dev/null 2>&1 || true
    else
      unset VIRTUAL_ENV 2>/dev/null || true
    fi
    # If user was in a previous venv before we activated blob-creation, restore it.
    if [ -n "${__OLD_VIRTUAL_ENV}" ] && [ -f "${__OLD_VIRTUAL_ENV}/bin/activate" ]; then
      # shellcheck disable=SC1090
      source "${__OLD_VIRTUAL_ENV}/bin/activate" >/dev/null 2>&1 || true
    fi
  fi
  # restore caller shell behavior
  eval "${__TSI_OLD_SET}" >/dev/null 2>&1 || true
  stty sane 2>/dev/null || true
  trap - RETURN EXIT 2>/dev/null || true
}

usage() {
  local p="${__TSI_SCRIPT_PATH}"
  if [ -r "$p" ]; then
    sed -n '1,320p' "$p" 2>/dev/null | sed 's/^# \{0,1\}//'
    return 0
  fi
  # fallback (should rarely happen)
  cat <<'EOF'
tsi-pkg-build.sh: unable to read script header for help output.
Try: cat tsi-pkg-build.sh | sed -n '1,320p'
EOF
  return 0
}

select_arch() {
  local m; m="$(uname -m)"
  case "$m" in
    x86_64|amd64) echo "x86_64" ;;
    aarch64|arm64) echo "aarch64" ;;
    *) log_error "Unsupported host arch from uname -m: $m"; return 2 ;;
  esac
}

# -------------------------
# Submodule logic (robust)
# -------------------------
MARKER_FILE=".tsi_submodules_initialized"
SUBMODULE_DIR="ggml-tsi-kernel"

submodule_self_heal_if_needed() {
  # If the path exists but is not a proper submodule checkout and is non-empty, wipe it.
  if [ -e "${SUBMODULE_DIR}" ]; then
    if [ ! -d "${SUBMODULE_DIR}/.git" ] && [ -n "$(ls -A "${SUBMODULE_DIR}" 2>/dev/null || true)" ]; then
      log_info "${SUBMODULE_DIR} exists and is non-empty (stale). Cleaning to allow submodule clone."
      run git submodule deinit -f -- "${SUBMODULE_DIR}" || true
      run rm -rf "${SUBMODULE_DIR}" || true
      run rm -rf ".git/modules/${SUBMODULE_DIR}" || true
      return 1
    fi
  fi
  return 0
}

ensure_submodules() {
  local want_update="$1" # 0/1 from user flag
  local force=0
  # If submodule directory missing, ALWAYS force init.
  if [ ! -d "${SUBMODULE_DIR}" ]; then
    log_info "${SUBMODULE_DIR} missing; forcing submodule init"
    force=1
  fi
  # If marker missing, treat as first-time repo.
  if [ ! -f "${MARKER_FILE}" ]; then
    force=1
  fi
  # User asked explicitly.
  if [ "${want_update}" -eq 1 ]; then
    force=1
  fi
  if [ "${force}" -eq 1 ]; then
    submodule_self_heal_if_needed || true
    run git submodule update --init --recursive || die "git submodule update failed"
    : > "${MARKER_FILE}" || true
  else
    log_info "Skipping git submodule update (already initialized). Use git-submodule-pull to refresh."
  fi
  [ -d "${SUBMODULE_DIR}" ] || die "${SUBMODULE_DIR} still missing after submodule init"
  return 0
}

# -------------------------
# Args/flags
# -------------------------
parse_args() {
  SHOW_HELP=0
  BUILD_TYPE=""
  # Always reset SDK-derived env on every script invocation.
  # SDK_VERSION is the only required input. Old exported paths must not leak
  # across runs when switching SDK versions.
  unset MLIR_COMPILER_DIR
  unset TOOLBOX_DIR
  unset TOOLBOX_DIR_EXPLICIT
  unset FPGA_TOOLBOX_DIR
  unset TSICommon_DIR
  unset MLIR_SDK_VERSION
  unset COMPILER_INSTALL_DIR
  unset FAU_LOOKUP_TABLE_PATH

  MLIR_COMPILER_DIR_IN=""
  TOOLBOX_DIR_IN=""
  ENABLE_COVERAGE_FLAG=""

  # submodules
  GIT_SUBMODULE_PULL=0

  # blobs
  DO_BLOB_FPGA=0
  DO_BLOB_POSIX=0

  # python venv
  OVERWRITE_VENV=0

  # build selection (default: posix+fpga+package)
  DO_BUILD_POSIX=1
  DO_BUILD_POSIX_TMU_ONLY=0
  DO_BUILD_POSIX_TMU_DISABLE=0
  DO_BUILD_FPGA=1
  DO_BUILD_FPGA_TMU_ONLY=0
  DO_BUILD_FPGA_TMU_DISABLE=0
  DO_PACKAGE_FPGA=1
  __USER_BUILD_SELECT=0

  # cleanup
  DO_CLEAN=0
  DO_CLEAN_ALL=0

  # cleaning build dirs before build (default ON)
  DO_CLEAN_BUILD_DIRS=1
  INCREMENTAL=0

  # auto blobs (default ON; applies to POSIX+FPGA host object link safety)
  AUTO_BLOBS=1

  # packaging selection
  PACKAGE_FPGA_BUILD_DIR=""
  # Triton kernel selection
  # Default behavior is equivalent to: triton all
  # TODO: keep "triton all" option for now for backward compatibility.
  # Later release can remove explicit "all" option once default behavior is stable.
  # User can override with:
  #   triton add
  #   triton mat_mul
  #   triton all
  ENABLE_TRITON_ADD=1
  ENABLE_TRITON_MAT_MUL=1
  __EXPECT_TRITON_ARG=0
  ENABLE_TRITON_DEBUG=0

  local a
  for a in "$@"; do

    case "$(tolower "$a")" in
      help|-h|--help|-help)
        SHOW_HELP=1
        return 0
        ;;
      release|debug|debug-tmu|debug-tmu-detail)
        [ -z "${BUILD_TYPE}" ] && BUILD_TYPE="$a"
        ;;
      enable_coverage)
        ENABLE_COVERAGE_FLAG="-DENABLE_COVERAGE=ON"
        log_info "enable_coverage detected"
        ;;
      git-submodule-pull)
        GIT_SUBMODULE_PULL=1
        log_info "git-submodule-pull detected"
        ;;
      triton)
        __EXPECT_TRITON_ARG=1
        log_info "triton option detected; expecting one of: add | mat_mul | all"
        ;;

      triton-debug)
         ENABLE_TRITON_DEBUG=1
         log_info "TRITON_DEBUG enabled"
         ;;

      add)
        if [ "${__EXPECT_TRITON_ARG}" -eq 1 ]; then
          ENABLE_TRITON_ADD=1
          ENABLE_TRITON_MAT_MUL=0
          __EXPECT_TRITON_ARG=0
          log_info "TRITON_ADD selected"
        elif [ -z "${MLIR_COMPILER_DIR_IN}" ]; then
          MLIR_COMPILER_DIR_IN="$a"
        elif [ -z "${TOOLBOX_DIR_IN}" ]; then
          TOOLBOX_DIR_IN="$a"
        fi
        ;;

      mat_mul|mat-mul|matmul)
        if [ "${__EXPECT_TRITON_ARG}" -eq 1 ]; then
          ENABLE_TRITON_ADD=0
          ENABLE_TRITON_MAT_MUL=1
          __EXPECT_TRITON_ARG=0
          log_info "TRITON_MAT_MUL selected"
        elif [ -z "${MLIR_COMPILER_DIR_IN}" ]; then
          MLIR_COMPILER_DIR_IN="$a"
        elif [ -z "${TOOLBOX_DIR_IN}" ]; then
          TOOLBOX_DIR_IN="$a"
        fi
        ;;

      all)
        if [ "${__EXPECT_TRITON_ARG}" -eq 1 ]; then
          ENABLE_TRITON_ADD=1
          ENABLE_TRITON_MAT_MUL=1
          __EXPECT_TRITON_ARG=0
          log_info "TRITON_ADD + TRITON_MAT_MUL selected"
        elif [ -z "${MLIR_COMPILER_DIR_IN}" ]; then
          MLIR_COMPILER_DIR_IN="$a"
        elif [ -z "${TOOLBOX_DIR_IN}" ]; then
          TOOLBOX_DIR_IN="$a"
        fi
        ;;
      build-fpga-blobs)
        DO_BLOB_FPGA=1
        log_info "build-fpga-blobs detected"
        ;;
      build-posix-blobs)
        DO_BLOB_POSIX=1
        log_info "build-posix-blobs detected"
        ;;
      build-all-blobs)
        DO_BLOB_FPGA=1
        DO_BLOB_POSIX=1
        log_info "build-all-blobs detected"
        ;;
      overwrite-venv)
        OVERWRITE_VENV=1
        log_info "overwrite-venv detected"
        ;;
      no-auto-blobs)
        AUTO_BLOBS=0
        log_info "no-auto-blobs detected"
        ;;
      incremental)
        INCREMENTAL=1
        DO_CLEAN_BUILD_DIRS=0
        log_info "incremental build selected (no rm -rf build dirs)"
        ;;
      build-posix|posix)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_POSIX=1
        log_info "build-posix selected"
        ;;
      build-posix-tmu-only)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_POSIX_TMU_ONLY=1
        log_info "build-posix-tmu-only selected"
        ;;
      build-posix-tmu-disable)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_POSIX_TMU_DISABLE=1
        log_info "build-posix-tmu-disable selected"
        ;;
      build-fpga|fpga)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_FPGA=1
        PACKAGE_FPGA_BUILD_DIR="build-fpga"
        log_info "build-fpga selected"
        ;;
      build-fpga-tmu-only)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_FPGA_TMU_ONLY=1
        PACKAGE_FPGA_BUILD_DIR="build-fpga-tmu-only"
        log_info "build-fpga-tmu-only selected"
        ;;
      build-fpga-tmu-disable)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_FPGA_TMU_DISABLE=1
        PACKAGE_FPGA_BUILD_DIR="build-fpga-tmu-disable"
        log_info "build-fpga-tmu-disable selected"
        ;;
      package|bundle)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_PACKAGE_FPGA=1
        log_info "package selected"
        ;;
      clean)
        DO_CLEAN=1
        log_info "clean selected"
        ;;
      clean-all)
        DO_CLEAN_ALL=1
        log_info "clean-all selected"
        ;;
      *)
        # positional paths
        if [ -z "${MLIR_COMPILER_DIR_IN}" ]; then
          MLIR_COMPILER_DIR_IN="$a"
        elif [ -z "${TOOLBOX_DIR_IN}" ]; then
          TOOLBOX_DIR_IN="$a"
        fi
        ;;
    esac
  done

  # git-submodule-pull ALSO deletes+recreates venv and builds all blobs.
  if [ "${GIT_SUBMODULE_PULL}" -eq 1 ]; then
    OVERWRITE_VENV=1
    DO_BLOB_FPGA=1
    DO_BLOB_POSIX=1
    log_info "git-submodule-pull => forcing overwrite-venv + build-all-blobs"
  fi

  # Default build type if none provided
  if [ -z "${BUILD_TYPE}" ]; then
    BUILD_TYPE="debug"
  fi

  # If user wrote only "triton" without argument → error
  if [ "${__EXPECT_TRITON_ARG}" -eq 1 ]; then
    die "Missing Triton kernel argument. Use: triton add OR triton mat_mul OR triton all"
  fi

  # Export so other functions/scripts can use it
  export ENABLE_TRITON_ADD
  export ENABLE_TRITON_MAT_MUL
  export ENABLE_TRITON_DEBUG
  return 0
}

resolve_paths() {
    local arch="$1"

    if [ -z "${MLIR_COMPILER_DIR_IN}" ]; then
        MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-/proj/rel/sw/tsi-sw/staging/sdk/sdk-r.${SDK_VERSION}/${arch}}"
        MLIR_COMPILER_DIR_IN="${MLIR_SDK_VERSION}/compiler"
    fi

    TOOLBOX_DIR_EXPLICIT=1
    if [ -z "${TOOLBOX_DIR_IN}" ]; then
        TOOLBOX_DIR_EXPLICIT=0
        MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-$(dirname "${MLIR_COMPILER_DIR_IN}")}"
        # Toolbox comes from the SDK (${MLIR_SDK_VERSION}/toolbox/build/install-<target>),
        # matching llama.cpp's single SDK_VERSION-driven build. Default now follows the
        # requested build target -- previously this always fell back to install-fpga,
        # even for posix-only builds. The documented trailing positional TOOLBOX_DIR
        # argument still overrides (TOOLBOX_DIR_EXPLICIT=1 above), including for the
        # FPGA-specific steps resolved by resolve_fpga_toolbox_dir() below. Note a bare
        # `TOOLBOX_DIR=...` environment variable does NOT: it's unset unconditionally
        # above (and TOOLBOX_DIR_IN is always reset to "") to avoid a stale exported
        # value leaking into a later run with a different SDK_VERSION in the same shell.
        if { [ "${DO_BUILD_FPGA:-0}" -eq 1 ] || [ "${DO_BUILD_FPGA_TMU_ONLY:-0}" -eq 1 ] || [ "${DO_BUILD_FPGA_TMU_DISABLE:-0}" -eq 1 ]; } \
          && ! { [ "${DO_BUILD_POSIX:-0}" -eq 1 ] || [ "${DO_BUILD_POSIX_TMU_ONLY:-0}" -eq 1 ] || [ "${DO_BUILD_POSIX_TMU_DISABLE:-0}" -eq 1 ]; }; then
            TOOLBOX_DIR_IN="${MLIR_SDK_VERSION}/toolbox/build/install-fpga"
        else
            TOOLBOX_DIR_IN="${MLIR_SDK_VERSION}/toolbox/build/install-posix"
        fi
    fi

    MLIR_COMPILER_DIR="$(absdir "${MLIR_COMPILER_DIR_IN}")"
    [ -n "${MLIR_COMPILER_DIR}" ] || die "MLIR_COMPILER_DIR not found: ${MLIR_COMPILER_DIR_IN}"

    TOOLBOX_DIR="$(absdir "${TOOLBOX_DIR_IN}")"
    [ -n "${TOOLBOX_DIR}" ] || die "TOOLBOX_DIR not found: ${TOOLBOX_DIR_IN}"

    TSICommon_DIR="${TOOLBOX_DIR}/lib/cmake/TSICommon"
    [ -d "${TSICommon_DIR}" ] || die "TSICommon_DIR not found: ${TSICommon_DIR}"

    export TSICommon_DIR
    export MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-$(dirname "${MLIR_COMPILER_DIR}")}"
    export MLIR_COMPILER_DIR
    export COMPILER_INSTALL_DIR="${MLIR_COMPILER_DIR}"
    export TOOLBOX_DIR
    export TOOLBOX_DIR_EXPLICIT
    export FAU_LOOKUP_TABLE_PATH="${MLIR_SDK_VERSION}/ffm/txe-ffm-cpp/third-party/FAU/include/"

    log_info "SDK_VERSION:        ${SDK_VERSION}"
    log_info "MLIR_COMPILER_DIR: ${MLIR_COMPILER_DIR}"
    log_info "TOOLBOX_DIR:       ${TOOLBOX_DIR}"
    log_info "TSICommon_DIR:     ${TSICommon_DIR}"
}

# Resolves the toolbox path that FPGA-specific build steps (blob cross-compile
# toolchain resolution, ARM linker flags) must use. Independent of whichever
# target the general TOOLBOX_DIR above resolved to -- which is install-posix
# in a combined `build-posix build-fpga` invocation (the default) -- so FPGA
# steps don't end up consuming host-native posix toolbox content instead of
# the ARM/Xtensa-flavored install-fpga content they actually need. Honors an
# explicit TOOLBOX_DIR override (the documented trailing positional argument --
# a bare TOOLBOX_DIR environment variable is not captured, see resolve_paths())
# rather than silently replacing it with the SDK's install-fpga.
resolve_fpga_toolbox_dir() {
    if [ "${TOOLBOX_DIR_EXPLICIT:-0}" -eq 1 ]; then
        # An explicit override is a single directory the caller asserts is correct
        # for everything they're building -- same contract the (pre-target-aware)
        # TOOLBOX_DIR override always had. We can't reliably auto-detect "is this
        # install-fpga vs install-posix" without hardcoding SDK-version-specific
        # file names (the toolchain cmake files themselves are byte-identical
        # between the two installs on every SDK release checked so far), which is
        # exactly the kind of hardcode this PR removes elsewhere. So: a cheap
        # sanity check that it's a toolbox install at all, plus a visible NOTE
        # instead of silently trusting it for FPGA-specific steps too.
        [ -f "${TOOLBOX_DIR}/lib/cmake/toolchains/arm.cmake" ] || { die "Explicit TOOLBOX_DIR doesn't look like a toolbox install (missing lib/cmake/toolchains/arm.cmake): ${TOOLBOX_DIR}"; return 1; }
        FPGA_TOOLBOX_DIR="${TOOLBOX_DIR}"
        log_info "NOTE: explicit TOOLBOX_DIR is also being used for FPGA-specific steps (ARM toolchain file + FPGA link libs) -- caller is responsible for it being FPGA-appropriate if building for FPGA."
    else
        FPGA_TOOLBOX_DIR="$(absdir "${MLIR_SDK_VERSION}/toolbox/build/install-fpga")"
        [ -n "${FPGA_TOOLBOX_DIR}" ] || { die "FPGA_TOOLBOX_DIR not found: ${MLIR_SDK_VERSION}/toolbox/build/install-fpga"; return 1; }
    fi
    export FPGA_TOOLBOX_DIR
    log_info "FPGA_TOOLBOX_DIR:  ${FPGA_TOOLBOX_DIR}"
}

setup_toolchain() {
  export CC="/proj/local/gcc-13.3.0/bin/gcc"
  export CXX="/proj/local/gcc-13.3.0/bin/g++"
  export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:${LD_LIBRARY_PATH:-}"
}

# -------------------------
# Python venv (only when needed for blob generation)
# -------------------------
setup_python() {
  # Save caller venv so cleanup() can restore it.
  __OLD_VIRTUAL_ENV="${VIRTUAL_ENV:-}"

  if [ "${OVERWRITE_VENV}" -eq 1 ] && [ -d "blob-creation" ]; then
    log_info "overwrite-venv: removing existing blob-creation venv"
    rm -rf blob-creation || return 1
  fi

  if [ -d "blob-creation" ] && [ -f "blob-creation/bin/activate" ]; then
    run bash -c 'source blob-creation/bin/activate && python -V >/dev/null' || return 1
    # shellcheck disable=SC1091
    source blob-creation/bin/activate || return 1
    [ "${VIRTUAL_ENV:-}" != "${__OLD_VIRTUAL_ENV:-}" ] && __TSI_CHANGED_VENV=1 || true
  else
    run /proj/local/Python-3.11.12/bin/python3 -m venv blob-creation || return 1
    run bash -c 'source blob-creation/bin/activate && python -V >/dev/null' || return 1
    # shellcheck disable=SC1091
    source blob-creation/bin/activate || return 1
    [ "${VIRTUAL_ENV:-}" != "${__OLD_VIRTUAL_ENV:-}" ] && __TSI_CHANGED_VENV=1 || true
  fi

  log_info "installing mlir / triton python dependencies from SDK"
  run pip install --upgrade pip || return 1

  local REQ_DIR="${MLIR_COMPILER_DIR}/python"
  local TRITON_DIR="${MLIR_SDK_VERSION}/triton"

  [ -d "${REQ_DIR}" ] || die "MLIR python directory not found: ${REQ_DIR}"
  [ -d "${TRITON_DIR}" ] || die "SDK Triton directory not found: ${TRITON_DIR}"

  # ---------------------------------------------------------------------------
  # Install latest mlir_external_packages wheel from SDK compiler/python
  # (your manual flow tried 1.8.2, 1.8.3, 1.9.1; safest is pick highest version)
  # ---------------------------------------------------------------------------
  local MLIR_WHL
  MLIR_WHL="$(ls -1 "${REQ_DIR}"/mlir_external_packages-*.whl 2>/dev/null | sort -V | tail -1 || true)"
  [ -n "${MLIR_WHL}" ] || die "No mlir_external_packages-*.whl found in ${REQ_DIR}"
  log_info "Installing MLIR wheel: ${MLIR_WHL}"
  run pip install "${MLIR_WHL}" || return 1

  # ---------------------------------------------------------------------------
  # Install latest Tsavorite Triton wheel from SDK triton/
  # (manual flow used triton_tsiai-1.0.0 / 0.1.3)
  # ---------------------------------------------------------------------------
  local TRITON_WHL
  TRITON_WHL="$(ls -1 "${TRITON_DIR}"/triton_tsiai-*.whl 2>/dev/null | sort -V | tail -1 || true)"
  [ -n "${TRITON_WHL}" ] || die "No triton_tsiai-*.whl found in ${TRITON_DIR}"
  log_info "Installing Triton wheel: ${TRITON_WHL}"
  run pip install "${TRITON_WHL}" || return 1

  # ---------------------------------------------------------------------------
  # Install compiler python requirements
  # Keep the existing rewrite for bad '-r /python/...' includes
  # ---------------------------------------------------------------------------
  local REQ_MAIN="${REQ_DIR}/requirements-common.txt"
  if [ ! -f "${REQ_MAIN}" ]; then
    die "requirements-common.txt not found: ${REQ_MAIN}"
  fi

  if grep -qE '^[[:space:]]*-r[[:space:]]+/python/' "${REQ_MAIN}"; then
    log_info "requirements-common.txt contains absolute /python includes; rewriting to ${REQ_DIR}"
    local REQ_TMP
    REQ_TMP="$(mktemp -t tsi-req-XXXXXX.txt)" || return 1
    sed -E "s|(^[[:space:]]*-r[[:space:]]+)/python/|\\1${REQ_DIR}/|g" "${REQ_MAIN}" > "${REQ_TMP}"
    run pip install -r "${REQ_TMP}" || {
      rm -f "${REQ_TMP}" >/dev/null 2>&1 || true
      return 1
    }
    rm -f "${REQ_TMP}" >/dev/null 2>&1 || true
  else
    run pip install -r "${REQ_MAIN}" || return 1
  fi

  # Optional packages that your manual flow used later
  if ! pip show torch >/dev/null 2>&1; then
    run pip install torch==2.7.0 || return 1
  fi

  if ! pip show onnxruntime-training >/dev/null 2>&1; then
    run pip install onnxruntime-training || return 1
  fi

  # ---------------------------------------------------------------------------
  # Export runtime/library paths needed by create-all-kernels.sh
  # based on your manual env setup
  # ---------------------------------------------------------------------------
  # This venv/python toolchain always runs natively on the host (x86_64), regardless
  # of build target, so it needs the native posix toolbox libs here specifically --
  # not the possibly-fpga-flavored TOOLBOX_DIR resolved above.
  export LD_LIBRARY_PATH="${MLIR_SDK_VERSION}/toolbox/build/install-posix/lib:${LD_LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${MLIR_SDK_VERSION}/ffm/txe-ffm-cpp/lib:${LD_LIBRARY_PATH}"
  export LD_LIBRARY_PATH="${MLIR_SDK_VERSION}/ffm/txe-ffm-wrapper/lib:${LD_LIBRARY_PATH}"

  export LIBRARY_PATH="${MLIR_SDK_VERSION}/ffm/txe-ffm-cpp/lib:${LIBRARY_PATH:-}"
  export LIBRARY_PATH="${MLIR_SDK_VERSION}/ffm/txe-ffm-wrapper/lib:${LIBRARY_PATH}"

  # Helpful trace
  log_info "Python: $(python -V 2>/dev/null)"
  log_info "Using MLIR wheel: ${MLIR_WHL}"
  log_info "Using Triton wheel: ${TRITON_WHL}"
  python -c "import triton; print('INFO: triton module:', triton.__file__)" || return 1

  return 0
}

# -------------------------
# Blob presence + build helpers
# -------------------------
posix_host_objs_present() {
  [ -d "posix-kernel/build-posix" ] || return 1
  find "posix-kernel/build-posix" -name "host.o" -print -quit 2>/dev/null | grep -q . || return 1
  return 0
}

fpga_host_objs_present() {
  [ -d "fpga-kernel/build-fpga" ] || return 1
  find "fpga-kernel/build-fpga" -name "host.o" -print -quit 2>/dev/null | grep -q . || return 1
  return 0
}

build_fpga_blobs() {
  log_info "BLOB: building FPGA kernels/blobs"
  resolve_fpga_toolbox_dir || return 1
  cd fpga-kernel || return 1
  run cmake -B build-fpga -DTOOLBOX_DIR="${FPGA_TOOLBOX_DIR}" -DCOMPILER_INSTALL_DIR="${MLIR_COMPILER_DIR}" || return 1
  run ./create-all-kernels.sh || return 1
  cd .. || return 1
  return 0
}

build_posix_blobs() {
  log_info "BLOB: building POSIX kernels/blobs"
  cd posix-kernel || return 1
  run ./create-all-kernels.sh || return 1
  cd .. || return 1
  return 0
}

# -------------------------
# PERF/DEBUG defs
# -------------------------
compute_perf_and_debug_defs() {
  local target="$1" # posix|fpga
  local bt; bt="$(tolower "${BUILD_TYPE}")"

  PERF_DEF="-DGGML_PERF"
  DBG_DEFS=""

  if [ "$bt" = "release" ]; then
    PERF_DEF="-DGGML_PERF_RELEASE"
    DBG_DEFS=""
    return 0
  fi

  if [ "$bt" = "debug" ]; then
    if [ "$target" = "fpga" ]; then
      # FPGA debug: disable GGML_PERF_DETAIL (expensive file logging on FPGA)
      PERF_DEF="-DGGML_PERF"
    else
      PERF_DEF="-DGGML_PERF_DETAIL"
    fi
    DBG_DEFS=""
    return 0
  fi

  if [ "$bt" = "debug-tmu" ]; then
    PERF_DEF="-DGGML_PERF_DETAIL"
    DBG_DEFS="-DTMU_DEBUG"
    return 0
  fi

  if [ "$bt" = "debug-tmu-detail" ]; then
    PERF_DEF="-DGGML_PERF_DETAIL"
    DBG_DEFS="-DTMU_DEBUG -DTMU_DEBUG_VALIDATE"
    return 0
  fi

  return 0
}

# -------------------------
# POSIX build (clean rebuild by default)
# -------------------------
build_posix_impl() {
  local build_dir="$1" # build-posix / build-posix-tmu-only / build-posix-tmu-disable
  local want_tmu="$2"  # 1/0
  local want_tvu="$3"  # 1/0

  log_info "building llama.cpp/ggml for posix (${build_dir})"

  if [ "${DO_CLEAN_BUILD_DIRS}" -eq 1 ]; then
    log_info "clean rebuild: rm -rf ./${build_dir}"
    rm -rf "${build_dir}" || return 1
  fi

  compute_perf_and_debug_defs "posix"

  local common="-DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DGGML_NATIVE=ON -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF"

  local supported=""
  [ "${want_tmu}" -eq 1 ] && supported="${supported} -DTMU_SUPPORTED"
  [ "${want_tvu}" -eq 1 ] && supported="${supported} -DTVU_SUPPORTED"

  local triton_defs="-DTRITON_ADD=${ENABLE_TRITON_ADD} -DTRITON_MAT_MUL=${ENABLE_TRITON_MAT_MUL} -DTRITON_DEBUG=${ENABLE_TRITON_DEBUG}"

  local cflags_base="-DGGML_TARGET_POSIX -DGGML_TSAVORITE ${supported} ${triton_defs} -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"

  run cmake -B "${build_dir}" ${common} \
    -DCMAKE_C_COMPILER="${CC}" -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="${PERF_DEF} ${DBG_DEFS} ${cflags_base}" \
    -DCMAKE_CXX_FLAGS="${PERF_DEF} ${DBG_DEFS} ${cflags_base}" \
-DCMAKE_EXE_LINKER_FLAGS="-L/proj/local/gcc-13.3.0/lib64 -Wl,-rpath-link,/proj/local/gcc-13.3.0/lib64 -Wl,-rpath,/proj/local/gcc-13.3.0/lib64 -L/usr/lib64 -lomp -lgcc_s" \
-DCMAKE_SHARED_LINKER_FLAGS="-L/proj/local/gcc-13.3.0/lib64 -Wl,-rpath-link,/proj/local/gcc-13.3.0/lib64 -Wl,-rpath,/proj/local/gcc-13.3.0/lib64 -L/usr/lib64 -lomp -lgcc_s" \
    ${ENABLE_COVERAGE_FLAG} || return 1

  run cmake --build "${build_dir}" --config Release || return 1
  return 0
}

build_posix() { build_posix_impl "build-posix" 1 1; }
build_posix_tmu_only() { build_posix_impl "build-posix-tmu-only" 1 0; }
build_posix_tmu_disable() { build_posix_impl "build-posix-tmu-disable" 0 1; }

wrap_glibc_bins() {
  local build_dir="$1"
  log_info "fixing GLIBC compatibility for TSI binaries (${build_dir})"

  if [ -f "${build_dir}/bin/simple-backend-tsi" ] && [ ! -f "${build_dir}/bin/simple-backend-tsi-original" ]; then
    mv "${build_dir}/bin/simple-backend-tsi" "${build_dir}/bin/simple-backend-tsi-original" || return 1
    cat > "${build_dir}/bin/simple-backend-tsi" <<'EOL'
#!/bin/bash
export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
exec "$(dirname "$0")/simple-backend-tsi-original" "$@"
EOL
    chmod +x "${build_dir}/bin/simple-backend-tsi" || return 1
  fi

  if [ -f "${build_dir}/bin/llama-cli" ] && [ ! -f "${build_dir}/bin/llama-cli-original" ]; then
    mv "${build_dir}/bin/llama-cli" "${build_dir}/bin/llama-cli-original" || return 1
    cat > "${build_dir}/bin/llama-cli" <<'EOL'
#!/bin/bash
export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
exec "$(dirname "$0")/llama-cli-original" "$@"
EOL
    chmod +x "${build_dir}/bin/llama-cli" || return 1
  fi

  return 0
}

# -------------------------
# FPGA build (clean rebuild by default)
# -------------------------
build_fpga_impl() {
  local build_dir="$1" # build-fpga / build-fpga-tmu-only / build-fpga-tmu-disable
  local want_tmu="$2"  # 1/0
  local want_tvu="$3"  # 1/0

  log_info "building llama.cpp/ggml for fpga (${build_dir})"

  if [ "${DO_CLEAN_BUILD_DIRS}" -eq 1 ]; then
    log_info "clean rebuild: rm -rf ./${build_dir}"
    rm -rf "${build_dir}" || return 1
  fi

  compute_perf_and_debug_defs "fpga"

  resolve_fpga_toolbox_dir || return 1

  # The ARM cross-compile toolchain file and link libs always need the
  # FPGA/ARM-flavored toolbox content, regardless of what the general
  # TOOLBOX_DIR resolved to above (which follows the requested build target
  # and would be install-posix for a combined posix+fpga invocation) -- so
  # both are derived from FPGA_TOOLBOX_DIR (resolve_fpga_toolbox_dir()),
  # not the possibly-posix-flavored TOOLBOX_DIR.
  local ARM_TOOLCHAIN_FILE="${FPGA_TOOLBOX_DIR}/lib/cmake/toolchains/arm.cmake"
  local FPGA_TOOLBOX_LIB_DIR="${FPGA_TOOLBOX_DIR}/lib"

  local supported=""
  [ "${want_tmu}" -eq 1 ] && supported="${supported} -DTMU_SUPPORTED"
  [ "${want_tvu}" -eq 1 ] && supported="${supported} -DTVU_SUPPORTED"

  local triton_defs="-DTRITON_ADD=${ENABLE_TRITON_ADD} -DTRITON_MAT_MUL=${ENABLE_TRITON_MAT_MUL} -DTRITON_DEBUG=${ENABLE_TRITON_DEBUG}"

  run cmake -B "${build_dir}" \
    -DCMAKE_TOOLCHAIN_FILE="${ARM_TOOLCHAIN_FILE}" \
    -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DLLAMA_CURL=OFF \
    -DCMAKE_C_FLAGS="${PERF_DEF} ${DBG_DEFS} -DGGML_TSAVORITE ${supported} ${triton_defs}" \
    -DCMAKE_CXX_FLAGS="${PERF_DEF} ${DBG_DEFS} -DGGML_TSAVORITE ${supported} ${triton_defs}" \
-DCMAKE_EXE_LINKER_FLAGS="-L${FPGA_TOOLBOX_LIB_DIR} -Wl,-rpath-link,${FPGA_TOOLBOX_LIB_DIR} -Wl,-rpath,${FPGA_TOOLBOX_LIB_DIR} -lomp" \
-DCMAKE_SHARED_LINKER_FLAGS="-L${FPGA_TOOLBOX_LIB_DIR} -Wl,-rpath-link,${FPGA_TOOLBOX_LIB_DIR} -Wl,-rpath,${FPGA_TOOLBOX_LIB_DIR} -lomp" \
    ${ENABLE_COVERAGE_FLAG} || return 1

  run cmake --build "${build_dir}" --config Release || return 1
  return 0
}

build_fpga() { build_fpga_impl "build-fpga" 1 1; }
build_fpga_tmu_only() { build_fpga_impl "build-fpga-tmu-only" 1 0; }
build_fpga_tmu_disable() { build_fpga_impl "build-fpga-tmu-disable" 0 1; }

choose_existing_fpga_build_dir_for_package() {
  # If user explicitly selected a package build dir, prefer it.
  if [ -n "${PACKAGE_FPGA_BUILD_DIR}" ] && [ -f "${PACKAGE_FPGA_BUILD_DIR}/bin/llama-cli" ]; then
    echo "${PACKAGE_FPGA_BUILD_DIR}"
    return 0
  fi
  # Otherwise, pick the first viable build dir in priority order.
  local d
  for d in build-fpga build-fpga-tmu-only build-fpga-tmu-disable; do
    if [ -f "${d}/bin/llama-cli" ]; then
      echo "${d}"
      return 0
    fi
  done
  # None found.
  echo ""
  return 0
}

bundle_fpga() {
  local build_dir="$1"
  log_info "creating tar bundle for fpga (${build_dir})"

  local TSI_GGML_VERSION="${SDK_VERSION}"
  local TSI_GGML_BUNDLE_INSTALL_DIR=tsi-ggml
  local GGML_TSI_INSTALL_DIR=ggml-tsi-kernel
  local TSI_GGML_RELEASE_DIR=/proj/rel/sw/ggml
  local TSI_BLOB_INSTALL_DIR

  TSI_BLOB_INSTALL_DIR="$(pwd)/${GGML_TSI_INSTALL_DIR}/fpga-kernel/build-fpga"

  [ -f "${build_dir}/bin/llama-cli" ] || die "package requested but ${build_dir}/bin/llama-cli not found. Run an FPGA build first."

  mkdir -p "${TSI_GGML_BUNDLE_INSTALL_DIR}"
  rm -f "${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh"

cat > "./${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh" <<'EOL'
#!/bin/bash

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(pwd)

TAOS_CONFIG_PATH="/etc/taos/taos.json"

extract_deployment_yaml_value() {
  local deployment_yaml_path="$1"
  local yaml_key="$2"

  awk -F: -v key="${yaml_key}" '
    $0 ~ "^[[:space:]]*" key "[[:space:]]*:" {
      v=$2

      # Remove an inline YAML comment before quote normalization.
      sub(/[[:space:]]+#.*/, "", v)

      # Trim whitespace.
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)

      # Normalize matching single or double quotes.
      dq=sprintf("%c", 34)
      sq=sprintf("%c", 39)
      if ((substr(v, 1, 1) == dq && substr(v, length(v), 1) == dq) ||
          (substr(v, 1, 1) == sq && substr(v, length(v), 1) == sq)) {
        v = substr(v, 2, length(v) - 2)
      }

      print v
      exit
    }
  ' "${deployment_yaml_path}"
}

update_one_tsavorite_deployment_yaml() {
  local deployment_yaml_path="$1"
  local txe_count="$2"
  local advanced_matmul_shape_offload="false"
  local advanced_matmul_broadcast_offload="false"
  local triton_matmul_small_n_transpose_opt="false"
local user_dram_size_gb="8"

  mkdir -p "$(dirname "${deployment_yaml_path}")" || return 1

  if [ -f "${deployment_yaml_path}" ]; then
    local existing_advanced
    local existing_broadcast
    local existing_small_n_opt
local existing_user_dram_size_gb

    existing_advanced="$(extract_deployment_yaml_value "${deployment_yaml_path}" "advanced_matmul_shape_offload")"
    existing_broadcast="$(extract_deployment_yaml_value "${deployment_yaml_path}" "advanced_matmul_broadcast_offload")"
    existing_small_n_opt="$(extract_deployment_yaml_value "${deployment_yaml_path}" "triton_matmul_small_n_transpose_opt")"

    if [ -n "${existing_advanced}" ]; then
      advanced_matmul_shape_offload="${existing_advanced}"
    fi

    if [ -n "${existing_broadcast}" ]; then
      advanced_matmul_broadcast_offload="${existing_broadcast}"
    fi
    if [ -n "${existing_small_n_opt}" ]; then
      triton_matmul_small_n_transpose_opt="${existing_small_n_opt}"
    fi
  fi

  
existing_user_dram_size_gb="$(
    awk -F: '
    /^[[:space:]]*user_dram_size_gb[[:space:]]*:/ {
        v=$2
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
        print v
        exit
    }' "$deployment_yaml_path"
)"

if [ -n "$existing_user_dram_size_gb" ]; then
    user_dram_size_gb="$existing_user_dram_size_gb"
fi

cat > "${deployment_yaml_path}" <<EOF
# Tsavorite deployment config
txe_count: ${txe_count}
multi_thread_enable: true

## Runtime user DRAM size in GiB.
## Example: 1 = 1GB, 2 = 2GB.
## If this key is missing, runtime DeviceConfig default is used.

user_dram_size_gb: $user_dram_size_gb


# Enable additional Triton MAT_MUL shapes beyond stable baseline.
# false = old behavior
# true  = new offload shapes
advanced_matmul_shape_offload: ${advanced_matmul_shape_offload}

## Enable Triton MAT_MUL broadcast/batched D2/D3 offload.
## false = keep broadcast MAT_MUL on fallback path
## true  = allow advanced MAT_MUL helper to offload supported broadcast shapes
advanced_matmul_broadcast_offload: ${advanced_matmul_broadcast_offload}

# Enable Triton MAT_MUL small-N transpose optimization.
# false = old behavior
# true  = for M >> N, compute swapped [N x M] and transpose copyback to [M x N]
triton_matmul_small_n_transpose_opt: ${triton_matmul_small_n_transpose_opt}
EOF

  echo "INFO: updated ${deployment_yaml_path} with txe_count:${txe_count}, multi_thread_enable:true; preserved advanced_matmul_shape_offload:${advanced_matmul_shape_offload}, advanced_matmul_broadcast_offload:${advanced_matmul_broadcast_offload}, triton_matmul_small_n_transpose_opt:${triton_matmul_small_n_transpose_opt}"
  return 0
}

read_txe_count_from_taos_json() {
  if [ ! -f "${TAOS_CONFIG_PATH}" ]; then
    echo "WARNING: ${TAOS_CONFIG_PATH} not found; using conservative default txe_count=1" >&2
    echo "1"
    return 0
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: ${TAOS_CONFIG_PATH} exists but python3 was not found; cannot parse JSON." >&2
    return 1
  fi

  python3 - <<'PY'
import json
import sys

path = "/etc/taos/taos.json"

try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    print(f"ERROR: failed to parse {path}: {e}", file=sys.stderr)
    sys.exit(2)

if not isinstance(data, dict):
    print(f"ERROR: {path} must contain a JSON object like {{\"txe_count\": 20}}", file=sys.stderr)
    sys.exit(2)

if set(data.keys()) != {"txe_count"}:
    print(f"ERROR: {path} must contain exactly one field: txe_count", file=sys.stderr)
    sys.exit(2)

txe_count = data.get("txe_count")

if isinstance(txe_count, bool) or not isinstance(txe_count, int) or txe_count < 1:
    print(f"ERROR: {path} field txe_count must be an integer >= 1", file=sys.stderr)
    sys.exit(2)

print(txe_count)
PY
}

update_tsavorite_deployment_yaml_from_taos() {
  local txe_count=""
  local script_dir=""

  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

  txe_count="$(read_txe_count_from_taos_json)" || return 1

  update_one_tsavorite_deployment_yaml "${script_dir}/tsavorite-model-deployment.yaml" "${txe_count}" || return 1

  if [ -d "${script_dir}/../bin" ] || [ -f "${script_dir}/../bin/tsavorite-model-deployment.yaml" ]; then
    update_one_tsavorite_deployment_yaml "${script_dir}/../bin/tsavorite-model-deployment.yaml" "${txe_count}" || return 1
  fi

  return 0
}

update_tsavorite_deployment_yaml_from_taos || exit 1

tsi_kernels=(
  "add" "sub" "mult" "div" "abs" "inv" "neg" "sin" "sqrt" "sqr" "sigmoid" "silu" "rms_norm" "swiglu"
  "add_16" "sub_16" "mult_16" "div_16" "abs_16" "inv_16" "neg_16" "sin_16" "sqrt" "sqr" "sigmoid_16" "silu_16" "rms_norm_16" "swiglu_16"
  "mul_mat_tile_f32_k32" "mul_mat_tile_f32_k64" "mul_mat_tile_f32_k128"
)

triton_kernels=(
  "triton_add"
  "triton_mat_mul_1x8"
  "triton_mat_mul_2x4"
)

for kernel in "${tsi_kernels[@]}"; do
  dst="__TSI_BLOB_INSTALL_DIR__/txe_${kernel}/blobs"
  rm -rf "${dst}"
  mkdir -p "${dst}"

  if [ -f "blobs/txe_${kernel}.blob" ]; then
    cp "blobs/txe_${kernel}.blob" "${dst}/txe_${kernel}.blob"
  fi
done

for kernel in "${triton_kernels[@]}"; do
  dst="__TSI_BLOB_INSTALL_DIR__/txe_${kernel}/blobs"
  rm -rf "${dst}"
  mkdir -p "${dst}"

  if [ -f "blobs/txe_${kernel}/txe_blob_0.blob" ]; then
    cp "blobs/txe_${kernel}/txe_blob_0.blob" "${dst}/txe_blob_0.blob"
  fi
done
EOL

  sed -i "s|__TSI_BLOB_INSTALL_DIR__|${TSI_BLOB_INSTALL_DIR}|g" "./${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh"
  chmod +x "./${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh" || return 1

  cp "${GGML_TSI_INSTALL_DIR}/fpga/blobs" "${TSI_GGML_BUNDLE_INSTALL_DIR}/" -r || return 1
  cp "${build_dir}/bin/llama-cli" "${TSI_GGML_BUNDLE_INSTALL_DIR}/" || return 1
  cp "${build_dir}/bin/libggml"*.so "${TSI_GGML_BUNDLE_INSTALL_DIR}/" || return 1
  cp "${build_dir}/bin/libllama"*.so "${TSI_GGML_BUNDLE_INSTALL_DIR}/" || return 1
  cp "${build_dir}/bin/simple-backend-tsi" "${TSI_GGML_BUNDLE_INSTALL_DIR}/" || return 1

if [ ! -f "./tsavorite-model-deployment.yaml" ]; then
    die "required ./tsavorite-model-deployment.yaml not found for FPGA package"
fi
cp "./tsavorite-model-deployment.yaml" "$TSI_GGML_BUNDLE_INSTALL_DIR/tsavorite-model-deployment.yaml" || return 1
log_info "included ./tsavorite-model-deployment.yaml in FPGA package"


  tar -cvzf "${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz" "${TSI_GGML_BUNDLE_INSTALL_DIR}"/* || return 1

  if [ "$(tolower "$BUILD_TYPE")" = "release" ]; then
    cp "${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz" "${TSI_GGML_RELEASE_DIR}/" || return 1

    local LATEST_TZ="${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz"
    local LATEST_FULL_PATH="${TSI_GGML_RELEASE_DIR}/$(basename "$LATEST_TZ")"

    rm -f "${TSI_GGML_RELEASE_DIR}/tsi-ggml-aws-latest.tz" "${TSI_GGML_RELEASE_DIR}/tsi-ggml-latest.tz"
    ln -s "/aws${LATEST_FULL_PATH}" "${TSI_GGML_RELEASE_DIR}/tsi-ggml-aws-latest.tz"
    ln -s "${LATEST_FULL_PATH}" "${TSI_GGML_RELEASE_DIR}/tsi-ggml-latest.tz"

    log_info "Symlinks updated to point to $(basename "$LATEST_FULL_PATH")"
  fi

  return 0
}

# -------------------------
# Cleanup commands
# -------------------------
do_clean() {
  log_info "clean: removing build directories"
  rm -rf \
    build-posix build-posix-tmu-only build-posix-tmu-disable \
    build-fpga build-fpga-tmu-only build-fpga-tmu-disable 2>/dev/null || true
  if [ -d "${SUBMODULE_DIR}" ]; then
    rm -rf "${SUBMODULE_DIR}/fpga-kernel/build-fpga" 2>/dev/null || true
    rm -rf "${SUBMODULE_DIR}/posix-kernel/build-posix" 2>/dev/null || true
  fi
  return 0
}

do_clean_all() {
  do_clean || return 1
  if [ -d "${SUBMODULE_DIR}/blob-creation" ]; then
    log_info "clean-all: removing python venv blob-creation"
    rm -rf "${SUBMODULE_DIR}/blob-creation" || true
  fi
  return 0
}

main() {
  set -o pipefail

  local arch
  local ORIG_PWD
  ORIG_PWD="$(pwd)"

  arch="$(select_arch)" || return $?
  parse_args "$@" || return $?

  if [ "${SHOW_HELP}" -eq 1 ]; then
    usage
    return 0
  fi

  if [ "${DO_CLEAN_ALL}" -eq 1 ]; then
    do_clean_all
    cd "${ORIG_PWD}" >/dev/null 2>&1 || true
    return 0
  fi

  if [ "${DO_CLEAN}" -eq 1 ]; then
    do_clean
    cd "${ORIG_PWD}" >/dev/null 2>&1 || true
    return 0
  fi

  resolve_paths "$arch" || {
    cd "${ORIG_PWD}" >/dev/null 2>&1 || true
    return $?
  }

  setup_toolchain || {
    cd "${ORIG_PWD}" >/dev/null 2>&1 || true
    return 1
  }

  ensure_submodules "${GIT_SUBMODULE_PULL}" || {
    cd "${ORIG_PWD}" >/dev/null 2>&1 || true
    return 1
  }

  local need_python=0
  if [ "${OVERWRITE_VENV}" -eq 1 ] || [ "${DO_BLOB_FPGA}" -eq 1 ] || [ "${DO_BLOB_POSIX}" -eq 1 ]; then
    need_python=1
  fi

  local auto_posix_blob=0
  local auto_fpga_blob=0

  if [ "${AUTO_BLOBS}" -eq 1 ]; then
    cd "${SUBMODULE_DIR}" || {
      cd "${ORIG_PWD}" >/dev/null 2>&1 || true
      return 1
    }

    if [ "${DO_BUILD_POSIX}" -eq 1 ] || [ "${DO_BUILD_POSIX_TMU_ONLY}" -eq 1 ] || [ "${DO_BUILD_POSIX_TMU_DISABLE}" -eq 1 ]; then
      if ! posix_host_objs_present; then
        auto_posix_blob=1
        log_info "POSIX host objects missing => auto-building POSIX blobs to avoid undefined _mlir_ciface_*_host"
        need_python=1
        DO_BLOB_POSIX=1
      fi
    fi

    if [ "${DO_BUILD_FPGA}" -eq 1 ] || [ "${DO_BUILD_FPGA_TMU_ONLY}" -eq 1 ] || [ "${DO_BUILD_FPGA_TMU_DISABLE}" -eq 1 ]; then
      if ! fpga_host_objs_present; then
        auto_fpga_blob=1
        log_info "FPGA host objects missing => auto-building FPGA blobs to avoid undefined _mlir_ciface_*_host"
        need_python=1
        DO_BLOB_FPGA=1
      fi
    fi

    cd "${ORIG_PWD}" || return 1
  fi

  if [ "${need_python}" -eq 1 ] && ( [ "${DO_BLOB_FPGA}" -eq 1 ] || [ "${DO_BLOB_POSIX}" -eq 1 ] ); then
    (
      cd "${SUBMODULE_DIR}" || exit 1
      setup_python || exit 1

      if [ "${DO_BLOB_FPGA}" -eq 1 ]; then
        build_fpga_blobs || exit 1
      fi

      if [ "${DO_BLOB_POSIX}" -eq 1 ]; then
        build_posix_blobs || exit 1
      fi
    )
    local rc=$?
    cd "${ORIG_PWD}" >/dev/null 2>&1 || true
    [ $rc -eq 0 ] || return $rc
  fi

  if [ "${DO_BUILD_POSIX}" -eq 1 ]; then
    build_posix || {
      cd "${ORIG_PWD}" >/dev/null 2>&1 || true
      return 1
    }
    wrap_glibc_bins "build-posix" || {
      cd "${ORIG_PWD}" >/dev/null 2>&1 || true
      return 1
    }
  fi

  if [ "${DO_BUILD_POSIX_TMU_ONLY}" -eq 1 ]; then
    build_posix_tmu_only || {
      cd "${ORIG_PWD}" >/dev/null 2>&1 || true
      return 1
    }
    wrap_glibc_bins "build-posix-tmu-only" || {
      cd "${ORIG_PWD}" >/dev/null 2>&1 || true
      return 1
    }
  fi

  if [ "${DO_BUILD_POSIX_TMU_DISABLE}" -eq 1 ]; then
    build_posix_tmu_disable || {
      cd "${ORIG_PWD}" >/dev/null 2>&1 || true
      return 1
    }
    wrap_glibc_bins "build-posix-tmu-disable" || {
      cd "${ORIG_PWD}" >/dev/null 2>&1 || true
      return 1
    }
  fi

  if [ "${DO_BUILD_FPGA}" -eq 1 ]; then
    build_fpga || {
      cd "${ORIG_PWD}" >/dev/null 2>&1 || true
      return 1
    }
    PACKAGE_FPGA_BUILD_DIR="${PACKAGE_FPGA_BUILD_DIR:-build-fpga}"
  fi

  if [ "${DO_BUILD_FPGA_TMU_ONLY}" -eq 1 ]; then
    build_fpga_tmu_only || {
      cd "${ORIG_PWD}" >/dev/null 2>&1 || true
      return 1
    }
    PACKAGE_FPGA_BUILD_DIR="${PACKAGE_FPGA_BUILD_DIR:-build-fpga-tmu-only}"
  fi

  if [ "${DO_BUILD_FPGA_TMU_DISABLE}" -eq 1 ]; then
    build_fpga_tmu_disable || {
      cd "${ORIG_PWD}" >/dev/null 2>&1 || true
      return 1
    }
    PACKAGE_FPGA_BUILD_DIR="${PACKAGE_FPGA_BUILD_DIR:-build-fpga-tmu-disable}"
  fi

  if [ "${DO_PACKAGE_FPGA}" -eq 1 ]; then
    local pkg_dir
    pkg_dir="$(choose_existing_fpga_build_dir_for_package)"
    [ -n "${pkg_dir}" ] || {
      cd "${ORIG_PWD}" >/dev/null 2>&1 || true
      die "package requested but no FPGA build output found (expected build-fpga / build-fpga-tmu-only / build-fpga-tmu-disable)."
    }
    bundle_fpga "${pkg_dir}" || {
      cd "${ORIG_PWD}" >/dev/null 2>&1 || true
      return 1
    }
  fi

  if [ "${auto_posix_blob}" -eq 1 ]; then
    log_info "NOTE: POSIX blobs were auto-built because they are required for linking _mlir_ciface_*_host."
  fi

  if [ "${auto_fpga_blob}" -eq 1 ]; then
    log_info "NOTE: FPGA blobs were auto-built because they are required for linking _mlir_ciface_*_host."
  fi

  cd "${ORIG_PWD}" >/dev/null 2>&1 || true
  return 0
}


if [ "$__TSI_SOURCED" -eq 1 ]; then
  trap cleanup RETURN
else
  trap cleanup EXIT
fi

main "$@"; __rc=$?
if [ "$__TSI_SOURCED" -eq 1 ]; then
  return "$__rc"
else
  exit "$__rc"
fi
