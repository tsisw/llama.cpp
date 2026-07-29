"""pytest plumbing for the ggml->MLIR export suite.

Cases come from the C++ generator. Either point --cases-root at a directory that already holds
case dirs (stage 2 in isolation), or pass --cases-bin and let the session fixture generate them.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_addoption(parser):
    parser.addoption("--cases-bin", default=None,
                     help="path to the mlir-export-cases binary; generates cases into a tmp dir")
    parser.addoption("--cases-root", default=None,
                     help="directory of pre-generated case dirs (skips generation)")
    parser.addoption("--target", default="ffm", choices=["ffm", "ten"],
                     help="ffm = host-native JIT (default); ten = TXE sim, needs Cadence xt-clang")


XT_CLANG = Path("/proj/vendors/cadence/xtensa/XtDevTools/install/tools/"
                "RJ-2025.5-linux/XtensaTools/bin/xt-clang")


@pytest.fixture(scope="session")
def target(request):
    t = request.config.getoption("--target")
    if t == "ten" and not XT_CLANG.exists():
        pytest.skip(f"--target ten needs the Cadence Xtensa toolchain ({XT_CLANG}); "
                    f"not present on this machine, so TXE blobs cannot be built")
    return t


@pytest.fixture(scope="session")
def cases_root(request, tmp_path_factory):
    root = request.config.getoption("--cases-root")
    if root:
        return Path(root)
    bin_path = request.config.getoption("--cases-bin")
    if not bin_path or not Path(bin_path).exists():
        pytest.skip("pass --cases-bin <mlir-export-cases> or --cases-root <dir>")
    out = tmp_path_factory.mktemp("mlir-export-cases")
    subprocess.run([str(bin_path), "--emit-all", str(out)], check=True)
    return out


def _discover(config):
    """Case names, resolved at collection time so each case is its own test id."""
    root = config.getoption("--cases-root")
    if root:
        return sorted(p.parent.name for p in Path(root).glob("*/case.json"))
    bin_path = config.getoption("--cases-bin")
    if bin_path and Path(bin_path).exists():
        out = subprocess.run([str(bin_path), "--list"], capture_output=True, text=True, check=True)
        return sorted(out.stdout.split())
    return []


def pytest_generate_tests(metafunc):
    if "case_name" in metafunc.fixturenames:
        names = _discover(metafunc.config)
        metafunc.parametrize("case_name", names or [pytest.param("<none>", marks=pytest.mark.skip(
            reason="no cases discovered; pass --cases-bin or --cases-root"))])


@pytest.fixture
def case(case_name, cases_root):
    d = cases_root / case_name
    meta = json.loads((d / "case.json").read_text())
    return d, meta
