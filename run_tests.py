"""轻量测试运行器。

装了 pytest 就直接用 pytest；没装就临时提供一个极简的 ``pytest`` 垫片（只实现 ``raises``），
再运行 ``tests/`` 下所有 ``test_*`` 函数。测试文件本身是普通 pytest 用例，在装了 pytest 的
机器上可以原样运行。
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
TESTS = ROOT / "tests"


def _ensure_pytest_shim() -> bool:
    try:
        import pytest  # noqa: F401

        return True
    except ModuleNotFoundError:
        pass

    shim = types.ModuleType("pytest")

    @contextlib.contextmanager
    def raises(expected):  # type: ignore[no-untyped-def]
        try:
            yield
        except expected:
            return
        except Exception as unexpected:  # 抛出的异常类型不对
            raise AssertionError(f"期望 {expected!r}，实际抛出 {type(unexpected)!r}") from unexpected
        raise AssertionError(f"期望抛出 {expected!r}，但什么都没抛")

    shim.raises = raises  # type: ignore[attr-defined]
    sys.modules["pytest"] = shim
    return False


def _load_module(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _force_utf8_stdout() -> None:
    # 让中文输出在任意控制台编码下都不报错。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def main() -> int:
    _force_utf8_stdout()
    sys.path.insert(0, str(SRC))
    have_pytest = _ensure_pytest_shim()
    if have_pytest:
        import pytest

        return int(pytest.main(["-q", str(TESTS)]))

    passed = failed = 0
    failures: list[str] = []
    for test_file in sorted(TESTS.glob("test_*.py")):
        module = _load_module(test_file)
        for name in sorted(vars(module)):
            if not name.startswith("test_"):
                continue
            func = getattr(module, name)
            if not callable(func):
                continue
            try:
                func()
                passed += 1
            except Exception:  # noqa: BLE001 - 汇报每一个失败
                failed += 1
                failures.append(f"{test_file.name}::{name}\n{traceback.format_exc()}")
    print(f"\n通过 {passed} 个，失败 {failed} 个（内置运行器；未安装 pytest）")
    for failure in failures:
        print("\n失败 " + failure)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
