"""切片结果合并器的测试。

最要紧的一条是**冲突不许自作主张**: 两块给出不同的非空值时, 必须标出来,
不能随便选一个。藏起来的错比看得见的冲突危险得多。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1] / "training"
SCRIPT = TRAINING / "merge_tiles.py"


def _setup(tmp_path: Path, tiles: list[dict], results: dict[str, dict]):
    td = tmp_path / "tiles"
    rd = tmp_path / "res"
    od = tmp_path / "out"
    td.mkdir(parents=True)
    rd.mkdir(parents=True)
    (td / "_tiles.json").write_text(json.dumps(tiles, ensure_ascii=False),
                                    encoding="utf-8")
    for name, obj in results.items():
        (rd / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False),
                                         encoding="utf-8")
    # 批处理产物混进来, 必须被跳过
    (rd / "batch-summary.json").write_text('{"total": 9}', encoding="utf-8")
    return td, rd, od


def _run(td: Path, rd: Path, od: Path) -> str:
    p = subprocess.run(
        [sys.executable, str(SCRIPT), "--tiles", str(td),
         "--results", str(rd), "--out", str(od)],
        capture_output=True, text=True, encoding="utf-8")
    return p.stdout + p.stderr


def _tiles_for(src: str, n: int) -> list[dict]:
    return [{"src": src, "tile": f"{Path(src).stem}__t{i:02d}.png", "index": i,
             "y0": i * 100, "y1": (i + 1) * 100} for i in range(n)]


def test_各块各有一部分字段_合起来是完整的(tmp_path: Path) -> None:
    tiles = _tiles_for("a.jpg", 3)
    td, rd, od = _setup(tmp_path, tiles, {
        "a__t00": {"amount": "-99.90", "recipient": None},
        "a__t01": {"amount": None, "recipient": "张三"},
        "a__t02": {"amount": None, "recipient": None, "order": "2026070512345678"},
    })
    _run(td, rd, od)
    m = json.loads((od / "a.json").read_text(encoding="utf-8"))
    assert m["amount"] == "-99.90"
    assert m["recipient"] == "张三"
    assert m["order"] == "2026070512345678"
    assert m["_tiles_used"] == 3


def test_多块给出同一个值_不算冲突(tmp_path: Path) -> None:
    """块之间有重叠, 同一行可能出现在两块里。值一样就不是冲突。"""
    tiles = _tiles_for("b.jpg", 2)
    td, rd, od = _setup(tmp_path, tiles, {
        "b__t00": {"amount": "-100.00"},
        "b__t01": {"amount": "-100.00"},
    })
    out = _run(td, rd, od)
    m = json.loads((od / "b.json").read_text(encoding="utf-8"))
    assert m["amount"] == "-100.00"
    assert "amount__conflict" not in m
    assert "有冲突的图: 0 张" in out


def test_冲突时不替它选_而是标出来(tmp_path: Path) -> None:
    """★ 最要紧的一条。两块说法不一样, 至少有一块错了, 不许随便选。"""
    tiles = _tiles_for("c.jpg", 2)
    td, rd, od = _setup(tmp_path, tiles, {
        "c__t00": {"amount": "-99.90"},
        "c__t01": {"amount": "-9.90"},          # 少读了一位
    })
    out = _run(td, rd, od)
    m = json.loads((od / "c.json").read_text(encoding="utf-8"))
    assert m["amount"] is None, "冲突时不能自作主张选一个"
    assert set(m["amount__conflict"]) == {"-99.90", "-9.90"}
    assert "有冲突的图: 1 张" in out


def test_所有块都没有这个字段_合出来是null(tmp_path: Path) -> None:
    tiles = _tiles_for("d.jpg", 2)
    td, rd, od = _setup(tmp_path, tiles, {
        "d__t00": {"amount": None},
        "d__t01": {"amount": None},
    })
    _run(td, rd, od)
    m = json.loads((od / "d.json").read_text(encoding="utf-8"))
    assert m["amount"] is None
    assert "amount__conflict" not in m


def test_空字符串当成没有(tmp_path: Path) -> None:
    tiles = _tiles_for("e.jpg", 2)
    td, rd, od = _setup(tmp_path, tiles, {
        "e__t00": {"note": "   "},
        "e__t01": {"note": "转账"},
    })
    _run(td, rd, od)
    m = json.loads((od / "e.json").read_text(encoding="utf-8"))
    assert m["note"] == "转账"


def test_有块没跑出结果_要报出来而不是当没这回事(tmp_path: Path) -> None:
    tiles = _tiles_for("f.jpg", 3)
    td, rd, od = _setup(tmp_path, tiles, {
        "f__t00": {"amount": "-1.00"},
        # t01 缺
        "f__t02": {"amount": None},
    })
    out = _run(td, rd, od)
    assert "找不到结果的 1 块" in out
    m = json.loads((od / "f.json").read_text(encoding="utf-8"))
    assert m["_tiles_used"] == 2


def test_字段名不写死_见到什么合什么(tmp_path: Path) -> None:
    """没见过真的输出文件, 写死 14 个字段名很可能对不上。"""
    tiles = _tiles_for("g.jpg", 2)
    td, rd, od = _setup(tmp_path, tiles, {
        "g__t00": {"完全没见过的字段": "值A"},
        "g__t01": {"another_unexpected": "值B"},
    })
    _run(td, rd, od)
    m = json.loads((od / "g.json").read_text(encoding="utf-8"))
    assert m["完全没见过的字段"] == "值A"
    assert m["another_unexpected"] == "值B"
