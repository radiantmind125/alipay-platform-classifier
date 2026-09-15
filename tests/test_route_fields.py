"""按字段分流的测试。

最要紧的两条:
  1. 每个字段确实取到了**该取的那一路**的值
  2. 在同一批数据上定路线又在同一批上报成绩, 必须**明确警告**是自己考自己
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1] / "training"
SCRIPT = TRAINING / "route_fields.py"


def _mk(tmp_path: Path, name: str, rows: dict[str, dict]) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for stem, obj in rows.items():
        o = dict(obj)
        o["input_image"] = f"{stem}.jpg"
        (d / f"{stem}.json").write_text(json.dumps(o, ensure_ascii=False),
                                        encoding="utf-8")
    (d / "batch-report.json").write_text('{"n":1}', encoding="utf-8")
    return d


def _run(*args: str) -> str:
    p = subprocess.run([sys.executable, str(SCRIPT), *args],
                       capture_output=True, text=True, encoding="utf-8")
    return p.stdout + p.stderr


def test_每个字段取该取的那一路(tmp_path: Path) -> None:
    """A 在 f1 上好, B 在 f2 上好 —— 合出来应当两个都是好的那个值。"""
    A = _mk(tmp_path, "A", {
        "img1": {"f1": "A1", "f2": None},
        "img2": {"f1": "A2", "f2": None},
    })
    B = _mk(tmp_path, "B", {
        "img1": {"f1": None, "f2": "B1"},
        "img2": {"f1": None, "f2": "B2"},
    })
    out = tmp_path / "out"
    _run("--run", f"擦={A}", "--run", f"切={B}", "--out", str(out))
    m1 = json.loads((out / "img1.json").read_text(encoding="utf-8"))
    assert m1["f1"] == "A1", "f1 应当取擦的"
    assert m1["f2"] == "B1", "f2 应当取切的"


def test_同批定路线要明确警告(tmp_path: Path) -> None:
    A = _mk(tmp_path, "A", {"i": {"f": "x"}})
    B = _mk(tmp_path, "B", {"i": {"f": None}})
    txt = _run("--run", f"擦={A}", "--run", f"切={B}", "--out", str(tmp_path / "o"))
    assert "自己考自己" in txt
    assert "成绩会偏高" in txt


def test_用另一批定的路线时不警告(tmp_path: Path) -> None:
    route = tmp_path / "route.json"
    route.write_text(json.dumps({"f": "擦"}, ensure_ascii=False), encoding="utf-8")
    A = _mk(tmp_path, "A", {"i": {"f": "x"}})
    B = _mk(tmp_path, "B", {"i": {"f": "y"}})
    txt = _run("--run", f"擦={A}", "--run", f"切={B}",
               "--out", str(tmp_path / "o"), "--route-from", str(route))
    assert "自己考自己" not in txt
    assert "真验证" in txt
    m = json.loads((tmp_path / "o" / "i.json").read_text(encoding="utf-8"))
    assert m["f"] == "x", "路线说走擦, 就该取擦的"


def test_路线存得下来给下一批用(tmp_path: Path) -> None:
    A = _mk(tmp_path, "A", {"i": {"f1": "a", "f2": None}})
    B = _mk(tmp_path, "B", {"i": {"f1": None, "f2": "b"}})
    rp = tmp_path / "r.json"
    _run("--run", f"擦={A}", "--run", f"切={B}", "--out", str(tmp_path / "o"),
         "--save-route", str(rp))
    r = json.loads(rp.read_text(encoding="utf-8"))
    assert r["f1"] == "擦"
    assert r["f2"] == "切"


def test_分流后的成绩应当等于各路最好值(tmp_path: Path) -> None:
    """分流就是每个字段取最好的那一路, 所以合出来每个字段都该等于单路最好值。"""
    A = _mk(tmp_path, "A", {
        "i1": {"f": "a"}, "i2": {"f": "a"}, "i3": {"f": None}, "i4": {"f": None},
    })
    B = _mk(tmp_path, "B", {
        "i1": {"f": "b"}, "i2": {"f": None}, "i3": {"f": None}, "i4": {"f": None},
    })
    txt = _run("--run", f"擦={A}", "--run", f"切={B}", "--out", str(tmp_path / "o"))
    # A 是 50%, B 是 25%, 分流后应当 50%
    assert "查一下" not in txt, f"分流成绩和单路最好值对不上:\n{txt}"


def test_只给一条路要拒绝(tmp_path: Path) -> None:
    A = _mk(tmp_path, "A", {"i": {"f": "x"}})
    txt = _run("--run", f"擦={A}", "--out", str(tmp_path / "o"))
    assert "至少要两条路" in txt
