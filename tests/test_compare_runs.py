"""跑法对比工具的测试。

这个工具要在**没有人工真值**的情况下判哪种跑法更好, 判据是
"输出里有没有混进带声调的拼音字母"。所以关键是:
    真有拼音混进来 -> 必须抓到
    正常的中文/数字/邮箱 -> 绝不能误报
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1] / "training"
SCRIPT = TRAINING / "compare_runs.py"


def _mk(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        (d / f"img{i:03d}.json").write_text(json.dumps(r, ensure_ascii=False),
                                            encoding="utf-8")
    (d / "batch-summary.json").write_text('{"n": 1}', encoding="utf-8")
    return d


def _run(*specs: str) -> str:
    args = [sys.executable, str(SCRIPT)]
    for s in specs:
        args += ["--run", s]
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    return p.stdout + p.stderr


def test_抓得到混进来的拼音(tmp_path: Path) -> None:
    dirty = _mk(tmp_path, "dirty", [
        {"label": "dìngdānhào订单号", "amount": "-99.90"},
        {"label": "fùkuǎnfāngshì付款方式", "amount": "-1.00"},
    ])
    out = _run(f"原图={dirty}")
    assert "100.0%" in out or "50.0%" in out
    assert "dìngdānhào" in out


def test_正常内容一个都不误报(tmp_path: Path) -> None:
    """★ 金额 / 中文姓名 / 邮箱 / 长订单号 都不带声调, 绝不能被当成拼音。"""
    clean = _mk(tmp_path, "clean", [
        {"amount": "-99.90", "name": "铭绪(**绪)",
         "mail": "ash***@hotmail.com",
         "order": "20260705200040011100560011316963",
         "time": "2026-07-05 14:00:58", "note": "转账"},
    ])
    out = _run(f"原图={clean}")
    assert "污染率" in out
    # 污染那一列应当是 0
    line = [l for l in out.splitlines() if l.startswith("原图")][0]
    assert line.split()[-1] == "0.0%", f"正常内容被误报了: {line}"


def test_能看出哪一种更好(tmp_path: Path) -> None:
    before = _mk(tmp_path, "before", [
        {"label": "dìngdānhào订单号"}, {"label": "zhuǎnzhàng转账"},
        {"label": "duìfāng对方"}, {"label": "正常"},
    ])
    after = _mk(tmp_path, "after", [
        {"label": "订单号"}, {"label": "转账"},
        {"label": "对方"}, {"label": "正常"},
    ])
    out = _run(f"原图={before}", f"擦完={after}")
    assert "**改善**" in out


def test_变差了也要如实报(tmp_path: Path) -> None:
    good = _mk(tmp_path, "good", [{"label": "订单号"}, {"label": "转账"}])
    bad = _mk(tmp_path, "bad", [{"label": "dìngdānhào订单号"},
                                {"label": "zhuǎnzhàng转账"}])
    out = _run(f"原图={good}", f"某方案={bad}")
    assert "**变差了**" in out


def test_字段变null也要看得出来(tmp_path: Path) -> None:
    """检测把整行丢掉时字段会变 null, 污染率抓不到, 要靠非空率。"""
    full = _mk(tmp_path, "full", [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}])
    holes = _mk(tmp_path, "holes", [{"a": "1", "b": None}, {"a": None, "b": None}])
    out = _run(f"原图={full}", f"有洞={holes}")
    assert "**读出来的反而少了**" in out


def test_合并器写的元字段不参与统计(tmp_path: Path) -> None:
    d = _mk(tmp_path, "meta", [
        {"_source_image": "x.jpg", "_tiles_used": 3,
         "amount__conflict": ["-9.90", "-99.90"], "amount": None, "note": "转账"},
    ])
    out = _run(f"切片={d}")
    line = [l for l in out.splitlines() if l.startswith("切片")][0]
    # 只有 amount 和 note 两个真字段
    assert line.split()[2] == "2", f"元字段混进统计了: {line}"


def test_文件名带点时不能把不同的图数成同一张(tmp_path: Path) -> None:
    """★ 实测踩到的坑: 我们的文件名里带点(r0.252_f0.69_s3_voucher_...),

    对已经是主干的名字再取一次 Path.stem, 会把最后一段当扩展名剥掉:
        "r0.252_f0.69_s3_x"  --stem-->  "r0.252_f0"
    于是好几个不同的文件塌成同一个键, 后面的覆盖前面的, 图数被数少。
    实测 12 张被数成 8 张。
    """
    d = tmp_path / "dotted"
    d.mkdir(parents=True)
    names = ["r0.252_f0.69_s3_voucher_AAA", "r0.253_f0.70_s3_voucher_BBB",
             "r0.254_f0.71_s3_voucher_CCC", "r0.255_f0.72_s3_voucher_DDD"]
    for n in names:
        (d / f"{n}.json").write_text(json.dumps({"amount": "-1.00"}),
                                     encoding="utf-8")
    out = _run(f"原图={d}")
    line = [l for l in out.splitlines() if l.startswith("原图")][0]
    assert line.split()[1] == str(len(names)), \
        f"{len(names)} 张被数成了 {line.split()[1]} 张: {line}"
