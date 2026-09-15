"""诊断结果分析器的测试 —— 四种病因各自能不能认出来。

这个分析器要判的是"订单号读不全"到底是哪种毛病, 判错了会让人白训几天,
所以每一种结论都要有对应的用例。
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

TRAINING = Path(__file__).resolve().parents[1] / "training"
SCRIPT = TRAINING / "analyze_voucher.py"

# 测试自己造的字段名和上限, 和任何真实配置无关
FIELD = "order_id"
MAXLEN = 20

# 造一份和真实输出结构相仿的记录, 字段名都是测试自己定的
FIELDS14 = [
    "input_image", "device", "voucher_type", "transfer_status", "amount",
    "recipient_name", "recipient_account", "payer_name", "payer_account",
    "transfer_time", "transfer_note", FIELD, "status_bar_time",
    "payment_method",
]


def _record(order_id: str | None) -> dict:
    rec = {k: None for k in FIELDS14}
    rec["input_image"] = "x.jpg"
    rec["voucher_type"] = "bill_detail"
    rec[FIELD] = order_id
    return rec


def _write_group(d: Path, values: list[str | None]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for i, v in enumerate(values):
        (d / f"img{i:04d}.json").write_text(
            json.dumps(_record(v), ensure_ascii=False), encoding="utf-8")
    # 批处理产物混在同一个目录里, 必须被跳过, 不能当成单张结果
    (d / "batch-summary.json").write_text('{"total": 999}', encoding="utf-8")


def _digits(n: int, rng: random.Random) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def _run(pinyin: Path, control: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--pinyin", str(pinyin), "--control", str(control),
         "--field", FIELD, "--max-len", str(MAXLEN)],
        capture_output=True, text=True, encoding="utf-8",
    )
    return proc.stdout + proc.stderr


@pytest.fixture
def rng() -> random.Random:
    return random.Random(1)


def test_上限截断_两组都恰好卡在上限(tmp_path: Path, rng: random.Random) -> None:
    """订单号实测 28/32 位, 都超过识别流程的单行长度上限。

    两组都大量出现长度恰好等于上限, 说明是那个上限在生效, 跟拼音无关。
    """
    _write_group(tmp_path / "pin", [_digits(MAXLEN, rng) for _ in range(180)] + [None] * 20)
    _write_group(tmp_path / "ctl", [_digits(MAXLEN, rng) for _ in range(175)] + [None] * 25)
    out = _run(tmp_path / "pin", tmp_path / "ctl")
    assert "恰好卡在上限" in out
    assert "跟拼音无关" in out


def test_两组一样烂_判为与拼音无关(tmp_path: Path, rng: random.Random) -> None:
    """对照组同样读不出, 那这个字段本身就没做完, 不是拼音的锅。

    ★ 这一条最要紧: 认错了就会拿归集的那批图去解一道根本不存在的题。
    """
    _write_group(tmp_path / "pin", [None] * 120 + [_digits(28, rng) for _ in range(80)])
    _write_group(tmp_path / "ctl", [None] * 118 + [_digits(32, rng) for _ in range(82)])
    out = _run(tmp_path / "pin", tmp_path / "ctl")
    assert "不是拼音特有的问题" in out
    assert "别拿归集的那批图开训" in out


def test_拼音特有_且能分出是null还是截短(tmp_path: Path, rng: random.Random) -> None:
    """拼音组 null 率显著高, 而且非 null 的都是满长度 -> 偏向标签污染那条路。"""
    _write_group(tmp_path / "pin", [None] * 150 + [_digits(28, rng) for _ in range(50)])
    _write_group(tmp_path / "ctl", [None] * 10 + [_digits(32, rng) for _ in range(190)])
    out = _run(tmp_path / "pin", tmp_path / "ctl")
    assert "显著高于" in out
    assert "偏向 b" in out


def test_字段名对不上时_报出实际的键而不是瞎猜(tmp_path: Path) -> None:
    """没见过真的输出文件, 字段名可能和文档不一样。

    这种时候必须把实际见到的键打出来并叫停, 不能硬套一个结论。
    """
    for grp in ("pin", "ctl"):
        d = tmp_path / grp
        d.mkdir(parents=True)
        for i in range(30):
            (d / f"a{i}.json").write_text(
                json.dumps({"input_image": "x.jpg", "some_other_field": "abc"}),
                encoding="utf-8")
    out = _run(tmp_path / "pin", tmp_path / "ctl")
    assert "字段名对不上" in out
    assert "some_other_field" in out


def test_置信区间重叠时不下结论(tmp_path: Path, rng: random.Random) -> None:
    """样本小、差别不大的时候, 不许把噪声读成结论。"""
    _write_group(tmp_path / "pin", [None] * 6 + [_digits(28, rng) for _ in range(14)])
    _write_group(tmp_path / "ctl", [None] * 5 + [_digits(28, rng) for _ in range(15)])
    out = _run(tmp_path / "pin", tmp_path / "ctl")
    assert "区间重叠" in out
