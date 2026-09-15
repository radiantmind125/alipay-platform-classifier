"""看图是不是原始截图的测试。

判据靠的是一件很硬的事: **截图的纯色底方差精确等于 0**。
本机 939 张真实截图里 937 张是精确的 0, 两个例外都是带处理前缀
(`r0.414_f0.38_...`)的派生文件 —— 判据抓对了东西。

所以测试也照这个来: 合成一张纯色的必须是 0, 加了噪点的必须被抓出来。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING))

from check_screenshot import SUSPECT, flatness, report  # noqa: E402

SCRIPT = TRAINING / "check_screenshot.py"


def _write(p: Path, img: np.ndarray) -> Path:
    cv2.imencode(".png", img)[1].tofile(str(p))
    return p


def _screenshot(p: Path) -> Path:
    """模拟截图: 纯色底 + 几个方块当文字, 底色**一个数值精确重复**。"""
    img = np.full((800, 400, 3), 245, np.uint8)
    img[100:140, 40:300] = 30
    img[200:240, 40:260] = 30
    return _write(p, img)


def _photographed(p: Path) -> Path:
    """模拟拍屏: 同样的版面, 但底色带噪点。"""
    rng = np.random.default_rng(7)
    img = np.full((800, 400, 3), 245, np.int16)
    img += rng.integers(-6, 7, img.shape, dtype=np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img[100:140, 40:300] = 30
    img[200:240, 40:260] = 30
    return _write(p, img)


def test_纯色底的截图平整度是精确的零(tmp_path: Path) -> None:
    """★★★★★ 整个判据就立在这一条上。不是"接近 0", 是**精确 0**。"""
    r = flatness(str(_screenshot(tmp_path / "s.png")))
    assert r["flat"] == 0.0, f"截图的平整度应当精确等于 0, 量到 {r['flat']}"


def test_带噪点的要被抓出来(tmp_path: Path) -> None:
    r = flatness(str(_photographed(tmp_path / "p.png")))
    assert r["flat"] > SUSPECT, f"有噪点却没被抓出来, 平整度 {r['flat']}"


def test_两者差得足够开(tmp_path: Path) -> None:
    """★ 不能只是"一个大一个小", 要拉得开才不会被 JPEG 之类的抖动淹掉。"""
    a = flatness(str(_screenshot(tmp_path / "s.png")))["flat"]
    b = flatness(str(_photographed(tmp_path / "p.png")))["flat"]
    assert b > a + SUSPECT * 5, f"差距太小: 截图 {a} vs 拍屏 {b}"


def test_全干净时要明说没有可疑的(tmp_path: Path, capsys) -> None:
    rows = [flatness(str(_screenshot(tmp_path / f"s{i}.png"))) for i in range(4)]
    suspect = report(rows)
    out = capsys.readouterr().out
    assert suspect == []
    assert "一张可疑的都没有" in out, f"干净的时候没说清楚:\n{out}"


def test_混进一张脏的要点名(tmp_path: Path, capsys) -> None:
    rows = [flatness(str(_screenshot(tmp_path / f"s{i}.png"))) for i in range(3)]
    rows.append(flatness(str(_photographed(tmp_path / "bad.png"))))
    suspect = report(rows)
    out = capsys.readouterr().out
    assert len(suspect) == 1
    assert "bad.png" in out, f"没点出是哪一张:\n{out}"


def test_图里有一块纯色也不能被骗过去(tmp_path: Path) -> None:
    """★★★★★ 这是写测试时抓到的真问题, 不是凑数的用例。

    判据一开始取的是"最平的 5% 小块的中位数"。合成用例时我在噪点图上又画了
    几块纯色的方块当文字 —— 结果最平的 5% 全落在那几块纯色上, 报出来是
    **0.0000**, 等于"这是原始截图", 可底色噪点明明很重。

    真实拍屏每个像素都带噪声, 所以线上未必踩得到; 但判据不该立在
    "整张图里没有任何一块是纯色的"这种运气上。改成取下四分位之后,
    少数几块纯色淹不掉大多数带噪点的块。
    """
    rng = np.random.default_rng(3)
    img = np.full((800, 400, 3), 245, np.int16)
    img += rng.integers(-6, 7, img.shape, dtype=np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img[100:300, 40:360] = 30          # ★ 一大块**精确纯色**, 就是拿来骗的
    p = _write(tmp_path / "trap.png", img)

    r = flatness(str(p))
    assert r["flat"] > SUSPECT, (
        f"被一块纯色骗过去了, 报成 {r['flat']} —— "
        "说明又退回成只看最平的那几块了"
    )


def test_读不出来的不算可疑(tmp_path: Path) -> None:
    """★ 坏文件要单独算, 不能混进"可疑"里 —— 那是两回事。"""
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    rows = [flatness(str(broken)), flatness(str(_screenshot(tmp_path / "s.png")))]
    suspect = report(rows)
    assert suspect == [], "读不出来的被当成可疑了"
    assert rows[0]["err"] == "decode"


def test_可以只看指定的几个文件(tmp_path: Path) -> None:
    a = _screenshot(tmp_path / "a.png")
    b = _photographed(tmp_path / "b.png")
    p = subprocess.run([sys.executable, str(SCRIPT), "--files", str(a), str(b)],
                       capture_output=True, text=True, encoding="utf-8")
    txt = p.stdout + p.stderr
    assert "b.png" in txt, f"没报出那张脏的:\n{txt}"


def test_结果能存下来(tmp_path: Path) -> None:
    a = _screenshot(tmp_path / "a.png")
    out = tmp_path / "r.jsonl"
    subprocess.run([sys.executable, str(SCRIPT), "--files", str(a),
                    "--out", str(out)],
                   capture_output=True, text=True, encoding="utf-8")
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 1 and rows[0]["flat"] == 0.0
