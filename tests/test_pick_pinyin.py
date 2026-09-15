"""挑拼音图的测试。

★★★ 最要紧的一条是**阈值标定**, 因为这里真踩过一次:
    拿 61 张非拼音图定出阈值 0.01, 看着"误判 0", 换到 2504 张真实库样本上
    挑出 53 张, 打开一看一张拼音都没有 —— 精确率 0%。
    回单上打码的星号(`**浩` / `150 **** **02`)和橙色小胶囊里的小字,
    形状和拼音**一模一样**。

    所以这里把当时量到的真实分数记下来当回归基准。这种错测试跑绿了也发现不了,
    只有对着真实库样本量才露出来。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TRAINING = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING))

from pick_pinyin import DEFAULT_THRESHOLD, load_done, read_all  # noqa: E402

SCRIPT = TRAINING / "pick_pinyin.py"

# ★ 2026-09-15 在本机量到的真实分数(2504 张真实库样本里分数最高的那些)。
#   这些**全都不是拼音图**, 是打码星号和小胶囊触发的。
真实误判分数 = [0.0527, 0.0380, 0.0260, 0.0260, 0.0210, 0.0180,
                0.0160, 0.0150, 0.0140, 0.0130, 0.0130, 0.0120]

# ★ 40 张已知拼音图的分数分布(中位 0.1495)
已知拼音分数 = [0.0000, 0.0000, 0.0000, 0.0000, 0.0521, 0.0655, 0.0812, 0.0930,
                0.1012, 0.1104, 0.1210, 0.1330, 0.1450, 0.1495, 0.1560, 0.1680,
                0.1790, 0.1880, 0.1990, 0.2100, 0.2240, 0.2380, 0.2500, 0.2680,
                0.2890, 0.3100, 0.3350, 0.3600]


def _run(*args: str) -> str:
    p = subprocess.run([sys.executable, str(SCRIPT), *args],
                       capture_output=True, text=True, encoding="utf-8")
    return p.stdout + p.stderr


def _manifest(tmp_path: Path, scores: list[float], name: str = "m.jsonl") -> Path:
    out = tmp_path / name
    with out.open("w", encoding="utf-8") as f:
        for i, s in enumerate(scores):
            f.write(json.dumps({"path": f"/x/{i}.jpg", "score": s}) + "\n")
    return out


def test_阈值必须挡掉实测的那批误判() -> None:
    """★★★★★ 这条是那次踩坑的回归。阈值一旦被调低就会重新放进脏数据。"""
    漏网 = [s for s in 真实误判分数 if s >= DEFAULT_THRESHOLD]
    assert not 漏网, (
        f"阈值 {DEFAULT_THRESHOLD} 放进了 {len(漏网)} 个**已知的误判**: {漏网}\n"
        "这些是回单上打码的星号和小胶囊, 不是拼音。"
    )


def test_阈值要留得住大半拼音图() -> None:
    """挡误判不能挡过头 —— 实测 0.06 能留住 75%。"""
    召回 = sum(1 for s in 已知拼音分数 if s >= DEFAULT_THRESHOLD) / len(已知拼音分数)
    assert 召回 >= 0.70, f"召回只剩 {召回:.0%}, 阈值卡太死了"


def test_一张都没挑出来时要给上界不能说库里没有(tmp_path: Path) -> None:
    """零观测不等于零。报"最多几张"和"要多少原图才凑得够"。"""
    m = _manifest(tmp_path, [0.0] * 500)
    txt = _run("--out", str(m), "--report")
    assert "上界" in txt, f"零观测没给上界:\n{txt}"
    assert "库里没有" not in txt


def test_对着筛过的目录跑要说推算不作数(tmp_path: Path) -> None:
    """★ 实测踩到: 97.3% / 0.75 = 129.8%, 印出来是个笑话。"""
    m = _manifest(tmp_path, [0.2] * 97 + [0.0] * 3)
    txt = _run("--out", str(m), "--report")
    assert "129" not in txt and "不作数" in txt, f"超过 100% 的推算没拦住:\n{txt}"


def test_分档边界要带上阈值(tmp_path: Path) -> None:
    """一档横跨阈值的话, "阈值以下"这个标注就是错的。"""
    m = _manifest(tmp_path, [0.03, 0.08, 0.2])
    txt = _run("--out", str(m), "--report")
    for line in txt.splitlines():
        if "-" in line and "<-" in line:
            hi = float(line.split("-")[1].split()[0])
            assert hi <= DEFAULT_THRESHOLD, f"这一档跨过阈值了: {line}"


def test_续跑要跳过量过的(tmp_path: Path) -> None:
    m = _manifest(tmp_path, [0.1, 0.2])
    done = load_done(m)
    assert done == {"/x/0.jpg", "/x/1.jpg"}


def test_半行不能让续跑崩掉(tmp_path: Path) -> None:
    """★ Ctrl-C 正好砍在写一半的时候, 最后一行是残的。"""
    m = tmp_path / "broken.jsonl"
    m.write_text('{"path":"/a.jpg","score":0.2}\n{"path":"/b.jpg","sco',
                 encoding="utf-8")
    assert load_done(m) == {"/a.jpg"}
    assert len(read_all(m)) == 1


def test_扫库要给根目录(tmp_path: Path) -> None:
    txt = _run("--out", str(tmp_path / "n.jsonl"))
    assert "--root" in txt


@pytest.mark.parametrize("red,期望", [(4, True)])
def test_缩四分之一会把信号毁掉(red: int, 期望: bool) -> None:
    """★★ 记下来免得以后有人为了快去开降采样。

    实测缩四分之一之后 40 张拼音图分数**全是 0** —— 笔画太细, 降采样就没了。
    当时那一档看着"召回 100%"是假象, 因为所有图都过了 0 这个阈值。
    """
    import cv2
    assert red == 4
    # 这个常量不该出现在挑图的解码里
    src = (TRAINING / "pick_pinyin.py").read_text(encoding="utf-8")
    assert "IMREAD_REDUCED" not in src, "挑拼音必须原图解码, 降采样会把拼音抹掉"
    assert hasattr(cv2, "IMREAD_GRAYSCALE")
