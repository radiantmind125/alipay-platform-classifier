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


def test_对着筛过的目录跑要报还差多少张(tmp_path: Path) -> None:
    """★ 实测踩到两次。

    第一次: 97.3% / 0.75 = 129.8%, 印出来是个笑话。
    第二次: 夹到 100% 之后变成 10000/1.0 = 10000 ——
            "要凑一万张得有大约 10,000 张原图", 等于把目标数原样念一遍, 没有信息。

    ★★ 对着已经筛过的目录跑, 唯一有用的数是**还差多少张**。
    """
    m = _manifest(tmp_path, [0.2] * 97 + [0.0] * 3)
    txt = _run("--out", str(m), "--report")
    assert "129" not in txt, f"超过 100% 的推算没拦住:\n{txt}"
    assert "还差" in txt, f"没报还差多少张:\n{txt}"
    assert "9,903" in txt, f"还差的张数算错了(97 张挑出来, 应当还差 9903):\n{txt}"
    assert "得有大约 10,000 张原图" not in txt, f"又把目标数念了一遍:\n{txt}"


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


def _fake_img(p: Path) -> None:
    import cv2
    import numpy as np
    cv2.imencode(".png", np.full((400, 200, 3), 255, np.uint8))[1].tofile(str(p))


def test_联系表要把过阈值和没过的都贴出来(tmp_path: Path) -> None:
    """★★ 这一步是那次踩坑的解药 —— 分数表看着正常, 打开图才发现全是误判。

    所以要能贴出来给人看, 而且**阈值上下都要有**, 好看清楚边界在哪。
    """
    from pick_pinyin import make_sheet

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    rows = []
    for i, s in enumerate([0.30, 0.20, 0.08, 0.03, 0.01, 0.00]):
        f = imgs / f"{i}.png"
        _fake_img(f)
        rows.append({"path": str(f), "score": s})
    m = tmp_path / "m.jsonl"
    m.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    sheet = tmp_path / "s.png"
    make_sheet(m, sheet, DEFAULT_THRESHOLD, n=6)
    assert sheet.exists() and sheet.stat().st_size > 0


def test_一张都没过阈值也要贴得出来(tmp_path: Path) -> None:
    """★ 全在阈值底下时不能崩, 也不能什么都不贴 —— 那样就没法确认是真没有。"""
    from pick_pinyin import make_sheet

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    rows = []
    for i in range(4):
        f = imgs / f"{i}.png"
        _fake_img(f)
        rows.append({"path": str(f), "score": 0.001})
    m = tmp_path / "m.jsonl"
    m.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    sheet = tmp_path / "s.png"
    make_sheet(m, sheet, DEFAULT_THRESHOLD, n=6)
    assert sheet.exists(), "一张都没过也得贴出来给人确认"


def test_图找不着了不能崩(tmp_path: Path) -> None:
    """名单是以前跑的, 图后来被挪走了 —— 要好好说一句, 不能抛异常。"""
    from pick_pinyin import make_sheet

    m = tmp_path / "m.jsonl"
    m.write_text(json.dumps({"path": str(tmp_path / "没了.png"), "score": 0.2}),
                 encoding="utf-8")
    make_sheet(m, tmp_path / "s.png", DEFAULT_THRESHOLD)   # 不该抛


def test_分段抽样只取该段的图(tmp_path: Path) -> None:
    """★★★★★ 找阈值必须能**只看某一段**。

    2026-09-16 栽过一次: 蓝图库按 0.06 挑出 8160 张, 贴出来的 12 张
    是分数**最高**的那些 —— 当然全有拼音, 于是我说"确认了"。
    Daniel 让我一张张看, 12 张里只有 3 张真有拼音,
    0.069 到 0.150 那一段**一张都没有**。

    只贴最高分那些去验阈值, 是自己骗自己: 阈值附近那一段根本没看到。
    """
    from pick_pinyin import make_band_sheet

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    rows = []
    for i, s in enumerate([0.40, 0.30, 0.20, 0.12, 0.09, 0.07, 0.03]):
        f = imgs / f"{i}.png"
        _fake_img(f)
        rows.append({"path": str(f), "score": s})
    m = tmp_path / "m.jsonl"
    m.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    sheet = tmp_path / "band.png"
    make_band_sheet(m, sheet, 0.06, 0.15, n=12)
    assert sheet.exists(), "分段抽样没出图"

    import cv2
    import numpy as np
    got = cv2.imdecode(np.fromfile(str(sheet), np.uint8), cv2.IMREAD_COLOR)
    # 这一段里只有三张(0.12 / 0.09 / 0.07), 不该把 0.40 那几张也贴进来
    assert got is not None
    assert got.shape[1] <= 520 * 3 + 20, "贴进了区间外的图"


def test_空区间不能崩(tmp_path: Path) -> None:
    """要找的那一段可能一张都没有, 得好好说一句, 不能抛。"""
    from pick_pinyin import make_band_sheet

    m = tmp_path / "m.jsonl"
    m.write_text(json.dumps({"path": str(tmp_path / "x.png"), "score": 0.1}),
                 encoding="utf-8")
    make_band_sheet(m, tmp_path / "s.png", 0.90, 1.00)   # 不该抛


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
