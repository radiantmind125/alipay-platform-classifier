r"""拼音行读取器的训练数据: 从每一段的旧裁图(seg_input, 上面带着拼音那行)切出拼音带, 标签 = 这段汉字逐字的默认读音(不带调)。

为什么(2026-10-23)
------------------
金标准上所有模型都读错的那批段, 大半是花体字 / 生僻字; 但每个字上面那行拼音是普通字母。
拿拼音给认字结果打分, 用现成 OCR 读拼音(音节只对 77%)就已经只改对不改坏;
拼音全对时 i2 能从 92.1 到 94.4。拼音读得越准, 能改对的越多, 所以专门训一个读拼音行的小模型。

切法
----
    seg_input/<段> = 原图[input_y0:input_y1, x0:x1]; 拼音带 = 这张图最上面 label_y0 - input_y0 行(切行时就记下了)
    往下多留 25%(至少 2 像素): 推理时拼音带下边估得差几像素也不怕
    标签: 这段文字里每个汉字查 pinyin_table.tsv 的读音, 空格连起来("bai ru liang"); 数字 / 字母 / 符号上面没有拼音, 不算
    ★ has_pinyin=0 的行也要(10-23 复核: 多数是页面上的拼音检测漏了, 拼音其实在, 只是上半截常被切掉;
      蓝页一半以上是这种, 金标准里占 36%)。拼音带里没有墨(真没拼音)的不要 —— 计 noink
    ★ 老师把图标读成汉字的几种固定错, 打标签前去掉(那里图上没有拼音): 结尾的 + 读成 十(转账红包十),
      红包干, 结尾的 > 读成 入 / 一(了入, 宝入), 开头小图标读成 羊 / 司 / 心(几句固定的)。计 icon_fix
    跳过: 没有汉字、有表里没有的字、拼音带太矮、拼音带里没墨
    ★ 金标准(gold / gold2)里的段一律不进训练也不进验证 —— 留着评(目录名 / 文件名不分大小写)
    ★ 只写切好的图(先写 .tmp 再改名); 参数 + 切法代码 + 读音表的指纹记在 _params.txt, 不一样不接着跑; train.txt 最后写

用法
----
    python training/pinyin_bands.py build --src E:\alipay-ai-data\train-n1009i --out E:\alipay-ai-data\pinyin-band-v1 --gold E:\_move_logs\gold E:\_move_logs\gold2 --workers 20
出: out/img/<目录名>/<段文件名>, out/train.txt, out/val.txt, out/charset.txt(空格 + a..z),
    out/gold_band_split.csv(金标准每段拼音带的下边和 has_pinyin, 不含答案, 本地评拼音读取器用)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HAN = re.compile(r"[\u4e00-\u9fff]")
MARGIN = 0.25          # 拼音带往下多留这么多倍(拼音带高), 至少 MIN_MARGIN 像素
MIN_MARGIN = 2
MIN_SPLIT = 6          # 拼音带比这还矮的不要(读不出来, 多半是切行出错)
INK_DIFF, INK_MIN = 40, 0.01    # 拼音带里和底色差 > 40 的像素至少占 1%, 才算有拼音
LETTERS = [" "] + [chr(c) for c in range(97, 123)]      # 类别 1..27(0 = blank), 和本地拼音实验的 lexdecode.LET 一样
TABLE = Path(__file__).resolve().parent / "pinyin_table.tsv"
NUMERAL = set("零一二三四五六七八九十百千万两第")     # 这些后面的 十 是真的(五十 / 第十)
ICON_PREFIX = ("羊开通记账本", "司查看往来记录", "心联系商家")


def load_table(p: Path = TABLE) -> dict:
    t = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln or ln.startswith("#"):
            continue
        c, d, _all = ln.split("\t")
        t[c] = d
    return t


def clean_icons(text: str) -> tuple:
    """老师把图标读成汉字的几种固定错(10-23 在服务器标签里数过, 约 1%)。返回 (去掉以后的文字, 改了没有)。"""
    t = text
    if len(t) >= 2 and t[-1] == "十" and HAN.match(t[-2]) and t[-2] not in NUMERAL:
        t = t[:-1]                          # 转账红包十 / 游戏充值十 / 下馆子十: 结尾的 + 小图标; 五十 这种不动
    elif t.endswith("红包干"):
        t = t[:-1]                          # 转账红包干: 同一个 + 图标(饼干 / 牛肉干 这种不动)
    elif len(t) >= 2 and t[-1] in "入一" and t[-2] in "了吧宝～~":
        t = t[:-1]                          # 结尾的 > 箭头读成 入 / 一
    for p in ICON_PREFIX:
        if t.startswith(p):
            t = t[1:]                       # 开头的小图标读成 羊 / 司 / 心
            break
    return t, t != text


def label_of(text: str, table: dict) -> tuple:
    """(标签, 原因): 汉字逐字读音空格连起来; 有表里没有的汉字 -> (None, 'table'), 没有汉字 -> (None, 'nocjk')。"""
    syl = []
    for c in text:
        if HAN.match(c):
            d = table.get(c)
            if d is None:
                return None, "table"
            syl.append(d)
    if not syl:
        return None, "nocjk"
    return " ".join(syl), "ok"


def band_rows(split: int, h: int) -> int:
    """切到第几行: 拼音带下边 + 25% 余量(至少 2 像素), 不超过图高。"""
    return min(h, split + max(MIN_MARGIN, math.ceil(MARGIN * split)))


def has_ink(img: np.ndarray, split: int) -> bool:
    """拼音带 [0, split) 里有没有字: 和整张图的底色(中位数)差得多的像素够不够多。白字蓝底也算(底色是蓝)。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    band = g[:split]
    return bool(band.size) and float((np.abs(band - int(np.median(g))) > INK_DIFF).mean()) >= INK_MIN


def params_text() -> str:
    """参数 + 切法代码 + 读音表的指纹: 改了任何一样, 接着跑就停(不混两版的图)。"""
    src = "".join(inspect.getsource(f) for f in (band_rows, has_ink, _cut, label_of, clean_icons))
    code = hashlib.sha1(src.replace("\r\n", "\n").encode("utf-8")).hexdigest()[:10]
    tab = hashlib.sha1(TABLE.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:10]
    return f"bands-v2 MARGIN={MARGIN} MIN_MARGIN={MIN_MARGIN} MIN_SPLIT={MIN_SPLIT} INK={INK_DIFF},{INK_MIN} code={code} table={tab}\n"


def _done(p: Path) -> bool:
    try:
        return p.stat().st_size > 0
    except OSError:
        return False


def _cut(job) -> list:
    """一批: 读旧裁图, 拼音带里有墨才切最上面的拼音带(+余量), 先写 .tmp 再改名。"""
    out = []
    for src, dst, split in job:
        d = Path(dst)
        if _done(d):
            out.append("done_before")
            continue
        img = None
        try:
            img = cv2.imdecode(np.fromfile(src, np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            img = None
        if img is None:
            out.append("unreadable")
            continue
        if split >= img.shape[0]:
            out.append("small")                    # 拼音带比整张图还高: 行的记录和图对不上, 不要
            continue
        if not has_ink(img, split):
            out.append("noink")                    # 拼音带里没有字(这一行真没拼音)
            continue
        band = img[: band_rows(split, img.shape[0])]
        tmp = d.with_name(d.name + ".tmp")
        cv2.imencode(".png", band)[1].tofile(str(tmp))
        os.replace(tmp, d)
        out.append("ok")
    return out


class _Rows:
    """每个配对目录: 段文件名 -> 行文件名, 行文件名 -> (label_y0, input_y0, has_pinyin)。用到才读。"""

    def __init__(self, pairs_root: Path):
        self.root, self.cache = pairs_root, {}

    def dir_of(self, seg_path: Path) -> Path:
        d = seg_path.parent.parent
        return d if (d / "_segments.csv").exists() else self.root / d.name

    def get(self, d: Path):
        if d not in self.cache:
            seg, row = {}, {}
            if (d / "_segments.csv").exists() and (d / "_pairs_labeled.csv").exists():
                with (d / "_segments.csv").open(encoding="utf-8-sig", newline="") as f:
                    seg = {r["file"]: r["row_file"] for r in csv.DictReader(f)}
                with (d / "_pairs_labeled.csv").open(encoding="utf-8-sig", newline="") as f:
                    row = {r["file"]: (int(r["label_y0"]), int(r["input_y0"]), r.get("has_pinyin", "")) for r in csv.DictReader(f)}
            self.cache[d] = (seg, row)
        return self.cache[d]

    def lookup(self, seg_path: Path):
        """(split, has_pinyin) 或 None(找不到段或行)。"""
        seg, row = self.get(self.dir_of(seg_path))
        rf = seg.get(seg_path.name)
        if rf is None or rf not in row:
            return None
        la, ia, hp = row[rf]
        return la - ia, hp


def _key(p: Path) -> tuple:
    return p.parent.parent.name.lower(), p.name.lower()


def gold_keys(golds: list) -> set:
    keys = set()
    for g in golds:
        with (g / "_gold_candidates.csv").open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                keys.add(_key(Path(r["src_path"])))
    return keys


def build(a) -> None:
    from concurrent.futures import ProcessPoolExecutor
    if (a.out / "train.txt").exists():
        print(f"  ★ {a.out} 已经做完了(train.txt 在), 不覆盖", flush=True)
        sys.exit(2)
    pf = a.out / "_params.txt"
    if (a.out / "img").exists() and any((a.out / "img").iterdir()):
        have = pf.read_text(encoding="utf-8") if pf.exists() else "none\n"
        if have != params_text():
            print(f"BANDS_PARAMS_MISMATCH have={have.strip()} code={params_text().strip()}", flush=True)
            sys.exit(3)
    (a.out / "img").mkdir(parents=True, exist_ok=True)
    pf.write_text(params_text(), encoding="utf-8")
    for t in (a.out / "img").rglob("*.tmp"):            # 上次停在写到一半的
        t.unlink()
    table = load_table()
    rows = _Rows(a.pairs_root)
    gk = gold_keys(a.gold)
    print(f"  拼音表 {len(table):,} 字   金标准的段 {len(gk):,} 段(不进训练也不进验证)   {params_text().strip()}", flush=True)
    lists = {}
    for name in ("train", "val"):
        cnt, jobs, lines = Counter(), [], []
        nsyl = 0
        for ln in (a.src / f"{name}.txt").read_text(encoding="utf-8").splitlines():
            if "\t" not in ln:
                continue
            p, txt = ln.split("\t", 1)
            cnt["total"] += 1
            sp = Path(p)
            if _key(sp) in gk:
                cnt["gold"] += 1
                continue
            info = rows.lookup(sp)
            if info is None:
                cnt["missing_row"] += 1
                continue
            split, hp = info
            txt2, fixed = clean_icons(txt)
            lab, why = label_of(txt2, table)
            if lab is None:
                cnt[why] += 1
                continue
            if split < MIN_SPLIT:
                cnt["small"] += 1
                continue
            cnt["icon_fix"] += fixed
            dst = a.out / "img" / sp.parent.parent.name / sp.name
            dst.parent.mkdir(exist_ok=True)
            jobs.append((str(sp), str(dst), split))
            lines.append((str(dst), lab, hp != "1"))
            nsyl += lab.count(" ") + 1
        print(f"  {name}: {cnt['total']:,} 段, 要切 {len(jobs):,} 段(平均 {nsyl / max(1, len(jobs)):.1f} 个音节)", flush=True)
        res = Counter()
        chunks = [jobs[i:i + 500] for i in range(0, len(jobs), 500)]
        with ProcessPoolExecutor(a.workers) as ex:
            for k, r in enumerate(ex.map(_cut, chunks), 1):
                res.update(r)
                if k % 100 == 0:
                    print(f"    {k * 500:,}/{len(jobs):,}", flush=True)
        keep = [(d, lab, hp0) for d, lab, hp0 in lines if _done(Path(d))]   # 只列真切好的(读不出 / 没墨 / 拼音带比图还高的没写文件)
        lists[name] = keep
        print(f"BANDS_STATS {name} total={cnt['total']} listed={len(keep)} ok={res['ok']} done_before={res['done_before']} "
              f"gold={cnt['gold']} missing_row={cnt['missing_row']} nocjk={cnt['nocjk']} table={cnt['table']} "
              f"small={cnt['small'] + res['small']} noink={res['noink']} unreadable={res['unreadable']} "
              f"icon_fix={cnt['icon_fix']} hp0={sum(1 for x in keep if x[2])}", flush=True)
    # 金标准每段拼音带的下边(不含答案): 本地评拼音读取器时用真的下边, 不受估下边的误差影响
    with (a.out / "gold_band_split.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "id", "file", "has_pinyin", "split"])
        ng = nm = 0
        for g in a.gold:
            with (g / "_gold_candidates.csv").open(encoding="utf-8-sig", newline="") as fg:
                for r in csv.DictReader(fg):
                    ng += 1
                    info = rows.lookup(Path(r["src_path"]))
                    if info is None:
                        nm += 1
                        w.writerow([g.name, r["id"], r["file"], "", ""])
                    else:
                        w.writerow([g.name, r["id"], r["file"], info[1], info[0]])
    print(f"GOLDSPLIT_STATS total={ng} missing={nm}", flush=True)
    (a.out / "charset.txt").write_text("\n".join(LETTERS) + "\n", encoding="utf-8")
    (a.out / "val.txt").write_text("".join(f"{d}\t{lab}\n" for d, lab, _h in lists["val"]), encoding="utf-8")
    tmp = a.out / "train.txt.tmp"
    tmp.write_text("".join(f"{d}\t{lab}\n" for d, lab, _h in lists["train"]), encoding="utf-8")
    os.replace(tmp, a.out / "train.txt")
    print("BANDS_OK", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="拼音行读取器的训练数据")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", type=Path, required=True, help="认字模型的训练集目录(train.txt / val.txt, 路径指向 seg_input)")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--gold", type=Path, nargs="+", required=True, help="金标准目录(里面的段不进训练也不进验证)")
    b.add_argument("--pairs-root", type=Path, default=Path(r"E:\alipay-ai-data"), help="清单里的目录找不到时到这里按目录名找")
    b.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a)


if __name__ == "__main__":
    main()
