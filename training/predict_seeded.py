r"""线路A 的**确定性外壳**: 同一张图永远得到同一个分数(只读源图)。

要解决什么
----------
官方 SSP 的取块法(`utils/patch.py`)每次**随机**取 64 个 32x32 小块, **留纹理能量最大的那一块**,
`--repeat 16` 重复 16 次求平均。整条预测路径**从没设过种子**, 于是同一张图两次跑分数不一样。
(注: 服务器上那份是 `patch_list[-1]` 取最大; 另有副本写的是 `[0]` 取最小 —— **以服务器那份为准**,
所有阈值都是在它上面标的。详见 `patch_select.py` 文件头。)

实测(352 张阈值附近的图, 每张打 5 次):

    标准差中位数 0.0305 | 单张最大极差 0.3019 | **判定不一致 165/352 = 46.9%**
    有一张图五次跑出了全部三种判定: 0.6252~0.8585 -> 放行/人工复核/自动拒

★ **这不只是不公平, 它可以被利用。** 同一张图重复提交 = 每次重新抽样:

    | 假图真实均分 | 单次躲过自动拒 | 试几次有五成机会溜过去 |
    | 0.86 |  3.1% | 22 次 |
    | 0.84 | 11.4% |  6 次 |
    | 0.82 | 29.0% |  2 次 |

**交同一个文件多试几次就能换个结果** —— 不用改图, 不用任何技巧。这个洞必须堵。

怎么堵
------
按**文件内容**算一个种子, 每张图打分前设一次。于是:

- 同一个文件永远同一个分数 -> **重试不再有任何收益**
- 改文件名**没用**(种子来自内容, 不是路径)
- 与批次大小、处理顺序、是否多进程**都无关**(种子只由这张图自己决定)

**为什么不改 SSP 仓库**: `predict_all_models.py` 在 `D:\SSP` 里, 不在我们 git 管的仓库里,
直接改要手工同步到服务器。这里改成**外壳**: 转交全部参数, 只把 `predict_image_with_model`
包一层加种子, `git pull` 就能带过去, SSP 仓库一个字都不动。

★ 一个必须知道的副作用
----------------------
种子只由图片内容决定, 所以**多个模型会看到同一批小块**(以前每个模型各随机各的)。
`final_ai_score` 是多模型取 max —— 以前那个 max 里含有"哪个模型运气好抽到怪块"的成分,
现在没有了, 于是**分数整体会略微偏低**, 误杀和召回都可能跟着动一点点。
**所以换上之后必须重跑一遍验收**, 确认 2.7/万 和 18.4/万 还站得住。

用法
----
  # 和 predict_all_models.py 参数完全一致, 只多一个 --ssp-repo
  python training/predict_seeded.py --ssp-repo D:\SSP --model_root ... --input ... --output_dir ...

**只读源图**: 只往 --output_dir 写。
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

_SEED_MASK = 0x7FFFFFFF


def _seed_of(image_path) -> int:
    """按**文件内容**定种子。

    读整个文件看着像多花一次 I/O, 其实**几乎不花钱**: 紧接着 PIL 就要打开同一个文件,
    这一读会把它带进系统缓存, PIL 那次就变成热读了 —— 总的磁盘读次数没变。
    (真花钱的是**冷读**, 线路C 那次 98,013 张 x 400KB 就是这么慢的。)

    不同图撞到同一个种子完全无所谓: 种子只决定"取哪些小块", 两张不同的图
    即使取块位置一样, 像素也不一样, 分数照样不同。
    """
    try:
        return zlib.crc32(Path(image_path).read_bytes()) & _SEED_MASK
    except OSError:
        # 读不了就退回文件名 —— 这张图后面多半也会打分失败, 走"两条线都没分"->人工复核
        return zlib.crc32(str(image_path).encode("utf-8", "replace")) & _SEED_MASK


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    argv = sys.argv[1:]
    repo = None
    patch_mode = "torch"
    rest = []
    i = 0
    while i < len(argv):                       # 只截 --ssp-repo, 其余原样转交
        if argv[i] == "--ssp-repo":
            repo = argv[i + 1]; i += 2; continue
        if argv[i].startswith("--ssp-repo="):
            repo = argv[i].split("=", 1)[1]; i += 1; continue
        if argv[i] == "--patch-mode":
            patch_mode = argv[i + 1]; i += 2; continue
        if argv[i].startswith("--patch-mode="):
            patch_mode = argv[i].split("=", 1)[1]; i += 1; continue
        rest.append(argv[i]); i += 1
    if patch_mode not in ("torch", "lcg"):
        raise SystemExit(f"--patch-mode 只能是 torch 或 lcg, 收到 {patch_mode!r}")
    if not repo:
        raise SystemExit("要给 --ssp-repo(SSP 仓库路径, 里面得有 predict_all_models.py)")
    repo_p = Path(repo)
    if not (repo_p / "predict_all_models.py").is_file():
        raise SystemExit(f"{repo_p} 底下没有 predict_all_models.py")

    sys.path.insert(0, str(repo_p))
    import torch                                # noqa: E402
    import predict_all_models as pam            # noqa: E402

    _orig = pam.predict_image_with_model

    # ---- --patch-mode lcg: 连取块也换成跨语言可复现的算法 ----
    if patch_mode == "lcg":
        import numpy as np
        from PIL import Image
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from patch_select import select_patches            # noqa: E402

        _queue: list = []
        _orig_patch_img = pam.patch_img

        def _patch_img_det(img, patch_size, height):
            # predict_image_with_model 会**恰好调 repeat 次**, 我们预先算好按序发给它。
            # ★ 队列空了必须**直接报错**, 绝不能悄悄退回随机取块 ——
            #   那样分数会变而没人知道, 正是这条线上最怕的一类错。
            if not _queue:
                raise RuntimeError("确定性取块的队列空了 —— --repeat 与预先算好的轮数对不上")
            return _queue.pop(0)

        pam.patch_img = _patch_img_det

        def _prep(image_path, kw: dict) -> None:
            # ★ patch_size / trainsize / repeat **全部从调用方原样取**, 不能写死。
            #   `patch_img` 里 num_patch = (trainsize // patch_size)^2, 三者是联动的;
            #   这里若写死 32/64, 一旦有人改 --patch_size 或 --trainsize,
            #   预先算好的块**大小或数量就对不上**, 而且不会报错, 只会安静地给出错的分。
            ps = int(kw.get("patch_size", 32))
            ts = int(kw.get("trainsize", 256))
            npatch = (ts // ps) * (ts // ps)
            img = Image.open(image_path).convert("RGB")     # 与原版同一句, 不做 exif 旋转
            patches, _ = select_patches(np.asarray(img), _seed_of(image_path),
                                        patch_size=ps, num_patch=npatch,
                                        repeat=int(kw.get("repeat", 16)))
            _queue[:] = [Image.fromarray(p) for p in patches]
    else:
        _prep = None

    def _seeded(*a, **kw):
        # ★ 种子必须**每张图设一次**, 而且要在那 16 次 repeat **之前**。
        #   设在 repeat 里面的话 16 次会取到一模一样的小块, 等于把 repeat 作废。
        ip = kw.get("image_path") if "image_path" in kw else (a[1] if len(a) > 1 else None)
        if ip is not None:
            torch.manual_seed(_seed_of(ip))
            if _prep is not None:
                _prep(ip, kw)
        r = _orig(*a, **kw)
        if _prep is not None and _queue:
            # 没取完说明 repeat 对不上, 那这张图的分数就不是按预期算的 —— 必须叫停
            raise RuntimeError(f"确定性取块还剩 {len(_queue)} 块没用 —— repeat 与预算的轮数不一致")
        return r

    # 同一个进程里 main() 被调第二次时**不能再包一层** —— 包多层虽然结果一样(种子值相同),
    # 但以后谁往种子里加点别的东西, 叠加就会变成真 bug。
    if not getattr(pam, "_seeded_wrapped", False):
        pam.predict_image_with_model = _seeded
        pam._seeded_wrapped = True
    print(f"(取块方式: {patch_mode}" + ("  <- 跨语言可复现" if patch_mode == "lcg" else
          "  <- 原版随机取块, 只有种子是定的") + ")", flush=True)
    print("(确定性外壳: 每张图按文件内容定种子 —— 同一个文件永远同一个分数, 重试拿不到新结果)",
          flush=True)

    sys.argv = ["predict_all_models.py"] + rest
    pam.main()


if __name__ == "__main__":
    main()
