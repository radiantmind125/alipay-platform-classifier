r"""整页分类(第二版): 先判一张原图是哪一类页面, 再交给这一类自己的模型 —— 规则为主, OCR 关键词补规则分不开的。

为什么(2026-10-12)
------------------
经理 10/12: "直接对图片进行分类 然后根据算法识别结果 使用对应的分类模型"。
拼音就是这么做的: PinyinCheck 先判有没有拼音, 有就走 rec_pinyin.onnx。这里把同样的路数推到所有页面 ——
先分出是哪一种模板, 后面才能各用各的模型、各跑各的假图检查(111/112/118 只对支付宝账单详情有意义)。
这一版先在本地图上把类别和规则跑通、出联系表人眼核, 规则定下来以后再写成 C# 类(和 demo/PinyinCheck.cs 一样)。

本地看到的(10/12 在本地 9,068 张拼音页和 118/111/112 核对图上看的)
    蓝图库   三种支付宝结果页, 各占三分之一左右: 转账成功 / 到账成功 / 支付成功(含代付成功)
    白图库   86% 是支付宝账单详情(左灰标签 + 右深色取值的卡片), 其余是转账凭证图(1304 宽)、
             浏览器里的 H5 支付成功页、支付消息列表、红包页、深色模式(1.1%)、拍屏照片(1.5%)、
             微信账单 / 银行 App / 聊天 / 平台页 / 游戏(不是支付宝)
    拼音     安卓的字体设置, 哪一类都会有 —— 是标记, 不是类别

类别
----
    B1 蓝图转账成功   B2 蓝图到账成功   B3 蓝图支付成功(含代付成功、付款成功)
    B4 蓝图异常页(v2: 到账失败 / 提交成功 / 等待对方领取, 钱没到)   B0 蓝图说不清(话费充值、授权成功、订单展示 ...)
    W1 账单详情(子类: 银行卡转账 / 转给个人 / 商户消费 / 收钱码 / 其它)
    W2 转账凭证(v2 子类: 页面截图 / 转账凭证图 / 转账回执 / 其它尺寸)
    W3 白底结果页(H5 支付成功、支付成功、支付消息、聊天转账、帮我付、红包 ...)
    W4 余额明细(v2: 余额明细详情页, 支付宝的另一个模板)
    D  深色模式       X  非支付宝(微信、银行、平台、收款码图、聊天 ...)   P 翻拍照片   U 说不清
每张图另外记几个标记: 页头(蓝底/白底/其它)、深色、平整度、拼音、设备(按分辨率)、读到的标题;
v2 加 state: status_icon 只看图认出的异常页(到账失败 / 提交成功 / 等待对方领取 / 白图未完成 / 白图已全额退款 / 白图退款记录),
白图的这三种照常分进 W1(模板一样), 只记在 state 里。

判定顺序(便宜的在前, 能早停就早停)
------------------------------------
1. 尺寸   宽 1304、高 2000~2080 -> W2(支付宝生成的转账凭证图就是这个尺寸);
          横图 / 短边 > 1500 / 长宽比不在 1.6~2.6 -> 可能是照片, 等第 3 步确认
2. 解码一次, 1/4 缩小图上量颜色: 页头 5%~20% 那一段的平均色(和 new_batch.page_kind 一样),
   深色模式(9.5%~96% 高度的中位灰度 < 60, 量法和 color_scan 一样), 上半截的支付宝蓝占比(页头被通知条盖住时用)
3. 照片: 平整度(check_screenshot 的量法: 16x16 小块标准差的下四分位, 截图精确是 0)
   + 出现最多的灰度占比(pinyin_probe 的 flat: 截图 0.3 以上, 照片 0.05 以下)
4. 白底 / 其它: 支付宝账单详情卡片(color_scan)、处理进度时间轴(timeline_scan)、账单管理的 > 箭头行(arrow_scan)
5. RapidOCR: 缩小到宽 540 检测一次, 框从原图上裁下来识别, 按关键词分蓝图三类、账单详情子类、转账凭证、
   白底结果页、非支付宝
   ★ 识别是大头(检测 0.2 秒, 识别一个框 0.03~0.1 秒), 所以先只读标题那几个框, 读到就停;
     读不到再读其余的框。账单详情已经由卡片定下来的, 只读左边的标签列。
--no-ocr 只跑 1~4 步: 蓝图一律 B0(找得到"通知TA"按钮的子类写 有通知卡片), 白底没有卡片的一律 U。
每一张都写一句中文理由(reason), 联系表上也画出来, 人眼核的时候对着看。

拼音标记: 用 pinyin_probe 原样的判据(C# 是 PinyinCheck); 苹果原尺寸截图不判(拼音是安卓的字体设置)。

run2 人眼核过以后改的(10/13, 每类 48 格; 错的大头是 B0 / U / D)
----------------------------------------------------------------
* 蓝图长屏(高宽比 >= 2.1): 0~0.115 的条没读到标题, 换 0~0.14 的条再检测一次(标题在 0.085~0.114, 0.115 切到字底,
  读成半个框或者拆成单字); 标题区不算右上角的框(回首页)和通知条的字(提醒 / 你有一笔 / 支出 / 现在 / 刚刚)
* 蓝图标题读坏的: 付城功 / 支什成功 / 付成访 / 转岁坊 / 转账城动 也认; 拼音 fochenggong 一类认成支付成功
* 蓝图标题被通知条盖住: 按顺序看字段 —— 话费充值 -> 提交成功+到账(B2) -> 到账时间一类(B2) -> 立即通知收款人 / 预约转账 /
  通知TA / 通知按钮(B1) -> 收款方(认读坏的 收默方 收款芳 ...)(B1) -> 交易商家 / 交易方式 / 支付有礼 / 两个 ¥ / 商家那一行 ¥(B3)
* 平台收银台、存款页(嵌了支付宝截图的也算)一律 X, 在 W1 / W3 / D / U 之前判; 蓝色结果页是浮窗(两边不蓝)的也是 X
* 微信: 页头微信绿 + 没有支付宝的字 -> X; 字太少的再读 0.75~0.92 那一段找 返回商家 / 摇一摇 / 收单机构 / 财付通
* 支付宝的线索多认几个: H5 的 便捷生活 / 一点就好 / 支付宝网页版、交易号 20xxxxxx200040011、蓝色 支 字图标;
  有这些就不走"读到 >= 12 个汉字、没有支付宝字段 -> X"
* 白底原生支付成功页(右上 完成 + 上面有 ¥ 金额 + 收款方)-> W3; 白底按标题判 W3 要有支付宝的线索, "付款成功后"不算标题
* 账单详情: 标题区的 账单详情 才算; 余额明细不是; 箭头行单独不够; 交易号 + 1 个账单标签也算; 子类按 支付时间 / 创建时间 分
* W2 不是凭证尺寸的: 先过银行 / 微信的字, 再读下面 0.70~0.95 找 财付通
* >= 750 万像素的直接算照片(截图没有这么大); 横图 / 相机尺寸的平整度只量灰度 > 40 的地方(黑屋子里拍的四周压成一个灰度)
* 同一个 GWCZ 编号的图在几个目录里重复(pinyin-all / pinyin_hits / suspect ...), 只留第一张

v2(10/19): census 1.8 万张人眼核过的错(scratchpad census_verify), 每条新关键词都先在 census + 拼音库 + 本地 5.5 万行 OCR 字上
查过会碰上哪些类(只碰上该碰的才加)
* 蓝图: 异常页(到账失败 / 提交成功 / 等待对方领取)先按 status_icon 的页头图标定 B4, 图标看不到的按字定(abnormal_text);
  census 8 张到账失败进了 B3、提交成功在 B2。没读到标题时: 收款码图 / 平台扫码收银台 -> X, 授权成功 / 订单展示 -> B0 子类,
  右边 ¥ 那一行左边是 银行名 + 卡尾号 -> B2, 还定不下来再读 0.34~0.60(支付有礼 / 到账时间 / 半屏的到账成功)
* 页头被盖住走蓝图那一路要上半截蓝 >= 0.30(0.18~0.30 的全是平台页 / 收款码 / 红包详情 / 浮窗; 真蓝图最低 0.36), 浮窗先判
* 白底: 余额明细详情单独一类 W4(在"不是支付宝的字"之前); 支付消息列表在账单详情的规则之前; 账单详情 要有 账(订单详情不算);
  完成 要单独一个框; 苹果状态栏的"◀ 支付宝"不算; 转账首页 / 交易中断 -> U; 标题和左边图标是微信绿的支付成功 -> X 微信
  (census U 里 83 张); 没有金额、框很少的 支付成功 -> H5支付成功; 银行 App 的 付款户名 / 付款账号、微信 零钱提现 -> X
* 账单详情子类: 余额充值 / 代充值 / 余额宝转入 / 退款转账 -> 其它(以前进了 商户消费 / 转给个人)
* 转账凭证: 标题区是 转账凭证 的只读标题就定(这种页 v1 平均 25 秒一张, 占 census 四分之一的 CPU); 凭证图的尺寸放宽、按高宽比认
* 深色: 平台页 / 微信分出去, 一点支付宝线索都没有的不算 D; 照片: 亮处不到 15% 的不用亮处的量法
* --compare 上一版的 page_type.csv: 只出变了的 diff.csv 和联系表

用法
----
    python training/page_type.py --src 目录或名单 [...] --out 输出目录 [--n 3000 --seed 1] [--workers 8] [--sheets 2] [--no-ocr]
                                 [--compare 上一版的page_type.csv --diff-sheets 30]
    名单 = .txt(一行一个路径) 或 .csv(按 path / file / source 列找文件)
    ★ 要在同一批图上和上一版比, --src 直接给上一版的 page_type.csv(--n --seed 重抽时目录里的图一变, 抽到的就全变了)
输出: page_type.csv、summary.txt、sheets\<类别>_NN.png + .txt(格子编号 -> 路径);
      --compare 时另有 diff.csv、diff_sheets\diff_NN.png(每格写 上一版 -> 这一版)
★ 只读原图, 不改不挪。
★ 一张图出任何错都记成 U, 理由里写错误, 不会让整批停下来。
"""
from __future__ import annotations

import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # 必须在 import numpy / cv2 之前, 见 pick_pinyin.py

import argparse  # noqa: E402
import csv  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import arrow_scan  # noqa: E402
import color_scan  # noqa: E402
import pinyin_probe as pp  # noqa: E402
import status_icon  # noqa: E402
import timeline_scan  # noqa: E402
from alipay_platform.bootstrap import resolution_platform  # noqa: E402
from alipay_platform.metadata_seed import IPHONE_RESOLUTIONS  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = {"B1": "蓝图转账成功", "B2": "蓝图到账成功", "B3": "蓝图支付成功", "B4": "蓝图异常页", "B0": "蓝图说不清",
         "W1": "账单详情", "W2": "转账凭证", "W3": "白底结果页", "W4": "余额明细",
         "D": "深色模式", "X": "非支付宝", "P": "翻拍照片", "U": "说不清"}
DEVICE = {"ios": "苹果", "android": "安卓", "abstain": "说不清"}
# 目录名是这些的不进(联系表、裁图), 只要整页原图
SKIP_DIRS = {"sheets", "seg_input", "seg_label", "label", "input", "cmp", "__pycache__"}

# 支付宝"转账凭证"生成的图: 宽 1304, 高 2014~2042(转出成功, 到银行卡) / 2114~2122(转账成功, 下面有 支付宝转账 到账特别快 + 二维码);
#   v2: 还有一种二维码在灰框里的 1304x2386(census 一张), 上限放到 2450
VOUCHER_W, VOUCHER_H = 1304, (1950, 2450)
VOUCHER_RATIO = (1.45, 1.66)   # v2: 存下来又缩小 / 裁过的凭证图(宽 1112~1288, 高宽比 1.49~1.63; 手机截图都在 1.77 以上), 要有凭证的字才算
# 中位灰度 < 这个算深色模式。★ 不用 color_scan 的 128: 本地 run1 量过, 真深色页是 0~36,
#   128 以下还混进了弹窗把白页压暗的(97~127)、零花钱红包页(64), 它们走白底那一路
DARK_MEDIAN = 60
BLUE_TOP = 0.18            # 上半截支付宝蓝占比 >= 这个: 先看是不是嵌在平台页里的浮窗(和 split_blue_white 一样)
# v2(10/19): 页头不是蓝底的, 上半截蓝 >= 这个才走蓝图那一路(页头被通知条盖住)。census + 拼音库 + 本地 5.5 万行量过:
#   0.18~0.30 的全是平台收银台、收款码图、红包详情、浮窗, 还有 1 张账单详情; 真蓝图被盖住的最低 0.36
BLUE_TOP_ROUTE = 0.30
OCR_W = 540                # 检测用的缩小宽度
STATUS = 0.03              # 框中心在这个高度以上是状态栏, 不读
MIN_BOX_H = 12             # 缩小图上框高 < 这个像素: 拼音那一行或者小字, 不读
BLUE_FRAC, PAGE_FRAC = 0.34, 0.70   # 蓝图只读上面 34%(标题、字段、通知卡片头); 别的读上面 70%
TITLE_Y = 0.115            # 框中心在这个高度以上算标题区
# 长屏(1080x2340/2400/2412、1260x2800 ...)标题在 0.085~0.114, 0~0.115 的检测条会切到字底:
#   检测出来是半个框(读成 得 / 得箴), 或者 到账成功 拆成 账 / 成 两个框、功 太小被扔。
#   本地 run2 的 198 张 B0 只把检测条放到 0~0.14, 77 张读出了标题, 没有错的。标题区(框中心 < TITLE_Y)不变。
#   ★ 只在 0~0.115 没读到标题时再用(见 _blue), 白底不用(见 _white)
TALL, TITLE_STRIP_TALL = 2.1, 0.14
BANNER_KW = ("提醒", "你有一笔", "支出", "现在", "刚刚")    # 通知条上的字, 不算标题
PHOTO_MP = 7.5e6           # 像素 >= 这个直接算照片: 本地最大的截图是平板 2136x3200(680 万), 相机最小 3264x2448(800 万)
PHOTO_MIN_GRAY = 40        # 横图 / 相机尺寸的平整度、最多灰度只量灰度 > 这个的地方(黑屋子里拍屏, 四周压成同一个灰度)
GREEN_MIN = 0.0004         # 上面 15% 里微信绿的占比 >= 这个(而且没有支付宝蓝)算微信页头
# 照片(本地 12,581 张定的, 见 size_cue / classify 第 3 步):
#   截图的平整度几乎都是 0(99.9% 分位 0.56), 最多灰度占比白底 0.5 以上、蓝底 0.15 以上;
#   相机拍的 平整度 1~3、最多灰度占比 0.01~0.06
PHOTO_Q, PHOTO_MODE = 0.30, 0.08     # 横图 / 相机尺寸: 两条占一条就算
PHOTO_Q_ODD = 1.0                    # 只是长宽比怪的(商品图、长截图拼图也是这样): 平整度要 >= 1 或最多灰度占比 < 0.08
PHOTO_Q_FREE, PHOTO_MODE_FREE = 0.30, 0.05   # 尺寸正常的(拍下来又裁成手机比例): 两条都要过, 而且不是蓝底

_OCR = None
_T = str.maketrans("單詳帳賬轉間時們錢額歸條戶號碼說體識別寶進態憑證筆", "单详账账转间时们钱额归条户号码说体识别宝进态凭证笔")

# ---- 关键词(都是 norm 以后的样子: 只留汉字、字母、数字和 ¥ * @) ----
BLUE_TITLES = [("转账成功", "B1"), ("到账成功", "B2"), ("支付成功", "B3"), ("代付成功", "B3"), ("付款成功", "B3")]
ALIPAY_KW = ["账单详情", "创建时间", "付款方式", "订单号", "账单管理", "账单分类", "计入收支", "再转一笔", "支付奖励",
             "交易成功", "对方账户", "转账备注", "商品说明", "收款方全称", "商家订单号", "处理进度", "转账到",
             "支付宝", "消费图鉴", "回首页", "收款方", "立即通知收款人", "预约转账", "到账时间", "支付有礼",
             "交易商家", "一键领", "凭证编号", "收款方姓名", "付款方账号", "转账附言", "余额宝", "花呗",
             "账户余额", "储蓄卡", "信用卡", "通知TA", "完成", "下载支付宝", "验证姓名", "转账保障"]
# 不是支付宝的: 微信账单的标签、银行 App、平台页、套图网站水印
X_KW = ["微信支付", "财付通", "收单机构", "当前状态", "商户全称", "账单服务", "对订单有疑惑", "发起群收款",
        "在此商户的交易", "申请电子凭证", "交易单号", "商户单号", "零钱通", "微信", "交易摘要",
        "交易附言", "对方户名", "交易卡号", "手机银行", "收银台", "凭证已提交", "继续支付",
        "tugaigai", "返回商家", "继续转账", "收款账号", "收款银行", "明细详情", "交易账户", "记账日", "交易地点",
        "存款通道", "交易币种", "交易对象", "可用余额", "动账提醒", "明细查询", "账户查询", "账户信息",
        "交易描述", "账户交易"]
# run2 核出来的银行 App、淘宝订单、数字货币、第三方支付完成页(这些页上只有对方名字里带"支付宝")。
# ★ 只认原样的: 允许错一个字时 网商银行 会认上 工商银行 / 招商银行, 转账单号 会认上 转账账号
X_EXACT = ["转出账号", "转账路径", "超级网银", "转账渠道", "收款人户名", "流水号", "查看回执", "交易动向", "交易流水号",
           "网商银行", "微银行", "农信", "资金明细", "交易渠道", "收货信息", "实付款", "天猫店", "数字货币", "货币地址",
           "提现金额", "支付来源", "平台客服", "转账单号", "所属账本", "微信向银行卡转账",
           # v2: 收支详情 / 存款详情 从 X_KW 挪过来只认原样: 错一个字时 收款详情(支付宝收钱码的收款详情页)两个都认上了(census 3 张);
           #   付款户名 / 付款账号: 银行 App 的转账详情(census 一张农行的因此进了 W1)
           "收支详情", "存款详情", "付款户名", "付款账号"]
# ★ 收单机构 不算强的: 支付宝的商户消费账单详情也有这一行(本地实测)
# ★ 不能放"存款方式": 允许错一个字的匹配会把"付款方式"也认成它(run1 之后实测把 40 多张支付宝深色页判成了非支付宝)
# ★ 强的要原样读到才算: 允许错一个字时 "电信支付" 也算 微信支付(run2 里一张支付宝 H5 交电信费的页因此判成 X)
X_STRONG = ["微信支付", "财付通", "存款详情", "存款通道", "收支详情", "tugaigai", "微信向银行卡转账"]
X_LATIN = ("tugai", "gaigai", "payeeobj", "ttlmessage")   # 套图网站水印、没渲染出来的模板变量(小写拉丁字母里找)
# 平台收银台 / 存款页: 原样匹配(不用 has 的错一个字), 读到一个就是 X, 不和支付宝的字比多少 ——
#   这类页常嵌一张支付宝截图(账单详情、转账成功浮窗), 支付宝的字总比平台的多。
#   ★ 只认汉字: mclient.alipay.com/h5pay/cashierActivity 是支付宝自己的 H5 收银台, 拉丁字母里的 cashier 不算
PLATFORM_KW = ["收银台", "收银苔", "银台", "拉单", "上传截图", "截图付款明细", "不予退款", "补款", "可能被冻结",
               "凭证已提交", "我已付款", "取消订单",
               # v2: census 核出来漏掉的平台页(上传付款截图 / 立即上传图片 的扫码收银台; 嵌了示例账单的存款说明页)
               "上传付款截图", "上传转账截图", "否则无法到账", "账单示例图", "立即上传图片"]
# v2: 收款码图(支付宝个人收款码海报)和平台扫码收银台: 蓝图那一路没读到标题时才认(census: 6 张收款码图 + 8 张收银台在 B0 / B1)。
#   ★ 不能有标题也认: 支付成功页、账单详情的推荐卡片上也有 推荐使用支付宝(拼音库各 1 张)
QR_RE = re.compile(r"推.?使用.?付.|推荐使用|打开支付宝.{0,3}扫一扫|扫码付款|收款姓名")
# v2: 蓝图异常页(钱没到): status_icon 看页头图标定; 页头被盖住、图标看不到的, 按读到的字定(见 abnormal_text)
BLUE_ABNORMAL = ("到账失败", "提交成功", "等待对方领取")
# v2: 右边 ¥ 金额那一行左边是银行卡(银行名 + 卡尾号 4 位)的是到账成功一类(census 72 张 B2 每张都有, B3 一张没有)
BANK_TAIL = re.compile(r"(银行|农信|信用社|农商|储蓄|邮政|村镇|信用联社|农料信)\d{4}")
DEPOSIT_KW = ["类型存款", "存款方式", "存款账号", "存款金额", "预计到账总金额", "每日开门红"]   # 原样匹配, 见 X_STRONG 的注
PLATFORM_ORDER = re.compile(r"(?<!家)订单号[A-Z]{2,}\d{8,}")   # 平台单号 GWCZ2085...; 支付宝的订单号全是数字
TRADE_NO = re.compile(r"20\d{6}200040011")                    # 支付宝转账的交易号; run2 里只在支付宝页上出现
AL_NOISE = re.compile(r"向支付宝|支付宝支付科技|支付宝中国网络技术|支付宝网络技术")   # 对方名字里的"支付宝", 不算支付宝的字
W2_KW = ["凭证编号", "收款方姓名", "付款方账号", "收款方账号", "转账附言", "到账特别快", "转账回执", "向你转账",
         "收方账户实际到账",      # 凭证上那句"请以收方账户实际到账为准"(微信的凭证是"收款方账户实际入账")
         "收款人账号", "通知对方已转账"]   # v2: 转账回执卡片(通知对方已转账 ... XX向你转账 ... 收款人账号; census 一张因此进了 U)
W2_TITLE = re.compile(r"转账.?凭证|转账凭|转宪证")   # v2: 标题区的 转账凭证(支付宝 转账凭证 页面截图的导航栏), 读坏的也认
# ★ 红包页不能只认"红包"两个字: 账单详情的账单分类常是"转账红包", 蓝图的券也带红包(本地 400 张里 4 张因此判错)
# v2: census 里 确认付款 碰上的 4 张都是 帮我付(朋友的代付请求, 那句"实际金额以付款人确认付款时为准"), 先认 帮我付;
#   ★ 确认付款 本身留着记 付款确认页: 本地有一张是 支付宝付款 还没付的确认页(订单信息 / 付款方式 / 确认付款 按钮)。
#   帮对方充值、给孩子发的红包(红包已经发给孩子啦)census 在 U 里各一两张
W3_KW = [("帮我付", "帮我付"), ("确认付款", "付款确认页"), ("支付消息", "支付消息"), ("付款给个人", "付款给个人"),
         ("加为好友", "聊天转账"), ("聊天更方便", "聊天转账"),           # 支付宝聊天里的转账气泡(转账成功 马上通知TA)
         ("零花钱红包", "红包"), ("红包发送成功", "红包"), ("已发送至", "红包"), ("你的零花钱", "红包"),
         ("红包已经发给", "红包"), ("帮对方充值", "帮对方充值")]
RED_KW = ["零花钱红包", "红包发送成功", "已发送至", "你的零花钱", "红包已经发给"]
# v2: 支付消息列表(服务消息 > 支付消息): 卡片闸门会把它的卡片当成账单详情(census 8 张进了 W1)。认它的筛选条和统计条
PAYMSG_RE = re.compile(r"支付消息|服务消息|统计支出|回顾\d*月支出")
# v2: 支付宝 转账 首页(转到支付宝 / 转到银行卡 / 二维码转账 + 最近转账卡片): 卡片上的 到账成功 / 转账成功 不是结果页(census 2 张进了 W3)
HOME_RE = re.compile(r"转到银行卡|转账记录|二维码转账|转到支付宝")
# v2: 余额明细详情(余额那一页的明细: 关联账单 查看账单详情 / 可用余额 / 交易类型): 支付宝的另一个模板, 单独一类 W4
#   (v1 只认标题, 标题被盖住的进了 W1; 正文里的 明细详情 / 资金明细 又让 census 一张进了 X)
YUE_TITLE = re.compile(r"余额明细|明细详情|明細")
# 支付宝 H5 支付成功页(浏览器里): 地址栏、下载条、页脚口号。字母不分大小写
H5_KW = ["mclientalipay", "alipaycom", "支付宝付款", "下载支付宝", "便捷生活", "一点就好", "支付宝网页版", "支付宝官网",
         "app立即下载"]
H5_LATIN = ("bianjieshenghuo", "yidianjiuhao")
# 微信支付成功页下面那一段(0.75~0.92)的字: 页面上面字太少、读不出东西的时候看
WX_LOW = ["返回商家", "返回商户", "摇一摇", "收单机构", "财付通"]     # v2: 返回商户(繁体 返回商戶, norm 以后)
WX_PAGE = re.compile(r"零钱提现|发起提现申请")     # v2: 微信 零钱提现 详情页(census 一张进了 W3 到账成功)
DARK_PLATFORM = re.compile(r"投注|提现记录|出款中|利息宝|晋级VIP|申请提现")   # v2: 深色的博彩平台页(census D 里 2 张); 只在深色那一路用
YEN = re.compile(r"[¥￥]\s*[\dUIZlOS]")     # ¥ 金额; 通知条压住半截的大金额常读成 ¥1UUUU / ¥IUUUU / ¥ZUUUU / ¥SUUUU
SHOP_B3 = re.compile(r"有限公司|商家|电商|商城|超市|便利店|杂货|旗舰店|专营店")
SHOP_W1 = re.compile(r"有限公司|公司|商行|店|超市|商城|便利|药房|餐饮|商家|电商")
# 账单详情卡片上的标签: 卡片闸门没过(缩得太小、浮窗里、重压过)但读到 >= 2 个, 也算账单详情
W1_LABELS = ["创建时间", "订单号", "对方账户", "转账备注", "商品说明", "收款方全称", "商家订单号", "支付奖励",
             "账单管理", "账单分类", "计入收支", "处理进度", "再转一笔"]
SUB_BANK = ["处理进度", "提交银行处理", "银行处理中", "银行入账成功", "转账到"]
SUB_PERSON = ["对方账户", "转账备注"]
SUB_SHOP = ["商品说明", "收款方全称", "商家订单号"]


def norm(s: str) -> str:
    """全半角统一、几个繁体字改简体, 只留汉字、字母、数字和 ¥ * @。"""
    s = unicodedata.normalize("NFKC", s or "").translate(_T)
    return "".join(c for c in s if c.isalnum() or c in "¥*@")


def has(text: str, kw: str) -> bool:
    """kw 在 text 里: 先找原样的; 4 个字以上的允许错 1 个字(位置要对上, 比如 付款芳式、付默方式)。
    ★ 蓝图标题不用这个: 转账成功 和 到账成功 只差一个字, 见 title_code。"""
    if kw in text:
        return True
    n = len(kw)
    if n < 4:
        return False
    for i in range(len(text) - n + 1):
        if sum(a == b for a, b in zip(text[i:i + n], kw)) >= n - 1:
            return True
    return False


def hits(text: str, kws) -> list[str]:
    return [k for k in kws if has(text, k)]


def title_code(t: str):
    """蓝图标题 -> (类别, 标题)。原样的优先; 错一个字的只在不会混的时候认:
    "X账成功" 只有 X 是 转/到 才分得出 B1/B2, 别的字(读错的)交给字段去判。"""
    for kw, c in BLUE_TITLES:
        if kw in t:
            return c, kw
    if "付成功" in t:                    # 第一个字常和前面的对勾圆圈粘在一起读丢; 以"付成功"结尾的只有支付成功 / 代付成功
        return "B3", "付成功"
    for i in range(len(t) - 3):
        w = t[i:i + 4]
        codes = {c for kw, c in BLUE_TITLES if sum(a == b for a, b in zip(w, kw)) >= 3}
        if len(codes) == 1:
            c = codes.pop()
            if c != "B3" or "付" in w:
                return c, w
        elif codes == {"B1", "B2"}:
            if w[0] in "转传砖专":
                return "B1", w
            if w[0] in "到倒":
                return "B2", w
    # 拼音字体常见的读坏(run2 的 B0 里看到的): 付城功 / 付减功 / 支什成功 / 变符成功 / 付成访; 转岁坊 / 转账城动 / 转账动
    m = B3_FUZZY.search(t) or B1_FUZZY.search(t)
    if m:
        return ("B3" if m.re is B3_FUZZY else "B1"), m.group()
    return None, ""


B3_FUZZY = re.compile(r"[付符什][成城减][功访坊]")
B1_FUZZY = re.compile(r"转[账岁帐](?:[成城][功坊动访]|[功坊访]|动(?![态账]))")


def pinyin_code(latin: str):
    """标题上面那一行拼音也能读出来(zhuanzhang cheng gong), 汉字读坏了的时候用。"""
    if "cheng" not in latin and "gong" not in latin:
        return None
    if "uanzh" in latin:
        return "B1"
    if "daozh" in latin or "aozhang" in latin:
        return "B2"
    if "zhifu" in latin or "daifu" in latin or "hifuch" in latin:
        return "B3"
    if re.search(r"f[uo]?che?n?g", latin):    # fochenggong / fchengong: 转账 / 到账 都以 zhang 结尾, 没有 f
        return "B3"
    return None


def zd_hit(t: str) -> bool:
    """v2: 账单详情(标题)。v1 用 has 允许错任意一个字, 订单详情(店铺的订单页)也认上了(census 一张进了 W1);
    现在要有 账单 / 账.详情 / 前面不是 订 的 单详情(账 读丢了或者读成 帐 的)。"""
    return "账单详情" in t or bool(re.search(r"账单.情|账.详情|(?<!订)单详情", t))


def alipay_hits(txt: str, items=None) -> list[str]:
    """支付宝的字; 对方名字里的"支付宝"(向支付宝-xx、支付宝支付科技有限公司)先去掉。
    v2: 账单详情 要过 zd_hit; 给了 items(读过的框)时, 完成 要单独一个框才算(census: 商城 H5 页"您的订单已完成支付"里的完成被当成了支付宝的字)。"""
    t = AL_NOISE.sub("", txt)
    out = hits(t, ALIPAY_KW)
    if "账单详情" in out and not zd_hit(t):
        out.remove("账单详情")
    if "完成" in out and items is not None and not any(norm(b[4]) == "完成" for b in items):
        out.remove("完成")
    return out


def platform_hit(txt: str, lat: str) -> str:
    """平台收银台 / 存款页的字(原样匹配), 读到一个就返回它, 没有返回空。"""
    for k in PLATFORM_KW + DEPOSIT_KW:
        if k in txt:
            return k
    m = PLATFORM_ORDER.search(txt)
    if m:
        return "平台单号" + m.group()[3:9]
    # v2: 银行短信通知条上的"本机构吸收的本外币存款依照存款保险条例受到保护"不算(census 一张到账成功因此进了 X 平台页)
    dep = re.sub(r"存款保险|本外币存款", "", txt).count("存款")
    if dep >= 2:
        return f"存款 x{dep}"
    if has(txt, "类型存款"):          # 拼音字体常读成 易型存款; 它不会和支付宝的字混(存款方式 会混 付款方式, 不能这样)
        return "类型存款"
    if "shouyintai" in lat:
        return "shouyintai"
    return ""


def h5_hit(txt: str, lat: str) -> str:
    """支付宝 H5 页的线索(地址栏、下载条、页脚口号), 没有返回空。允许错一个字(has),
    只有 支付宝付款 要原样: 错一个字就是平台收银台上的"支付宝收款"。"""
    low = txt.lower()
    for k in H5_KW:
        if (k in low) if k == "支付宝付款" else has(low, k):
            return k
    for k in H5_LATIN:
        if k in lat:
            return k
    return ""


def is_banner(b) -> bool:
    """读过的框(带字)是不是通知条上的字。"""
    return len(b) > 4 and any(k in norm(b[4]) for k in BANNER_KW)


def title_strip(H: int, W: int) -> float:
    """标题检测条的下沿(占页面高度的比例), 见 TALL。"""
    return TITLE_STRIP_TALL if H / W >= TALL else TITLE_Y


# ---------------------------------------------------------------- 读图和便宜的量

def decode(path: str):
    buf = np.fromfile(path, np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if bgr is None and buf.size > 2 and buf[0] == 0xFF and buf[1] == 0xD8:
        # 截断的 JPEG(结尾少了 FFD9), 补上再解一次, 和 timeline_scan 一样
        bgr = cv2.imdecode(np.concatenate([buf, np.array([0xFF, 0xD9], np.uint8)]), cv2.IMREAD_COLOR)
    return bgr


def header_kind(small) -> str:
    """页头颜色, 规则和 new_batch.page_kind 一模一样(5%~20% 高度那一段的平均色)。"""
    H = small.shape[0]
    b, g, r = small[int(H * 0.05):int(H * 0.20)].reshape(-1, 3).mean(0)
    if b > 150 and b > r + 60:
        return "蓝底"
    if min(b, g, r) > 200:
        return "白底"
    return "其它"


def blue_top(small) -> float:
    """上半截支付宝蓝(HSV 色相 95~120、饱和度 >= 90、亮度 >= 80)的占比, 和 split_blue_white.blue_fraction 一样。"""
    hsv = cv2.cvtColor(small[:small.shape[0] // 2], cv2.COLOR_BGR2HSV)
    m = (hsv[..., 0] >= 95) & (hsv[..., 0] <= 120) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 80)
    return float(m.mean())


def flatness(gray, min_gray: int = -1) -> float:
    """check_screenshot.flatness 的量法(那边只收路径, 这里用已经解好的图): 16x16 小块标准差的下四分位。
    原始截图精确是 0; 转存/重压过的 0.01~0.3; 拍屏的 1 以上。
    ★ 蓝图的渐变底色会把它顶上去, 所以蓝底的图只在尺寸本来就可疑时才拿它判照片。
    min_gray >= 0: 只量平均灰度 > 它的小块(黑屋子里拍屏, 四周黑成同一个灰度, 不量它们)。"""
    h, w = gray.shape
    hh, ww = h // 16, w // 16
    if hh < 4 or ww < 4:
        return 0.0
    blocks = gray[:hh * 16, :ww * 16].reshape(hh, 16, ww, 16).swapaxes(1, 2).reshape(hh * ww, 256)
    if min_gray >= 0:
        blocks = blocks[blocks.mean(axis=1) > min_gray]
        if len(blocks) < 16:
            return 0.0
    return float(np.percentile(blocks.std(axis=1), 25))


def mode_share(gray, min_gray: int = -1) -> float:
    """出现最多的那个灰度占多少像素(pinyin_probe 的 flat): 截图有大片完全一样的底色, 照片没有。
    缩图必须用 INTER_NEAREST, 插值会把"完全一样"抹掉。min_gray >= 0: 只数灰度 > 它的像素。"""
    g = gray
    if max(g.shape) > 1400:
        sc = 1400 / max(g.shape)
        g = cv2.resize(g, (int(g.shape[1] * sc), int(g.shape[0] * sc)), interpolation=cv2.INTER_NEAREST)
    v = g.ravel()
    if min_gray >= 0:
        v = v[v > min_gray]
        if v.size < 1000:
            return 1.0
    return float(np.bincount(v, minlength=256).max()) / v.size


def wechat_green(small) -> float:
    """上面 15% 里微信绿(#07C160 一带: 色相 65~85、饱和度 >= 150、亮度 >= 120)占整页的比例。
    同一段里支付宝蓝 >= 0.0002 就返回 0 —— 支付宝 H5 页地址栏也有一点绿, 但它们页头有蓝色的 支 字图标。
    本地 run2: 微信支付成功页 0.0006~0.0022(被通知条盖住的读不出, 交给下面那一段的字);
    ★ 只在一个支付宝的线索都没有的时候用(账单详情上绿色的商家图标也有 0.001~0.003)"""
    h, w = small.shape[:2]
    hsv = cv2.cvtColor(small[:max(1, int(h * 0.15))], cv2.COLOR_BGR2HSV)
    hh, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = ((hh >= 95) & (hh <= 120) & (s >= 90) & (v >= 80)).sum() / (h * w)
    if blue >= 0.0002:
        return 0.0
    return float(((hh >= 65) & (hh <= 85) & (s >= 150) & (v >= 120)).sum() / (h * w))


def blue_logo(bgr) -> int:
    """上面 40% 里蓝色的方块 / 圆形图标(支付宝的 支 字图标、蓝底白勾)有几个: 宽 0.05~0.3 页宽、近似方形、
    填满 45%~92%(里面有白字 / 白勾, 白占 >= 6%), 不碰边。通知TA 那种胶囊按钮宽高比不对, 不算。
    只拿来当支付宝的线索(挡住"读到很多汉字、没有支付宝的字 -> X"), 不拿来定类。"""
    H, W = bgr.shape[:2]
    y1 = int(0.40 * H)
    if y1 < 20:
        return 0
    roi = cv2.resize(bgr[:y1], (360, max(1, int(y1 * 360 / W))), interpolation=cv2.INTER_AREA)
    m = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), (95, 150, 170), (120, 255, 255))
    n, _lab, st, _cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    found = 0
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if not (18 <= w <= 108 and 0.75 <= w / max(h, 1) <= 1.33) or x == 0 or y == 0 or x + w >= 360:
            continue
        if not (0.45 <= a / (w * h) <= 0.92):
            continue
        if (roi[y:y + h, x:x + w].min(axis=2) >= 200).mean() >= 0.06:
            found += 1
    return found


def inset_window(small) -> bool:
    """蓝色结果页是不是浮在别的页面上的小窗(平台收银台里嵌的支付宝截图、安卓多任务界面)。
    真的蓝图蓝色铺满整宽: 12%~50% 高度那一段, 左右两边 3% 的列都是支付宝蓝;
    浮窗两边是平台页的白 / 灰 / 黑底。本地 run2 的 2,536 张蓝图量过: 两边都 < 0.3、中间 >= 0.3 的 14 张,
    人眼看全是浮窗(13 张收银台 + 1 张多任务界面); 左边有悬浮工具条的真蓝图一边 0.47, 不会算进来"""
    h, w = small.shape[:2]
    hsv = cv2.cvtColor(small[int(h * 0.12):int(h * 0.50)], cv2.COLOR_BGR2HSV)
    m = (hsv[..., 0] >= 95) & (hsv[..., 0] <= 120) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 80)
    if m.size == 0:
        return False
    e = max(1, int(w * 0.03))
    side = max(float(m[:, :e].mean()), float(m[:, w - e:].mean()))
    return side < 0.3 and float(m[:, int(w * 0.3):int(w * 0.7)].mean()) >= 0.3


def pinyin_flag(gray, mode: float) -> int:
    """pinyin_probe 原样的判据(比例 >= 0.35、压住数 >= 15、平坦占比 >= 0.15), 在已经解好的灰度图上算。"""
    H, W = gray.shape
    cs = pp._components(pp._text_mask(gray), H, W)
    if len(cs) < 60:
        return 0
    ratio, n_small, stacked, _big_h = pp._stacking(cs)
    if n_small == 0:
        return 0
    return int(pp.has_pinyin({"pinyin_ratio": ratio, "stacked": stacked, "flat": mode}))


def size_cue(W: int, H: int) -> str:
    """尺寸像不像手机竖屏截图(screenshot_filter 的两条 + 横图)。返回可疑的种类, 不可疑返回空。"""
    if W > H:
        return "横图"
    if min(W, H) > 1500:
        return "相机尺寸"            # 短边 > 1500: 手机截图没有这么宽的
    if not (1.6 <= H / W <= 2.6):
        return "长宽比怪"
    return ""


# ---------------------------------------------------------------- 白底的规则闸门

def card_gate(bgr) -> tuple[str, str]:
    """账单详情上面那张"左灰标签 + 右深色取值"的卡片, 用 color_scan 的量法。返回 (卡片种类, 说明):
        A   支付宝的卡片: 对齐的行 >= 3, 页边是支付宝灰底 238~250, 标签是 #999 一带
        A0  对齐的行 >= 3, 页边整页白, 但取值是 #333(支付宝 8 月以前的老版就是整页白)
        O   对齐的行 >= 3, 页边整页白, 取值不是 #333(微信账单、别的 App 多半是这样)
        ''  没有这种卡片
    """
    r = color_scan.measure_img(bgr)
    if not r or not r.get("rows") or int(r["rows"]) < 3:
        return "", ""
    edge, lab = int(r["edge"]), int(r["label_core"])
    info = f"行{r['rows']} 页边{edge} 标签{lab}"
    if color_scan.EDGE_LO <= edge <= color_scan.EDGE_HI and color_scan.LABEL_LO <= lab <= color_scan.LABEL_HI:
        return "A", info
    if edge > color_scan.EDGE_HI and float(r["grey_share"] or 0) >= color_scan.SHARE:
        return "A0", info
    return "O", info


def arrow_rows(bgr) -> int:
    """账单管理那一列"灰字 + >"的行数(arrow_scan.measure_img 的取列办法, 不做后面的判定)。"""
    H, W = bgr.shape[:2]
    if W < 500 or H < 800:
        return 0
    rows = arrow_scan.find_rows(bgr)
    if not rows:
        return 0
    mc = Counter(r["margin"] for r in rows).most_common(1)[0][0]
    col = [r for r in rows if abs(r["margin"] - mc) <= 2]
    ah = float(np.median([r["arrow_h"] for r in col]))
    return sum(1 for r in col if abs(r["arrow_h"] - ah) <= 0.2 * ah)


def notify_button(bgr) -> int:
    """蓝图上"立即通知收款人"卡片里的蓝色胶囊按钮(通知TA)有几个: 白卡片上的支付宝蓝、宽是高的 1.8~6 倍。
    转账成功 / 到账成功 有这张卡片, 支付成功没有。本地 run1 抽查(按 OCR 标题分好的各 150 张):
        转账成功 87% 找得到   到账成功 61%   支付成功 1%
    所以只用一边: 找得到 = 转账或到账; 找不到什么也说明不了。
    标题和字段都没读到时拿来定 B1(到账成功的字段在它前面先判; run2 的 B0 里有按钮的 17 张人眼看全是转账成功)。"""
    H, W = bgr.shape[:2]
    y0, y1, x0, x1 = int(0.10 * H), int(0.50 * H), int(0.60 * W), int(0.98 * W)
    roi = bgr[y0:y1, x0:x1]
    m = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), (98, 140, 180), (118, 255, 255))
    k = max(3, W // 100)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, _lab, st, _cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    found = 0
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if y == 0:
            continue        # ★ 碰到 ROI 上沿: 通知条里支付宝图标被 0.10H 切下来的半块(run2 一张支付成功页因此报了有通知卡片)
        if not (0.08 * W <= w <= 0.26 * W and 0.022 * W <= h <= 0.075 * W and 1.8 <= w / h <= 6 and a >= 0.6 * w * h):
            continue
        # 按钮四周要是白卡片(灰度中位数 >= 225); 蓝色页头本身也是这个蓝, 但四周不白
        pad = max(4, int(0.02 * W))
        ya, yb, xa, xb = max(0, y - pad), min(roi.shape[0], y + h + pad), max(0, x - pad), min(roi.shape[1], x + w + pad)
        ring = np.ones((yb - ya, xb - xa), bool)
        ring[y - ya:y - ya + h, x - xa:x - xa + w] = False
        vals = g[ya:yb, xa:xb][ring]
        if vals.size and np.median(vals) >= 225:
            found += 1
    return found


def has_timeline(bgr) -> bool:
    return len(timeline_scan.timeline(timeline_scan.find_circles(bgr))) >= 2


# ---------------------------------------------------------------- OCR

def cut_pinyin(c, lo: float = 0.15, hi: float = 0.6, ratio: float = 0.2):
    """拼音页上一个框常是"上面一行拼音 + 下面一行汉字", 识别模型只认一行, 整框送进去常读成空或乱字。
    在框高 15%~60% 之间找墨最少的一行(墨 = 和框边底色差 > 50), 少到最多那行的 20% 以下就从那里切开, 只留下面。
    本地 120 张蓝图拼音页: 标题读出来的从 75 张升到 88 张。只在整页判出拼音的时候用。"""
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    h = g.shape[0]
    if h < 12:
        return c
    bg = int(np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])))
    ink = (np.abs(g.astype(np.int16) - bg) > 50).sum(1)
    a, z = int(h * lo), max(int(h * lo) + 1, int(h * hi))
    r = a + int(np.argmin(ink[a:z]))
    if ink[r] <= ratio * ink.max() and ink[:r].sum() > 0:
        return c[r:]
    return c


def _engine():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        # ★ 检测用 limit_type=max: 送进去的是宽 540 的横条, 不超过 960 就原样检测。
        #   默认的 min 会把短边放大到 736 —— 540x60 的标题条会被放大成 6600x736, 反而慢十倍
        _OCR = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1,
                        det_limit_type="max", det_limit_side_len=960)
        # 识别时一批里的框都补到同一个宽度, 最窄补到 320(48 高)。我们读的多是 2~6 个字的短框,
        # 补到 160 就够: 本地 12 张实测识别快 40%, 读出来的字一样(这个模型宽度本来就不固定)
        _OCR.text_rec.rec_image_shape = [3, 48, 160]
    return _OCR


class Reader:
    """一张图的 OCR: 缩小到宽 540, 按横条分段检测(先标题那一条, 要的时候再检测下面), 框按需要分批识别。
    ★ 检测按面积算钱, 识别按框数算钱: 标题那一条只有 540x60 左右, 几个框, 大部分图读完这一条就定了。
    框从原图上裁(比缩小图清楚), 识别前不旋转、不分类方向。"""

    def __init__(self, bgr, pinyin: bool = False):
        H, W = bgr.shape[:2]
        self.bgr, self.H, self.W, self.pinyin = bgr, H, W, pinyin
        self.s = OCR_W / W
        self.small = cv2.resize(bgr, (OCR_W, max(1, round(H * self.s))), interpolation=cv2.INTER_AREA)
        self.boxes = []                 # (x0, y0, x1, y1) 原图坐标
        self.text = {}                  # 框序号 -> 读出来的字(原样)

    def detect(self, y0f: float, y1f: float, x0f: float = 0.0, x1f: float = 1.0) -> None:
        """检测一个横条(按页面高度、宽度的比例给)里的字框, 加到 self.boxes。"""
        h, w = self.small.shape[:2]
        ya, yb = int(h * y0f), max(int(h * y0f) + 2, int(h * y1f))
        xa, xb = int(w * x0f), max(int(w * x0f) + 2, int(w * x1f))
        crop = self.small[ya:yb, xa:xb]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return
        det, _ = _engine().text_det(np.ascontiguousarray(crop))
        s = self.s
        for b in (det if det is not None else []):
            x0, y0 = float(b[:, 0].min()) + xa, float(b[:, 1].min()) + ya
            x1, y1 = float(b[:, 0].max()) + xa, float(b[:, 1].max()) + ya
            if y1 - y0 < MIN_BOX_H or x1 - x0 > 0.8 * OCR_W:
                continue
            if (y0 + y1) / 2 / s < STATUS * self.H:
                continue
            self.boxes.append((int(x0 / s), int(y0 / s), int(np.ceil(x1 / s)), int(np.ceil(y1 / s))))

    def read(self, pick) -> None:
        idx = [i for i, b in enumerate(self.boxes) if i not in self.text and pick(b)]
        crops = []
        for i in idx:
            x0, y0, x1, y1 = self.boxes[i]
            c = self.bgr[max(0, y0):min(self.H, y1), max(0, x0):min(self.W, x1)]
            if self.pinyin and c.size:
                c = cut_pinyin(c)
            crops.append(c if c.size else np.full((8, 8, 3), 255, np.uint8))
        if not crops:
            return
        res, _ = _engine().text_rec(crops)
        for i, r in zip(idx, res):
            self.text[i] = r[0] if float(r[1]) >= 0.3 else ""

    def items(self, pick=None):
        """读过的框, 按行从上到下、行里从左到右: [(x0, y0, x1, y1, 原文)]。
        ★ 不能直接按框顶排: 字距大的标题每个字一个框, 框顶高低差一两个像素, 会排成"功成账到"。
          所以先按框的竖直中心分行(和上一个框中心差不到半个框高算同一行), 行里再按 x 排。"""
        # pick 拿到的是 (x0, y0, x1, y1, 原文), 可以按字挑(比如去掉通知条的字)
        out = [(*self.boxes[i], t) for i, t in self.text.items() if t and (pick is None or pick((*self.boxes[i], t)))]
        out.sort(key=lambda b: (b[1] + b[3]) / 2)
        lines, last = [], None
        for b in out:
            yc, h = (b[1] + b[3]) / 2, b[3] - b[1]
            if last is not None and abs(yc - last[0]) < 0.5 * min(h, last[1]):
                lines[-1].append(b)
            else:
                lines.append([b])
                last = (yc, h)
        return [b for ln in lines for b in sorted(ln, key=lambda b: b[0])]

    def joined(self, pick=None) -> str:
        return "".join(norm(b[4]) for b in self.items(pick))

    def latin(self, pick=None) -> str:
        return "".join(c for b in self.items(pick) for c in b[4].lower() if "a" <= c <= "z")

    def first_cjk(self, pick=None) -> str:
        for b in self.items(pick):
            t = norm(b[4])
            if sum("一" <= c <= "鿿" for c in t) >= 2:
                return t[:12]
        return ""

    def band(self, y0f: float, y1f: float) -> str:
        """再检测、识别一段(按高度比例), 返回这一段读到的字。"""
        self.detect(y0f, y1f)
        pick = lambda b: y0f * self.H <= (b[1] + b[3]) / 2 < y1f * self.H     # noqa: E731
        self.read(pick)
        return self.joined(pick)


# ---------------------------------------------------------------- 分类

def classify(path: str, use_ocr: bool) -> dict:
    out = {"path": path, "W": "", "H": "", "cls": "U", "sub": "", "state": "", "header": "", "dark": "", "flat": "",
           "pinyin": "", "device": "", "title": "", "reason": "", "card": "", "mode": "", "ocr": ""}
    bgr = decode(path)
    if bgr is None:
        out["reason"] = "读不出"
        return out
    H, W = bgr.shape[:2]
    out.update(W=W, H=H, device=DEVICE.get(resolution_platform(W, H), "说不清"))
    if W < 300 or H < 400:
        out["reason"] = "图太小"
        return out

    # ---- 1. 尺寸
    if W == VOUCHER_W and VOUCHER_H[0] <= H <= VOUCHER_H[1]:
        out.update(cls="W2", sub="转账凭证图", header="白底", reason=f"尺寸 {W}x{H}: 支付宝生成的转账凭证图")
        return out
    cue = size_cue(W, H)

    # ---- 2. 颜色(1/4 缩小图)
    small = cv2.resize(bgr, (max(1, W // 4), max(1, H // 4)), interpolation=cv2.INTER_AREA)
    head = header_kind(small)
    gs = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    dark = int(np.median(gs[int(gs.shape[0] * 0.095):int(gs.shape[0] * 0.96)]) < DARK_MEDIAN)
    btop = blue_top(small)
    out.update(header=head, dark=dark)

    # ---- 3. 照片
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    q = flatness(gray)
    mode = mode_share(gray)
    out.update(flat=round(q, 3), mode=round(mode, 3))
    photo = ""
    sz = f"{W}x{H}"
    if cue in ("横图", "相机尺寸"):
        # 只量灰度 > 40 的地方: 黑屋子里横着拍的屏, 四周黑成同一个灰度, 平整度 0、最多灰度占 0.37, 像截图(run2 判成了深色)
        # v2: 亮处不到 15% 的(黑底视频截图上只有一行字)亮处只剩字的笔画, 量出来的平整度没意义, 用整张的(census P 一张)
        if float((gray > PHOTO_MIN_GRAY).mean()) >= 0.15:
            q2, mode2 = flatness(gray, PHOTO_MIN_GRAY), mode_share(gray, PHOTO_MIN_GRAY)
        else:
            q2, mode2 = q, mode
    if W * H >= PHOTO_MP:
        photo = f"{sz} 有 {W * H / 1e6:.1f} 百万像素, 手机 / 平板截图没有这么大"
    elif cue in ("横图", "相机尺寸") and (q2 >= PHOTO_Q or mode2 < PHOTO_MODE):
        photo = f"{cue} {sz}, 亮处平整度 {q2:.2f} 最多灰度占 {mode2:.3f}"
    elif cue == "长宽比怪" and (q >= PHOTO_Q_ODD or mode < PHOTO_MODE):
        photo = f"长宽比 {H / W:.2f} {sz}, 平整度 {q:.2f} 最多灰度占 {mode:.3f}"
    elif head != "蓝底" and q >= PHOTO_Q_FREE and mode < PHOTO_MODE_FREE:
        # 蓝底不判: 渐变底色 + 重压过的蓝图最多灰度占比也会掉到 0.04(本地 1 张)
        photo = f"底色有噪点(平整度 {q:.2f}), 最多灰度只占 {mode:.3f}"
    if photo:
        out.update(cls="P", sub=cue or "尺寸正常", reason="像照片: " + photo)
        return out

    # 拼音标记: 苹果原尺寸截图不判
    if (W, H) not in IPHONE_RESOLUTIONS:
        out["pinyin"] = pinyin_flag(gray, mode)

    # v2: 异常页(钱没到 / 退回来了 / 还在处理), status_icon 只看图、便宜(蓝图几毫秒)。蓝图的三种在 _blue 里直接定 B4;
    #   白图的三种照常分类(都是账单详情的模板), 只记在 state 里
    st = status_icon.check(bgr)
    if st["verdict"] == "Suspicious":
        out["state"] = st["kind"]

    # 页头不是蓝的但上半截蓝很多: 通知条(白的、灰的)盖住了蓝图的页头, 或者弹窗把蓝图压暗了。
    # 本地 run1 量过: 白底页头的账单详情上半截蓝最多 0.17。
    # v2: 0.18 以上先看是不是平台页里嵌的浮窗; 走蓝图那一路要 0.30 以上(见 BLUE_TOP_ROUTE)
    if not dark and head != "蓝底" and btop >= BLUE_TOP and inset_window(small):
        return _x(out, f"页头被盖住, 上半截蓝 {btop:.2f}, 蓝色结果页是浮窗(两边不是蓝的), 像平台页里嵌的截图", "", "浮窗")
    is_blue = not dark and (head == "蓝底" or btop >= BLUE_TOP_ROUTE)
    if dark:
        return _dark(out, bgr, small, use_ocr)
    if is_blue:
        return _blue(out, bgr, small, use_ocr, head, btop)
    return _white(out, bgr, small, use_ocr, head)


def x_words(txt: str) -> list[str]:
    """不是支付宝的字: X_KW 允许错一个字, X_EXACT 要原样。"""
    return hits(txt, X_KW) + [k for k in X_EXACT if k in txt]


def _xcheck(txt: str, al: list[str], lat: str = "", strict_card: bool = False):
    """不是支付宝的字: 强的(微信支付、财付通 ...)原样读到一个就够, 别的要 >= 2 个而且不比支付宝的字少。"""
    xs = x_words(txt)
    strong = [k for k in xs if k in X_STRONG and k in txt] + [k for k in X_LATIN if k in lat]
    if strong and not strict_card:
        return strong + [k for k in xs if k not in strong]
    if len(xs) >= 2 and len(xs) >= len(al):
        return xs
    return None


def _x(out, why: str, title: str = "", sub: str = ""):
    out.update(cls="X", sub=sub, title=title, reason=why)
    return out


def _dark(out, bgr, small, use_ocr):
    if not use_ocr:
        out.update(cls="D", reason=f"深色(中位灰度 < {DARK_MEDIAN})")
        return out
    rd = Reader(bgr, str(out.get("pinyin")) == "1")
    rd.detect(0.0, PAGE_FRAC)
    rd.read(lambda b: True)
    txt = rd.joined()
    lat = rd.latin()
    out["ocr"] = f"[{len(rd.text)}] " + txt[:120]
    pf = platform_hit(txt, lat)
    if pf:
        return _x(out, "深色, 平台收银台 / 存款页: " + pf, rd.first_cjk(), "平台页")
    al = alipay_hits(txt)
    xs = _xcheck(txt, al, lat)
    if xs:
        return _x(out, "深色, 不是支付宝: " + "/".join(xs[:3]), rd.first_cjk())
    if hits(txt, RED_KW):
        out.update(cls="W3", sub="红包", title="红包", reason="红包页(底色本来就深)")
        return out
    h5 = h5_hit(txt, lat)
    trade = TRADE_NO.search(txt)
    if not (al or h5 or trade) and wechat_green(small) >= GREEN_MIN:
        return _x(out, "深色, 页头有微信绿, 没有支付宝的字", rd.first_cjk(), "微信")
    # v2: census 的 D 20 张里只有 10 张是支付宝(深色浏览器里的 H5 结果页), 另外 6 张微信、2 张平台页、2 张不是页面。
    #   平台页的字(提现 / 投注 / VIP)、微信(绿色标题 / 返回商家 / XX的店铺 的账单)分出去; 一点支付宝线索都没有的不算 D
    m = DARK_PLATFORM.search(txt)
    if m:
        return _x(out, "深色, 平台页: " + m.group(), rd.first_cjk(), "平台页")
    items = rd.items()
    if title_icon(bgr, items) == "green" and not trade:
        return _x(out, "深色, 支付成功 标题和左边的图标是微信绿", rd.first_cjk(), "微信")
    if not (al or h5 or trade or blue_logo(bgr)):
        k = [w for w in WX_LOW if w in rd.band(0.75, 0.92)] + (["的店铺"] if "的店铺" in txt else [])
        if k:
            return _x(out, "深色, 没有支付宝的线索, 有 " + "/".join(k), rd.first_cjk(), "微信")
        ncjk = sum("一" <= c <= "鿿" for c in txt)
        if ncjk >= 12:
            return _x(out, f"深色, 读到 {ncjk} 个汉字, 没有支付宝的线索", rd.first_cjk())
        out.update(cls="U", sub="深色", title=rd.first_cjk(), reason=f"深色, 没有支付宝的线索, 字太少({ncjk} 个汉字)")
        return out
    sub = ""
    t2 = txt.replace("成功后", "后")          # "付款成功后请截图" 是平台页的提示, 不是标题
    for kw in ("账单详情", "转账成功", "到账成功", "支付成功", "代付成功", "付款成功", "交易成功"):
        if (has(t2, kw) if kw == "账单详情" else kw in t2):
            sub = kw
            break
    if sub == "支付成功" and h5:
        sub = "H5支付成功"
    out.update(cls="D", sub=sub, title=sub or rd.first_cjk(),
               reason=f"深色(中位灰度 < {DARK_MEDIAN})" + (f", 读到 {sub}" if sub else "") + ("" if al else ", 没读到支付宝的字"))
    return out


def _blue(out, bgr, small, use_ocr, head, btop):
    why = "蓝底" if head == "蓝底" else f"页头被盖住, 上半截蓝 {btop:.2f}"
    if inset_window(small):
        # 平台收银台里嵌的转账成功浮窗、安卓多任务界面: 不是整页的支付宝结果页, 不能交给蓝图的模型
        return _x(out, why + ", 蓝色结果页是浮窗(两边不是蓝的), 像平台页里嵌的截图", "", "浮窗")
    if out.get("state") in BLUE_ABNORMAL:
        # v2: 到账失败 / 提交成功 / 等待对方领取 的页头图标(status_icon 看图定的), 不用读字。
        #   census 里 8 张到账失败被分进了 B3 支付成功、提交成功在 B2、等待对方领取在 B0, 钱没到的页不能交给成功页的模型
        k = out["state"]
        out.update(cls="B4", sub=k, title=k, reason=why + f", 页头图标是 {k}(看图)")
        return out
    if not use_ocr:
        btn = notify_button(bgr)
        out.update(cls="B0", sub="有通知卡片(转账或到账)" if btn else "",
                   reason=why + ", 不做 OCR 分不了三类" + (", 有通知TA按钮" if btn else ""))
        return out
    py = str(out.get("pinyin")) == "1"
    rd = Reader(bgr, py)
    H, W = rd.H, rd.W
    # 标题区: 框中心 < TITLE_Y, 不算右上角的框(回首页, 拼音字体常读成 简曾贵)和通知条的字
    in_title = lambda b: (b[1] + b[3]) / 2 < TITLE_Y * H and b[0] <= 0.75 * W and not is_banner(b)   # noqa: E731
    ts = TITLE_Y
    rd.detect(0.0, ts)
    rd.read(lambda b: True)
    code, kw = title_code(rd.joined(in_title))
    if not code and title_strip(H, W) > ts:
        # 长屏: 0~0.115 的条没读到标题, 换 0~0.14 的条从头再检测一次(见 TALL)。
        # ★ 不能一上来就用 0.14: 也有反过来的, 0.14 的条把 到账成功 切成 到 / 账成功(run2 随机抽 75 张 B2 坏了 3 张)
        rd2 = Reader(bgr, py)
        rd2.detect(0.0, title_strip(H, W))
        rd2.read(lambda b: True)
        code, kw = title_code(rd2.joined(in_title))
        if code or pinyin_code(rd2.latin(in_title)):
            rd, ts = rd2, title_strip(H, W)
    how = f"标题 {kw}" if code else ""
    # ★ 蓝底 + 标题读到就停(大部分蓝图只读这 3~5 个框)。标题没读到、或者页头不是蓝底(被通知条盖住的)、
    #   或者有不是支付宝的字, 再检测下面一段, 读字段、看是不是支付宝
    head_txt = rd.joined()
    if not code or head != "蓝底" or x_words(head_txt) or platform_hit(head_txt, ""):
        rd.detect(ts, BLUE_FRAC)
        rd.read(lambda b: True)
    cls, sub, title, reason = _blue_decide(rd, bgr, why, head, code, kw, how, in_title, BLUE_FRAC)
    if cls == "B0" and not sub and not code:
        # v2: 0~0.34 没定下来(标题被通知条盖住、字段不够), 再读 0.34~0.60: 支付有礼 / 到账时间 / 立即通知收款人 多在这一段;
        #   从 0.37 起的半屏到账成功浮层也要这一段才看得到(census B0 里一张)。只有定不下来的才多读, 大部分蓝图不受影响
        rd.detect(BLUE_FRAC, 0.60)
        rd.read(lambda b: True)
        c2 = _blue_decide(rd, bgr, why, head, code, kw, how, in_title, 0.60)
        if c2[0] != "B0" or c2[1]:
            cls, sub, title, reason = c2[0], c2[1], c2[2], c2[3] + ("" if c2[0] == "X" else ", 读到 0.60")
    out["ocr"] = f"[{len(rd.text)}] " + rd.joined()[:120]
    if cls == "B0":
        out.update(cls="B0", sub=sub, title=title or rd.first_cjk(in_title),
                   reason=why + ", " + (reason or "没读到标题和分得开的字段"))
        return out
    out.update(cls=cls, sub=sub, title=title, reason=reason)
    return out


def _blue_decide(rd, bgr, why, head, code, kw, how, in_title, ylim):
    """蓝图读完字以后的判定: 返回 (类别, 子类, 标题, 理由)。类别 B0 时理由不带开头的 why(调用方补)。
    ylim: 字段找 ¥ 金额只看这个高度以上(第一次 0.34, 再读一段以后 0.60)。"""
    txt = rd.joined()
    lat = rd.latin()
    pf = platform_hit(txt, lat)
    if pf:
        # 平台收银台里嵌的支付宝截图: 订单号 / 付款方式 / 收款方都有, 支付宝的字总比平台的多, 所以不比多少
        return "X", "平台页", kw or rd.first_cjk(in_title), why + ", 平台收银台: " + pf
    al = alipay_hits(txt)
    xs = _xcheck(txt, al, lat)
    if xs and len(xs) > len(al):
        return "X", "", kw or rd.first_cjk(in_title), why + ", 不是支付宝: " + "/".join(xs[:3])
    ab = abnormal_text(txt, lat, rd.joined(in_title))
    if ab:
        return "B4", ab[0], ab[0], why + ", " + ab[1]
    ym = yue_hit(txt, lat)
    if ym:
        # v2: 余额明细详情的导航栏是蓝的(页头算蓝底), 本地一张 v1 判 B0、加了读 0.60 以后被商家名带成了 B3
        return "W4", "", rd.first_cjk(in_title) or "余额明细", why + f", 余额明细详情: 有 {ym}"
    sub = "代付成功" if "代付" in kw else ""
    if not code:
        code, kw = title_code(txt)
        how = f"标题 {kw}(不在标题区)" if code else ""
        sub = "代付成功" if "代付" in kw else ""
    if not code:
        pc = pinyin_code(rd.latin(in_title))
        if pc:
            code, how = pc, "标题汉字没读出来, 按上面的拼音"
    if not code:
        sp = _blue_special(txt)
        if sp:
            c, s, t, h = sp
            return c, s, t, (why + ", " + h) if c != "B0" else h
        code, sub, how = _blue_fields(rd, txt, lat, bgr, ylim)
        how = "没读到标题, " + how if how else ""
    if not code or code == "B0":
        return "B0", sub, rd.first_cjk(in_title), how
    sure = al or (head == "蓝底" and (kw in dict(BLUE_TITLES) or kw == "付成功"))
    return code, sub, kw or rd.first_cjk(in_title), why + ", " + how + ("" if sure else ", ★ 没读到支付宝的字段")


def abnormal_text(txt: str, lat: str, tt: str):
    """v2: 蓝图异常页按读到的字定(页头图标被通知条盖住、拼音字体看不出图标的): 返回 (种类, 理由) 或 None。
    ★ 原路退回 不单独认: 等待对方领取 页上也有"逾期未完善资金将原路退回"; 资金将原路返回 是到账失败页那一句。
    ★ 话费充值页(提交成功 -> 运营商处理中 -> 充值成功)的 提交成功 是进度, 不算, 交给 _blue_fields 记 B0 话费充值。"""
    if "到账失败" in txt or "资金将原路返回" in txt or "daozhangshibai" in lat:
        return "到账失败", "读到 " + ("到账失败" if "到账失败" in txt else "资金将原路返回" if "资金将原路返回" in txt else "daozhangshibai")
    if "等待对方领取" in txt:
        return "等待对方领取", "读到 等待对方领取"
    if re.search(r"运营商|充值成功|充值中|话费充值", txt):
        return None
    if "提交成功" in tt:
        return "提交成功", "标题 提交成功"
    if "提交成功" in txt and "到账" in txt:
        return "提交成功", "读到 提交成功 + 到账"
    return None


def _blue_special(txt: str):
    """v2: 没读到标题的蓝图里不是三种结果页的(census 核出来的): 返回 (类别, 子类, 标题, 理由) 或 None。
        收款码图(支付宝个人收款码海报) / 平台扫码收银台(订单金额 复制 上传截图)        -> X
        授权成功(订单金额 + 付款方式, 没有商家那一行, 不是支付成功的模板) / 订单展示     -> B0 子类"""
    m = QR_RE.search(txt)
    if m:
        sub = "平台页" if re.search(r"订单金额|复制|上传|取消订单|已付款", txt) else "收款码图"
        return "X", sub, "", f"没读到标题, 有 {m.group()}(收款码 / 扫码付款页)"
    if "授权成功" in txt:
        return "B0", "授权成功", "授权成功", "标题 授权成功(不是支付成功的模板)"
    if "订单展示" in txt:
        return "B0", "订单展示", "订单展示", "有 订单展示"
    return None


def _bank_row(rd, ys, W: int) -> str:
    """右边 ¥ 金额那一行(框左边 > 0.55 W)左边的字里有 银行名 + 卡尾号 4 位(到账成功的收款银行卡), 返回那段字; 没有返回空。
    ¥ 和银行卡读成一个框的也算。census: 72 张 B2 每张都有这一行, B3 一张都没有(B3 那一行是商家名)。"""
    items = rd.items()
    for y in ys:
        if BANK_TAIL.search(norm(y[4])):
            return norm(y[4])[:16]
        if y[0] <= 0.55 * W:
            continue
        yc, h = (y[1] + y[3]) / 2, y[3] - y[1]
        left = "".join(norm(b[4]) for b in items
                       if b[2] <= y[0] + 2 and abs((b[1] + b[3]) / 2 - yc) < 0.6 * max(h, b[3] - b[1]))
        if BANK_TAIL.search(left):
            return left[:16]
    return ""


def _blue_fields(rd, txt: str, lat: str, bgr, ylim: float = BLUE_FRAC):
    """蓝图标题没读到(多半是通知条盖住了标题和大金额的上半截)时按字段分, 返回 (类别, 子类, 理由)。
    各条线索在标题读得到的蓝图上量过(run2 随机 B1 150 / B2 100 / B3 150 张, 读 0~0.34):
                      B1    B2    B3
        到账时间一类     0    42     0      立即通知收款人一类  14   1   0      收款方(含读坏的) 106   2   0
        通知按钮       137    55     7      两个 ¥              2  99 145      商家名(有限公司 ...) 1   0  60
    所以顺序是:
        话费充值页(提交成功 -> 运营商处理中 -> 充值成功)单独记 B0, 不是到账成功
        (v2: 提交成功 + 到账 改在 abnormal_text 里记 B4, 不再算到账成功)
        到账时间 / 使用转账凭证 / 预计..到账 / 查看到账进度 / 拼音 daozhangshi -> 到账成功
            ★ 不能认拼音 daozhang: 支付成功页的券"红包已到账"读出来是 hongbaoyidaozhang
        立即通知收款人 / 预约转账 / 通知TA、收款方(读坏的 收默方 / 收款芳 / 收款王涛 也算)-> 转账成功
            ★ 要在 支付有礼 前面: 转账成功页也有 支付有礼 卡片(本地有一张)
            ★ v2: 收款姓名 不算(平台扫码收银台的, census 3 张因此判成了 B1)
        v2: 右边 ¥ 那一行左边是 银行名 + 卡尾号 -> 到账成功(census 3 张标题读坏的到账成功被判成了支付成功)
        读到 账成功(第一个字丢了): 只能是转账 / 到账, 两个 ¥ -> 到账成功(银行卡那一行带 ¥), 有通知按钮 -> 转账成功
        交易商家 / 交易方式 / 给商家 / 支付有礼 / 商家名 -> 支付成功
        通知按钮: 右边没有 ¥ 金额 -> 转账成功; 有的分不开到账 / 支付, 记 B0
        两个 ¥ / 商家那一行的 ¥ + 付款方式 -> 支付成功(到账成功也有两个 ¥, 但它多半有按钮或者到账时间, 前面已经分走)"""
    H, W = rd.H, rd.W
    if re.search(r"运营商|充值成功|充值中|话费充值", txt):
        return "B0", "话费充值", "话费充值页"
    b2 = hits(txt, ["到账时间", "使用转账凭证"])
    m = re.search(r"预计.{0,8}到账|查看到账进度", txt)
    if m:
        b2.append(m.group())
    if "daozhangshi" in lat:
        b2.append("daozhangshi")
    if b2:
        return "B2", "", "有 " + "/".join(b2)
    b1 = hits(txt, ["立即通知收款人", "预约转账"]) + (["通知TA"] if "通知ta" in txt.lower() else [])
    if b1:
        return "B1", "", "有 " + "/".join(b1)
    m = re.search(r"(?<!码)收[款默](?!方全称|方式|码|姓名)", txt)
    if m:
        return "B1", "", f"有 {m.group()}(收款方)"
    ys = [b for b in rd.items(lambda b: (b[1] + b[3]) / 2 < ylim * H) if YEN.search(b[4])]
    bank = _bank_row(rd, ys, W)
    if bank:
        return "B2", "", f"右边 ¥ 那一行是银行卡({bank})"
    btn = notify_button(bgr) > 0
    if "账成功" in txt:
        if len(ys) >= 2 or any(b[0] > 0.55 * W for b in ys):
            return "B2", "", "有 账成功 + 银行卡那一行的 ¥ 金额"
        if btn:
            return "B1", "", "有 账成功 + 通知TA按钮"
        return "B0", "", "有 账成功(转账或到账), 字段分不开"
    b3 = [k for k in ("交易方式", "给商家") if k in txt] + hits(txt, ["交易商家", "支付有礼"])
    m = SHOP_B3.search(txt)
    if m:
        b3.append(m.group())
    if b3:
        return "B3", "", "有 " + "/".join(b3)
    if btn:
        # 转账成功页右边没有 ¥ 金额(150 张里 0 张); 到账成功的银行卡那一行、支付成功的商家那一行有。
        # 支付成功页的奖励卡片上也有蓝色胶囊按钮(150 张里 7 张), 所以有按钮又有右边的 ¥ 就分不开
        if len(ys) >= 2 or any(b[0] > 0.55 * W for b in ys):
            return "B0", "", "有通知TA按钮又有右边的 ¥ 金额, 分不开到账 / 支付"
        return "B1", "", "有通知TA按钮"
    if len(ys) >= 2:
        return "B3", "", "上面有两个 ¥ 金额"
    if any(b[0] > 0.55 * W for b in ys) and re.search(r"付款方式|[付符]款|付默", txt):
        return "B3", "", "商家那一行有 ¥ 金额 + 付款方式"
    return None, "", ""


def yue_hit(txt: str, lat: str) -> str:
    """v2: 余额明细详情页(W4)的线索, 返回读到的那段, 没有返回空。
    census + 拼音库 + 本地 5.5 万行: 余额明细 / 额明細(读坏的)只出现在余额明细详情上(进了 U 47 / W1 4 / X 3);
    明细详情 单独不算: 银行 App 的"明细详情"有 28 张(X), 它们没有 关联账单。"""
    m = re.search(r"余额明[细細]|额明[细細]", txt)
    if m:
        return m.group()
    if "mingxixiangqing" in lat:
        return "mingxixiangqing"
    if "关联账单" in txt and ("查看账单详情" in txt or re.search(r"明[细細]详?情", txt)):
        return "关联账单 + 查看账单详情"
    return ""


def title_icon(bgr, items) -> str:
    """v2: 白底 支付成功 / 付款成功 标题(连同左边的图标)的颜色: 'green' 微信 / 'blue' 支付宝 / '' 说不清。
    微信支付结果页: 标题左边绿色的微信支付图标、标题字也是绿的(#07C160, OpenCV 色相 ~74);
    支付宝的白底支付成功: 蓝色 支 字图标、标题字蓝(#1677FF, 色相 ~107)。
    census: U 里 83 张"只有标题 支付成功、没有支付宝的线索"全是微信支付的结果页(返回商家在 0.75~0.85, 上面 70% 读不到)。
    ★ 不用 wechat_green: 它看整条页头, 状态栏 / 灵动岛上的支付宝蓝图标会把它否掉(census 两张)。"""
    H, W = bgr.shape[:2]
    for b in items:
        if (b[1] + b[3]) / 2 > 0.35 * H or not re.search(r"支付成功|付款成功", norm(b[4])):
            continue
        h = b[3] - b[1]
        x0, x1 = max(0, int(b[0] - 2.5 * h)), min(W, int(b[2]))
        y0, y1 = max(0, int(b[1] - 0.6 * h)), min(H, int(b[3] + 0.6 * h))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hh, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        g = int(((hh >= 65) & (hh <= 85) & (s >= 120) & (v >= 100)).sum())
        bl = int(((hh >= 95) & (hh <= 120) & (s >= 120) & (v >= 100)).sum())
        n = (y1 - y0) * (x1 - x0)
        if g >= 0.02 * n and g > 3 * bl:
            return "green"
        if bl >= 0.02 * n and bl > 3 * g:
            return "blue"
        return ""
    return ""


def _white(out, bgr, small, use_ocr, head):
    H, W = bgr.shape[:2]
    card, cinfo = card_gate(bgr)
    tl = has_timeline(bgr)
    ar = arrow_rows(bgr) if not (card == "A" or tl) else 0
    out["card"] = card + ("T" if tl else "") + (f"R{ar}" if ar >= 2 else "")
    rule_w1 = ""
    if tl:
        rule_w1 = "处理进度时间轴"
    elif card == "A":
        rule_w1 = "支付宝账单详情卡片(" + cinfo + ")"
    elif card == "A0":
        rule_w1 = "左标签右取值的卡片, 整页白但取值是 #333(支付宝老版)"
    # ★ 账单管理的 > 箭头行单独不够(run2 抽 24 张错 2 张: 亲情卡、套图网站的模板), 要读字再定, 见下面
    if not use_ocr:
        if rule_w1 or ar >= 2:
            out.update(cls="W1", sub="银行卡转账" if tl else "", reason=rule_w1 or f"账单管理 > 箭头行 {ar} 行")
        elif card == "O":
            out.update(cls="U", reason="左标签右取值但整页白、取值不是 #333(像微信账单), 不做 OCR 分不清")
        else:
            out.update(cls="U", reason=f"{head}, 没有账单详情卡片, 不做 OCR 分不清")
        return out
    if tl:
        # 时间轴是支付宝账单详情才有的(蓝色带对勾的圆点竖着排), 子类也定了, 不用读字
        out.update(cls="W1", sub="银行卡转账", reason=rule_w1)
        return out

    rd = Reader(bgr, str(out.get("pinyin")) == "1")
    # ★ 白底不用长屏的 0.14 检测条: 改了以后检测框的切法变了, 聊天转账页的"聊天更方便"、H5 页的"下载支付宝"、
    #   凭证截图的"收款方姓名"反而读丢了(人眼核过的 511 格里 W3 坏了 5 格、W2 坏了 1 格), 白底的标题也没有被切的问题
    ts = TITLE_Y
    # 标题区: 框中心 < TITLE_Y, 不算右上角的框(完成 / 回首页)和通知条的字
    in_title = lambda b: (b[1] + b[3]) / 2 < TITLE_Y * H and b[0] <= 0.75 * W and not is_banner(b)   # noqa: E731
    rd.detect(0.0, ts)
    if rule_w1:
        # 已经知道是账单详情: 只检测左边 30%、只读从左边 15% 以内起头的框(标签列), 够分子类。
        # 拼音页的标签常读坏(手写字体), 子类没分出来再读右边的取值, 按取值的样子猜;
        # 普通字体标签读不出子类的, 多半本来就是"其它"(充值、缴费 ...), 不再多花时间
        rd.detect(ts, PAGE_FRAC, 0.0, 0.30)   # ★ 不能截到 0.64: 商户消费的卡片长, 收款方全称/商家订单号在 0.6~0.7
        rd.read(lambda b: b[0] < 0.15 * W)
        # 收钱码收款: 标签和商户消费一样(商品说明 / 收款方全称), 只有 商品说明 的取值是"收钱码收款",
        # 所以读到 商品说明 就把同一行的取值也读一下(一小条, 一两个框)。
        # 拼音字体常把 商品说明 读成 品明 / 明 / 得式祭蜜明, 以"明"结尾的短框也算(左边标签里只有它以"明"结尾)
        for b in rd.items(lambda b: b[0] < 0.15 * W):
            t = norm(b[4])
            if has(t, "商品说明") or "说明" in t or "品明" in t or (t.endswith("明") and len(t) <= 5):
                h = b[3] - b[1]
                rd.detect((b[1] - 0.2 * h) / H, (b[3] + 0.2 * h) / H, 0.18, 1.0)
                rd.read(lambda v: v[0] > 0.18 * W and abs((v[1] + v[3]) / 2 - (b[1] + b[3]) / 2) < h)
                break
        if _w1_sub(rd.joined())[0] == "其它" and rd.pinyin:
            rd.detect(ts, 0.62, 0.18, 1.0)
            rd.read(lambda b: b[2] - b[0] < 0.55 * W)
    else:
        # v2: 先只读标题条。标题是 转账凭证 的(支付宝 转账凭证 页面的截图, 导航栏的标题)直接定 W2 页面截图, 不读下面 ——
        #   这种页 21 个框左右, v1 平均 25 秒一张, 占 census 全部 CPU 的四分之一。只再读 0.70~0.95 一段, 看有没有微信 / 财付通。
        #   census + 拼音库 + 本地 5.5 万行: 标题附近有 转账凭证 的 1,127 张是 W2, 另有 3 张凭证页 v1 判错了, 没有银行 / 微信的页
        #   ★ 标题条用另一个 Reader 读: 识别是一批一批做的、按一批里最宽的框补边, 标题的框单独一批识别,
        #     正文的框读出来就和 v1 不一样了(本地 8 张拼音账单因此 商家订单号 读成 家军号, 从 W1 掉成了 U)。
        #     所以不是凭证的页, 下面照 v1 原样(标题条 + 正文一起检测、一批识别), 多花的只是标题条的几个框
        tr = Reader(bgr, rd.pinyin)
        tr.detect(0.0, ts)
        tr.read(lambda b: True)
        t0 = tr.joined(in_title) + tr.latin(in_title)
        if W2_TITLE.search(t0) or "zhuanzhangpingzheng" in t0:
            low = tr.band(0.70, 0.95)
            xl = [k for k in X_STRONG if k in low]
            if not xl and not platform_hit(tr.joined(), tr.latin()):
                out["ocr"] = f"[{len(tr.text)}] " + tr.joined()[:120]
                out.update(cls="W2", sub="页面截图", title="转账凭证",
                           reason=f"{head}, 标题区是 转账凭证(凭证页面的截图), 只读了标题和下面一段")
                return out
        rd.detect(ts, PAGE_FRAC)
        rd.read(lambda b: True)
    # v2: 苹果从别的 App 跳过来时状态栏左上角的"◀ 支付宝"(返回支付宝)不算支付宝的字(census: 一张商城的订单页因此进了 W3)
    nobl = lambda b: not ((b[1] + b[3]) / 2 < 0.06 * H and b[0] < 0.3 * W and norm(b[4]) == "支付宝")   # noqa: E731
    items = rd.items(nobl)
    txt = rd.joined(nobl)
    lat = rd.latin(nobl)
    tt = rd.joined(lambda b: in_title(b) and nobl(b))
    first = rd.first_cjk(in_title)
    out["ocr"] = f"[{len(rd.text)}] " + txt[:120]
    # ---- 平台收银台 / 存款页: 在一切支付宝模板之前(它们常嵌一张账单详情 / 转账成功截图)
    pf = platform_hit(txt, lat)
    if pf:
        return _x(out, "平台收银台 / 存款页: " + pf, first, "平台页")
    # ---- v2: 余额明细详情(W4), 在"不是支付宝的字"之前: 它正文里的 明细详情 / 资金明细 / 可用余额 都在 X 的字里
    ym = yue_hit(txt, lat)
    if ym:
        out.update(cls="W4", sub="", title=first or "余额明细", reason=f"{head}, 余额明细详情: 有 {ym}")
        return out
    al = alipay_hits(txt, items)
    xs = _xcheck(txt, al, lat, strict_card=card == "A")
    if xs and not (rule_w1 and len(al) >= len(xs)):
        return _x(out, "不是支付宝: " + "/".join(xs[:4]), first)
    # ---- v2: 支付消息列表, 在账单详情的规则之前(卡片闸门会把它的卡片当成账单详情)
    pm = PAYMSG_RE.search(txt)
    if pm and not hits(txt, ["创建时间", "支付时间", "订单号", "商品说明"]):
        out.update(cls="W3", sub="支付消息", title=first or "支付消息",
                   reason=f"{head}, 支付消息列表: 有 {pm.group()}, 没有账单详情的时间 / 订单号标签")
        return out
    w2 = hits(txt, W2_KW)
    if len(w2) >= 2 or (w2 and (has(txt, "转账凭证") or "转出成功" in txt)):
        # 不是凭证尺寸、按字认的: 微信的"向银行卡转账电子凭证"字段很像, 财付通的章在 0.76 上下, 再读下面一段看看
        low = rd.band(0.70, 0.95)
        xl = [k for k in X_STRONG if k in low]
        if xl:
            return _x(out, "像转账凭证, 但下面有 " + "/".join(xl), first)
        out.update(cls="W2", sub=_w2_sub(txt, tt, W, H), title="转账凭证",
                   reason=f"尺寸 {W}x{H} 不对, 但有 " + "/".join(w2[:3]))
        return out
    trade = TRADE_NO.search(txt)
    w1l = hits(txt, W1_LABELS)
    if not rule_w1 and zd_hit(tt) and len(al) >= 3:
        rule_w1 = "标题 账单详情 + " + "/".join([k for k in al if k != "账单详情"][:3])
    if not rule_w1 and len(w1l) >= 2:
        rule_w1 = "没有卡片, 但读到账单详情的标签 " + "/".join(w1l[:3])
    # 装饰字体 / 很淡的拼音字体: 卡片闸门过不了、标签只读出 1 个, 但交易号的数字还读得准
    extra = (["交易号"] if trade else []) + [k for k in ("交易成功", "全部账单") if k in txt]
    if re.search(r"(?<!订)单详情|账单情", tt):           # v2: 订单详情 不算
        extra.append("账单详情")
    if not rule_w1 and ar >= 2 and (extra or w1l or al):
        # 箭头行 + 随便一个支付宝的字(拼音字体的账单常只读出 付款方式); 亲情卡页一个都没有
        rule_w1 = f"账单管理 > 箭头行 {ar} 行 + " + "/".join((extra + w1l + al)[:3])
    if not rule_w1 and w1l and extra:
        rule_w1 = f"读到账单标签 {w1l[0]} + " + "/".join(extra)
    if rule_w1:
        sub, why = _w1_sub(txt)
        out.update(cls="W1", sub=sub, title="账单详情" if zd_hit(txt) else first,
                   reason=rule_w1 + (", " + why if why else ""))
        return out
    # ---- v2: 不是结果页的支付宝页: 转账首页(卡片上有 到账成功 / 转账成功)、交易中断的付款页
    home = sorted({m.group() for m in HOME_RE.finditer(txt)})
    if len(home) >= 2:
        out.update(cls="U", sub="转账首页", title=first, reason=f"{head}, 支付宝转账首页: 有 " + "/".join(home))
        return out
    if "交易中断" in txt:
        out.update(cls="U", sub="交易中断", title=first, reason=f"{head}, 交易中断(付款没成功)")
        return out
    h5 = h5_hit(txt, lat)
    if h5:
        out.update(cls="W3", sub="H5支付成功", title=h5, reason=f"{head}, 没有账单详情卡片, 有 {h5}")
        return out
    for kw, sub in W3_KW:
        if kw in txt:
            out.update(cls="W3", sub=sub, title=kw, reason=f"{head}, 没有账单详情卡片, 有 {kw}")
            return out
    # ---- v2: 微信 零钱提现 详情页(它的 到账成功 / 银行处理中 会被当成白底结果页)
    wp = WX_PAGE.search(txt)
    if wp and not (trade or card == "A" or blue_logo(bgr)):
        return _x(out, f"{head}, 微信 {wp.group()}", first, "微信")
    # ---- 白底原生支付成功页(扫码付款常见): 标题是拼音手写字体读不出来, 但右上 完成、上面的 ¥ 金额、收款方 认得出
    done = any(norm(b[4]) == "完成" for b in items)
    yen = any(YEN.search(b[4]) for b in items if (b[1] + b[3]) / 2 < 0.35 * H)
    payee = re.search(r"收款方(?!.?称|式)", txt)            # ★ 收款方式 是平台收银台的, 收款方金称 是读坏的 收款方全称
    if payee and yen and (done or blue_logo(bgr)):           # 完成 读丢了的, 看标题上那个蓝底白勾
        out.update(cls="W3", sub="支付成功", title=first,
                   reason=f"{head}, 标题没读出来, 有 收款方 + 上面的 ¥ 金额 + " + ("完成" if done else "蓝色对勾"))
        return out
    if "付款方式" in txt and (payee or done):
        # ★ 原样的 付款方式 / 收款方 / 单独一个框的 完成: 错一个字的 付款方式 会认上 收款方式, "付款完成后"里也有 完成
        out.update(cls="W3", sub="支付成功", title=first,
                   reason=f"{head}, 没有账单详情卡片, 标题没读出来, 有 付款方式 + 收款方/完成")
        return out
    code, kw = title_code(txt.replace("成功后", "后"))      # "付款成功后请截图" 是平台页的提示, 不是标题
    if code:
        # v2: 标题(连左边的图标)是微信绿的 -> 微信支付的结果页(census U 里 83 张; 支付宝的是蓝的)
        if title_icon(bgr, items) == "green" and not (trade or card == "A"):
            return _x(out, f"{head}, 标题 {kw} 和左边的图标是微信绿", first, "微信")
        # 只有一个标题: 要有支付宝的线索(支付宝的字 / H5 / 交易号 / 蓝色 支 字图标), 不然微信账单的"当前状态 支付成功"、
        # 别的商城 H5 页也会进来(run2 两张)
        if not (al or trade or blue_logo(bgr)):
            xw = x_words(txt)
            if xw:
                return _x(out, f"只有标题 {kw}, 没有支付宝的线索, 有 " + "/".join(xw[:3]), first)
            # v2: 微信支付结果页的 返回商家 / 摇一摇 在 0.75~0.92(上面 70% 读不到), 读一下这一段
            k = [w for w in WX_LOW if w in rd.band(0.75, 0.92)]
            if k:
                return _x(out, f"{head}, 只有标题 {kw}, 下面有 " + "/".join(k), first, "微信")
            out.update(cls="U", title=kw, reason=f"{head}, 只有标题 {kw}, 没有支付宝的线索")
            return out
        sub = kw if head == "白底" else f"{kw}(页头{head})"
        if code == "B1" and (title_code(tt)[0] != "B1" or "通知ta" in txt.lower() or "liaotian" in lat):
            sub = "聊天转账"         # 聊天里的转账气泡: 转账成功 不在标题区, 后面跟着 马上通知TA
        elif code == "B3" and not yen and len(rd.text) <= 9:
            # v2: H5 收银台的结果页只有 支 字图标 + 支付成功 / 付款成功, 没有金额(浏览器 / App 里的网页, census 36 张记成了 支付成功)
            sub = "H5支付成功"
        out.update(cls="W3", sub=sub, title=kw, reason=f"{head}, 没有账单详情卡片, 标题 {kw}")
        return out
    # ---- v2: 没有标题的收款码图 / 平台扫码收银台(页头被盖住的蓝图那一路现在要上半截蓝 >= 0.30, 低于它的走到这里)
    m = QR_RE.search(txt)
    if m:
        sub = "平台页" if re.search(r"订单金额|复制|上传|取消订单|已付款", txt) else "收款码图"
        return _x(out, f"{head}, 有 {m.group()}(收款码 / 扫码付款页)", first, sub)
    ncjk = sum("一" <= c <= "鿿" for c in txt)
    if card == "O" and not al:
        return _x(out, "左标签右取值但整页白、取值不是 #333, 没有支付宝的字", first)
    if al:
        out.update(cls="U", title=first, reason=f"{head}, 有支付宝的字(" + "/".join(al[:3]) + "), 但不是认得的模板")
        return out
    if not (trade or ar >= 2 or blue_logo(bgr)):        # 账单管理的箭头行也算支付宝的线索
        if wechat_green(small) >= GREEN_MIN:
            return _x(out, f"{head}, 页头有微信绿, 没有支付宝的字", first, "微信")
        if ncjk < 12:
            k = [w for w in WX_LOW if w in rd.band(0.75, 0.92)]
            if k:
                return _x(out, f"{head}, 上面字太少, 下面有 " + "/".join(k), first, "微信")
        if ncjk >= 12:
            return _x(out, f"读到 {ncjk} 个汉字, 没有一个支付宝的字段", first)
    out.update(cls="U", title=first, reason=f"{head}, 字太少({ncjk} 个汉字)" if ncjk < 12
               else f"{head}, 没有支付宝的字段, 但有交易号 / 箭头行 / 支 字图标")
    return out


def _w2_sub(txt: str, tt: str, W: int, H: int) -> str:
    """v2: 按字认出来的转账凭证的子类(census 821 张里: 页面截图 628 / 凭证图 184 / 转账回执 8)。
        转账回执   通知对方已转账 ... XX向你转账 ... 收款人账号(支付宝的回执分享卡片)
        转账凭证图 存下来又缩小 / 裁过的凭证图: 高宽比 1.45~1.66(凭证图 1304x2038 / 1304x2122 的比例; 手机截图 1.77 以上)
        页面截图   转账凭证 页面的截图: 标题区有 转账凭证, 或者下面的 保存图片 / 分享给好友
        其它尺寸   都不是"""
    if re.search(r"通知对方已转账|收款人账号", txt):
        return "转账回执"
    if VOUCHER_RATIO[0] <= H / W <= VOUCHER_RATIO[1]:
        return "转账凭证图"
    if W2_TITLE.search(tt) or re.search(r"保存图片|分享给好友", txt):
        return "页面截图"
    return "其它尺寸"


def _w1_sub(txt: str) -> tuple[str, str]:
    """账单详情的子类。先看标签; 拼音手写字体的标签常读坏, 再看时间标签, 最后按取值的样子猜。
    ★ 时间标签: 商户付款(含收钱码)是 支付时间, 转账是 创建时间。run2 里按"打码的账号"猜成转给个人的 387 张,
      有 31 张读到的是 支付时间(多半是联通交费: 商品说明里"为186****8921交费"的打码手机号)。"""
    k = hits(txt, SUB_BANK)
    if k:
        return "银行卡转账", "有 " + k[0]
    m = re.search(r"收钱码|收码收款|钱码收款", txt)
    if m:
        return "收钱码", "有 " + m.group()
    # v2: 余额充值 / 找人代充 / 余额宝转入 / 退款转账 都是"其它"(census: 15 + 2 + 4 + 6 张分进了 商户消费 / 转给个人);
    #   ★ 充值专营店(游戏充值商家, 约 580 张)是真的商户消费, 正则里没有"充值"两个字单独出现的
    m = re.search(r"充值说明|余额充值|代充值|余额宝.{0,2}转入|退款转账|退款方式", txt)
    if m:
        return "其它", "有 " + m.group()
    k = hits(txt, SUB_PERSON)
    if k:
        return "转给个人", "有 " + k[0]
    k = hits(txt, SUB_SHOP)
    if k:
        return "商户消费", "有 " + k[0]
    m = re.search(r"交费|缴费|充值|话费", txt)
    if m:
        return "商户消费", "有 " + m.group()
    if has(txt, "支付时间") or "付时间" in txt:
        return "商户消费", "有 支付时间(商户付款的标签)"
    if has(txt, "创建时间"):
        return "转给个人", "有 创建时间(转账的标签)"
    m = SHOP_W1.search(txt)
    if m:
        return "商户消费", "按取值猜: 有 " + m.group()
    if re.search(r"\*{2,}\d{2}", txt) or re.search(r"@[0-9a-z]+(com|cn|net)", txt, re.I):
        return "转给个人", "按取值猜: 有打码的账号"
    # v2: 拼音页标签读坏了的, 按订单号的样子猜(census + 拼音库 + 本地已经分好子类的 W1 上量过):
    #   20xxxxxx20004001...  转给个人 435 / 商户消费 12 / 银行卡转账 7 / 收钱码 2
    #   20xxxxxx2[123]00...  商户消费 259 / 收钱码 81 / 转给个人 7;   品明(商品说明读坏的) 商户消费 1,397 / 收钱码 415 / 转给个人 25
    if re.search(r"20\d{6}20004001", txt):
        return "转给个人", "按订单号猜: 20xxxxxx20004001 是转账的"
    if re.search(r"20\d{6}2[123]00\d", txt):
        return "商户消费", "按订单号猜: 20xxxxxx2x00 是商户付款的"
    if "品明" in txt:
        return "商户消费", "有 品明(拼音字体读坏的 商品说明)"
    return "其它", ""


def _init() -> None:
    cv2.setNumThreads(1)


def work(task):
    path, src, use_ocr = task
    t0 = time.perf_counter()
    try:
        r = classify(path, use_ocr)
    except Exception as e:  # noqa: BLE001
        r = {"path": path, "cls": "U", "reason": f"出错 {type(e).__name__}: {str(e)[:80]}"}
    r["name"] = NAMES.get(r["cls"], "")
    r["src"] = src
    r["secs"] = round(time.perf_counter() - t0, 3)
    return r


# ---------------------------------------------------------------- 找图、汇总、联系表

def collect(srcs) -> list[tuple[str, str]]:
    """--src 给的目录(递归)或名单, 返回 [(路径, 来源)]; 来源 = 目录的最后两级 / 名单文件名。"""
    out = []
    for s in srcs:
        p = Path(s)
        if p.is_dir():
            tag = "/".join(p.resolve().parts[-2:])
            for dp, dns, fs in os.walk(p):
                dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(("sheets", "pinyin_sheets"))]
                out += [(os.path.join(dp, f), tag) for f in sorted(fs) if os.path.splitext(f)[1].lower() in EXTS]
        elif p.suffix.lower() == ".txt":
            for ln in p.read_text(encoding="utf-8-sig").splitlines():
                ln = ln.strip().strip('"')
                if ln and os.path.splitext(ln)[1].lower() in EXTS:
                    out.append((ln, p.stem))
        elif p.suffix.lower() == ".csv":
            with p.open(encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    for cand in (r.get("path"), str(p.parent / r["file"]) if r.get("file") else None, r.get("source")):
                        if cand and os.path.exists(cand):
                            out.append((cand, p.stem))
                            break
        elif p.is_file() and p.suffix.lower() in EXTS:
            out.append((str(p), p.parent.name))
        else:
            print(f"  ★ 不认得的 --src: {s}")
    seen, uniq = set(), []
    for path, tag in out:
        k = os.path.normcase(os.path.abspath(path))
        if k not in seen:
            seen.add(k)
            uniq.append((path, tag))
    return uniq


def dedupe_gwcz(todo):
    """同一个 GWCZ 编号(同一张凭证)只留第一张。pinyin-all、New folder\\pinyin_hits、suspect、suspect2 里有同一张图,
    不去重的话各类的数目虚高, 联系表上同一张图出现两次(run2 有 1,755 行重复)。返回 (留下的, 跳过几张)。"""
    seen, keep = set(), []
    for path, tag in todo:
        m = re.search(r"GWCZ\d{10,}", os.path.basename(path))
        if m:
            if m.group() in seen:
                continue
            seen.add(m.group())
        keep.append((path, tag))
    return keep, len(todo) - len(keep)


KEYS = ["path", "W", "H", "cls", "name", "sub", "state", "header", "dark", "flat", "pinyin", "device", "title", "reason", "secs",
        "src", "card", "mode", "ocr"]


def table(rows, rk, ck, rorder=None):
    """交叉表的文字版。"""
    cnt = defaultdict(Counter)
    for r in rows:
        cnt[r.get(rk, "")][r.get(ck, "")] += 1
    cols = sorted({r.get(ck, "") for r in rows}, key=str)
    rws = rorder or sorted(cnt, key=str)
    w = max([7] + [len(str(c)) * 2 + 1 for c in cols])
    tw = max([14] + [len(str(x)) + 2 for x in rws])
    lines = ["  " + f"{'':<{tw}}" + "".join(f"{str(c):>{w}}" for c in cols) + f"{'合计':>8}"]
    for rv in rws:
        if rv not in cnt:
            continue
        lines.append("  " + f"{str(rv):<{tw}}" + "".join(f"{cnt[rv][c]:>{w},}" for c in cols)
                     + f"{sum(cnt[rv].values()):>8,}")
    return lines


def summary(rows, a, wall) -> list[str]:
    n = len(rows)
    order = list(NAMES)
    L = ["=" * 78, f"  整页分类 page_type   {time.strftime('%Y-%m-%d %H:%M')}   OCR {'关' if a.no_ocr else '开'}", "=" * 78]
    secs = np.array([float(r.get("secs") or 0) for r in rows]) if rows else np.zeros(1)
    L.append(f"  图 {n:,} 张   进程 {a.workers}   用时 {wall:.0f} 秒   {n / max(wall, 1e-9):.2f} 张/秒(整批)")
    L.append(f"  按 GWCZ 编号去重: 跳过 {getattr(a, 'dups', 0):,} 张(同一张凭证在几个目录里, 只留第一张)")
    L.append(f"  每张(单进程) 平均 {secs.mean():.3f} 秒  中位 {np.median(secs):.3f}  p90 {np.percentile(secs, 90):.3f}"
             f"  -> 单核约 {1 / max(secs.mean(), 1e-9):.1f} 张/秒")
    L.append(f"  出错 {sum(1 for r in rows if str(r.get('reason', '')).startswith('出错')):,}   "
             f"读不出 {sum(1 for r in rows if r.get('reason') == '读不出'):,}")
    L.append("")
    L.append("  ---- 各类 ----")
    c = Counter(r["cls"] for r in rows)
    for k in order:
        if not c[k]:
            continue
        sub = Counter(r.get("sub", "") for r in rows if r["cls"] == k)
        L.append(f"  {k:<3}{NAMES[k]:<8}{c[k]:>7,}  {c[k] / max(1, n):>6.1%}   "
                 + "  ".join(f"{s or '-'} {m:,}" for s, m in sub.most_common(6)))
    L.append("")
    L.append("  ---- 异常页(state, status_icon 只看图: 钱没到 / 退回 / 还在处理) x 类别 ----")
    ab = [r for r in rows if r.get("state")]
    L += table(ab, "state", "cls") if ab else ["  (没有)"]
    L.append("")
    L.append("  ---- 各类 x 来源 ----")
    L += table(rows, "src", "cls")
    L.append("")
    L.append("  ---- 页头颜色 x 来源(蓝图库应全是蓝底, 白图库应全是白底) ----")
    L += table(rows, "src", "header")
    for r in rows:
        k = r["cls"]
        r["_g"] = "蓝图B" if k.startswith("B") else ("白图W" if k.startswith("W") else k)
    L.append("")
    L.append("  ---- 大类 x 来源 ----")
    L += table(rows, "src", "_g")
    L.append("")
    L.append("  ---- 各类: 拼音标记 / 设备 / 每张秒数 ----")
    for k in order:
        rs = [r for r in rows if r["cls"] == k]
        if not rs:
            continue
        py = Counter(str(r.get("pinyin", "")) for r in rs)
        dv = Counter(r.get("device", "") for r in rs)
        s = np.array([float(r.get("secs") or 0) for r in rs])
        L.append(f"  {k:<3} 拼音 1:{py['1']:,} 0:{py['0']:,} 不判:{py['']:,}   "
                 + " ".join(f"{d}:{m:,}" for d, m in dv.most_common()) + f"   平均 {s.mean():.2f} 秒")
    L.append("")
    L.append("  ---- 各类最常见的理由(数字换成 #) ----")
    for k in order:
        rs = [r for r in rows if r["cls"] == k]
        if not rs:
            continue
        L.append(f"  {k} {NAMES[k]}")
        for why, m in Counter(re.sub(r"\d+(\.\d+)?", "#", str(r.get("reason", ""))) for r in rs).most_common(8):
            L.append(f"      {m:>6,}  {why[:90]}")
    L.append("=" * 78)
    return L


def _tile(r, tw=270, th=600):
    from PIL import Image, ImageDraw
    from make_check_sheet import _cjk_font
    font = _cjk_font(15)
    bgr = decode(r["path"])
    if bgr is None:
        bgr = np.full((th, tw, 3), 200, np.uint8)
    h, w = bgr.shape[:2]
    t = cv2.resize(bgr, (tw, max(1, int(h * tw / w))), interpolation=cv2.INTER_AREA)[:th]
    page = np.full((th, tw, 3), 255, np.uint8)
    page[:t.shape[0]] = t
    bar = np.full((76, tw, 3), 255, np.uint8)
    pil = Image.fromarray(cv2.cvtColor(np.vstack([bar, page]), cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    if font:
        old = f"{r['_old']} -> " if r.get("_old") else ""          # v2: --compare 的 diff 联系表写上一版的类别
        d.text((3, 1), f"{r['_i']}. {old}{r['cls']} {r.get('sub', '')}  {r.get('W', '')}x{r.get('H', '')}"
                       f"{'  拼音' if str(r.get('pinyin')) == '1' else ''}", font=font, fill=(190, 0, 0))
        d.text((3, 19), f"标题 {r.get('title', '')}  页头 {r.get('header', '')}", font=font, fill=(0, 0, 160))
        d.text((3, 37), str(r.get("reason", ""))[:30], font=font, fill=(60, 60, 60))
        d.text((3, 55), os.path.basename(r["path"])[-34:], font=font, fill=(110, 110, 110))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def make_sheet(job):
    """一张联系表: 12 格(4 列 x 3 行), 每格是缩小的整页 + 上面四行字(类别子类 / 标题 / 理由 / 文件名)。
    ★ 字一定画进图里(make_check_sheet 记的坑: 服务器控制台按 GBK 显示, 贴回来全是乱码)。"""
    rows, dst = job
    tiles = []
    for r in rows:
        try:
            tiles.append(_tile(r))
        except Exception:  # noqa: BLE001
            tiles.append(np.full((676, 270, 3), 200, np.uint8))
    tiles = [cv2.copyMakeBorder(t, 0, 4, 0, 4, cv2.BORDER_CONSTANT, value=(120, 120, 120)) for t in tiles]
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imencode(".png", grid)[1].tofile(dst)
    with open(dst[:-4] + ".txt", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"{r['_i']}\t{r['cls']}\t{r.get('sub', '')}\t{r.get('title', '')}\t{r.get('reason', '')}\t{r['path']}\n")
    return dst


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="整页分类(第二版, 规则 + OCR 关键词)")
    ap.add_argument("--src", nargs="+", required=True, help="目录(递归)或名单(.txt 一行一个路径 / .csv)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=0, help="随机抽这么多张(0 = 全部)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--sheets", type=int, default=2, help="每类最多出几张联系表(每张 12 格), 0 = 不出")
    ap.add_argument("--no-ocr", action="store_true", help="只跑规则, 不做 OCR")
    ap.add_argument("--compare", type=Path, default=None,
                    help="上一版的 page_type.csv: 类别 / 子类变了的写进 diff.csv, 再出 diff 联系表(只看变了的, 不用拷图)")
    ap.add_argument("--diff-sheets", type=int, default=30, help="diff 联系表最多几张(每张 12 格; 每种变法最多 2 张)")
    a = ap.parse_args()

    todo = collect(a.src)
    todo, a.dups = dedupe_gwcz(todo)
    if a.n and a.n < len(todo):
        random.Random(a.seed).shuffle(todo)          # ★ 必须随机抽, 文件名带时间戳, 取前 N 张只会取到一天
        todo = todo[:a.n]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"  图 {len(todo):,} 张   来源 {len({t for _p, t in todo})} 个   进程 {a.workers}   "
          f"OCR {'关' if a.no_ocr else '开'}   按 GWCZ 编号去重跳过 {a.dups:,} 张", flush=True)
    t0 = time.time()
    rows = []
    step = max(200, len(todo) // 20)
    with ProcessPoolExecutor(a.workers, initializer=_init) as pool:
        for i, r in enumerate(pool.map(work, [(p, t, not a.no_ocr) for p, t in todo], chunksize=4), 1):
            rows.append(r)
            if i % step == 0 or i == len(todo):
                el = time.time() - t0
                print(f"    {i:,}/{len(todo):,}   {i / max(el, 1e-9):.1f} 张/秒   "
                      + " ".join(f"{k}:{v}" for k, v in sorted(Counter(x['cls'] for x in rows).items())), flush=True)
        wall = time.time() - t0
        tmp = a.out / "page_type.csv.tmp"
        with tmp.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, a.out / "page_type.csv")
        lines = summary(rows, a, wall)
        (a.out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
        if a.sheets:
            sd = a.out / "sheets"
            sd.mkdir(exist_ok=True)
            jobs = []
            for k in NAMES:
                rs = [r for r in rows if r["cls"] == k]
                random.Random(a.seed).shuffle(rs)
                rs = rs[:12 * a.sheets]
                for j in range(0, len(rs), 12):
                    part = [dict(r, _i=i) for i, r in enumerate(rs[j:j + 12], 1)]
                    jobs.append((part, str(sd / f"{k}_{j // 12 + 1:02d}.png")))
            for _dst in pool.map(make_sheet, jobs):
                pass
            print(f"  联系表 {len(jobs)} 张 -> {sd}", flush=True)
        if a.compare:
            compare(rows, a, pool)
    print(f"  -> {a.out / 'page_type.csv'}", flush=True)


def compare(rows, a, pool) -> None:
    """v2: 和上一版的 page_type.csv 逐张比(按路径), 类别或子类变了的写 diff.csv、在 summary.txt 后面加一张变法表、
    出 diff 联系表(每种变法最多 24 格, 一共最多 --diff-sheets 张)。人眼只核变了的, 不用把图拷回来。"""
    with a.compare.open(encoding="utf-8-sig", newline="") as f:
        old = {r["path"]: r for r in csv.DictReader(f)}
    ch = []
    for r in rows:
        o = old.get(r["path"])
        if o and (o.get("cls", ""), o.get("sub", "")) != (r.get("cls", ""), r.get("sub", "")):
            ch.append(dict(r, _old=f"{o.get('cls', '')}/{o.get('sub', '')}", _new=f"{r.get('cls', '')}/{r.get('sub', '')}"))
    keys = ["path", "_old", "_new", "state", "title", "reason", "ocr"]
    with (a.out / "diff.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(ch)
    cnt = Counter((r["_old"], r["_new"]) for r in ch)
    lines = ["", f"  ---- 和上一版比({a.compare.name}): 比得上的 {sum(1 for r in rows if r['path'] in old):,} 张, "
                 f"类别或子类变了 {len(ch):,} 张 ----"]
    lines += [f"  {n:>6,}  {o:<22} -> {nw}" for (o, nw), n in cnt.most_common()]
    with (a.out / "summary.txt").open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    if not a.diff_sheets or not ch:
        return
    dd = a.out / "diff_sheets"
    dd.mkdir(exist_ok=True)
    pick = []
    for (o, nw), _n in cnt.most_common():
        rs = [r for r in ch if (r["_old"], r["_new"]) == (o, nw)]
        random.Random(a.seed).shuffle(rs)
        pick += rs[:24]
    pick = pick[:12 * a.diff_sheets]
    jobs = []
    for j in range(0, len(pick), 12):
        part = [dict(r, _i=j + i) for i, r in enumerate(pick[j:j + 12], 1)]
        jobs.append((part, str(dd / f"diff_{j // 12 + 1:02d}.png")))
    for _dst in pool.map(make_sheet, jobs):
        pass
    print(f"  diff 联系表 {len(jobs)} 张 -> {dd}", flush=True)


if __name__ == "__main__":
    main()
