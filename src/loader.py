# -*- coding: utf-8 -*-
"""
loader.py  ——  把 58 个 JSON 分片读成一张干净的表格

输入：data/json/poet.tang.*.json
输出：data/poems.csv  （5.76 万行 × 9 列）

处理链条：
    JSON 分片 ──► 拼接 ──► 繁转简 ──► 去重/去空 ──► 算句数/字数/言/诗体 ──► CSV
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
# 原始分片放在 data/json/ 子目录里，与产物 poems.csv 分开
JSON_DIR = os.path.join(DATA_DIR, "json")

# 句读符号：古诗用全角标点断句，这里把常见几种都算上
PUNCT = "，。！？；：、,.!?;:"  


# ----------------------------------------------------------------------------
# 1. 读原始数据
# ----------------------------------------------------------------------------
def load_raw():
    """读取所有分片，返回记录列表。"""
    # 分片优先从 data/json/ 找；同时兼容直接放在 data/ 下的老布局
    patterns = [
        os.path.join(JSON_DIR, "poet.tang.*.json"),
        os.path.join(DATA_DIR, "poet.tang.*.json"),
    ]
    files = sorted({fp for p in patterns for fp in glob.glob(p)})
    if not files:
        raise FileNotFoundError(
            "没有找到分片文件，请把 poet.tang.*.json 放进 %s" % JSON_DIR
        )

    records = []
    for fp in files:
        # encoding 必须显式写 utf-8：Windows 默认 GBK，读中文 JSON 会直接崩
        with open(fp, "r", encoding="utf-8") as f:
            records.extend(json.load(f))
    print("读取分片 %d 个，原始记录 %d 条" % (len(files), len(records)))
    return records


# ----------------------------------------------------------------------------
# 2. 繁转简
# ----------------------------------------------------------------------------
def make_converter():
    """
    返回一个繁转简函数。opencc 是可选依赖：
      装了就转简体，没装就原样返回（并给出提示）。
    这样即使别人电脑上没有 opencc，程序也不会直接崩。
    """
    try:
        import opencc
        cc = opencc.OpenCC("t2s")
        print("已加载 OpenCC：繁体 → 简体")
        return cc.convert
    except Exception:
        print("⚠ 未安装 opencc，保留繁体原文（意象词表已同时收录繁简写法，不影响统计）")

        def _noop(s):
            return s

        return _noop


# ----------------------------------------------------------------------------
# 3. 拼装 DataFrame
# ----------------------------------------------------------------------------
def to_frame(records, convert):
    rows = []
    for r in records:
        paragraphs = r.get("paragraphs") or []
        text = "".join(paragraphs).strip()
        if not text:
            continue
        rows.append({
            "作者": convert((r.get("author") or "").strip()),
            "标题": convert((r.get("title") or "").strip()),
            "正文": convert(text),
            "正文繁": text,          # 保留繁体原文：意象统计要用（见 features.py 说明）
            "编号": r.get("id", ""),
        })

    df = pd.DataFrame(rows)
    print("拼装完成：%d 行" % len(df))
    return df


# ----------------------------------------------------------------------------
# 4. 清洗
# ----------------------------------------------------------------------------
# 原始数据里“不知道作者是谁”有十种写法，如果不合并，
# 同一种东西会被拆成好几份分别参与排名（“不详”和“无名氏”会各占一个榜位）。
ANON = {
    "不详", "无名氏", "佚名", "缺名",
    "吴越失姓名人", "无名僧", "无名鬼", "无名释",
    "南唐失名僧", "无名女鬼",
}


def clean(df):
    n0 = len(df)
    df = df[df["作者"] != ""]                 # 作者为空
    df = df[df["作者"].str.len() <= 8]        # “作者”字段过长的一般是误录
    df = df.drop_duplicates(subset=["编号"])  # 同一首诗可能有多个副本
    df = df.drop_duplicates(subset=["作者", "标题", "正文"])
    df = df.reset_index(drop=True)

    # 佚名归一：十种写法合并成一个“佚名”
    is_anon = df["作者"].isin(ANON)
    n_anon = int(is_anon.sum())
    df["作者"] = df["作者"].where(~is_anon, "佚名")
    # 单独留一列标记，后面按“人”统计时一眼就能过滤掉
    df["是否佚名"] = df["作者"] == "佚名"

    print("清洗：%d → %d 行（去掉重复/空作者 %d 行）" % (n0, len(df), n0 - len(df)))
    print("佚名归一：%d 首（原写法 %d 种）" % (n_anon, len(ANON)))
    return df


# ----------------------------------------------------------------------------
# 5. 形态特征：句数、字数、言、诗体
# ----------------------------------------------------------------------------
def count_lines(text):
    """按标点断句，统计句子数。"""
    parts = re.split("[%s]" % re.escape(PUNCT), text)
    return len([p for p in parts if p.strip()])


def count_chars(text):
    """去掉标点后的纯字数。"""
    return len(re.sub("[%s]" % re.escape(PUNCT), "", text))


def add_shape(df):
    df = df.copy()
    df["句数"] = df["正文"].map(count_lines)
    df["字数"] = df["正文"].map(count_chars)

    # 每句几个字：5 言 / 7 言 / 其他。句数为 0 时会产生除零，先屏蔽
    df["言"] = np.where(df["句数"] > 0, df["字数"] / df["句数"].replace(0, np.nan), np.nan)
    df["言"] = df["言"].round()

    # 诗体判定：4 句 = 绝句，8 句 = 律诗，其余归为古体/杂言
    # np.select 是 numpy 里的“向量化 if-elif”，比逐行判断快几十倍，也可读
    df["诗体"] = np.select(
        [df["句数"] == 4, df["句数"] == 8],
        ["绝句", "律诗"],
        default="古体/杂言",
    )
    return df


# ----------------------------------------------------------------------------
# 6. 主流程
# ----------------------------------------------------------------------------
def build(save=True):
    convert = make_converter()
    df = to_frame(load_raw(), convert)
    df = clean(df)
    df = add_shape(df)

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        out = os.path.join(DATA_DIR, "poems.csv")
        # 正文繁 是中间变量，不写进 CSV（否则体积翻倍，人看也乱）
        df.drop(columns=["正文繁"]).to_csv(out, index=False, encoding="utf-8-sig")
        print("已保存：%s  (%.1f MB)" % (out, os.path.getsize(out) / 1024 / 1024))

    return df


if __name__ == "__main__":
    d = build()
    print("\n===== 数据概览 =====")
    print("诗作总数：%d" % len(d))
    print("诗人总数：%d（不含佚名）" % d.loc[~d["是否佚名"], "作者"].nunique())
    print("\n存诗量前 10：")
    print(d["作者"].value_counts().head(10).to_string())
    print("\n诗体分布：")
    print(d["诗体"].value_counts().to_string())
    print("\n正文长度：中位数 %d 字，最长 %d 字" % (d["字数"].median(), d["字数"].max()))

