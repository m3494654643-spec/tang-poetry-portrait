# -*- coding: utf-8 -*-
"""
analysis.py  ——  六个分析模块，每个函数返回一张表，供 visualize.py 画图

M1 基础画像：产量、长尾分布、诗体构成
M2 意象分析：意象频次、意象共现矩阵
M3 诗人特征词：按诗人聚合后做字符级 TF-IDF
M4 赠答网络：见 network.py
M5 季节意象：四季词频与密度
"""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from features import IMAGERY, SEASON

# 只保留汉字：标点、空格、序号（“其一”“其二”）都不该进 TF-IDF
# 否则特征词榜会被“，”“。”霸占——第一次跑真实数据时就踩了这个坑
_HAN = re.compile(r"[\u4e00-\u9fa5]")


def _only_han(text):
    """去掉一切非汉字字符，剩下的才参与 TF-IDF。"""
    return "".join(_HAN.findall(text))


# 虚词表：这些字人人都用，体现不出"个人风格"，取特征词前必须屏蔽
# 否则榜单会被 不、人、一、无、何 这种字霸占（第一次跑真实数据时就是这样）
STOP_CHARS = set(
    "不人一无有何日为君来我之其如见上下中时长得亦与而则且又已未可欲能被所于也矣乎哉兮莫"
    "一二三四五六七八九十百千万"
)


def _top_chars(row, words, top_n):
    """
    按 TF-IDF 权重取前 top_n 个字，跳过虚词。
    做法：把虚词的权重临时压成 -1，再取最大值的下标。
    """
    masked = np.where(np.isin(words, list(STOP_CHARS)), -1.0, row)
    return np.argsort(masked)[-top_n:][::-1], masked


def _drop_anon(df):
    """
    按“人”统计时排除佚名。
    “佚名”是上千首无名诗凑成的一个集合，不是一位诗人；
    把它当诗人排名，它会挤进第二名，榜单就失真了。
    """
    if "是否佚名" not in df.columns:
        return df
    return df[~df["是否佚名"]]


# ---------------------------------------------------------------------------
# M1 基础画像
# ---------------------------------------------------------------------------
def overview(df):
    """给出一组总体数字，直接写进报告正文。"""
    poets = _drop_anon(df)
    stats = {
        "诗作总数": len(df),
        "诗人数": poets["作者"].nunique(),
        "佚名诗数": int(df["是否佚名"].sum()) if "是否佚名" in df.columns else 0,
        "标题数": df["标题"].nunique(),
        "平均字数": round(df["字数"].mean(), 1),
        "中位字数": int(df["字数"].median()),
        "最长诗字数": int(df["字数"].max()),
        "平均句数": round(df["句数"].mean(), 1),
    }
    return pd.Series(stats, name="数值")


def top_authors(df, n=20):
    """存诗量前 n 名（不含佚名）。"""
    s = _drop_anon(df)["作者"].value_counts().head(n)
    return s.rename_axis("诗人").reset_index(name="诗作数")


def author_counts(df):
    """
    每位诗人的存诗数，从多到少。
    图 2 画的是排名-数量曲线，需要的是逐个诗人的原始计数，而不是分桶后的表。
    """
    return _drop_anon(df)["作者"].value_counts()


def author_distribution(df, bins=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 10 ** 9)):
    """
    存诗量分布：把诗人按“存诗几首”分桶，看长尾有多长。
    每一档的上界都是前一档的两倍（1、2、4、8……），这样各档在对数尺度上等宽，
    人数才可以直接比较；若用 1~10、11~100 这种等距分桶，宽档会被人为撑大。
    这是本项目最能说明问题的一张表——全唐诗远不是均匀语料库。
    """
    cnt = _drop_anon(df)["作者"].value_counts()
    labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1] - 1
        if hi >= 10 ** 9 - 1:
            labels.append("≥%d 首" % lo)
        elif lo == hi:
            labels.append("%d 首" % lo)
        else:
            labels.append("%d~%d 首" % (lo, hi))
    bucket = pd.cut(cnt, bins=bins, right=False, labels=labels)
    tab = bucket.value_counts().reindex(labels).rename_axis("存诗量").reset_index(name="诗人数")
    tab["占比%"] = (tab["诗人数"] / len(cnt) * 100).round(1)
    return tab


def genre_dist(df):
    """诗体构成。"""
    s = df["诗体"].value_counts()
    tab = s.rename_axis("诗体").reset_index(name="诗作数")
    tab["占比%"] = (tab["诗作数"] / len(df) * 100).round(1)
    return tab


def line_length_dist(df):
    """每句字数的分布：五言 / 七言 / 其他。"""
    s = df["言"].dropna().astype(int).value_counts().head(6)
    return s.rename_axis("每句字数").reset_index(name="诗作数")


# ---------------------------------------------------------------------------
# M2 意象分析
# ---------------------------------------------------------------------------
def imagery_counts(df):
    """每个意象的总出现次数，以及“含有该意象的诗”占比。"""
    rows = []
    for name in IMAGERY:
        hits = (df[name] > 0).sum()
        rows.append({
            "意象": name,
            "出现次数": int(df[name].sum()),
            "含此意象的诗": int(hits),
            "覆盖率%": round(hits / len(df) * 100, 1),
        })
    return pd.DataFrame(rows).sort_values("出现次数", ascending=False).reset_index(drop=True)


def imagery_cooccurrence(df):
    """
    意象共现矩阵。

    把 10 列意象转成 (N首诗 × 10意象) 的 0/1 矩阵 M，
    那么 M.T @ M 的第 (i, j) 个元素 = 同时出现意象 i 和 j 的诗有几首。
    一行矩阵乘法搞定 10×10 的共现表，比双重 for 循环快几百倍。
    """
    names = list(IMAGERY)
    M = (df[names].values > 0).astype(np.int32)     # (N, 10)
    co = M.T @ M                                   # (10, 10)
    np.fill_diagonal(co, 0)                        # 自己和自己不算共现
    return pd.DataFrame(co, index=names, columns=names)


def imagery_by_poet(df, poets, topk=5):
    """指定几位诗人最偏爱的意象（按出现次数排）。"""
    rows = []
    for p in poets:
        sub = df[df["作者"] == p]
        if len(sub) == 0:
            continue
        cnt = sub[list(IMAGERY)].sum().sort_values(ascending=False).head(topk)
        rows.append({"诗人": p, "存诗数": len(sub),
                     "偏爱意象": "、".join(cnt.index)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# M3 诗人特征词（字符级 TF-IDF）
# ---------------------------------------------------------------------------
def poet_tfidf(df, top_n=8, min_chars=3000):
    """
    把“一位诗人的全部作品”拼成一篇文档，用字符级 TF-IDF 找他的特征用字。

    为什么用 analyzer='char'（按单字切）：
      古诗不分词，单字就是最小的语义单位；绕开分词器，反而更准。

    TF-IDF 的直觉：
      TF  = 这个字在该诗人作品里出现得多不多
      IDF = 这个字在全唐诗里稀不稀罕（“之”“不”这种到处都有的字权重被压低）
      两者相乘，剩下的就是“这位诗人爱用、别人少用”的字。
    """
    docs = _drop_anon(df).groupby("作者")["正文"].apply(lambda s: _only_han("".join(s)))
    docs = docs[docs.str.len() >= min_chars]
    if len(docs) < 2:
        return pd.DataFrame()

    vec = TfidfVectorizer(analyzer="char", max_features=3000, min_df=1)
    X = vec.fit_transform(docs)                    # (诗人, 字) 稀疏矩阵
    words = np.array(vec.get_feature_names_out())
    order = docs.str.len().sort_values(ascending=False).index[:15]

    rows = []
    for poet in order:
        i = docs.index.get_loc(poet)
        row = X[i].toarray().ravel()
        top, _ = _top_chars(row, words, top_n)
        rows.append({
            "诗人": poet,
            "字数": len(docs[poet]),
            "特征词": "、".join(words[top]),
        })
    return pd.DataFrame(rows)


def poet_word_compare(df, poets, top_n=8):
    """指定几位诗人各自的 TF-IDF 前列词，用于画对比图。"""
    docs = _drop_anon(df).groupby("作者")["正文"].apply(lambda s: _only_han("".join(s)))
    docs = docs[docs.str.len() >= 3000]
    vec = TfidfVectorizer(analyzer="char", max_features=3000)
    X = vec.fit_transform(docs)
    words = np.array(vec.get_feature_names_out())

    out = {}
    for p in poets:
        if p not in docs.index:
            continue
        i = docs.index.get_loc(p)
        row = X[i].toarray().ravel()
        top, _ = _top_chars(row, words, top_n)
        out[p] = [(words[j], float(row[j])) for j in top]
    return out


# ---------------------------------------------------------------------------
# M5 季节意象
# ---------------------------------------------------------------------------
def season_stats(df):
    """四季词频总计。"""
    cols = ["季_" + s for s in SEASON]
    tab = pd.DataFrame({
        "季节": list(SEASON),
        "出现次数": [int(df[c].sum()) for c in cols],
        "含有该季词的诗": [int((df[c] > 0).sum()) for c in cols],
    })
    tab["占比%"] = (tab["出现次数"] / tab["出现次数"].sum() * 100).round(1)
    return tab


def season_density_by_poet(df, min_poems=300, top_n=10):
    """
    诗人的“秋意象密度”：每千字出现多少次秋相关词。
    用密度而不是绝对次数，才能公平比较存诗量不同的诗人。
    """
    g = _drop_anon(df).groupby("作者")
    agg = pd.DataFrame({
        "诗数": g.size(),
        "总字数": g["字数"].sum(),
        "秋词": g["季_秋"].sum(),
    })
    agg = agg[agg["诗数"] >= min_poems]
    agg["秋密度每千字"] = (agg["秋词"] / agg["总字数"] * 1000).round(2)
    return agg.sort_values("秋密度每千字", ascending=False).head(top_n).reset_index()

