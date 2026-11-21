# -*- coding: utf-8 -*-
"""
network.py  ——  M4 赠答网络：从诗题里抽出“这首诗是写给谁的”

【难点】
诗题长这样：赠李白、寄李十二白、酬乐天、送元二使安西、别董大、和張僕射塞下曲。
朴素正则 `^(赠|寄|酬|送)(.{2,4})$` 会误抓：“送友人”里的“友人”不是人名，
“和张仆射塞下曲”里真正的对象是“张仆射”，后面“塞下曲”是曲名。

【做法】最长前缀匹配 + 诗人词表校验
  ① 先拿“全唐诗里出现过的所有作者名”当词表（这是最强的约束：能赠诗的对象，
     大概率本身也是收录在册的诗人）；
  ② 去掉开头的赠答动词，剩下的部分从长到短逐段截取，第一个命中词表的就是答案。
  这样“张仆射塞下曲”会先试“张仆射塞下曲”（不在词表），再试“张仆射”（命中）。
"""

import pandas as pd

# 唐诗里表示“人际投赠”的动词。排在前面的优先匹配。
VERBS = ["赠", "寄", "酬", "答", "送", "别", "和", "怀", "忆", "呈", "上", "简", "悼", "哭"]


def build_poet_set(df, min_poems=2):
    """
    词表：只保留存诗 ≥ min_poems 的作者。
    存诗 1 首的人重名概率高，容易误匹配，先剔掉。
    """
    cnt = df["作者"].value_counts()
    # 佚名不是人，进词表只会在网络里凭空造出一个巨大节点
    if "是否佚名" in df.columns:
        cnt = cnt.drop(labels=["佚名"], errors="ignore")
    return set(cnt[cnt >= min_poems].index)


def extract_target(title, poet_set):
    """
    从一条诗题里抽“赠答对象”。抽不到返回 None。

    参数都是简体（loader 里已经统一转过），所以动词表也用简体。
    """
    if not isinstance(title, str):
        return None
    for verb in VERBS:
        if not title.startswith(verb):
            continue
        rest = title[len(verb):]
        # 从长到短试，最长匹配优先；至少 2 个字才算人名
        for L in range(min(len(rest), 6), 1, -1):
            if rest[:L] in poet_set:
                return rest[:L]
        return None          # 这个动词后面接的不是诗人名
    return None


def build_edges(df, min_poems=2):
    """返回 (边表, 覆盖率统计)。边表每行 = 谁给谁写了多少首。"""
    poet_set = build_poet_set(df, min_poems)
    targets = df["标题"].map(lambda t: extract_target(t, poet_set))

    hit = df[targets.notna()].copy()
    hit["目标"] = targets[targets.notna()]
    hit = hit[hit["作者"] != hit["目标"]]          # 去掉自己赠自己

    edges = (hit.groupby(["作者", "目标"]).size()
             .reset_index(name="次数")
             .sort_values("次数", ascending=False)
             .reset_index(drop=True))

    coverage = pd.Series({
        "诗作总数": len(df),
        "识别出赠答对象的诗": len(hit),
        "识别覆盖率%": round(len(hit) / len(df) * 100, 1),
        "涉及诗人": hit["作者"].nunique(),
        "关系条数(去重)": len(edges),
    })
    return edges, coverage


def build_graph(edges):
    """用 networkx 建有向图。权重 = 赠诗次数。"""
    import networkx as nx

    G = nx.DiGraph()
    for r in edges.itertuples(index=False):
        G.add_edge(r.作者, r.目标, weight=int(r.次数))
    return G


def hub_scores(G, top_n=15):
    """
    节点重要性：加权度中心性（收 + 发 的总诗数）。
    在有向图里用 degree(weight=) 会把出入度合并，这里另外分开统计。
    """
    rows = []
    for n in G.nodes():
        out_w = sum(d["weight"] for _, _, d in G.out_edges(n, data=True))
        in_w = sum(d["weight"] for _, _, d in G.in_edges(n, data=True))
        rows.append({"诗人": n, "赠出": out_w, "收到": in_w, "加权度": out_w + in_w,
                     "入度": G.in_degree(n), "出度": G.out_degree(n)})
    tab = pd.DataFrame(rows).sort_values("加权度", ascending=False).head(top_n)

    # 单向关系：A 赠 B，但 B 没回赠
    one_way = sum(1 for u, v in G.edges() if not G.has_edge(v, u))
    stats = pd.Series({
        "节点数(诗人)": G.number_of_nodes(),
        "边数(有向)": G.number_of_edges(),
        "单向关系占比%": round(one_way / G.number_of_edges() * 100, 1),
    })
    return tab.reset_index(drop=True), stats

