# -*- coding: utf-8 -*-
"""
visualize.py  ——  画图。每张图一个函数，统一风格，统一存到 output/

踩过的坑：matplotlib 默认字体不含中文字形，中文会显示成一个个方框（□□□）。
必须在画图前显式指定中文字体，并把负号设成正号形式，否则坐标轴上的负号也会变方块。
"""

import os

import matplotlib
matplotlib.use("Agg")            # 不弹窗，直接存文件（在 PyCharm 里跑不会卡住）
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import networkx as nx



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "output")

# 统一配色：取一组低饱和的颜色，比默认色板耐看
C_MAIN = "#4C72B0"
C_ACCENT = "#DD8452"
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def setup():
    """字体 + 全局样式。所有画图函数都要先调这个。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["axes.edgecolor"] = "#888888"
    os.makedirs(OUT_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print("  已保存 %s" % name)
    return path


# ---------------------------------------------------------------------------
# 图 1：诗人存诗量 Top20
# ---------------------------------------------------------------------------
def fig_top_authors(tab, name="fig1_诗人存诗量Top20.png"):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    y = np.arange(len(tab))[::-1]
    ax.barh(y, tab["诗作数"], color=C_MAIN, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(tab["诗人"])
    for yi, v in zip(y, tab["诗作数"]):
        ax.text(v + max(tab["诗作数"]) * 0.01, yi, str(v), va="center", fontsize=9)
    ax.set_xlabel("存诗数量（首）")
    ax.set_title("图1  全唐诗存诗量前 20 位诗人", fontsize=14, pad=12)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(tab["诗作数"]) * 1.12)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 图 2：存诗量长尾分布
# ---------------------------------------------------------------------------
def fig_long_tail(counts, name="fig2_存诗量长尾分布.png"):
    """
    图 2：长尾分布（对数等宽分桶的柱状图 + 累计占比曲线）。
    分桶按 1、2、4、8、16……逐档翻倍，每一档在对数尺度上宽度相等，
    所以柱高可以直接横向比较，不会像等距分桶那样把窄档压扁、宽档撑大。
    """
    val = counts.values
    # 上界取“最大值以上再翻一档”，避免数据变多时把最长的尾巴漏掉
    top = 512
    while top <= val.max():
        top *= 2
    edges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, top]
    labels = ["1 首", "2~3 首", "4~7 首", "8~15 首", "16~31 首", "32~63 首",
              "64~127 首", "128~255 首", "256~511 首", "512 首以上"]

    n_poets, _ = np.histogram(val, bins=edges)
    poems, _ = np.histogram(val, bins=edges, weights=val)
    cum = np.cumsum(poems) / poems.sum() * 100      # 累计诗作占比

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    x = np.arange(len(n_poets))
    bars = ax.bar(x, n_poets, color=C_ACCENT, width=0.66, label="该档诗人数")

    ax.set_yscale("log")                            # 人数跨三个数量级，必须用对数轴
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("诗人数量（对数坐标）")
    ax.set_xlabel("个人存诗量（每档宽度翻倍）")
    ax.set_title("图2  唐代诗人存诗量的长尾分布", fontsize=14, pad=12)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)

    for b, v in zip(bars, n_poets):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.18, str(v),
                ha="center", fontsize=9, color="#333333")

    # 右轴：累计诗作占比，看"少数人写了多少"
    ax2 = ax.twinx()
    ax2.plot(x, cum, "o-", color="#8B2E1F", lw=1.8, ms=4.5, label="累计诗作占比")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("累计诗作占比（%）")
    for xi, ci in zip(x, cum):
        if ci < 95:                                  # 末点 100% 会压到图例上，略去
            ax2.annotate("%.0f%%" % ci, xy=(xi, ci), xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize=8,
                         color="#8B2E1F")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper center", ncol=2, frameon=False)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 图 3：诗体构成
# ---------------------------------------------------------------------------
def fig_genre(tab, name="fig3_诗体构成.png"):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = PALETTE[:len(tab)]
    wedges, texts, autotexts = ax.pie(
        tab["诗作数"], labels=tab["诗体"], autopct="%1.1f%%",
        colors=colors, startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("图3  全唐诗诗体构成", fontsize=14, pad=12)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 图 4：意象频次
# ---------------------------------------------------------------------------
def fig_imagery(tab, name="fig4_意象频次.png"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(tab))
    bars = ax.bar(x, tab["出现次数"], color=PALETTE[:len(tab)], width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(tab["意象"], fontsize=13)
    ax.set_ylabel("出现总次数")
    ax.set_title("图4  唐诗十大意象出现频次", fontsize=14, pad=12)
    for b, v, r in zip(bars, tab["出现次数"], tab["覆盖率%"]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.01,
                "%d\n(%.0f%%)" % (v, r), ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(tab["出现次数"]) * 1.18)
    ax.grid(axis="x", visible=False)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 图 5：意象共现热力图
# ---------------------------------------------------------------------------
def fig_cooccurrence(co, name="fig5_意象共现热力图.png"):
    m = co.values.astype(float)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(m, cmap="YlOrRd")
    ax.set_xticks(np.arange(len(co)))
    ax.set_yticks(np.arange(len(co)))
    ax.set_xticklabels(co.columns, fontsize=12)
    ax.set_yticklabels(co.index, fontsize=12)
    ax.grid(False)

    vmax = m.max()
    for i in range(len(co)):
        for j in range(len(co)):
            if i == j:
                continue
            ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if m[i, j] > vmax * 0.6 else "#333333")
    ax.set_title("图5  意象共现热力图（同首诗中出现次数）", fontsize=14, pad=12)
    fig.colorbar(im, ax=ax, shrink=0.82, label="共现次数")
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 图 6：诗人特征词对比（TF-IDF）
# ---------------------------------------------------------------------------
def fig_tfidf(cmp, name="fig6_诗人特征词.png"):
    poets = list(cmp)
    fig, axes = plt.subplots(1, len(poets), figsize=(3.4 * len(poets), 5.2),
                             sharey=False)
    if len(poets) == 1:
        axes = [axes]
    for ax, p in zip(axes, poets):
        words = [w for w, _ in cmp[p]][::-1]
        vals = [v for _, v in cmp[p]][::-1]
        ax.barh(np.arange(len(words)), vals, color=C_MAIN, height=0.65)
        ax.set_yticks(np.arange(len(words)))
        ax.set_yticklabels(words, fontsize=13)
        ax.set_title(p, fontsize=13)
        ax.set_xlabel("TF-IDF 权重", fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.suptitle("图6  主要诗人的特征用字（字符级 TF-IDF）", fontsize=14, y=1.02)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 图 7：赠答关系网络
# ---------------------------------------------------------------------------
def fig_network(G, name="fig7_赠答网络.png"):
    # 第一步：只保留"唱和 ≥2 次"的强关系。弱关系占绝大多数，画上去就是毛线团。
    strong = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= 2]
    S = G.edge_subgraph(strong).copy() if strong else G.copy()

    # 第二步：取最大连通块。剩下的都是零散的一两对诗友，
    # 画在图上就是一堆互不相连的圆点，构不成"网络"。
    comp = max(nx.connected_components(S.to_undirected()), key=len)
    H = S.subgraph(comp).copy()

    deg = dict(H.degree(weight="weight"))
    pos = nx.spring_layout(H, k=1.25, iterations=600, seed=42, weight="weight")
    wdeg = np.array([deg[n] for n in H.nodes()])
    sizes = 300 + 1300 * (wdeg - wdeg.min()) / (np.ptp(wdeg) + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 9))
    widths = [0.4 + 0.7 * min(d["weight"], 20) for _, _, d in H.edges(data=True)]
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.3, edge_color="#555555",
                           width=widths, arrows=True, arrowsize=10,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=sizes,
                           node_color=wdeg, cmap="viridis", alpha=0.9,
                           linewidths=0.8, edgecolors="white")
    # 名字抬高一点，避免压在节点上；再加白色描边，压在连线上也看得清。
    # 两个点挨得太近时，后一个的名字翻到下方，免得两行字叠成一团。
    label_pos = {}
    placed = []
    for n in sorted(H.nodes(), key=lambda x: -deg[x]):
        x, y = pos[n]
        dy = 0.045
        if any(abs(x - px) < 0.13 and abs(y - py) < 0.08 for px, py in placed):
            dy = -0.075
        label_pos[n] = (x, y + dy)
        placed.append((x, y))
    labels = nx.draw_networkx_labels(H, label_pos, ax=ax, font_size=12,
                                     font_family="Microsoft YaHei",
                                     font_color="#222222")
    for t in labels.values():
        t.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])
    ax.set_title("图7  唐代诗人的核心唱和圈（唱和 ≥2 次的最大连通网络，共 %d 位诗人）"
                 % H.number_of_nodes(), fontsize=13, pad=12)
    ax.axis("off")
    ax.grid(False)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# 图 8：四季意象
# ---------------------------------------------------------------------------
def fig_season(tab, name="fig8_四季意象.png"):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(tab))
    bars = ax.bar(x, tab["出现次数"], color=PALETTE[:len(tab)], width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(tab["季节"], fontsize=15)
    ax.set_ylabel("相关字词出现总次数")
    ax.set_title("图8  唐诗中的四季意象分布", fontsize=14, pad=12)
    for b, v, p in zip(bars, tab["出现次数"], tab["占比%"]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.01, "%d\n%.1f%%" % (v, p),
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(tab["出现次数"]) * 1.18)
    ax.grid(axis="x", visible=False)
    return _save(fig, name)


