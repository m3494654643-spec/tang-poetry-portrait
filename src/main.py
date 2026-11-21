# -*- coding: utf-8 -*-
"""
main.py  ——  总入口，按顺序跑完六个模块，产出所有图表和表

前置条件：data/json/ 目录下已有 poet.tang.*.json 分片
"""

import os
import time

import numpy as np

import analysis
import features
import loader
import network
import visualize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "output")

# 图 6 要对比的诗人：李杜 + 白居易 + 王维，正好代表四种典型风格
COMPARE_POETS = ["李白", "杜甫", "白居易", "王维"]


def step(n, title):
    print("\n" + "=" * 58)
    print(" 步骤 %d：%s" % (n, title))
    print("=" * 58, flush=True)


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------- 1. 读数据 ----------------
    step(1, "读取并清洗全唐诗")
    df = loader.build()

    # ---------------- 2. 特征 ----------------
    step(2, "统计意象与季节词频")
    df = features.build_features(df)

    # ---------------- 3. M1 基础画像 ----------------
    step(3, "M1 基础画像")
    ov = analysis.overview(df)
    top = analysis.top_authors(df)
    dist = analysis.author_distribution(df)
    counts = analysis.author_counts(df)
    genre = analysis.genre_dist(df)
    linelen = analysis.line_length_dist(df)
    for name, tab in [("overview", ov), ("top_authors", top),
                      ("author_distribution", dist), ("genre", genre),
                      ("line_length", linelen)]:
        tab.to_csv(os.path.join(OUT_DIR, "M1_%s.csv" % name), encoding="utf-8-sig")
    print(ov.to_string())
    print("\n存诗量前 10：\n%s" % top.head(10).to_string(index=False))
    print("\n存诗量分布：\n%s" % dist.to_string(index=False))

    visualize.setup()
    print()
    visualize.fig_top_authors(top)
    visualize.fig_long_tail(counts)
    visualize.fig_genre(genre)

    # ---------------- 4. M2 意象 ----------------
    step(4, "M2 意象分析")
    img = analysis.imagery_counts(df)
    co = analysis.imagery_cooccurrence(df)
    by_poet = analysis.imagery_by_poet(df, COMPARE_POETS)
    img.to_csv(os.path.join(OUT_DIR, "M2_imagery.csv"), index=False, encoding="utf-8-sig")
    co.to_csv(os.path.join(OUT_DIR, "M2_cooccurrence.csv"), encoding="utf-8-sig")
    by_poet.to_csv(os.path.join(OUT_DIR, "M2_imagery_by_poet.csv"),
                   index=False, encoding="utf-8-sig")
    print(img.to_string(index=False))
    print("\n意象共现次数最高的一组：")
    # 共现矩阵是对称的，取上三角，否则 “山+水” 和 “水+山” 会各打一遍
    upper = co.where(np.triu(np.ones(co.shape), 1).astype(bool))
    flat = upper.stack().sort_values(ascending=False)
    for (a, b), v in flat.head(5).items():
        print("  %s + %s : %d 首" % (a, b, v))
    print("\n%s" % by_poet.to_string(index=False))

    print()
    visualize.fig_imagery(img)
    visualize.fig_cooccurrence(co)

    # ---------------- 5. M3 诗人特征词 ----------------
    step(5, "M3 诗人特征词（TF-IDF）")
    tf = analysis.poet_tfidf(df)
    tf.to_csv(os.path.join(OUT_DIR, "M3_poet_tfidf.csv"),
              index=False, encoding="utf-8-sig")
    print(tf.to_string(index=False))
    cmp_tf = analysis.poet_word_compare(df, COMPARE_POETS)
    print()
    visualize.fig_tfidf(cmp_tf)

    # ---------------- 6. M4 赠答网络 ----------------
    step(6, "M4 赠答网络")
    edges, cov = network.build_edges(df)
    edges.to_csv(os.path.join(OUT_DIR, "M4_edges.csv"),
                 index=False, encoding="utf-8-sig")
    print(cov.to_string())
    G = network.build_graph(edges)
    hubs, nstats = network.hub_scores(G)
    hubs.to_csv(os.path.join(OUT_DIR, "M4_hubs.csv"),
                index=False, encoding="utf-8-sig")
    print("\n%s" % nstats.to_string())
    print("\n社交枢纽（加权度前 15）：\n%s" % hubs.to_string(index=False))
    print()
    visualize.fig_network(G)

    # ---------------- 7. M5 季节 ----------------
    step(7, "M5 季节意象")
    season = analysis.season_stats(df)
    dens = analysis.season_density_by_poet(df)
    season.to_csv(os.path.join(OUT_DIR, "M5_season.csv"),
                  index=False, encoding="utf-8-sig")
    dens.to_csv(os.path.join(OUT_DIR, "M5_season_density.csv"),
                index=False, encoding="utf-8-sig")
    print(season.to_string(index=False))
    print("\n“秋意象密度”最高（存诗≥300 首）：\n%s" % dens.to_string(index=False))
    print()
    visualize.fig_season(season)

    # ---------------- 8. 汇总 ----------------
    step(8, "汇总")
    print("全部完成，用时 %.1f 秒" % (time.time() - t0))
    print("图表目录：%s" % OUT_DIR)
    for f in sorted(os.listdir(OUT_DIR)):
        print("  - %s" % f)


if __name__ == "__main__":
    main()
