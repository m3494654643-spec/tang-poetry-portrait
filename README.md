# 全唐诗的数字画像

> 用 pandas 读 57,598 首唐诗，回答三个问题：谁写得多、写什么、谁跟谁来往。

一个数据分析项目：从原始 JSON 到清洗、统计、可视化。

---

## 一、快速开始

### 第 1 步：拿到项目

```bash
git clone https://github.com/m3494654643-spec/tang-poetry-portrait.git
cd tang-poetry-portrait
```

不会用 git 的话：仓库页右上角绿色 `<> Code` → `Download ZIP` → 解压。

### 第 2 步：安装依赖

```bash
pip install -r requirements.txt
```

需要 Python 3.9 以上（本项目在 3.12 上开发）。

### 第 3 步：下载数据

**仓库里不含原始数据**（24 MB，版权属原项目），需要自己去下，方法见下一节 [二、数据在哪里下载](#二数据在哪里下载)。

下完放到 `data/json/`，目录长这样：

```
tang-poetry-portrait/
└── data/
    └── json/
        ├── poet.tang.0.json
        ├── poet.tang.1000.json
        └── …（共 58 个）
```

### 第 4 步：运行

```bash
cd src
python main.py
```

> **必须先进 `src` 目录。** 六个文件之间是同级 import，在仓库根目录跑 `python src/main.py` 会报 `ModuleNotFoundError`。

### 成功的标志

| 检查项 | 应该是 |
|---|---|
| `data/json/` 里的文件数 | **58 个** `poet.tang.*.json`，共约 24 MB |
| 运行耗时 | 十几秒 |
| 最后一行输出 | `全部完成，用时 xx 秒` |
| `output/` 里的文件数 | **21 个**（8 张 PNG + 13 张 CSV） |
| 抽样核对 | 打开 `output/M1_top_authors.csv`，第一行应是 **白居易 3009 首** |

五个都对上，说明数据、环境、代码三样都没问题。

---

## 二、数据在哪里下载

数据来自开源项目 **[chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)**（MIT 许可）。

1. 打开 https://github.com/chinese-poetry/chinese-poetry
2. 进入 **`全唐诗/`** 目录
3. 下载里面所有 `poet.tang.*.json`（共 58 个分片，约 24 MB）
4. 全部放进项目根的 `data/json/`

> ⚠️ **文件名必须保持原样**（`poet.tang.0.json`、`poet.tang.1000.json`…）。
> 代码用通配符 `poet.tang.*.json` 扫描，改了名就扫不到。



数据格式（每条一首诗）：

```json
{"author": "李白", "title": "静夜思",
 "paragraphs": ["床前明月光，", "疑是地上霜。"], "id": "..."}
```

只有四个字段；内容为繁体字，代码里用 OpenCC 转成简体，另存一列繁体原文供意象统计使用。

---

## 三、如果卡住了

| 报错 / 现象 | 原因 | 解决 |
|---|---|---|
| `FileNotFoundError: data/json 下没有找到分片文件` | 数据没放对位置 | 确认 json 在 **`data/json/`** 里（不是 `data/`，也不是随便哪个文件夹） |
| 同上，但文件确实在 | 文件名被改过 | 恢复成 `poet.tang.数字.json`，序号不能动 |
| `ModuleNotFoundError: No module named 'loader'` | 在仓库根目录运行 | 先 `cd src`，再 `python main.py` |
| `ModuleNotFoundError: No module named 'pandas'` | 依赖没装 | `pip install -r requirements.txt` |
| 图里中文全是方框 | 系统缺中文字体 | Windows 自带 SimHei 一般没问题；Linux 需自行安装中文字体 |
| 只跑了几秒就结束 | 数据没读全 | 看开头那行 `读取分片 N 个`，N 应该是 58 |

---

## 四、跑出来是什么样

### 三个问题，三个答案

| 问题 | 结论 | 看哪张图 |
|---|---|---|
| **谁写得多？** | 白居易 3,009 首居首；**51.7% 的诗人一生只留下 1 首** | 图 1、图 2 |
| **写什么？** | 第一意象是「水」（覆盖率 34.4%），**不是印象中的「月」**（19.6%） | 图 4、图 5 |
| **谁跟谁来往？** | 姚合是核心节点（赠出 45、收到 7）；全部关系里 **91.5% 是单向的** | 图 7 |

几个值得一看的数字：

- 存诗 1~63 首的前六档共 3,484 人，占诗人总数的 **95.7%**，却只贡献 **25%** 的诗作
- 存诗 512 首以上的 20 人，独占 **30%** 的诗作
- 李白排头的特征字是「天」，杜甫是「江」，王维是「山」——算法没读过文学史，算出的差别与文学史评价吻合
- 按每千字「秋」字密度排序，许浑（10.39）排第一，正是写「山雨欲来风满楼」那位

![存诗量长尾分布](docs/figures/fig2_存诗量长尾分布.png)

<p align="center">
  <img src="docs/figures/fig6_诗人特征词.png" width="620">
</p>

---

## 五、项目结构

```
.
├── src/
│   ├── loader.py      读 JSON、繁转简、清洗、算句数/字数/诗体  194 行
│   ├── features.py    意象词表与季节词表的字频统计             77 行
│   ├── analysis.py    产量、意象、共现、TF-IDF、季节密度       259 行
│   ├── network.py     从诗题抽人名、建赠答网络                 111 行
│   ├── visualize.py   8 张图的绘制                             275 行
│   └── main.py        总入口，按顺序跑完六步                   135 行
├── data/json/         原始分片（需自行下载）
├── output/            运行产物：8 图 + 13 表
└── docs/figures/      README 用的图
```


---

## 六、已知局限

- **赠答网络覆盖率只有 1.2%**（687 / 57,598）。只认诗题里以赠答动词开头、且后面紧跟的诗名能在作者名单里对上的情况；只写官职（「令狐相公」）、夹排行（「李十二白」）、写「友人」的，全部漏掉。图 7 反映的是**显性赠答**的局部，不是完整社交图。
- **十类意象是人工定的**，覆盖面有限。「剑」只占 2.7% 不代表唐人少写剑，可能只是词表没抓到其他说法。
- **四季词表有重叠**：「春」这一档含花、柳，一首秋日咏花的诗会同时计入两季，47.2% 衡量的是「春意词」而非严格季节。
- **TF-IDF 按单字算**，认不出「明月」「长安」这类双字词。
- 数据里没有年代和籍贯，做不了时间演化和地域分布。

---

## 七、关于代码

选题、指标口径、每个模块要回答什么问题由作者确定；具体代码在 AI 辅助下生成，作者负责运行、核对输出、修正错误。调试日志记录的四个问题（佚名写法不统一、分桶宽度不等、特征词被标点占据、网络图不可读）都是实际遇到并修改的。

---

## 八、数据来源与许可

- 数据：[chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（MIT License）
- 代码：[MIT License](LICENSE)
- 本仓库只包含分析与可视化代码，不包含原始数据

---

## 九、运行后会生成什么

一次完整运行会在 `output/` 下写出 **21 个文件**：

**8 张图（PNG）**

| 文件 | 内容 |
|---|---|
| `fig1_诗人存诗量Top20.png` | 存诗量前 20 名的横向条形图 |
| `fig2_存诗量长尾分布.png` | 对数分档柱状图 + 累计诗作占比折线 |
| `fig3_诗体构成.png` | 律诗 / 古体 / 绝句的环形图 |
| `fig4_意象频次.png` | 十类意象的出现次数 |
| `fig5_意象共现热力图.png` | 10×10 共现矩阵 |
| `fig6_诗人特征词.png` | 四位诗人的 TF-IDF 特征字对比 |
| `fig7_赠答网络.png` | 26 位诗人的核心唱和圈 |
| `fig8_四季意象.png` | 春夏秋冬的字词出现次数 |

**13 张表（CSV，UTF-8 with BOM，Excel 可直接打开）**

`M1_top_authors.csv`、`M1_overview.csv`、`M1_author_distribution.csv`、`M1_genre.csv`、`M1_line_length.csv`、`M2_imagery.csv`、`M2_cooccurrence.csv`、`M2_imagery_by_poet.csv`、`M3_poet_tfidf.csv`、`M4_edges.csv`、`M4_hubs.csv`、`M5_season.csv`、`M5_season_density.csv`

图表上的每一个数字都来自这些 CSV，报告中引用的数据可逐项回溯。

---

## 环境

Python 3.12 · numpy · pandas · matplotlib · opencc-python-reimplemented · networkx · scikit-learn


