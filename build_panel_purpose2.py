"""
研究目的二：数据清洗 + 变量构建 → 输出干净面板数据

输入：
  data/raw/微博热搜posts_analyzed.csv
  data/raw/微博热搜comments_BERT.csv
  data/raw/热搜核心诉求_LLM提取结果.csv
  data/raw/微博热搜博文_筛选后待爬评论.csv   （白名单：仅保留此文件中的话题）
  data/clean/topic_category_labels.csv      （LLM标注：is_demand=Y 且有领域分类）

输出（data/clean/）：
  panel_topic_window.csv      话题 × 2h时间窗口面板（用于聚合层面回归）
  topic_level.csv             话题层面变量（earlyStance / controls / IV变量）
  individual_comments.csv     个体评论层面（含 exposure_ijt，用于微观回归）

核心设计决策：
  - 分析单元：title（热搜话题）
  - t₀：从 source_md 解析热搜上榜日期，取该话题在上榜当天的最早 post 时间；
         若当天无 post，退化为上榜日期的 00:00:00
  - 时间窗口：2h（主分析），相对于 t₀ 的编号 window = floor(相对小时 / 2)
  - 活跃期：t₀ 后 72h 内的所有 posts + comments
  - 样本筛选：
      1. 仅保留白名单（微博热搜博文_筛选后待爬评论.csv）中的话题
      2. 剔除 is_demand=N 的话题（纯会议/人事/宣传性内容）
      3. 剔除水军/垃圾；保留 ≥6 个有效窗口（每窗口≥5条观测）的话题
  - 早期评论定义：前2h（主）；稳健性备用：前4h、前10%（在 topic_level 中均保留）
"""

import pandas as pd
import numpy as np
import os, re

RAW   = "/Users/wangmanlin/Desktop/citizens-will-and-politics/data/raw"
CLEAN = "/Users/wangmanlin/Desktop/citizens-will-and-politics/data/clean"
os.makedirs(CLEAN, exist_ok=True)

WINDOW_H       = 2      # 主分析窗口宽度（小时）
CUTOFF_H       = 72     # 活跃期上限（小时）
MIN_OBS_WIN    = 5      # 每个有效窗口最低观测数
MIN_WIN_TOPIC  = 6      # 每个话题最低有效窗口数
EARLY_H        = 2      # 早期评论定义（小时）
TOP_HOT_N      = 20     # exposure 变量：发评前累计热评前 N 条（按点赞数）

# ─────────────────────────────────────────────────────────────────────────────
# 1. 加载原始数据
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. 加载原始数据 + 白名单 + 领域标注")
print("=" * 60)

# ── 白名单：仅保留筛选后待爬评论的话题 ──
whitelist_df = pd.read_csv(
    os.path.join(RAW, "微博热搜博文_筛选后待爬评论.csv"),
    encoding="utf-8-sig", usecols=["title"]
)
WHITELIST_TITLES = set(whitelist_df["title"].dropna().unique())
print(f"白名单话题数：{len(WHITELIST_TITLES):,}")

# ── 领域标注 ──
category_labels = pd.read_csv(
    os.path.join(CLEAN, "topic_category_labels.csv"),
    encoding="utf-8-sig",
    usecols=["title", "category_L1", "category_L2"]
)

# ── 主数据 ──
posts_raw = pd.read_csv(
    os.path.join(RAW, "微博热搜posts_analyzed.csv"),
    encoding="utf-8-sig", low_memory=False,
)
posts_raw["发布时间"] = pd.to_datetime(posts_raw["发布时间"], format="mixed", errors="coerce")

comments_raw = pd.read_csv(
    os.path.join(RAW, "微博热搜comments_BERT.csv"),
    encoding="utf-8-sig", low_memory=False,
)
comments_raw["评论时间"] = pd.to_datetime(comments_raw["评论时间"], format="mixed", errors="coerce")

demand_raw = pd.read_csv(
    os.path.join(RAW, "热搜核心诉求_LLM提取结果.csv"),
    encoding="utf-8-sig", low_memory=False
)

print(f"posts_analyzed：{len(posts_raw):,} 行，{posts_raw['title'].nunique():,} 个话题（过滤前）")
print(f"comments_BERT：{len(comments_raw):,} 行（过滤前）")
print(f"热搜核心诉求：{len(demand_raw):,} 行，字段：{list(demand_raw.columns)}")

# ── 有效话题：仅用白名单过滤 ──
VALID_TITLES = WHITELIST_TITLES
print(f"\n有效话题数：{len(VALID_TITLES):,}（白名单全量）")
posts_raw = posts_raw[posts_raw["title"].isin(VALID_TITLES)].copy()
# comments 通过 博文id → title 映射过滤（在第4步 merge 后自动剔除）
print(f"过滤后 posts_raw：{len(posts_raw):,} 行，{posts_raw['title'].nunique():,} 个话题")

# ─────────────────────────────────────────────────────────────────────────────
# 2. 过滤水军 / 垃圾
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. 过滤水军 / 垃圾")
print("=" * 60)

posts = posts_raw[
    (posts_raw["is_bot_post"].astype(str).str.upper() != "Y") &
    (posts_raw["is_spam_post"].astype(str).str.upper() != "Y") &
    posts_raw["发布时间"].notna() &
    posts_raw["stance_score"].notna() &
    posts_raw["sentiment_score"].notna()
].copy()

comments = comments_raw[
    (comments_raw["is_bot_comment"].astype(str).str.upper() != "Y") &
    (comments_raw["is_spam_comment"].astype(str).str.upper() != "Y") &
    comments_raw["评论时间"].notna() &
    comments_raw["stance_score"].notna() &
    comments_raw["sentiment_score"].notna()
].copy()

print(f"有效 posts：{len(posts):,}（剔除 {len(posts_raw)-len(posts):,}）")
print(f"有效 comments：{len(comments):,}（剔除 {len(comments_raw)-len(comments):,}）")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 解析 t₀（从 source_md 获取上榜日期，精确到话题当天最早 post 时间）
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 解析 t₀")
print("=" * 60)

def parse_source_date(s):
    """从 '2021-01-29.md' 中提取日期"""
    if pd.isna(s):
        return pd.NaT
    m = re.search(r"(\d{4}-\d{2}-\d{2})", str(s))
    return pd.Timestamp(m.group(1)) if m else pd.NaT

# 每个话题的上榜日期（取该话题所有 source_md 中最早的日期）
posts["_post_上榜日期"] = posts["source_md"].apply(parse_source_date)
topic_date = (
    posts.groupby("title")["_post_上榜日期"].min()
    .reset_index()
    .rename(columns={"_post_上榜日期": "上榜日期"})
)

# t₀ = 该话题在上榜当天的最早 post 发布时间
# 若当天无 post，退化为上榜日期的 00:00:00
posts_with_date = posts.merge(topic_date, on="title", how="left")
posts_with_date["上榜日期"] = pd.to_datetime(posts_with_date["上榜日期"], errors="coerce")
posts_with_date["发布时间"] = pd.to_datetime(posts_with_date["发布时间"], format="mixed", errors="coerce")
posts_on_date = posts_with_date[
    posts_with_date["发布时间"].dt.date == posts_with_date["上榜日期"].dt.date
]
t0_on_date = posts_on_date.groupby("title")["发布时间"].min().reset_index()
t0_on_date.rename(columns={"发布时间": "t0"}, inplace=True)

# 合并 t₀，fallback 到上榜日期 00:00:00
t0_df = topic_date.merge(t0_on_date, on="title", how="left")
t0_df["t0"] = t0_df["t0"].where(
    t0_df["t0"].notna(),
    t0_df["上榜日期"]
)
t0_df = t0_df[["title", "上榜日期", "t0"]]

n_fallback = t0_df["t0"].dt.hour.eq(0).sum()  # 只是大致估计 fallback 数
print(f"话题数：{len(t0_df):,}")
print(f"t₀ 时间分布（小时）：")
print(t0_df["t0"].dt.hour.value_counts().sort_index().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 4. 将 comments 关联到话题，构建合并时间序列
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. 构建合并时间序列（posts + comments）")
print("=" * 60)

post_to_title = posts[["博文id", "title"]].drop_duplicates("博文id")
comments = comments.merge(post_to_title, on="博文id", how="left")
n_unmatched = comments["title"].isna().sum()
print(f"评论未匹配到话题：{n_unmatched:,}（{n_unmatched/len(comments)*100:.1f}%）")
comments = comments.dropna(subset=["title"])

# 统一格式：posts
posts_long = posts[[
    "title", "博文id", "用户id", "发布时间",
    "stance_score", "sentiment_score", "emotion",
    "点赞数", "转发数", "评论数"
]].copy().rename(columns={"发布时间": "时间戳"})
posts_long["来源"] = "post"
posts_long["评论id"] = np.nan
posts_long["父评论id"] = np.nan
for col in ["点赞数", "转发数", "评论数"]:
    posts_long[col] = pd.to_numeric(posts_long[col], errors="coerce")

# 统一格式：comments
comments_long = comments[[
    "title", "博文id", "用户id", "评论时间",
    "stance_score", "sentiment_score", "emotion",
    "点赞数", "评论id", "父评论id"
]].copy().rename(columns={"评论时间": "时间戳"})
comments_long["来源"] = "comment"
comments_long["转发数"] = np.nan
comments_long["评论数"] = np.nan
comments_long["点赞数"] = pd.to_numeric(comments_long["点赞数"], errors="coerce")

# 合并
combined = pd.concat([posts_long, comments_long], ignore_index=True)
combined = combined.merge(t0_df[["title", "t0", "上榜日期"]], on="title", how="left")
combined["时间戳"] = pd.to_datetime(combined["时间戳"], errors="coerce")
combined["t0"] = pd.to_datetime(combined["t0"], errors="coerce")
combined = combined.dropna(subset=["时间戳", "t0", "stance_score", "sentiment_score"])

# 计算相对小时 & 窗口编号
combined["相对小时_float"] = (combined["时间戳"] - combined["t0"]).dt.total_seconds() / 3600
combined["相对小时"] = combined["相对小时_float"].astype(int)
combined["窗口"] = combined["相对小时"] // WINDOW_H

# 过滤活跃期：t≥0 且 t<72h
combined = combined[(combined["相对小时"] >= 0) & (combined["相对小时"] < CUTOFF_H)]

# 确保数值列类型正确
for _col in ["stance_score", "sentiment_score", "点赞数", "转发数", "评论数"]:
    combined[_col] = pd.to_numeric(combined[_col], errors="coerce")

print(f"活跃期内总观测：{len(combined):,}（posts: {(combined['来源']=='post').sum():,}，comments: {(combined['来源']=='comment').sum():,}）")
print(f"覆盖话题数：{combined['title'].nunique():,}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. 筛选可行话题（≥6 个有效 2h 窗口，每窗口 ≥5 条观测）
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. 筛选可行话题")
print("=" * 60)

window_counts = combined.groupby(["title", "窗口"]).size().reset_index(name="窗口观测数")
valid_windows = window_counts[window_counts["窗口观测数"] >= MIN_OBS_WIN]
valid_per_topic = valid_windows.groupby("title").size().reset_index(name="有效窗口数")
eligible_titles = set(
    valid_per_topic[valid_per_topic["有效窗口数"] >= MIN_WIN_TOPIC]["title"]
)
print(f"可行话题数：{len(eligible_titles):,}（共 {combined['title'].nunique():,} 个话题）")

combined = combined[combined["title"].isin(eligible_titles)].copy()
print(f"筛选后观测数：{len(combined):,}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. 构建话题-时间窗口面板（聚合层面）
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. 构建话题-时间窗口面板")
print("=" * 60)

panel = combined.groupby(["title", "窗口"]).agg(
    stance_it=("stance_score", "mean"),
    sentiment_it=("sentiment_score", "mean"),
    stance_std_it=("stance_score", "std"),        # 滚动标准差（σ_it）用于收敛检验
    volume_it=("stance_score", "count"),
    post_count_it=("来源", lambda x: (x == "post").sum()),
    comment_count_it=("来源", lambda x: (x == "comment").sum()),
    avg_likes_it=("点赞数", "mean"),
    hour_of_day_it=("时间戳", lambda x: pd.to_datetime(x, errors="coerce").dt.hour.mode().iloc[0] if len(x) > 0 else np.nan),
).reset_index()

# 添加滞后一期因变量（同话题内）
panel = panel.sort_values(["title", "窗口"])
panel["stance_it_lag1"] = panel.groupby("title")["stance_it"].shift(1)
panel["sentiment_it_lag1"] = panel.groupby("title")["sentiment_it"].shift(1)
panel["volume_it_lag1"] = panel.groupby("title")["volume_it"].shift(1)
# 累积评论数（话题生命周期位置控制变量）
panel["cum_volume_it"] = panel.groupby("title")["volume_it"].cumsum()

print(f"面板总行数：{len(panel):,}（话题数 {panel['title'].nunique():,} × 平均窗口数 {len(panel)/panel['title'].nunique():.1f}）")
print(f"窗口编号范围：{panel['窗口'].min()} 到 {panel['窗口'].max()}（对应 0~{panel['窗口'].max()*WINDOW_H}h）")

# ─────────────────────────────────────────────────────────────────────────────
# 7. 构建话题层面变量（earlyStance / controls / IV 变量）
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. 构建话题层面变量")
print("=" * 60)

# ── 早期评论变量（三种定义，均保留用于稳健性）──
def early_stats(group, h_cutoff=None, pct_cutoff=None, label=""):
    if h_cutoff is not None:
        early = group[group["相对小时"] < h_cutoff]
    elif pct_cutoff is not None:
        n = max(1, int(len(group) * pct_cutoff))
        early = group.nsmallest(n, "时间戳")
    else:
        raise ValueError
    if len(early) == 0:
        return pd.Series({
            f"earlyStance_{label}": np.nan,
            f"earlySentiment_{label}": np.nan,
            f"earlyConsistency_{label}": np.nan,
            f"earlyVolume_{label}": 0,
            f"earlyLikes_{label}": np.nan,
        })
    std_early = early["stance_score"].std()
    return pd.Series({
        f"earlyStance_{label}": early["stance_score"].mean(),
        f"earlySentiment_{label}": early["sentiment_score"].mean(),
        # earlyConsistency = 1 - std/50（std最大约100，除50归一化到0~1；越高越一致）
        f"earlyConsistency_{label}": 1 - (std_early / 50) if not np.isnan(std_early) else np.nan,
        f"earlyVolume_{label}": len(early),
        f"earlyLikes_{label}": early["点赞数"].mean() if "点赞数" in early.columns else np.nan,
    })

print("  计算早期评论变量（前2h / 前4h / 前10%）...")
early_2h  = combined.groupby("title").apply(early_stats, h_cutoff=2,    label="2h",  include_groups=False)
early_4h  = combined.groupby("title").apply(early_stats, h_cutoff=4,    label="4h",  include_groups=False)
early_10p = combined.groupby("title").apply(early_stats, pct_cutoff=0.1, label="10p", include_groups=False)

# ── 话题整体统计 ──
topic_agg = combined.groupby("title").agg(
    total_obs=("stance_score", "count"),
    post_count=("来源", lambda x: (x == "post").sum()),
    comment_count=("来源", lambda x: (x == "comment").sum()),
    overall_stance=("stance_score", "mean"),
    overall_sentiment=("sentiment_score", "mean"),
    duration_hours=("相对小时", "max"),
).reset_index()

# ── IV 变量：上榜时刻 & 星期 ──
# 使用 t₀ 的小时数和星期几
t0_iv = t0_df[["title", "t0", "上榜日期"]].copy()
t0_iv["t0"] = pd.to_datetime(t0_iv["t0"])
t0_iv["上榜小时"] = t0_iv["t0"].dt.hour                         # Z1: 0~23
t0_iv["是否深夜"] = t0_iv["上榜小时"].isin(range(0, 7)).astype(int)  # 0~6时=深夜
t0_iv["上榜星期"] = t0_iv["t0"].dt.dayofweek                    # 0=周一...6=周日
t0_iv["是否周末"] = (t0_iv["上榜星期"] >= 5).astype(int)
t0_iv["深夜x周末"] = t0_iv["是否深夜"] * t0_iv["是否周末"]         # Z3 交互项
t0_iv["上榜年份"] = t0_iv["t0"].dt.year
t0_iv["上榜月份"] = t0_iv["t0"].dt.month

# ── 合并领域分类（来自 topic_category_labels.csv）+ is_government_related ──
demand_cols = demand_raw[["title", "is_government_related", "keywords_count"]].copy()
demand_cols = demand_cols.drop_duplicates("title")

category_cols = category_labels[["title", "category_L1", "category_L2"]].drop_duplicates("title")

# ── 汇总话题层面数据框 ──
topic_df = (
    topic_agg
    .merge(early_2h.reset_index(),  on="title", how="left")
    .merge(early_4h.reset_index(),  on="title", how="left")
    .merge(early_10p.reset_index(), on="title", how="left")
    .merge(t0_iv,                   on="title", how="left")
    .merge(demand_cols,             on="title", how="left")
    .merge(category_cols,           on="title", how="left")
)

print(f"话题层面数据框：{len(topic_df):,} 行 × {len(topic_df.columns):,} 列")
print(f"  - earlyStance_2h 覆盖率：{topic_df['earlyStance_2h'].notna().mean()*100:.1f}%")
print(f"  - category_L1 覆盖率：{topic_df['category_L1'].notna().mean()*100:.1f}%")
print(f"  - is_government_related 覆盖率：{topic_df['is_government_related'].notna().mean()*100:.1f}%")
print(f"\nearlyStance_2h 分布：")
print(topic_df["earlyStance_2h"].describe(percentiles=[.1,.25,.5,.75,.9]).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 8. 将话题层面变量合并到面板
# ─────────────────────────────────────────────────────────────────────────────
topic_merge_cols = [c for c in topic_df.columns if c != "title"]
panel = panel.merge(
    topic_df[["title"] + topic_merge_cols],
    on="title", how="left"
)
print(f"\n面板（含话题变量）：{len(panel):,} 行 × {len(panel.columns):,} 列")

# ─────────────────────────────────────────────────────────────────────────────
# 9. 构建个体评论层面数据集（含 exposure_ijt）
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. 构建个体评论层面数据集（含 exposure_ijt）")
print("=" * 60)

# 只处理 comments（posts 不存在"看到热评"的问题）
individual = combined[combined["来源"] == "comment"].copy()
individual = individual.sort_values(["title", "时间戳"])

print(f"个体评论观测数：{len(individual):,}")
print(f"计算 exposure_ijt（每条评论发布前，该话题累计热评前{TOP_HOT_N}条的平均态度）...")

# exposure_ijt：对每条评论 j（话题 i，时刻 t），
# 取该话题在时刻 t 之前所有评论中点赞数最高的 TOP_HOT_N 条的平均 stance_score
# 为降低计算量，逐话题处理
exposure_list = []
topics_with_comments = individual["title"].unique()
n_topics = len(topics_with_comments)

for idx, topic in enumerate(topics_with_comments):
    if (idx + 1) % 200 == 0:
        print(f"  处理进度：{idx+1}/{n_topics}")
    grp = individual[individual["title"] == topic].sort_values("时间戳").reset_index(drop=True)
    exposures = []
    for i, row in grp.iterrows():
        # 该评论发布时刻之前的所有评论
        prior = grp[grp["时间戳"] < row["时间戳"]]
        if len(prior) == 0:
            exposures.append(np.nan)
        else:
            # 取点赞数最高的前 TOP_HOT_N 条
            top_hot = prior.nlargest(TOP_HOT_N, "点赞数")
            exposures.append(top_hot["stance_score"].mean())
    grp["exposure_ijt"] = exposures
    exposure_list.append(grp)

individual_out = pd.concat(exposure_list, ignore_index=True)

# 合并话题层面变量
ind_topic_cols = [
    "title", "earlyStance_2h", "earlySentiment_2h", "earlyConsistency_2h",
    "earlyVolume_2h", "category_L1", "category_L2", "is_government_related",
    "上榜小时", "是否深夜", "是否周末", "上榜年份",
    "overall_stance", "overall_sentiment"
]
individual_out = individual_out.merge(
    topic_df[ind_topic_cols], on="title", how="left"
)
# 添加相对时间函数形式
individual_out["log_time"] = np.log1p(individual_out["相对小时_float"])
individual_out["窗口"] = individual_out["相对小时"] // WINDOW_H

print(f"个体数据集：{len(individual_out):,} 行 × {len(individual_out.columns):,} 列")
print(f"exposure_ijt 覆盖率（非NaN）：{individual_out['exposure_ijt'].notna().mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 10. 输出
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. 输出清洗数据")
print("=" * 60)

panel_out_path     = os.path.join(CLEAN, "panel_topic_window.csv")
topic_out_path     = os.path.join(CLEAN, "topic_level.csv")
individual_out_path = os.path.join(CLEAN, "individual_comments.csv")

panel.to_csv(panel_out_path, index=False, encoding="utf-8-sig")
topic_df.to_csv(topic_out_path, index=False, encoding="utf-8-sig")
individual_out.to_csv(individual_out_path, index=False, encoding="utf-8-sig")

print(f"panel_topic_window.csv     → {len(panel):,} 行，{panel['title'].nunique():,} 个话题")
print(f"topic_level.csv            → {len(topic_df):,} 行")
print(f"individual_comments.csv    → {len(individual_out):,} 行")
print("\n完成。")
