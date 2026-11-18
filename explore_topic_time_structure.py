"""
以热搜话题（title）为单元，合并 posts_analyzed + comments_BERT，
重新评估研究目的二（锚定效应）数据可行性。

分析层次：
  - 最高单元：title（热搜话题）
  - 一个 title 下包含多个 博文id（posts），每个博文下有若干 comments
  - 时间序列 = posts(发布时间) + comments(评论时间) 的合并流
  - t₀ = 该 title 所有 posts 中最早的发布时间（或从 source_md 推断日期）

关键问题：
  1. 共有多少个唯一热搜话题？posts / comments 分别覆盖多少？
  2. 合并后，每个话题可构建多少个有效时间窗口（1h / 2h）？
  3. 满足 ≥6 个有效窗口的话题有多少？
  4. earlyStance 的分布是否合理？
  5. 话题内 posts 和 comments 时间分布的连续性如何？
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RAW = "/Users/wangmanlin/Desktop/citizens-will-and-politics/data/raw"
FIG = "/Users/wangmanlin/Desktop/citizens-will-and-politics/results/figures"
os.makedirs(FIG, exist_ok=True)

# ── 1. 加载数据 ───────────────────────────────────────────────────────────────
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

posts = pd.read_csv(
    os.path.join(RAW, "微博热搜posts_analyzed.csv"),
    encoding="utf-8-sig", low_memory=False,
    parse_dates=["发布时间"]
)
comments = pd.read_csv(
    os.path.join(RAW, "微博热搜comments_BERT.csv"),
    encoding="utf-8-sig", low_memory=False,
    parse_dates=["评论时间"]
)
print(f"posts_analyzed 行数：{len(posts):,}，字段：{list(posts.columns)}")
print(f"comments_BERT  行数：{len(comments):,}")
print(f"\nposts 中 title 唯一值数：{posts['title'].nunique():,}")
print(f"posts 中 博文id 唯一值数：{posts['博文id'].nunique():,}")

# ── 2. 有效数据过滤 ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. 有效数据过滤（剔除水军 / 垃圾）")
print("=" * 60)

# posts：剔除 is_bot_post=Y 或 is_spam_post=Y
before_p = len(posts)
posts_valid = posts[
    (posts["is_bot_post"].astype(str).str.upper() != "Y") &
    (posts["is_spam_post"].astype(str).str.upper() != "Y") &
    posts["发布时间"].notna()
].copy()
print(f"有效 posts：{len(posts_valid):,}（原 {before_p:,}，剔除 {before_p - len(posts_valid):,}）")

# comments：剔除 is_bot_comment=Y 或 is_spam_comment=Y
before_c = len(comments)
comments_valid = comments[
    (comments["is_bot_comment"].astype(str).str.upper() != "Y") &
    (comments["is_spam_comment"].astype(str).str.upper() != "Y") &
    comments["评论时间"].notna()
].copy()
print(f"有效 comments：{len(comments_valid):,}（原 {before_c:,}，剔除 {before_c - len(comments_valid):,}）")

# ── 3. 将 comments 关联到 title ────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 将 comments 关联到热搜话题（title）")
print("=" * 60)

# 构建 博文id → title 的映射（用有效 posts）
post_to_title = posts_valid[["博文id", "title", "source_md"]].drop_duplicates("博文id")
comments_valid = comments_valid.merge(post_to_title, on="博文id", how="left")

n_matched = comments_valid["title"].notna().sum()
n_unmatched = comments_valid["title"].isna().sum()
print(f"评论成功匹配到 title：{n_matched:,}（未匹配：{n_unmatched:,}，{n_unmatched/len(comments_valid)*100:.1f}%）")
print(f"注：未匹配原因通常是对应 post 被水军过滤或未被抓取")

comments_valid = comments_valid.dropna(subset=["title"])

# ── 4. 合并 posts 和 comments 成统一时间序列 ──────────────────────────────
print("\n" + "=" * 60)
print("4. 构建合并时间序列（posts + comments）")
print("=" * 60)

# 构建统一的观测记录：每条记录有 title、时间戳、stance_score、sentiment_score、数据来源
posts_long = posts_valid[["title", "source_md", "发布时间", "stance_score", "sentiment_score"]].copy()
posts_long.rename(columns={"发布时间": "时间戳"}, inplace=True)
posts_long["来源"] = "post"

comments_long = comments_valid[["title", "source_md", "评论时间", "stance_score", "sentiment_score"]].copy()
comments_long.rename(columns={"评论时间": "时间戳"}, inplace=True)
comments_long["来源"] = "comment"

combined = pd.concat([posts_long, comments_long], ignore_index=True)
combined = combined.dropna(subset=["时间戳", "stance_score", "sentiment_score"])
print(f"合并后总观测数：{len(combined):,}（posts: {len(posts_long):,}，comments: {len(comments_long):,}）")
print(f"覆盖热搜话题数：{combined['title'].nunique():,}")

# ── 5. t₀ 推断：每个话题最早的 post 发布时间 ──────────────────────────────
t0 = posts_valid.groupby("title")["发布时间"].min().reset_index()
t0.rename(columns={"发布时间": "t0"}, inplace=True)
combined = combined.merge(t0, on="title", how="left")
combined["时间戳"] = pd.to_datetime(combined["时间戳"], errors="coerce")
combined["t0"] = pd.to_datetime(combined["t0"], errors="coerce")
combined = combined.dropna(subset=["时间戳", "t0"])
combined["相对小时"] = ((combined["时间戳"] - combined["t0"]).dt.total_seconds() / 3600).fillna(-1).astype(int)

# 仅保留 t >= 0（t₀ 之后）
combined = combined[combined["相对小时"] >= 0]
print(f"过滤 t<0 后剩余观测数：{len(combined):,}")

# ── 6. 话题层面基本统计 ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. 话题（title）层面基本统计")
print("=" * 60)

topic_stats = combined.groupby("title").agg(
    总观测数=("stance_score", "count"),
    post数=("来源", lambda x: (x == "post").sum()),
    comment数=("来源", lambda x: (x == "comment").sum()),
    时间跨度_小时=("相对小时", "max"),
).reset_index()

print(f"话题总数：{len(topic_stats):,}")
print(f"\n每话题总观测数（posts+comments）分布：")
print(topic_stats["总观测数"].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))
print(f"\n每话题 comment 数分布：")
print(topic_stats["comment数"].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))
print(f"\n每话题时间跨度（小时）分布：")
print(topic_stats["时间跨度_小时"].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

# 有多少话题有至少1条 comment
n_has_comment = (topic_stats["comment数"] > 0).sum()
print(f"\n有评论数据的话题：{n_has_comment:,}（占 {n_has_comment/len(topic_stats)*100:.1f}%）")

# ── 7. 时间窗口分析（1h 和 2h 两种窗口）──────────────────────────────────
print("\n" + "=" * 60)
print("6. 时间窗口分析")
print("=" * 60)

def analyze_windows(df, window_h, min_obs_per_window, min_windows_per_topic):
    """对给定窗口宽度和最低观测数，计算满足条件的话题数"""
    df = df.copy()
    df["窗口"] = df["相对小时"] // window_h
    window_counts = df.groupby(["title", "窗口"]).size().reset_index(name="窗口观测数")
    valid_w = window_counts[window_counts["窗口观测数"] >= min_obs_per_window]
    valid_per_topic = valid_w.groupby("title").size().reset_index(name="有效窗口数")
    eligible = valid_per_topic[valid_per_topic["有效窗口数"] >= min_windows_per_topic]
    return valid_per_topic, eligible

for wh in [1, 2, 4]:
    vpt, elig = analyze_windows(combined, window_h=wh, min_obs_per_window=5, min_windows_per_topic=6)
    print(f"\n  窗口宽度={wh}h，每窗口≥5条，≥6个有效窗口：")
    print(f"    满足条件话题数：{len(elig):,}")
    print(f"    有效窗口数分布（仅有效话题内）：")
    merged = vpt[vpt["title"].isin(elig["title"])]
    print(f"      中位数 {merged['有效窗口数'].median():.0f}  P75={merged['有效窗口数'].quantile(.75):.0f}  P90={merged['有效窗口数'].quantile(.90):.0f}  最大={merged['有效窗口数'].max():.0f}")

# 主力方案：2h 窗口，≥6个有效窗口
vpt_2h, elig_2h = analyze_windows(combined, window_h=2, min_obs_per_window=5, min_windows_per_topic=6)
print(f"\n【主力方案：2h 窗口，≥6个有效窗口】话题数：{len(elig_2h):,}")

# 阈值敏感性
print(f"\n不同「最少有效窗口数」阈值（2h 窗口）：")
for threshold in [3, 4, 5, 6, 8, 10, 12]:
    n = (vpt_2h["有效窗口数"] >= threshold).sum()
    print(f"  ≥{threshold:>2} 个窗口：{n:>4,} 个话题")

# ── 8. 早期评论定义（在 eligible topics 上）──────────────────────────────
print("\n" + "=" * 60)
print("7. 早期评论定义（在 2h 窗口可行话题上）")
print("=" * 60)

elig_titles = set(elig_2h["title"])
df_elig = combined[combined["title"].isin(elig_titles)].copy()
print(f"可行话题数：{len(elig_titles):,}，对应观测数：{len(df_elig):,}")

for early_def, label in [("前2小时", 2), ("前4小时", 4)]:
    early = df_elig[df_elig["相对小时"] < label]
    es = early.groupby("title")["stance_score"].mean()
    n_covered = es.notna().sum()
    print(f"\nearlyStance（{early_def}）：覆盖 {n_covered}/{len(elig_titles)} 个话题")
    print(es.describe(percentiles=[.1, .25, .5, .75, .9]).to_string())

# 前10%（按话题总观测数）
def get_early_pct(group, pct=0.10):
    n = max(1, int(len(group) * pct))
    return group.nsmallest(n, "时间戳")

early_10 = df_elig.groupby("title", group_keys=False).apply(get_early_pct)
es_10 = early_10.groupby("title")["stance_score"].mean()
print(f"\nearlyStance（前10%）：覆盖 {es_10.notna().sum()}/{len(elig_titles)} 个话题")
print(es_10.describe(percentiles=[.1, .25, .5, .75, .9]).to_string())

# ── 9. posts vs comments 数据来源对比 ────────────────────────────────────
print("\n" + "=" * 60)
print("8. 数据来源构成（可行话题内）")
print("=" * 60)

source_stats = df_elig.groupby("title")["来源"].value_counts().unstack(fill_value=0)
if "post" not in source_stats.columns:
    source_stats["post"] = 0
if "comment" not in source_stats.columns:
    source_stats["comment"] = 0
source_stats["comment_ratio"] = source_stats["comment"] / (source_stats["post"] + source_stats["comment"])
print(f"每话题 comment 占比分布：")
print(source_stats["comment_ratio"].describe(percentiles=[.1, .25, .5, .75, .9]).to_string())
print(f"\ncomment 占比中位数：{source_stats['comment_ratio'].median():.1%}")
print(f"（说明合并序列中 posts 和 comments 各自的权重）")

# ── 10. 窗口缺口率（2h 窗口，eligible topics）──────────────────────────
print("\n" + "=" * 60)
print("9. 窗口连续性（2h 窗口，eligible topics）")
print("=" * 60)

df_elig["窗口_2h"] = df_elig["相对小时"] // 2
window_counts_2h = df_elig.groupby(["title", "窗口_2h"]).size().reset_index(name="n")
valid_w_2h = window_counts_2h[window_counts_2h["n"] >= 5]

gap_rates = []
for title, grp in valid_w_2h.groupby("title"):
    ws = set(grp["窗口_2h"].values)
    max_w = max(ws)
    gap_rate = len(set(range(0, max_w + 1)) - ws) / max(1, max_w + 1)
    gap_rates.append(gap_rate)

gap_series = pd.Series(gap_rates)
print(f"窗口缺口率分布（2h 窗口）：")
print(gap_series.describe(percentiles=[.25, .5, .75, .9]).to_string())
print(f"\n对比：1h 窗口时缺口率中位数约 50%（上一版脚本），2h 窗口预期改善")

# ── 11. 图：话题层级窗口数 & 观测数分布 ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.hist(vpt_2h["有效窗口数"].clip(upper=60), bins=40, color="#4C72B0", edgecolor="white", lw=0.4)
ax.axvline(6, color="red", linestyle="--", lw=1.5, label="threshold=6")
ax.set_title("Valid Windows per Topic (2h)", fontsize=11)
ax.set_xlabel("# valid 2h-windows", fontsize=10)
ax.set_ylabel("# topics", fontsize=10)
ax.legend(fontsize=9)

ax = axes[1]
ax.hist(topic_stats["总观测数"].clip(upper=5000), bins=50, color="#55A868", edgecolor="white", lw=0.4)
ax.set_yscale("log")
ax.set_title("Total Observations per Topic (posts+comments)", fontsize=11)
ax.set_xlabel("# observations (capped at 5000)", fontsize=10)
ax.set_ylabel("# topics (log)", fontsize=10)

ax = axes[2]
ax.hist(gap_series, bins=30, color="#C44E52", edgecolor="white", lw=0.4)
ax.axvline(gap_series.median(), color="navy", linestyle="--", lw=1.5,
           label=f"median={gap_series.median():.2f}")
ax.set_title("Window Gap Rate (2h, eligible topics)", fontsize=11)
ax.set_xlabel("gap rate (fraction of hours without valid window)", fontsize=10)
ax.set_ylabel("# topics", fontsize=10)
ax.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(FIG, "explore_topic_time_structure.pdf")
plt.savefig(out, dpi=150)
plt.close()
print(f"\n图表已保存：{out}")

print("\n" + "=" * 60)
print("探索完成。")
print("=" * 60)
