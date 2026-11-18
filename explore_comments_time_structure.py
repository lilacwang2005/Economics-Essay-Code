"""
探索 comments_BERT.csv 的时间结构，评估研究目的二（锚定效应）的数据可行性。

分析单元：博文id（每条微博下的评论构成一个讨论池，对应"话题"）
关键问题：
  1. 时间戳格式与范围是否正常？
  2. 每个博文id下评论数量的分布？
  3. 以1小时为窗口，每个博文能构建多少个有效窗口（≥5条有效评论）？
  4. 满足"≥6个有效窗口"的博文有多少？（这决定模型是否可行）
  5. "早期评论"的定义验证：前2小时/前10%各有多少评论？
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ── 路径 ──────────────────────────────────────────────────────────────────────
RAW = "/Users/wangmanlin/Desktop/citizens-will-and-politics/data/raw"
FIG = "/Users/wangmanlin/Desktop/citizens-will-and-politics/results/figures"
os.makedirs(FIG, exist_ok=True)

COMMENTS_PATH = os.path.join(RAW, "微博热搜comments_BERT.csv")

# ── 1. 加载数据 ───────────────────────────────────────────────────────────────
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

df = pd.read_csv(COMMENTS_PATH, encoding="utf-8-sig", low_memory=False)
print(f"总行数：{len(df):,}")
print(f"字段：{list(df.columns)}")
print(f"\n各字段缺失值：")
print(df.isnull().sum())

# ── 2. 时间戳解析 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. 时间戳解析")
print("=" * 60)

df["评论时间"] = pd.to_datetime(df["评论时间"], errors="coerce")
n_bad_ts = df["评论时间"].isnull().sum()
print(f"时间戳解析失败行数：{n_bad_ts:,}（占比 {n_bad_ts/len(df)*100:.2f}%）")
print(f"时间范围：{df['评论时间'].min()} → {df['评论时间'].max()}")
print(f"\n按年份分布：")
print(df["评论时间"].dt.year.value_counts().sort_index())

# 丢弃时间戳缺失行
df = df.dropna(subset=["评论时间"])

# ── 3. 有效评论过滤（剔除水军/垃圾）────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 有效评论过滤")
print("=" * 60)

# is_bot_comment / is_spam_comment 字段取值情况
for col in ["is_bot_comment", "is_spam_comment"]:
    if col in df.columns:
        print(f"{col} 取值分布：")
        print(df[col].value_counts(dropna=False).to_string())
        print()

# 保留有效评论（非水军且非垃圾）
before = len(df)
if "is_bot_comment" in df.columns and "is_spam_comment" in df.columns:
    df_valid = df[
        (df["is_bot_comment"].astype(str).str.upper() != "Y") &
        (df["is_spam_comment"].astype(str).str.upper() != "Y")
    ].copy()
else:
    df_valid = df.copy()
print(f"过滤后剩余有效评论：{len(df_valid):,}（原 {before:,}，剔除 {before - len(df_valid):,}）")

# ── 4. 博文级别基本统计 ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. 博文级别基本统计")
print("=" * 60)

post_stats = df_valid.groupby("博文id").agg(
    评论数=("评论id", "count"),
    最早评论=("评论时间", "min"),
    最晚评论=("评论时间", "max"),
).reset_index()
post_stats["时间跨度_小时"] = (
    post_stats["最晚评论"] - post_stats["最早评论"]
).dt.total_seconds() / 3600

print(f"博文（话题）总数：{len(post_stats):,}")
print(f"\n每博文评论数分布：")
print(post_stats["评论数"].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))
print(f"\n评论时间跨度（小时）分布：")
print(post_stats["时间跨度_小时"].describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

# ── 5. 时间窗口分析（1小时窗口）──────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. 时间窗口分析（1小时窗口，每窗口≥5条评论视为有效）")
print("=" * 60)

MIN_COMMENTS_PER_WINDOW = 5
MIN_WINDOWS_PER_POST = 6

# 为每条评论计算相对于该博文最早评论的小时窗口编号
df_valid = df_valid.merge(
    post_stats[["博文id", "最早评论"]],
    on="博文id", how="left"
)
df_valid["相对小时"] = (
    (df_valid["评论时间"] - df_valid["最早评论"]).dt.total_seconds() / 3600
).astype(int)  # floor to hour bin

# 每博文 × 每窗口的评论数
window_counts = (
    df_valid.groupby(["博文id", "相对小时"])
    .size()
    .reset_index(name="窗口评论数")
)
# 有效窗口（评论数≥5）
valid_windows = window_counts[window_counts["窗口评论数"] >= MIN_COMMENTS_PER_WINDOW]
valid_windows_per_post = valid_windows.groupby("博文id").size().reset_index(name="有效窗口数")

print(f"\n每博文有效窗口数分布：")
print(valid_windows_per_post["有效窗口数"].describe(percentiles=[.25, .5, .75, .9, .95]))

# 满足≥6个有效窗口的博文
eligible_posts = valid_windows_per_post[
    valid_windows_per_post["有效窗口数"] >= MIN_WINDOWS_PER_POST
]
print(f"\n满足 ≥{MIN_WINDOWS_PER_POST} 个有效窗口的博文：{len(eligible_posts):,} 个")
print(f"占全部博文比例：{len(eligible_posts)/len(post_stats)*100:.1f}%")
print(f"对应有效评论数：{df_valid[df_valid['博文id'].isin(eligible_posts['博文id'])].shape[0]:,}")

# 不同阈值的敏感性
print(f"\n不同「最少有效窗口数」阈值下满足条件的博文数：")
for threshold in [3, 4, 5, 6, 8, 10, 12]:
    n = (valid_windows_per_post["有效窗口数"] >= threshold).sum()
    print(f"  ≥{threshold:>2} 个窗口：{n:>5,} 篇博文（{n/len(post_stats)*100:.1f}%）")

# ── 6. 早期评论定义验证 ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. 早期评论定义验证")
print("=" * 60)

# 仅在eligible posts中分析
df_elig = df_valid[df_valid["博文id"].isin(eligible_posts["博文id"])].copy()

# 方案A：前2小时
early_2h = df_elig[df_elig["相对小时"] < 2]
# 方案B：前10%（按博文）
def early_pct(group, pct=0.10):
    n = max(1, int(len(group) * pct))
    return group.nsmallest(n, "评论时间")

early_10pct = df_elig.groupby("博文id", group_keys=False).apply(early_pct)

print(f"可行样本博文数：{len(eligible_posts):,}")
print(f"早期评论定义 A（前2小时）：共 {len(early_2h):,} 条（均值 {len(early_2h)/len(eligible_posts):.1f} 条/博文）")
print(f"早期评论定义 B（前10%）：共 {len(early_10pct):,} 条（均值 {len(early_10pct)/len(eligible_posts):.1f} 条/博文）")

# early stance 的分布
for name, early_df in [("前2小时", early_2h), ("前10%", early_10pct)]:
    if "stance_score" in early_df.columns:
        early_stance = early_df.groupby("博文id")["stance_score"].mean()
        print(f"\nearlyStance 分布（{name}）：")
        print(early_stance.describe(percentiles=[.1, .25, .5, .75, .9]))

# ── 7. 图：每博文有效窗口数分布 ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左：有效窗口数分布直方图
ax = axes[0]
ax.hist(
    valid_windows_per_post["有效窗口数"].clip(upper=48),
    bins=48, color="#4C72B0", edgecolor="white", linewidth=0.4
)
ax.axvline(MIN_WINDOWS_PER_POST, color="red", linestyle="--", linewidth=1.5,
           label=f"最低阈值（{MIN_WINDOWS_PER_POST}个窗口）")
ax.set_xlabel("有效时间窗口数（每博文）", fontsize=11)
ax.set_ylabel("博文数", fontsize=11)
ax.set_title("每博文有效时间窗口数分布\n（窗口宽度=1小时，每窗口≥5条有效评论）", fontsize=11)
ax.legend(fontsize=9)
ax.xaxis.set_major_locator(mticker.MultipleLocator(6))

# 右：评论数分布（log scale）
ax = axes[1]
ax.hist(
    post_stats["评论数"].clip(upper=2000),
    bins=50, color="#55A868", edgecolor="white", linewidth=0.4
)
ax.set_xlabel("每博文评论数（截断至2000）", fontsize=11)
ax.set_ylabel("博文数", fontsize=11)
ax.set_title("每博文有效评论数分布", fontsize=11)
ax.set_yscale("log")

plt.tight_layout()
out_path = os.path.join(FIG, "explore_comment_time_structure.pdf")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\n图表已保存：{out_path}")

# ── 8. 时序连续性检验（eligible posts）──────────────────────────────────────
print("\n" + "=" * 60)
print("7. 时序连续性（eligible posts 中，相邻窗口缺口检验）")
print("=" * 60)

# 对eligible posts，检查从窗口0到最大窗口之间缺失窗口的比例
gap_rates = []
for post_id, grp in valid_windows[valid_windows["博文id"].isin(eligible_posts["博文id"])].groupby("博文id"):
    windows = set(grp["相对小时"].values)
    max_w = max(windows)
    all_possible = set(range(0, max_w + 1))
    gap_rate = len(all_possible - windows) / max(1, len(all_possible))
    gap_rates.append(gap_rate)

gap_rates = pd.Series(gap_rates)
print(f"窗口缺口率（相邻小时中无有效窗口的比例）分布：")
print(gap_rates.describe(percentiles=[.25, .5, .75, .9]))
print(f"\n> 若中位数缺口率较高，建议放宽窗口宽度至2小时")

print("\n" + "=" * 60)
print("探索完成。")
print("=" * 60)
