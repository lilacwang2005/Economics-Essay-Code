"""
研究目的一（政策落实）数据清洗主脚本
整合自变量构建（原 build_purpose1_vars.py）和因变量构建（原 build_purpose1_dv.py）

输入（只读）：
  data/raw/微博热搜博文_筛选后待爬评论.csv
  data/raw/热搜核心诉求_LLM提取结果_融合仅核心.csv
  data/raw/微博热搜posts_analyzed.csv
  data/raw/微博热搜comments_BERT.csv
  data/clean/topic_level.csv
  data/clean/topic_category_labels.csv
  data/clean/人民日报_BERT推理结果.csv

输出：
  data/clean/purpose1_topic_vars.csv   — 话题层面自变量（含 DaysToSolve_i / Censored_i，每话题一行）
  data/clean/purpose1_event_panel.csv  — 话题 × 日 事件研究面板（含因变量与自变量）

BERT solve_score 含义：
  0=无关, 20=仅讨论, 40=正在推进, 60=部分解决, 80=大部分解决, 100=已完全解决
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

RAW   = "data/raw"
CLEAN = "data/clean"

# ══════════════════════════════════════════════════════════════════════════════
# 0. 有效话题白名单（两个脚本共用）
# ══════════════════════════════════════════════════════════════════════════════
print("══ 0. 加载话题白名单 ══")
valid_titles = set(
    pd.read_csv(os.path.join(RAW, "微博热搜博文_筛选后待爬评论.csv"),
                encoding="utf-8-sig", usecols=["title"])
    ["title"].dropna().unique()
)
print(f"  白名单话题数：{len(valid_titles):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. 话题基础信息（含 hot_date，作为事件时间基准）
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 1. 加载 topic_level.csv ══")
tl = pd.read_csv(os.path.join(CLEAN, "topic_level.csv"), encoding="utf-8-sig")
tl = tl[tl["title"].isin(valid_titles)].copy()
tl["上榜日期"] = pd.to_datetime(tl["上榜日期"], errors="coerce")
tl["上榜年份"] = tl["上榜日期"].dt.year
tl["上榜月份"] = tl["上榜日期"].dt.month
print(f"  有效话题数：{len(tl):,}  其中有上榜日期：{tl['上榜日期'].notna().sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. 自变量：帖子数据 → 热度指标
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 2. 加载 posts_analyzed.csv → 热度指标 ══")
posts = pd.read_csv(
    os.path.join(RAW, "微博热搜posts_analyzed.csv"),
    encoding="utf-8-sig",
    usecols=["title", "博文id", "转发数", "评论数", "点赞数",
             "is_bot_post", "is_spam_post", "微博正文"],
    dtype=str,
)
posts = posts[posts["title"].isin(valid_titles)].copy()

for col in ["转发数", "评论数", "点赞数"]:
    posts[col] = pd.to_numeric(
        posts[col].astype(str).str.replace(r"[^\d.]", "", regex=True),
        errors="coerce",
    )

# 水军比例（分母为全部帖子）
bot_count   = posts.groupby("title")["is_bot_post"].apply(
    lambda x: (x.fillna("N") == "Y").sum())
total_count = posts.groupby("title").size()
bot_ratio   = (bot_count / total_count).rename("BotRatio_i").reset_index()

# 过滤水军/垃圾后聚合
posts_clean = posts[
    (posts["is_bot_post"].fillna("N") != "Y") &
    (posts["is_spam_post"].fillna("N") != "Y")
].copy()
heat = posts_clean.groupby("title", as_index=False).agg(
    总点赞数     = ("点赞数",   "sum"),
    总转发数     = ("转发数",   "sum"),
    总评论数     = ("评论数",   "sum"),
    NumPosts_i   = ("博文id",  "nunique"),
    TextLength_i = ("微博正文", lambda x: x.dropna().str.len().mean()),
)
heat = heat.merge(bot_ratio, on="title", how="left")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 自变量：评论数据 → 态度 / 情感指标
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 3. 加载 comments_BERT.csv → 态度/情感指标 ══")
comments = pd.read_csv(
    os.path.join(RAW, "微博热搜comments_BERT.csv"),
    encoding="utf-8-sig",
    usecols=["博文id", "stance_score", "sentiment_score",
             "emotion", "is_bot_comment", "is_spam_comment"],
)
id_title = posts[["博文id", "title"]].drop_duplicates("博文id").dropna(subset=["title"])
comments["博文id"]  = comments["博文id"].astype(str)
id_title["博文id"] = id_title["博文id"].astype(str)
comments  = comments.merge(id_title, on="博文id", how="inner")
comments_clean = comments[
    (comments["is_bot_comment"].fillna("N") != "Y") &
    (comments["is_spam_comment"].fillna("N") != "Y")
].copy()
comments_clean["stance_score"]    = pd.to_numeric(comments_clean["stance_score"],    errors="coerce")
comments_clean["sentiment_score"] = pd.to_numeric(comments_clean["sentiment_score"], errors="coerce")

stance_agg = comments_clean.groupby("title", as_index=False).agg(
    Stance_i              = ("stance_score",    "mean"),
    StanceSD_i            = ("stance_score",    "std"),
    Sentiment_i           = ("sentiment_score", "mean"),
    StancePositiveRatio_i = ("stance_score",    lambda x: (x.dropna() > 20).mean()),
    AngerRatio_i          = ("emotion",         lambda x: (x == "愤怒").mean()),
)
print(f"  聚合评论话题数：{len(stance_agg):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. 合并输出：话题层面自变量
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 4. 合并话题层面自变量 ══")
base_cols = [c for c in [
    "title", "上榜日期", "上榜年份", "上榜月份",
    "is_government_related", "category_L1", "category_L2",
    "post_count", "comment_count", "duration_hours", "keywords_count",
] if c in tl.columns]

topic_vars = (
    tl[base_cols]
    .merge(heat[["title", "总点赞数", "总转发数", "总评论数",
                 "NumPosts_i", "BotRatio_i", "TextLength_i"]],
           on="title", how="left")
    .merge(stance_agg, on="title", how="left")
)
print(f"  最终话题数：{len(topic_vars):,}")
print(f"  总评论数 非空：{topic_vars['总评论数'].notna().sum():,}  "
      f"Stance_i 非空：{topic_vars['Stance_i'].notna().sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. 加载诉求-话题映射 + BERT 推理结果
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 5. 加载诉求-话题映射 ══")
demands_raw = pd.read_csv(
    os.path.join(RAW, "热搜核心诉求_LLM提取结果_融合仅核心.csv"),
    encoding="utf-8-sig",
    usecols=["title", "core_demand_onlycore"],
    dtype=str,
).rename(columns={"core_demand_onlycore": "core_demand"})
demands_raw = demands_raw[
    demands_raw["core_demand"].notna() &
    (demands_raw["core_demand"].str.strip() != "无具体核心诉求")
].copy()
demands = demands_raw[demands_raw["title"].isin(valid_titles)].copy()
# panel_titles：有具体核心诉求的话题（有人民日报检索路径），是面板/生存分析的实际样本
panel_titles = set(demands["title"].unique())
print(f"  全量诉求行数：{len(demands_raw):,}（去重 core_demand：{demands_raw['core_demand'].nunique():,}）")
print(f"  有效话题诉求行数：{len(demands):,}（去重话题：{len(panel_titles):,}）")

print("\n══ 6. 加载 BERT 推理结果 ══")
llm_path = os.path.join(CLEAN, "人民日报_BERT推理结果.csv")
if not os.path.exists(llm_path):
    raise FileNotFoundError(f"BERT 推理结果尚未生成：{llm_path}")
bert = pd.read_csv(llm_path, encoding="utf-8-sig",
                   usecols=["article_id", "date", "core_demand",
                            "demand_related", "npc_related", "solve_score"])
bert["solve_score"] = pd.to_numeric(bert["solve_score"], errors="coerce")
bert_T = bert[bert["demand_related"] == "T"].copy()
print(f"  总行数：{len(bert):,}  demand_related=T：{len(bert_T):,}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. 诊断：匹配率统计
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 7. 匹配率诊断 ══")
bert_agg = (
    bert_T[bert_T["solve_score"].notna()]
    .groupby("core_demand", as_index=False)
    .agg(
        n_articles      = ("solve_score", "count"),
        max_solve_score = ("solve_score", "max"),
        any_nonzero     = ("solve_score", lambda x: (x > 0).any()),
    )
)

for label, df_d in [("全量话题", demands_raw), ("有效话题（白名单∩有核心诉求）", demands)]:
    diag = df_d.merge(bert_agg, on="core_demand", how="left")
    td = diag.groupby("title").agg(
        has_any_score = ("max_solve_score", lambda x: x.notna().any()),
        has_nonzero   = ("any_nonzero",     lambda x: x.any() if x.notna().any() else False),
    ).reset_index()
    n = len(td)
    print(f"  [{label}] 总话题数：{n:,}")
    print(f"    匹配到任意 solve_score（0-100）：{td['has_any_score'].sum():,}  ({td['has_any_score'].mean()*100:.1f}%)")
    print(f"    匹配到 solve_score > 0：          {td['has_nonzero'].sum():,}  ({td['has_nonzero'].mean()*100:.1f}%)")

score_dist = bert_T["solve_score"].value_counts(dropna=False).sort_index()
print(f"\n  BERT solve_score 分布（demand_related=T，共 {len(bert_T):,} 条）：")
for v, c in score_dist.items():
    print(f"    {str(v):>6} : {c:>8,}  ({c/len(bert_T)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 8. 构建文章-话题-日期数据集
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 8. 构建文章-话题关联 ══")
bert_valid = bert_T[bert_T["solve_score"].notna()].copy()

bert_valid["article_date"] = pd.to_datetime(bert_valid["date"], errors="coerce", dayfirst=False)
mask_fail = bert_valid["article_date"].isna()
if mask_fail.any():
    bert_valid.loc[mask_fail, "article_date"] = pd.to_datetime(
        bert_valid.loc[mask_fail, "date"], errors="coerce", format="%Y/%m/%d"
    )
bert_valid = bert_valid[bert_valid["article_date"].notna()].copy()
print(f"  有效 solve_score 且日期解析成功：{len(bert_valid):,} 条")

# core_demand → title（仅有效话题）
llm = bert_valid.merge(
    demands[["title", "core_demand"]].drop_duplicates(),
    on="core_demand", how="inner",
)
print(f"  关联到有效话题后：{len(llm):,} 条，涉及话题：{llm['title'].nunique():,} 个")

# 关联 hot_date
hot_dates = (
    tl[tl["上榜日期"].notna()][["title", "上榜日期"]]
    .rename(columns={"上榜日期": "hot_date"})
    .drop_duplicates("title")
)
llm = llm.merge(hot_dates, on="title", how="inner")
llm["rel_days"] = (llm["article_date"] - llm["hot_date"]).dt.days
print(f"  关联热搜日期后：{len(llm):,} 条，涉及话题：{llm['title'].nunique():,} 个")

# matched_titles：实际关联到人民日报文章的话题（530 个），作为最终回归样本
matched_titles = set(llm["title"].unique())
print(f"  最终回归样本（matched_titles）：{len(matched_titles):,} 个话题")

# ══════════════════════════════════════════════════════════════════════════════
# 9. 生存分析变量：DaysToSolve_i（Cox 模型用）
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 9. 构建 DaysToSolve_i ══")
CENSOR_DATE = pd.Timestamp("2026-03-31")   # 右删失截止日

# 上热搜后首条 solve_score >= 40 的文章日期
post_action = llm[(llm["rel_days"] >= 0) & (llm["solve_score"] >= 40)].copy()
first_action = (
    post_action.groupby("title")["article_date"].min()
    .rename("first_action_date")
    .reset_index()
)

# 合并到 530 个实际匹配话题（无 solve_score≥40 记录则视为右删失）
survival = hot_dates[hot_dates["title"].isin(matched_titles)].merge(
    first_action, on="title", how="left"
)
survival["DaysToSolve_i"] = (survival["first_action_date"] - survival["hot_date"]).dt.days
survival["Censored_i"] = survival["first_action_date"].isna().astype(int)
# 右删失话题：用截止日 - 上热搜日作为观测时长上限
mask_censor = survival["Censored_i"] == 1
survival.loc[mask_censor, "DaysToSolve_i"] = (
    CENSOR_DATE - survival.loc[mask_censor, "hot_date"]
).dt.days

n_solved   = (survival["Censored_i"] == 0).sum()
n_censored = (survival["Censored_i"] == 1).sum()
print(f"  已落实（solve_score≥40）：{n_solved:,}  右删失：{n_censored:,}")
print(f"  DaysToSolve_i 中位数（已落实）："
      f"{survival[survival['Censored_i']==0]['DaysToSolve_i'].median():.0f} 天")

# ══════════════════════════════════════════════════════════════════════════════
# 10. 事件研究面板（话题 × 日，完整日历骨架）
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 10. 构建事件研究面板 ══")
REL_DAY_MIN, REL_DAY_MAX = -365, 730   # 前1年到后2年

# 按 (title, article_date) 聚合同一天多条记录
daily = (
    llm.groupby(["title", "article_date", "hot_date"])
    .agg(
        y_max_score = ("solve_score", "max"),
        y_count     = ("solve_score", "count"),
        rel_days    = ("rel_days",    "first"),
    )
    .reset_index()
)
daily = daily[(daily["rel_days"] >= REL_DAY_MIN) & (daily["rel_days"] <= REL_DAY_MAX)].copy()

# 完整日历骨架：仅包含实际关联到人民日报文章的话题（530 个）
# 有 core_demand 但未匹配到任何人民日报文章的话题（共 2 个）同样无法构建因变量，
# 不纳入面板，确保回归样本一致。
all_hot = hot_dates[hot_dates["title"].isin(matched_titles)].copy()
print(f"  骨架话题数：{len(all_hot):,}（实际关联到人民日报文章的话题）")

skeleton_rows = []
for _, row in all_hot.iterrows():
    t, hd = row["title"], row["hot_date"]
    dates = pd.date_range(hd + pd.Timedelta(days=REL_DAY_MIN),
                          hd + pd.Timedelta(days=REL_DAY_MAX), freq="D")
    skeleton_rows.append(pd.DataFrame({
        "title":        t,
        "article_date": dates,
        "hot_date":     hd,
        "rel_days":     range(REL_DAY_MIN, REL_DAY_MAX + 1),
    }))
skeleton = pd.concat(skeleton_rows, ignore_index=True)

panel = skeleton.merge(
    daily[["title", "article_date", "y_max_score", "y_count"]],
    on=["title", "article_date"], how="left",
)
panel["y_max_score"] = panel["y_max_score"].fillna(0)
panel["y_count"]     = panel["y_count"].fillna(0).astype(int)

panel = panel.sort_values(["title", "article_date"])
panel["y_cummax_score"] = panel.groupby("title")["y_max_score"].cummax()

# PreBaseline_i：上热搜前365天内匹配到的相关文章数（话题层面控制变量）
pre_baseline = (
    llm[(llm["rel_days"] >= -365) & (llm["rel_days"] < 0)]
    .groupby("title")["solve_score"].count()
    .rename("PreBaseline_i")
    .reset_index()
)

# SolveDegree_score：上热搜后匹配到的最高 solve_score（话题层面截面 DV）
solve_deg = (
    llm[llm["rel_days"] >= 0]
    .groupby("title")["solve_score"].max()
    .rename("SolveDegree_score")
    .reset_index()
)

# 合并 DaysToSolve_i / Censored_i / SolveDegree_score / PreBaseline_i → 保存 topic_vars
survival_cols = survival[["title", "DaysToSolve_i", "Censored_i"]]
topic_vars = (
    topic_vars
    .merge(survival_cols, on="title", how="left")
    .merge(solve_deg, on="title", how="left")
    .merge(pre_baseline, on="title", how="left")
)
topic_vars["PreBaseline_i"] = topic_vars["PreBaseline_i"].fillna(0).astype(int)
topic_vars.to_csv(os.path.join(CLEAN, "purpose1_topic_vars.csv"),
                  index=False, encoding="utf-8-sig")
print("✓ 已保存 data/clean/purpose1_topic_vars.csv")

panel = panel.merge(topic_vars, on="title", how="left")

panel["is_base"] = (panel["rel_days"] == -1).astype(int)

n_topics       = panel["title"].nunique()
n_with_article = panel[panel["y_count"] > 0]["title"].nunique()
print(f"  面板行数：{len(panel):,}（{n_topics} 个话题 × {REL_DAY_MAX - REL_DAY_MIN + 1} 天）")
print(f"  有人民日报匹配文章的话题：{n_with_article:,} / {n_topics:,}")

panel.to_csv(os.path.join(CLEAN, "purpose1_event_panel.csv"), index=False, encoding="utf-8-sig")
print("✓ 已保存 data/clean/purpose1_event_panel.csv")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
