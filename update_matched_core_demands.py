"""
根据新核心诉求（core_demand_onlycore）及其 keywords，
重新匹配人民日报新闻，替换 matched_core_demands 列。

匹配规则：某新核心诉求对应的 keywords 中，至少有 2 个不同关键词
         同时出现在新闻的 matched_keywords 里 → 该新核心诉求匹配成功。

输出：data/raw/人民日报_关键词匹配结果_新诉求.csv
"""

import os
import pandas as pd
from collections import defaultdict

RAW = "data/raw"

# ── 1. 加载新核心诉求文件 ─────────────────────────────────────────────────────
print("加载新核心诉求文件…")
lkw = pd.read_csv(
    os.path.join(RAW, "热搜核心诉求_LLM提取结果_融合仅核心.csv"),
    encoding="utf-8-sig",
    usecols=["keywords", "core_demand_onlycore"],
    dtype=str,
).dropna(subset=["keywords", "core_demand_onlycore"])

# 去掉"无具体核心诉求"
lkw = lkw[lkw["core_demand_onlycore"].str.strip() != "无具体核心诉求"].copy()
lkw["keywords_list"] = lkw["keywords"].str.split("|")
print(f"  有效新核心诉求数：{len(lkw):,}（去除无诉求后）")

# ── 2. 构建倒排索引：keyword → set(新核心诉求) ─────────────────────────────
print("构建倒排索引…")
inverted: dict[str, list[str]] = defaultdict(list)
for _, row in lkw.iterrows():
    demand = row["core_demand_onlycore"].strip()
    for kw in row["keywords_list"]:
        kw = kw.strip()
        if kw:
            inverted[kw].append(demand)

print(f"  索引关键词数：{len(inverted):,}")

# ── 3. 加载人民日报数据（流式处理，减少内存） ─────────────────────────────
print("处理人民日报数据（逐块）…")
IN_PATH  = os.path.join(RAW, "人民日报_关键词匹配结果.csv")
OUT_PATH = os.path.join(RAW, "人民日报_关键词匹配结果_新诉求.csv")

CHUNK_SIZE = 5_000
first_chunk = True
total_rows  = 0
matched_rows = 0

reader = pd.read_csv(
    IN_PATH,
    encoding="utf-8-sig",
    dtype=str,
    chunksize=CHUNK_SIZE,
)

for chunk_idx, chunk in enumerate(reader):
    chunk = chunk.copy()
    new_demands_col = []

    for matched_kws_raw in chunk["matched_keywords"]:
        # 该文章命中的关键词集合
        if pd.isna(matched_kws_raw) or str(matched_kws_raw).strip() == "":
            new_demands_col.append("")
            continue

        article_kws = set(k.strip() for k in str(matched_kws_raw).split("/") if k.strip())

        # 统计每个新核心诉求被命中的关键词数
        demand_hit_count: dict[str, int] = defaultdict(int)
        for kw in article_kws:
            for demand in inverted.get(kw, []):
                demand_hit_count[demand] += 1

        # 保留命中 ≥ 1 个关键词的新核心诉求
        matched_demands = [d for d, cnt in demand_hit_count.items() if cnt >= 1]
        new_demands_col.append("/".join(sorted(matched_demands)))

    chunk["matched_core_demands"] = new_demands_col
    chunk = chunk[chunk["matched_core_demands"] != ""].copy()

    total_rows  += len(chunk)
    matched_rows += len(chunk)

    chunk.to_csv(
        OUT_PATH,
        mode="w" if first_chunk else "a",
        index=False,
        encoding="utf-8-sig",
        header=first_chunk,
    )
    first_chunk = False

    if (chunk_idx + 1) % 10 == 0:
        print(f"  已处理 {total_rows:,} 行…", flush=True)

# ── 4. 统计 ─────────────────────────────────────────────────────────────────
print(f"\n完成！")
print(f"  总行数：{total_rows:,}")
print(f"  有新核心诉求匹配的行数：{matched_rows:,}（{matched_rows/total_rows:.1%}）")
print(f"  输出：{OUT_PATH}")
