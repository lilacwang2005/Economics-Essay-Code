"""
用 DeepSeek API 对白名单全部热搜话题进行两步标注（单次 API 调用完成）：
  Step 1：判断该话题是否体现了公众的某种诉求（is_demand: Y/N，标准宽松）
  Step 2：若 Y，进行细粒度领域分类（category_L1 / category_L2）

"无诉求"话题示例（is_demand=N）：
  - 纯会议/活动报道："第十四届全国人民代表大会开幕式"
  - 纯人事任免："某某当选XXX职务"
  - 宣传性新闻无具体问题指向："建国XX周年庆典"
  - 娱乐/体育结果（无政策诉求）

"有诉求"话题示例（is_demand=Y，标准宽松）：
  - 明确政策建议/问题反映
  - 涉及民生痛点（即使是新闻报道形式，背后有公众关切）
  - 对某一社会现象的讨论（哪怕尚未形成明确建议）

输入：
  data/raw/微博热搜博文_筛选后待爬评论.csv   （白名单，提供全部待标注 title）
  data/raw/微博热搜posts_analyzed.csv         （每话题取点赞最高 5 条 post 作上下文）
  data/raw/热搜核心诉求_LLM提取结果_融合仅核心.csv  （core_demand_onlycore 列，已有核心诉求的话题作参考）

输出：
  data/clean/topic_category_labels.csv
    列：title | is_demand | category_L1 | category_L2 | reasoning

设计要点：
  - 7 个 API key 随机轮换，10 线程并发
  - 断点续跑，线程安全写入，自动重试失败条目
"""

import os, json, time, random, threading
import pandas as pd
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── API Keys（按顺序填入，脚本随机抽取）───────────────────────────────────
API_KEYS = [
    "sk-2171370a80414a6baf53668bf1600d15",
    "sk-fee82228633c42f4b040c4dae3287c30",
    "sk-644a8078ba114a30aa69f05e2063c354",
    "sk-d84e357c9d714becb8a8a9a5d0d9aaac",
    "sk-9cb5dc1255794cdc9776a8f82c41ed96",
    "sk-9b82f376b12a4c0bb91d4fb5d3d25abe",
    "sk-2ef731f4284c45fc91fc98b83f67d4d5"
]

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MODEL        = "deepseek-chat"
MAX_RETRIES  = 3
MAX_WORKERS  = 10    # 并发线程数
MAX_POSTS    = 5     # 每话题最多取几条 post 作上下文
MAX_POST_LEN = 150   # 每条 post 截断字数

RAW      = "/Users/wangmanlin/Desktop/citizens-will-and-politics/data/raw"
CLEAN    = "/Users/wangmanlin/Desktop/citizens-will-and-politics/data/clean"

# ─── 输入文件 ────────────────────────────────────────────────────────────────
IN_WHITELIST = os.path.join(RAW,   "微博热搜博文_筛选后待爬评论.csv")
IN_POSTS     = os.path.join(RAW,   "微博热搜posts_analyzed.csv")
IN_DEMANDS   = os.path.join(RAW,   "热搜核心诉求_LLM提取结果_融合仅核心.csv")  # 使用 core_demand_onlycore 列
OUT_PATH     = os.path.join(CLEAN, "topic_category_labels.csv")

# ─── 分类体系（修改这里即可调整粒度）────────────────────────────────────────
TAXONOMY = {
    "教育":   ["基础教育", "高等教育", "职业技能教育", "教育公平与资源"],
    "医疗健康": ["医疗改革与医保", "药品与食品安全", "公共卫生与疫情", "心理健康"],
    "就业劳动": ["就业政策", "劳动权益与工时", "职场平等与歧视"],
    "社会保障": ["养老与退休", "生育补贴与托育", "低收入群体保障"],
    "住房":   ["房地产调控", "住房保障与租房"],
    "生态环保": ["环境污染治理", "生态保护与野生动物", "碳中和与气候"],
    "经济产业": ["宏观经济与财政", "产业政策与营商环境", "农业农村", "消费与物价"],
    "科技创新": ["互联网与数字经济", "人工智能", "新能源与先进制造"],
    "法治司法": ["司法改革", "民事与个人权益", "刑事安全与涉罪"],
    "社会治理": ["反腐廉政", "基层治理与信访", "公共安全与应急"],
    "交通出行": ["道路交通安全", "新能源汽车政策", "公共交通与出行"],
    "文化体育": ["文化政策与遗产", "影视娱乐监管", "体育政策"],
    "人口家庭": ["生育政策", "老龄化社会", "婚恋与性别平等"],
    "其他":   ["其他"],
}

ALL_L2 = [f"{l1} / {l2}" for l1, l2_list in TAXONOMY.items() for l2 in l2_list]
L2_TO_L1 = {l2: l1 for l1, l2_list in TAXONOMY.items() for l2 in l2_list}

# ─── 线程安全的写入锁 & 进度计数器 ──────────────────────────────────────────
_write_lock   = threading.Lock()
_counter_lock = threading.Lock()
_done_count   = 0


# ─── 构建 Prompt ─────────────────────────────────────────────────────────────
def build_prompt(title: str, posts_text: list, core_demand) -> str:
    options_str = "\n".join(f"  - {opt}" for opt in ALL_L2)
    posts_str = "\n".join(
        f"  [{i+1}] {p[:MAX_POST_LEN]}{'…' if len(p) > MAX_POST_LEN else ''}"
        for i, p in enumerate(posts_text)
    ) if posts_text else "  （无）"
    demand_str = f"\n【已提取的核心诉求参考】{core_demand}" if core_demand and str(core_demand) != "nan" else ""

    return f"""你是一位中国政策与社会研究专家，需要对微博热搜话题进行两步判断。

【热搜话题标题】
{title}
{demand_str}

【话题下的代表性微博（按点赞数排序）】
{posts_str}

---

【第一步】判断该热搜是否体现了公众的某种诉求或关切。

判断标准（宽松）：
- Y（有诉求）：话题涉及任何社会问题、民生议题、政策期待、权益保护、不满情绪等，即使以新闻报道形式出现、即使诉求尚不明确
- N（无诉求）：纯会议/开幕式报道、纯人事任免公告、纯成就宣传、娱乐明星八卦、体育赛事结果（无政策诉求背景）

【第二步】若第一步为 Y，从下方列表选择最合适的一个分类。若第一步为 N，category_L1 和 category_L2 填 "无"。

【可选分类（格式：一级分类 / 二级分类）】
{options_str}

请输出 JSON，格式如下：
{{
  "is_demand": "Y 或 N",
  "category_L1": "一级分类名称（原文），无诉求时填'无'",
  "category_L2": "二级分类名称（原文），无诉求时填'无'",
  "reasoning": "一句话说明判断理由（不超过50字）"
}}

注意：
- is_demand 只能是 "Y" 或 "N"
- 分类必须从给定选项中选，不得自创
- 只输出 JSON，不要其他文字"""


# ─── API 调用（含重试）───────────────────────────────────────────────────────
def call_deepseek(prompt: str) -> dict | None:
    for attempt in range(MAX_RETRIES):
        key = random.choice(API_KEYS)  # 每次重试也随机换 key
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }
        try:
            resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return json.loads(resp.json()["choices"][0]["message"]["content"])
        except Exception as e:
            wait = 0.5 * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"  [重试 {attempt+1}/{MAX_RETRIES}] {str(e)[:60]}，等待 {wait:.1f}s")
            time.sleep(wait)
    return None


# ─── 单任务处理函数（供线程池调用）──────────────────────────────────────────
def label_one(title: str, posts_text: list, core_demand) -> dict:
    global _done_count

    prompt = build_prompt(title, posts_text, core_demand)
    result = call_deepseek(prompt)

    if result is None:
        row = {"title": title, "is_demand": "标注失败",
               "category_L1": "标注失败", "category_L2": "标注失败",
               "reasoning": "API调用失败"}
    else:
        is_demand = str(result.get("is_demand", "Y")).strip().upper()
        if is_demand not in ("Y", "N"):
            is_demand = "Y"  # 默认宽松保留

        if is_demand == "N":
            row = {"title": title, "is_demand": "N",
                   "category_L1": "无", "category_L2": "无",
                   "reasoning": str(result.get("reasoning", ""))}
        else:
            l2 = str(result.get("category_L2", "其他")).strip()
            l1 = str(result.get("category_L1", "")).strip()
            # 兜底：L2 不在预定义列表中时模糊匹配
            if l2 not in L2_TO_L1:
                matched = [k for k in L2_TO_L1 if k in l2 or l2 in k]
                l2 = matched[0] if matched else "其他"
            l1 = L2_TO_L1.get(l2, "其他")
            row = {"title": title, "is_demand": "Y",
                   "category_L1": l1, "category_L2": l2,
                   "reasoning": str(result.get("reasoning", ""))}

    # 线程安全写入
    with _write_lock:
        pd.DataFrame([row]).to_csv(OUT_PATH, mode="a", header=False,
                                   index=False, encoding="utf-8-sig")

    # 线程安全计数 & 打印进度
    with _counter_lock:
        _done_count += 1
        cnt = _done_count
    if cnt % 50 == 0 or cnt == 1:
        print(f"  [{cnt}] {title[:35]}… → {row['category_L1']} / {row['category_L2']}")

    return row


# ─── 主流程 ────────────────────────────────────────────────────────────────
def main():
    # 加载话题列表（白名单全量）
    titles_all = (
        pd.read_csv(IN_WHITELIST, encoding="utf-8-sig", usecols=["title"])
        ["title"].dropna().unique().tolist()
    )
    print(f"待标注话题总数：{len(titles_all):,}")

    # 每话题取点赞最高 MAX_POSTS 条 post
    posts_df = pd.read_csv(
        IN_POSTS,
        encoding="utf-8-sig", low_memory=False,
        usecols=["title", "微博正文", "点赞数"]
    )
    posts_df["点赞数"] = pd.to_numeric(posts_df["点赞数"], errors="coerce").fillna(0)
    posts_df["微博正文"] = posts_df["微博正文"].fillna("").astype(str)
    top_posts = (
        posts_df.sort_values("点赞数", ascending=False)
        .groupby("title").head(MAX_POSTS)
        .groupby("title")["微博正文"].apply(list).to_dict()
    )

    # 已有 core_demand
    demand_map = pd.read_csv(
        IN_DEMANDS,
        encoding="utf-8-sig", low_memory=False,
        usecols=["title", "core_demand_onlycore"]
    ).drop_duplicates("title").set_index("title")["core_demand_onlycore"].to_dict()

    # 断点续跑：读取已完成
    done_titles = set()
    if Path(OUT_PATH).exists():
        done_df = pd.read_csv(OUT_PATH, encoding="utf-8-sig")
        # 排除上次失败的条目，允许重跑
        done_titles = set(done_df[done_df["is_demand"] != "标注失败"]["title"].tolist())
        print(f"已完成（含上次成功）：{len(done_titles):,}，本次将重跑失败条目")
    else:
        pd.DataFrame(columns=["title", "is_demand", "category_L1", "category_L2", "reasoning"]
                     ).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    to_label = [t for t in titles_all if t not in done_titles]
    print(f"本次需标注：{len(to_label):,} 个话题，并发线程：{MAX_WORKERS}\n")

    start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                label_one,
                title,
                top_posts.get(title, []),
                demand_map.get(title, None)
            ): title
            for title in to_label
        }
        failed = []
        for future in as_completed(futures):
            row = future.result()
            if row["category_L1"] == "标注失败":
                failed.append(row["title"])

    elapsed = time.time() - start
    print(f"\n标注完成，耗时 {elapsed/60:.1f} 分钟，结果保存至：{OUT_PATH}")
    if failed:
        print(f"仍有 {len(failed)} 个话题标注失败，可再次运行脚本重试")

    # 汇总统计（去除重复，优先保留成功结果）
    final = pd.read_csv(OUT_PATH, encoding="utf-8-sig")
    final = final[final["is_demand"] != "标注失败"]
    final = final.drop_duplicates("title", keep="last")
    final.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    n_demand = (final["is_demand"] == "Y").sum()
    n_no_demand = (final["is_demand"] == "N").sum()
    print(f"\n去重后共 {len(final):,} 条")
    print(f"  有诉求（is_demand=Y）：{n_demand:,}（{n_demand/len(final)*100:.1f}%）")
    print(f"  无诉求（is_demand=N）：{n_no_demand:,}（{n_no_demand/len(final)*100:.1f}%）")
    print(f"\n各一级分类分布（有诉求话题）：")
    print(final[final["is_demand"]=="Y"]["category_L1"].value_counts().to_string())


if __name__ == "__main__":
    main()
