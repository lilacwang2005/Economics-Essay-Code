"""
人民日报匹配结果 LLM 标注脚本（行迭代版）
每行 = 一篇新闻 + 该行的所有核心诉求，直接按行迭代，无需 explode/groupby。
失败时自动拆回单条重试。

输出字段：
  article_id      : 原始行号
  date            : 新闻日期
  core_demand     : 核心诉求文本
  demand_related  : T/F
  npc_related     : T/F
  solve_score     : 0-100 或空
  solve_reason    : 评分理由或空
"""

import os, json, random, time, threading, csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────────────────────
API_KEYS = [
    "sk-ca9b4f66b2b24a3c950f7cc2625ef97a",
    "sk-e15ef82229444f1faac4cb99f103d888",
    "sk-748c1798dd0747f688093878e45eaf1f",
    "sk-52b6bde2424347d8b0030a150c8374d8",
    "sk-10b26a1d9d104ad6ba36a1a0a6f312fe",
    "sk-194432ce2c664be5b52fd5dcf0c71fb9",
    "sk-4fc1bb19bd01495d8ab3d7c48ffc9bd3"
]

BASE_URL    = "https://api.deepseek.com"
MODEL       = "deepseek-chat"
MAX_WORKERS = 30          # 并发行数
BATCH_SIZE  = 20          # 每行最多一次处理几个诉求
SAVE_EVERY  = 500         # 每累积多少条写一次断点

RAW_DIR   = "data/raw"
CLEAN_DIR = "data/clean"
OUT_PATH  = os.path.join(CLEAN_DIR, "人民日报_LLM标注结果.csv")
CKPT_PATH = os.path.join(CLEAN_DIR, "人民日报_LLM标注结果_ckpt.csv")

OUT_COLS  = ["article_id", "date", "core_demand",
             "demand_related", "npc_related", "solve_score", "solve_reason"]
CKPT_COLS = OUT_COLS  # 不再需要 _pair_idx

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是一名中国政策与舆论研究助理，专注于分析人民日报新闻与公众诉求之间的关联。"
    "请严格依据所提供的新闻文本和核心诉求作出判断，不得自行联系外部信息、进行网络检索或引入新闻之外的现实背景。"
    "请仅输出合法 JSON，不得输出任何其他内容。"
)

JUDGMENT_CRITERIA = """
请根据以上内容对每个核心诉求判断以下四项：

1. demand_related（布尔）：该新闻是否可能与上述核心诉求有关。
   判断标准宽松，只要有合理关联即为 true。关联包括但不限于：
   - 新闻报道政府/部门就该诉求作出回应、出台政策或措施（如"教育部发文要求…"）
   - 新闻报道与该诉求高度相关的社会事件、研究或调查（如医改报道 vs 医疗诉求）
   - 新闻反映该诉求尚未解决、引发争议或受到公众关注（如"XX问题仍未解决"）
   - 新闻报道相关立法、规划或财政投入（如"纳入医保"、"立法草案通过"）
   若新闻与诉求主题明显无关（如农业补贴新闻 vs 教育改革诉求），则为 false。

2. npc_related（布尔）：该新闻是否一定直接涉及人大代表或政协委员的提案/建议/答复。
   判断标准严格，必须有明确文字证据才为 true。包括：
   - 新闻明确提到"人大代表建议/提案"、"政协委员提案"、"两会提案"
   - 新闻报道某代表/委员的具体提案内容
   - 新闻标题或正文中出现"代表建议"、"委员提案"、"人大议案"等关键词
   若仅是政策报道、专家观点或政府公告，而未明确提及代表/委员提案，则为 false。

3. solve_score（整数 0-100 或 null）：
   仅当 demand_related 为 true 时填写，否则填 null。
   评分表示该核心诉求在新闻中体现的已解决/落实程度（0=尚未开始，100=完全解决）。
   评分示例：
   - 0分：新闻仅讨论问题存在，或政府尚未回应（如"专家呼吁关注XX问题"）
   - 20分：相关部门表示将研究、会着手解决，但无具体措施（如"教育部回应将研究该问题"）
   - 40分：已出台政策文件或规划，但尚未落地执行（如"国务院发布XX方案"）
   - 60分：政策已开始在部分地区或领域试点实施（如"XX省试点推行"）
   - 80分：政策已全面推行，但存在执行问题或覆盖不完全（如"全面推行但农村地区仍存在缺口"）
   - 100分：新闻反映该诉求已被完全解决，措施覆盖所有相关群体（如"全国全面落实，成效显著"）
   注意：评分仅基于新闻文本所呈现的信息，不得基于你对现实情况的了解。

4. solve_reason（字符串或 null）：
   仅当 demand_related 为 true 时填写，否则填 null。
   简要说明 solve_score 的评分依据（1-2句话），例如：
   "新闻反映相关部门已对该诉求作出官方回复，表示将着手解决，但未反映现实中具体落实进度，因而评为20分。"
"""

def make_batch_prompt(text: str, demands: list[str]) -> str:
    demands_str = "\n".join(f"{i+1}. {d}" for i, d in enumerate(demands))
    return f"""【新闻文本】
{text}

【待判断的核心诉求列表】
{demands_str}

{JUDGMENT_CRITERIA}
请以核心诉求原文为 key，输出 JSON 对象，每个 key 对应一个包含四个字段的对象。
示例（2个诉求）：
{{"延长义务教育至十二年": {{"demand_related": true, "npc_related": false, "solve_score": 40, "solve_reason": "..."}}, "提高教师工资待遇": {{"demand_related": false, "npc_related": false, "solve_score": null, "solve_reason": null}}}}"""

def make_single_prompt(text: str, demand: str) -> str:
    return f"""【核心诉求】
{demand}

【新闻文本】
{text}

{JUDGMENT_CRITERIA}
输出格式（严格遵守，不得添加任何额外字段或文字）：
{{"demand_related": true, "npc_related": false, "solve_score": 20, "solve_reason": "评分理由"}}
或（demand_related 为 false 时）：
{{"demand_related": false, "npc_related": false, "solve_score": null, "solve_reason": null}}"""

# ── API 调用：批量（一篇文章 × N 个诉求）────────────────────────────────────
def call_api_batch(article_id: str, date: str, text: str,
                   demands: list[str],
                   max_retries: int = 3) -> list[dict]:
    """
    返回 list of row dicts（每个诉求一行）。
    失败时返回空列表，由调用方决定是否拆回单条。
    """
    key    = random.choice(API_KEYS)
    client = OpenAI(api_key=key, base_url=BASE_URL)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": make_batch_prompt(text, demands)},
                ],
                temperature=0,
                max_tokens=200 * len(demands),
            )
            raw_text = resp.choices[0].message.content.strip()
            start = raw_text.find("{")
            end   = raw_text.rfind("}") + 1
            parsed = json.loads(raw_text[start:end])

            rows = []
            for demand in demands:
                result = parsed.get(demand)
                if result is None:
                    return []   # key 缺失 → 触发回退
                rows.append({
                    "article_id":  article_id,
                    "date":        date,
                    "core_demand": demand,
                    "demand_related": "T" if result.get("demand_related") else "F",
                    "npc_related":    "T" if result.get("npc_related")    else "F",
                    "solve_score":    result.get("solve_score"),
                    "solve_reason":   result.get("solve_reason"),
                })
            return rows
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return []   # 全部重试失败

# ── API 调用：单条（批量失败时的回退）────────────────────────────────────────
def call_api_single(article_id: str, date: str, text: str,
                    demand: str,
                    max_retries: int = 3) -> dict | None:
    key    = random.choice(API_KEYS)
    client = OpenAI(api_key=key, base_url=BASE_URL)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": make_single_prompt(text, demand)},
                ],
                temperature=0,
                max_tokens=400,
            )
            raw_text = resp.choices[0].message.content.strip()
            start = raw_text.find("{")
            end   = raw_text.rfind("}") + 1
            parsed = json.loads(raw_text[start:end])
            return {
                "article_id":  article_id,
                "date":        date,
                "core_demand": demand,
                "demand_related": "T" if parsed.get("demand_related") else "F",
                "npc_related":    "T" if parsed.get("npc_related")    else "F",
                "solve_score":    parsed.get("solve_score"),
                "solve_reason":   parsed.get("solve_reason"),
            }
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None

# ── 写入工具 ──────────────────────────────────────────────────────────────────
_write_lock = threading.Lock()

def _append_rows(rows: list, path: str, write_header: bool):
    with _write_lock:
        with open(path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CKPT_COLS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    print("加载原始数据…")
    raw = pd.read_csv(
        os.path.join(RAW_DIR, "人民日报_关键词匹配结果_新诉求.csv"),
        encoding="utf-8-sig",
        usecols=["date", "text", "matched_core_demands"],
        dtype=str,
    )
    raw["article_id"] = raw.index.astype(str)
    total_rows = len(raw)
    print(f"  原始行数：{total_rows:,}")

    # 断点续跑：读取已完成的 (article_id, core_demand) 对
    # 自动迁移旧格式（含 _pair_idx 列）→ 新格式
    done_demands: dict[str, set] = {}   # article_id → set of done demands
    done_count = 0
    need_header = True
    if os.path.exists(CKPT_PATH):
        try:
            ckpt = pd.read_csv(CKPT_PATH, encoding="utf-8-sig", dtype=str)
            # 迁移：去掉旧格式中的 _pair_idx 列，避免 append 时列对齐错位
            if "_pair_idx" in ckpt.columns:
                ckpt = ckpt.drop(columns=["_pair_idx"])
                ckpt.to_csv(CKPT_PATH, index=False, encoding="utf-8-sig")
                print("  已迁移旧断点格式（去除 _pair_idx 列）")
            done_demands = (
                ckpt[["article_id", "core_demand"]].dropna()
                .groupby("article_id")["core_demand"]
                .apply(set)
                .to_dict()
            )
            done_count = sum(len(v) for v in done_demands.values())
            need_header = False
            print(f"  已有断点：{done_count:,} 条，跳过重跑")
        except Exception as e:
            print(f"  断点读取失败（{e}），从头开始")

    # 构建待处理任务列表（向量化，避免 iterrows 慢）
    raw["date"] = raw["date"].fillna("")
    raw["text"] = raw["text"].fillna("")
    raw["matched_core_demands"] = raw["matched_core_demands"].fillna("")

    tasks = []
    for article_id, date, text, demands_raw in zip(
        raw["article_id"], raw["date"], raw["text"], raw["matched_core_demands"]
    ):
        if not demands_raw.strip():
            continue
        demands = [d.strip() for d in demands_raw.split("/") if d.strip()]
        done = done_demands.get(str(article_id), set())
        todo = [d for d in demands if d not in done]
        if not todo:
            continue
        tasks.append({
            "article_id": str(article_id),
            "date":       str(date),
            "text":       str(text),
            "demands":    todo,
        })

    total_todo = sum(len(t["demands"]) for t in tasks)
    print(f"  待处理行数：{len(tasks):,}，待标注对数：{total_todo:,}，并发：{MAX_WORKERS}")

    completed_pairs = 0
    batch_buf: list = []
    write_header_once = need_header

    def process_row(task: dict) -> list[dict]:
        """处理一行的所有待标注诉求，返回所有行结果。"""
        article_id = task["article_id"]
        date       = task["date"]
        text       = task["text"]
        demands    = task["demands"]

        all_rows = []
        for i in range(0, len(demands), BATCH_SIZE):
            chunk = demands[i: i + BATCH_SIZE]
            rows = call_api_batch(article_id, date, text, chunk)
            if not rows:
                # 批量失败 → 逐条回退
                for demand in chunk:
                    row = call_api_single(article_id, date, text, demand)
                    if row:
                        all_rows.append(row)
            else:
                all_rows.extend(rows)
        return all_rows

    SUBMIT_BATCH = MAX_WORKERS * 4
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in range(0, len(tasks), SUBMIT_BATCH):
            chunk = tasks[i: i + SUBMIT_BATCH]
            future_map = {executor.submit(process_row, t): t for t in chunk}
            for future in as_completed(future_map):
                rows = future.result()
                batch_buf.extend(rows)
                completed_pairs += len(rows)
                if completed_pairs % 1000 == 0:
                    print(f"  进度：{completed_pairs + done_count:,} 对已完成", flush=True)
                if len(batch_buf) >= SAVE_EVERY:
                    _append_rows(batch_buf, CKPT_PATH, write_header_once)
                    write_header_once = False
                    batch_buf.clear()

    if batch_buf:
        _append_rows(batch_buf, CKPT_PATH, write_header_once)
        batch_buf.clear()

    # 汇总最终输出
    print("  汇总最终结果…")
    final = pd.read_csv(CKPT_PATH, encoding="utf-8-sig", dtype=str)
    # 兼容旧格式：过滤无效行
    final = final[final["demand_related"].notna() & (final["demand_related"] != "None")].copy()
    final[OUT_COLS].to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    os.remove(CKPT_PATH)

    print(f"\n完成！已保存至 {OUT_PATH}（共 {len(final):,} 对）")
    print(f"  demand_related=T : {(final['demand_related'] == 'T').sum():,}")
    print(f"  npc_related=T    : {(final['npc_related'] == 'T').sum():,}")
    demand_t = pd.to_numeric(
        final[final["demand_related"] == "T"]["solve_score"], errors="coerce"
    ).dropna()
    if len(demand_t):
        print(f"  solve_score 均值 : {demand_t.mean():.1f}（共 {len(demand_t):,} 条）")


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    main()
