"""
人民日报 BERT 两阶段训练与全量推理脚本
========================================
模型: hfl/chinese-roberta-wwm-ext-large (330M)

两阶段训练设计:
  Stage 1 — demand_related + npc_related (二分类 × 2)
    · 输入: (article_text, core_demand) 句子对
    · 全量 LLM 标注数据训练 (~879k 对)
    · 输出: checkpoints/best_model_stage1.pt

  Stage 2 — solve_score (6 分类: 0/20/40/60/80/100)
    · 从 Stage 1 最佳模型热启动
    · 仅在 demand_related=T 的子集训练 (~144k 对)
    · 冻结 BERT 底层 16/24 层, 只 fine-tune 顶层 + solve_head
    · 输出: checkpoints/best_model_stage2.pt

全量推理:
  · Stage 1 模型 → 所有 2.36M (article, demand) 对预测 demand/npc
  · Stage 2 模型 → 仅对 demand_related=T 的对预测 solve_score
  · 合并输出到 data/clean/人民日报_BERT推理结果.csv

环境: RTX 4090 24GB, 16 vCPU Intel Xeon 8352V, 120GB RAM
路径: /root/autodl-tmp/citizens-will-and-politics/
"""

import os
import sys
import gc
import time
import json
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
PROJECT_DIR   = "/root/autodl-tmp/citizens_will_and_politics"
DATA_DIR      = os.path.join(PROJECT_DIR, "data")
MODEL_PATH    = os.path.join(DATA_DIR, "chinese-roberta-wwm-ext-large")

TRAIN_CKPT    = os.path.join(DATA_DIR, "clean", "人民日报_LLM标注结果_ckpt.csv")
NEWS_CSV      = os.path.join(DATA_DIR, "raw",   "人民日报_关键词匹配结果_新诉求.csv")
OUTPUT_CSV    = os.path.join(DATA_DIR, "clean", "人民日报_BERT推理结果.csv")

OUTPUT_DIR    = os.path.join(PROJECT_DIR, "log")
CKPT_DIR      = os.path.join(OUTPUT_DIR, "checkpoints_purpose1_bert")
LOG_FILE      = os.path.join(OUTPUT_DIR, "purpose1_bert.log")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "purpose1_bert_infer_progress.json")

os.makedirs(CKPT_DIR, exist_ok=True)

# ============================================================
# 日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================
# 超参数 (RTX 4090 24GB, large 模型)
# ============================================================
MAX_LEN            = 512    # BERT 上限; 新闻首段约 490 字已含主要政策信息

# Stage 1: 全量数据, sentence-pair
S1_BATCH_SIZE      = 24     # large + FP16 + MAX_LEN=512 → ~16-18 GB
S1_EPOCHS          = 6      # 小 LR 收敛慢, 多给 epoch; early stopping 会自动提前退出
S1_LR              = 5e-6   # 原 2e-5 导致 Epoch2 起梯度震荡、train_loss 反升, 降至 5e-6

# Stage 2: demand=T 子集, 热启动微调
S2_BATCH_SIZE      = 48     # 数据量小, 可加大 batch
S2_EPOCHS          = 5
S2_LR_BERT         = 5e-6   # BERT 顶层用更小 LR
S2_LR_HEAD         = 1e-4   # solve_head 用较大 LR
S2_FREEZE_LAYERS   = 16     # 冻结 BERT 底层 16/24 层

# 推理
INFER_BATCH_SIZE   = 128
INFER_SAVE_EVERY   = 20000

WARMUP_RATIO       = 0.15  # 增大 warmup, 开头更平稳
VAL_RATIO          = 0.1
TEST_RATIO         = 0.1
SEED               = 42
NUM_WORKERS        = 4
PIN_MEMORY         = True
USE_AMP            = True
AUTO_SHUTDOWN      = False

# 早停: 连续 N 个 epoch val_loss 不下降则提前退出
S1_PATIENCE        = 2
S2_PATIENCE        = 2

# solve_score 6 类映射
SCORE_CLASSES  = [0, 20, 40, 60, 80, 100]
SCORE2ID       = {s: i for i, s in enumerate(SCORE_CLASSES)}
ID2SCORE       = {i: s for s, i in SCORE2ID.items()}
NUM_SCORES     = len(SCORE_CLASSES)

# 类别权重 (基于实际分布, 对稀有类做上限约束)
# 分布: 0→45742, 20→22131, 40→50639, 60→17163, 80→8659, 100→461
DEMAND_POS_WEIGHT = 5.0     # F:T ≈ 5:1
NPC_POS_WEIGHT    = 81.0    # F:T ≈ 81:1
# solve_score: 逆频率权重, 100分极稀有故上限设 20
_SOLVE_FREQ   = [45742, 22131, 50639, 17163, 8659, 461]
_SOLVE_MED    = float(np.median(_SOLVE_FREQ))
SOLVE_WEIGHTS = [min(_SOLVE_MED / f, 20.0) for f in _SOLVE_FREQ]


def format_time(s: float) -> str:
    return str(timedelta(seconds=int(s)))


# ============================================================
# Datasets
# ============================================================
class Stage1Dataset(Dataset):
    """(article_text, core_demand) → demand_related, npc_related"""

    def __init__(self, texts, demands, demand_labels, npc_labels, tokenizer, max_len):
        self.texts         = texts
        self.demands       = demands
        self.demand_labels = demand_labels
        self.npc_labels    = npc_labels
        self.tokenizer     = tokenizer
        self.max_len       = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            str(self.demands[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "demand_label":   torch.tensor(self.demand_labels[idx], dtype=torch.long),
            "npc_label":      torch.tensor(self.npc_labels[idx],    dtype=torch.long),
        }


class Stage2Dataset(Dataset):
    """demand_related=T 子集: (article_text, core_demand) → solve_score (6类)"""

    def __init__(self, texts, demands, score_labels, tokenizer, max_len):
        self.texts        = texts
        self.demands      = demands
        self.score_labels = score_labels   # 已映射为 0-5
        self.tokenizer    = tokenizer
        self.max_len      = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            str(self.demands[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":    enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "score_label":  torch.tensor(self.score_labels[idx], dtype=torch.long),
        }


class InferDataset(Dataset):
    """推理专用: 只需要 (text, demand)"""

    def __init__(self, texts, demands, tokenizer, max_len):
        self.texts     = texts
        self.demands   = demands
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            str(self.demands[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ============================================================
# 模型定义
# ============================================================
class Stage1Model(nn.Module):
    """BERT + demand_head + npc_head"""

    def __init__(self, bert_path: str, dropout: float = 0.1):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(bert_path)
        h            = self.bert.config.hidden_size   # 1024 for large
        self.dropout = nn.Dropout(dropout)
        self.demand_head = nn.Sequential(
            nn.Linear(h, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 2)
        )
        self.npc_head = nn.Sequential(
            nn.Linear(h, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        cls = self.dropout(
            self.bert(input_ids=input_ids, attention_mask=attention_mask)
            .last_hidden_state[:, 0, :]
        )
        return self.demand_head(cls), self.npc_head(cls)


class Stage2Model(nn.Module):
    """BERT (从 Stage1 热启动) + solve_head (6分类)"""

    def __init__(self, bert_path: str, dropout: float = 0.1):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(bert_path)
        h            = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # 更深的分类头: 捕捉政策落实语义
        self.solve_head = nn.Sequential(
            nn.Linear(h, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, NUM_SCORES),
        )

    def forward(self, input_ids, attention_mask):
        cls = self.dropout(
            self.bert(input_ids=input_ids, attention_mask=attention_mask)
            .last_hidden_state[:, 0, :]
        )
        return self.solve_head(cls)   # (B, 6)

    def freeze_bottom_layers(self, n: int):
        """冻结 BERT embedding + 底层 n 个 transformer 层"""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False
        frozen = sum(not p.requires_grad for p in self.bert.parameters())
        total  = sum(1 for _ in self.bert.parameters())
        logger.info(f"  冻结 BERT 底层 {n} 层: {frozen}/{total} 参数不参与梯度")


# ============================================================
# 数据加载
# ============================================================
def load_labeled_data():
    """读取 LLM 标注结果并与新闻原文合并"""
    logger.info("读取 LLM 标注结果...")
    t0  = time.time()
    ckpt = pd.read_csv(TRAIN_CKPT, encoding="utf-8-sig", dtype=str, low_memory=False)
    ckpt = ckpt[ckpt["demand_related"].isin(["T", "F"])].copy()
    logger.info(f"  有效标注行: {len(ckpt):,}")

    logger.info("读取新闻原文...")
    news = pd.read_csv(NEWS_CSV, encoding="utf-8-sig", dtype=str,
                       low_memory=False, usecols=["text"])
    news["article_id"] = news.index.astype(str)

    df = ckpt.merge(news.rename(columns={"text": "article_text"}),
                    on="article_id", how="inner")
    logger.info(f"  合并后: {len(df):,} 行, 耗时: {format_time(time.time()-t0)}")

    df["demand_id"]  = (df["demand_related"] == "T").astype(int)
    df["npc_id"]     = (df["npc_related"]    == "T").astype(int)
    df["solve_num"]  = pd.to_numeric(df["solve_score"], errors="coerce")

    d_t = df["demand_id"].sum();  d_f = (df["demand_id"] == 0).sum()
    n_t = df["npc_id"].sum();     n_f = (df["npc_id"] == 0).sum()
    logger.info(f"  demand T/F: {d_t:,}/{d_f:,}  npc T/F: {n_t:,}/{n_f:,}")
    logger.info(f"  solve_score 有效: {df['solve_num'].notna().sum():,}")

    del ckpt, news; gc.collect()
    return df


def prepare_stage1(df, tokenizer):
    logger.info("准备 Stage1 数据集...")
    train_df, temp_df = train_test_split(df, test_size=VAL_RATIO+TEST_RATIO, random_state=SEED)
    val_df, test_df   = train_test_split(temp_df,
                                         test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO),
                                         random_state=SEED)
    logger.info(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    def mk(sub):
        return Stage1Dataset(
            texts=sub["article_text"].values,
            demands=sub["core_demand"].values,
            demand_labels=sub["demand_id"].values.astype(np.int64),
            npc_labels=sub["npc_id"].values.astype(np.int64),
            tokenizer=tokenizer, max_len=MAX_LEN,
        )
    return mk(train_df), mk(val_df), mk(test_df), test_df


def prepare_stage2(df, tokenizer):
    logger.info("准备 Stage2 数据集 (demand=T 子集)...")
    sub = df[df["demand_id"] == 1].copy()

    # 将 solve_score 四舍五入到最近的 6 档, 映射为 class id
    def snap(v):
        if pd.isna(v):
            return -1
        nearest = min(SCORE_CLASSES, key=lambda c: abs(c - v))
        return SCORE2ID[nearest]

    sub["score_id"] = sub["solve_num"].map(snap)
    sub = sub[sub["score_id"] >= 0].copy()
    logger.info(f"  有效 solve_score 样本: {len(sub):,}")

    score_dist = sub["score_id"].value_counts().sort_index()
    for sid, cnt in score_dist.items():
        logger.info(f"    {ID2SCORE[sid]:>3}分: {cnt:,}")

    train_df, temp_df = train_test_split(sub, test_size=VAL_RATIO+TEST_RATIO, random_state=SEED)
    val_df, test_df   = train_test_split(temp_df,
                                         test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO),
                                         random_state=SEED)
    logger.info(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    def mk(s):
        return Stage2Dataset(
            texts=s["article_text"].values,
            demands=s["core_demand"].values,
            score_labels=s["score_id"].values.astype(np.int64),
            tokenizer=tokenizer, max_len=MAX_LEN,
        )
    return mk(train_df), mk(val_df), mk(test_df), test_df


# ============================================================
# 通用训练工具
# ============================================================
def make_loader(ds, batch_size, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


def make_optimizer_scheduler(model, lr, total_steps, warmup_ratio=WARMUP_RATIO):
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler   = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


# ============================================================
# Stage 1 训练
# ============================================================
def train_stage1(model, train_ds, val_ds, device):
    logger.info("=" * 60)
    logger.info("Stage 1 训练: demand_related + npc_related")
    logger.info("=" * 60)

    train_loader = make_loader(train_ds, S1_BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(val_ds,   S1_BATCH_SIZE * 2, shuffle=False)

    demand_ce = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, DEMAND_POS_WEIGHT], device=device))
    npc_ce    = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, NPC_POS_WEIGHT], device=device))

    total_steps = len(train_loader) * S1_EPOCHS
    optimizer, scheduler = make_optimizer_scheduler(model, S1_LR, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # 断点续训
    start_epoch = 0
    ckpt_path   = os.path.join(CKPT_DIR, "s1_checkpoint.pt")
    if os.path.exists(ckpt_path):
        logger.info(f"发现 Stage1 断点, 加载...")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck["epoch"] + 1
        logger.info(f"  从 epoch {start_epoch} 继续")
        del ck; gc.collect()

    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, S1_EPOCHS):
        t_ep = time.time()
        model.train()
        ep_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"S1 Epoch {epoch+1}/{S1_EPOCHS} [Train]", leave=True)
        for step, batch in enumerate(pbar):
            ids   = batch["input_ids"].to(device,      non_blocking=True)
            mask  = batch["attention_mask"].to(device,  non_blocking=True)
            d_lbl = batch["demand_label"].to(device,    non_blocking=True)
            n_lbl = batch["npc_label"].to(device,       non_blocking=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                d_pred, n_pred = model(ids, mask)
                loss = demand_ce(d_pred, d_lbl) + 0.5 * npc_ce(n_pred, n_lbl)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step(); optimizer.zero_grad()

            ep_loss += loss.item()
            eta = (time.time()-t_ep)/(step+1) * (len(train_loader)-step-1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "ETA": format_time(eta)})

        avg_train = ep_loss / len(train_loader)
        logger.info(f"S1 Epoch {epoch+1} Train — loss: {avg_train:.4f}, "
                    f"耗时: {format_time(time.time()-t_ep)}")

        # Validate
        model.eval()
        val_loss = 0.0
        vd_true, vd_pred, vn_true, vn_pred = [], [], [], []
        t_val = time.time()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"S1 Epoch {epoch+1}/{S1_EPOCHS} [Val]", leave=True):
                ids  = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                d_lbl = batch["demand_label"].to(device, non_blocking=True)
                n_lbl = batch["npc_label"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    d_pred, n_pred = model(ids, mask)
                    val_loss += (demand_ce(d_pred, d_lbl) + 0.5 * npc_ce(n_pred, n_lbl)).item()
                vd_true.extend(d_lbl.cpu().numpy()); vd_pred.extend(d_pred.argmax(1).cpu().numpy())
                vn_true.extend(n_lbl.cpu().numpy()); vn_pred.extend(n_pred.argmax(1).cpu().numpy())

        avg_val = val_loss / len(val_loader)
        d_f1 = f1_score(vd_true, vd_pred, pos_label=1, zero_division=0)
        n_f1 = f1_score(vn_true, vn_pred, pos_label=1, zero_division=0)
        logger.info(f"S1 Epoch {epoch+1} Val   — loss: {avg_val:.4f}, "
                    f"demand_F1(T)={d_f1:.4f}, npc_F1(T)={n_f1:.4f}, "
                    f"耗时: {format_time(time.time()-t_val)}")

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_loss": avg_val}, ckpt_path)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_path = os.path.join(CKPT_DIR, "best_model_stage1.pt")
            torch.save(model.state_dict(), best_path)
            logger.info(f"  ✓ S1 最佳模型更新 (val_loss={avg_val:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  S1 早停计数: {patience_counter}/{S1_PATIENCE} "
                        f"(best={best_val_loss:.4f}, current={avg_val:.4f})")
            if patience_counter >= S1_PATIENCE:
                logger.info(f"  S1 早停触发, 在 epoch {epoch+1} 停止训练")
                break

    best_path = os.path.join(CKPT_DIR, "best_model_stage1.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
    return model


# ============================================================
# Stage 1 评估
# ============================================================
def evaluate_stage1(model, test_ds, device):
    logger.info("Stage 1 测试集评估...")
    loader = make_loader(test_ds, INFER_BATCH_SIZE, shuffle=False)
    model.eval()
    d_true, d_pred, n_true, n_pred = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="S1 评估"):
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                dp, np_ = model(ids, mask)
            d_true.extend(batch["demand_label"].numpy()); d_pred.extend(dp.argmax(1).cpu().numpy())
            n_true.extend(batch["npc_label"].numpy());    n_pred.extend(np_.argmax(1).cpu().numpy())

    logger.info(f"[demand_related] Acc={accuracy_score(d_true, d_pred):.4f}, "
                f"F1(T)={f1_score(d_true, d_pred, pos_label=1, zero_division=0):.4f}")
    logger.info(classification_report(d_true, d_pred, target_names=["F","T"], zero_division=0))
    logger.info(f"[npc_related]    Acc={accuracy_score(n_true, n_pred):.4f}, "
                f"F1(T)={f1_score(n_true, n_pred, pos_label=1, zero_division=0):.4f}")
    logger.info(classification_report(n_true, n_pred, target_names=["F","T"], zero_division=0))
    logger.info(f"S1 评估耗时: {format_time(time.time()-t0)}")

    results = {
        "demand_related": {"accuracy": accuracy_score(d_true, d_pred),
                           "f1_T": f1_score(d_true, d_pred, pos_label=1, zero_division=0)},
        "npc_related":    {"accuracy": accuracy_score(n_true, n_pred),
                           "f1_T": f1_score(n_true, n_pred, pos_label=1, zero_division=0)},
    }
    with open(os.path.join(CKPT_DIR, "eval_stage1.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


# ============================================================
# Stage 2 训练
# ============================================================
def train_stage2(model, train_ds, val_ds, device):
    logger.info("=" * 60)
    logger.info("Stage 2 训练: solve_score (6分类, 热启动)")
    logger.info("=" * 60)

    # 冻结底层
    model.freeze_bottom_layers(S2_FREEZE_LAYERS)

    train_loader = make_loader(train_ds, S2_BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(val_ds,   S2_BATCH_SIZE * 2, shuffle=False)

    solve_ce = nn.CrossEntropyLoss(
        weight=torch.tensor(SOLVE_WEIGHTS, dtype=torch.float, device=device))

    # 差分学习率: BERT 顶层用小 LR, solve_head 用大 LR
    bert_params  = [p for n, p in model.named_parameters()
                    if p.requires_grad and "solve_head" not in n]
    head_params  = [p for n, p in model.named_parameters() if "solve_head" in n]
    optimizer    = torch.optim.AdamW([
        {"params": bert_params, "lr": S2_LR_BERT},
        {"params": head_params, "lr": S2_LR_HEAD},
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * S2_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # 断点续训
    start_epoch = 0
    ckpt_path   = os.path.join(CKPT_DIR, "s2_checkpoint.pt")
    if os.path.exists(ckpt_path):
        logger.info("发现 Stage2 断点, 加载...")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck["epoch"] + 1
        logger.info(f"  从 epoch {start_epoch} 继续")
        del ck; gc.collect()

    best_val_loss    = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, S2_EPOCHS):
        t_ep = time.time()
        model.train()
        ep_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"S2 Epoch {epoch+1}/{S2_EPOCHS} [Train]", leave=True)
        for step, batch in enumerate(pbar):
            ids   = batch["input_ids"].to(device,      non_blocking=True)
            mask  = batch["attention_mask"].to(device,  non_blocking=True)
            s_lbl = batch["score_label"].to(device,     non_blocking=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                s_pred = model(ids, mask)
                loss   = solve_ce(s_pred, s_lbl)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step(); optimizer.zero_grad()

            ep_loss += loss.item()
            eta = (time.time()-t_ep)/(step+1) * (len(train_loader)-step-1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "ETA": format_time(eta)})

        avg_train = ep_loss / len(train_loader)
        logger.info(f"S2 Epoch {epoch+1} Train — loss: {avg_train:.4f}, "
                    f"耗时: {format_time(time.time()-t_ep)}")

        # Validate
        model.eval()
        val_loss = 0.0
        vs_true, vs_pred = [], []
        t_val = time.time()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"S2 Epoch {epoch+1}/{S2_EPOCHS} [Val]", leave=True):
                ids  = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                s_lbl = batch["score_label"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    s_pred = model(ids, mask)
                    val_loss += solve_ce(s_pred, s_lbl).item()
                vs_true.extend(s_lbl.cpu().numpy())
                vs_pred.extend(s_pred.argmax(1).cpu().numpy())

        avg_val  = val_loss / len(val_loader)
        s_acc    = accuracy_score(vs_true, vs_pred)
        # 邻近准确率: 预测档位与真实档位相差 ≤ 1
        near_acc = np.mean([abs(t - p) <= 1 for t, p in zip(vs_true, vs_pred)])
        # MAE (还原到实际分数)
        true_scores = [ID2SCORE[t] for t in vs_true]
        pred_scores = [ID2SCORE[p] for p in vs_pred]
        mae = float(np.mean([abs(t - p) for t, p in zip(true_scores, pred_scores)]))
        logger.info(f"S2 Epoch {epoch+1} Val   — loss: {avg_val:.4f}, "
                    f"Acc={s_acc:.4f}, 邻近Acc={near_acc:.4f}, MAE={mae:.1f}分, "
                    f"耗时: {format_time(time.time()-t_val)}")

        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_loss": avg_val}, ckpt_path)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_path = os.path.join(CKPT_DIR, "best_model_stage2.pt")
            torch.save(model.state_dict(), best_path)
            logger.info(f"  ✓ S2 最佳模型更新 (val_loss={avg_val:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  S2 早停计数: {patience_counter}/{S2_PATIENCE} "
                        f"(best={best_val_loss:.4f}, current={avg_val:.4f})")
            if patience_counter >= S2_PATIENCE:
                logger.info(f"  S2 早停触发, 在 epoch {epoch+1} 停止训练")
                break

    best_path = os.path.join(CKPT_DIR, "best_model_stage2.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
    return model


# ============================================================
# Stage 2 评估
# ============================================================
def evaluate_stage2(model, test_ds, device):
    logger.info("Stage 2 测试集评估...")
    loader = make_loader(test_ds, INFER_BATCH_SIZE, shuffle=False)
    model.eval()
    s_true, s_pred = [], []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="S2 评估"):
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                sp = model(ids, mask)
            s_true.extend(batch["score_label"].numpy())
            s_pred.extend(sp.argmax(1).cpu().numpy())

    true_scores = [ID2SCORE[t] for t in s_true]
    pred_scores = [ID2SCORE[p] for p in s_pred]
    mae      = float(np.mean([abs(t - p) for t, p in zip(true_scores, pred_scores)]))
    near_acc = np.mean([abs(t - p) <= 1 for t, p in zip(s_true, s_pred)])
    logger.info(f"[solve_score] Acc={accuracy_score(s_true, s_pred):.4f}, "
                f"邻近Acc(±1档)={near_acc:.4f}, MAE={mae:.1f}分")
    logger.info(classification_report(s_true, s_pred,
                                      target_names=[f"{s}分" for s in SCORE_CLASSES],
                                      zero_division=0))
    logger.info(f"  注: solve_score 依赖深层政策推理, 邻近Acc 更有参考价值")
    logger.info(f"S2 评估耗时: {format_time(time.time()-t0)}")

    results = {"solve_score": {"accuracy": accuracy_score(s_true, s_pred),
                                "near_acc": float(near_acc), "MAE": mae}}
    with open(os.path.join(CKPT_DIR, "eval_stage2.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


# ============================================================
# 全量推理
# ============================================================
def run_inference(s1_model, s2_model, tokenizer, device):
    logger.info("=" * 60)
    logger.info("全量推理")
    logger.info("=" * 60)

    # 展开 (article_id, date, text, core_demand) 对
    logger.info("展开 (article, demand) 对...")
    t0   = time.time()
    news = pd.read_csv(NEWS_CSV, encoding="utf-8-sig", dtype=str, low_memory=False,
                       usecols=["date", "text", "matched_core_demands"])
    news["article_id"] = news.index.astype(str)

    records = []
    for _, row in tqdm(news.iterrows(), total=len(news), desc="展开诉求"):
        raw = str(row.get("matched_core_demands", ""))
        if not raw.strip():
            continue
        for d in raw.split("/"):
            d = d.strip()
            if d:
                records.append({"article_id": row["article_id"],
                                 "date":       row.get("date", ""),
                                 "text":       row.get("text", ""),
                                 "core_demand": d})

    infer_df    = pd.DataFrame(records)
    total_pairs = len(infer_df)
    logger.info(f"  总对数: {total_pairs:,}, 展开耗时: {format_time(time.time()-t0)}")
    del news, records; gc.collect()

    # 断点续传
    start_idx    = 0
    write_header = True
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            prog = json.load(f)
        start_idx = prog.get("last_completed_idx", 0)
        if start_idx > 0:
            logger.info(f"断点续传: 从第 {start_idx:,} 对继续")
            write_header = False

    if start_idx >= total_pairs:
        logger.info("全量推理已完成。")
        return

    OUT_COLS = ["article_id", "date", "core_demand",
                "demand_related", "npc_related", "solve_score"]

    s1_model.eval(); s2_model.eval()
    subset    = infer_df.iloc[start_idx:].reset_index(drop=True)
    t_total   = time.time()
    processed = 0
    pbar      = tqdm(total=len(subset), desc="全量推理", unit="对")

    for chunk_start in range(0, len(subset), INFER_SAVE_EVERY):
        chunk   = subset.iloc[chunk_start: chunk_start + INFER_SAVE_EVERY].copy()
        texts   = chunk["text"].fillna("").tolist()
        demands = chunk["core_demand"].fillna("").tolist()

        ds     = InferDataset(texts, demands, tokenizer, MAX_LEN)
        loader = DataLoader(ds, batch_size=INFER_BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        # Stage 1: demand_related + npc_related
        all_d, all_n = [], []
        with torch.no_grad():
            for batch in loader:
                ids  = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    dp, np_ = s1_model(ids, mask)
                all_d.extend(["T" if x == 1 else "F" for x in dp.argmax(1).cpu().numpy()])
                all_n.extend(["T" if x == 1 else "F" for x in np_.argmax(1).cpu().numpy()])

        chunk["demand_related"] = all_d
        chunk["npc_related"]    = all_n

        # Stage 2: solve_score (仅对 demand=T 的行)
        solve_scores = [None] * len(chunk)
        demand_t_idx = [i for i, d in enumerate(all_d) if d == "T"]

        if demand_t_idx:
            t_texts   = [texts[i]   for i in demand_t_idx]
            t_demands = [demands[i] for i in demand_t_idx]
            s2_ds     = InferDataset(t_texts, t_demands, tokenizer, MAX_LEN)
            s2_loader = DataLoader(s2_ds, batch_size=INFER_BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            s2_preds  = []
            with torch.no_grad():
                for batch in s2_loader:
                    ids  = batch["input_ids"].to(device, non_blocking=True)
                    mask = batch["attention_mask"].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=USE_AMP):
                        sp = s2_model(ids, mask)
                    s2_preds.extend([ID2SCORE[x] for x in sp.argmax(1).cpu().numpy()])
            for list_pos, chunk_idx in enumerate(demand_t_idx):
                solve_scores[chunk_idx] = s2_preds[list_pos]

        chunk["solve_score"] = solve_scores

        chunk[OUT_COLS].to_csv(
            OUTPUT_CSV,
            mode="a" if not write_header else "w",
            header=write_header,
            index=False, encoding="utf-8-sig",
        )
        write_header = False

        processed += len(chunk)
        pbar.update(len(chunk))

        with open(PROGRESS_FILE, "w") as f:
            json.dump({"last_completed_idx": start_idx + processed,
                       "total_pairs": total_pairs,
                       "timestamp": datetime.now().isoformat()}, f)

        elapsed   = time.time() - t_total
        speed     = processed / elapsed if elapsed > 0 else 1
        remaining = (total_pairs - start_idx - processed) / speed if speed > 0 else 0
        pbar.set_postfix({"速度": f"{speed:.0f}对/s", "剩余": format_time(remaining)})

        del chunk, ds; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()
    logger.info(f"全量推理完成, 共 {total_pairs:,} 对, 总耗时: {format_time(time.time()-t_total)}")
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)


# ============================================================
# 主流程
# ============================================================
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("人民日报 BERT 两阶段训练与全量推理")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    for path, name in [(MODEL_PATH, "BERT模型"), (TRAIN_CKPT, "训练数据"), (NEWS_CSV, "推理数据")]:
        if not os.path.exists(path):
            logger.error(f"{name} 不存在: {path}")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    logger.info("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # ── Stage 1 ──────────────────────────────────────────────
    s1_best = os.path.join(CKPT_DIR, "best_model_stage1.pt")
    if not os.path.exists(s1_best):
        logger.info("Stage 1 未训练, 开始训练...")
        df = load_labeled_data()
        train_ds, val_ds, test_ds, test_df = prepare_stage1(df, tokenizer)

        s1_model = Stage1Model(MODEL_PATH).to(device)
        logger.info(f"S1 参数量: {sum(p.numel() for p in s1_model.parameters())/1e6:.1f}M")

        t_s1 = time.time()
        s1_model = train_stage1(s1_model, train_ds, val_ds, device)
        logger.info(f"Stage 1 训练耗时: {format_time(time.time()-t_s1)}")
        evaluate_stage1(s1_model, test_ds, device)

        del train_ds, val_ds, test_ds, test_df, df; gc.collect()
    else:
        logger.info(f"发现 Stage1 已训练模型, 跳过训练")
        s1_model = Stage1Model(MODEL_PATH).to(device)
        s1_model.load_state_dict(torch.load(s1_best, map_location=device, weights_only=False))
        eval_path = os.path.join(CKPT_DIR, "eval_stage1.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                logger.info(f"S1 已有评估结果: {json.load(f)}")

    s1_model.eval()

    # ── Stage 2 ──────────────────────────────────────────────
    s2_best = os.path.join(CKPT_DIR, "best_model_stage2.pt")
    if not os.path.exists(s2_best):
        logger.info("Stage 2 未训练, 开始训练...")
        if "df" not in dir():
            df = load_labeled_data()
        train_ds2, val_ds2, test_ds2, test_df2 = prepare_stage2(df, tokenizer)

        # Stage 2 模型: 用 BERT 路径初始化, 然后载入 Stage 1 的 BERT 权重
        s2_model = Stage2Model(MODEL_PATH).to(device)
        # 从 Stage1 迁移 BERT 权重 (solve_head 随机初始化)
        s1_state  = s1_model.state_dict()
        s2_state  = s2_model.state_dict()
        bert_keys = {k: v for k, v in s1_state.items() if k.startswith("bert.")}
        s2_state.update(bert_keys)
        s2_model.load_state_dict(s2_state)
        logger.info(f"  S2 模型从 Stage1 热启动 (迁移 {len(bert_keys)} 个 BERT 参数)")
        logger.info(f"  S2 参数量: {sum(p.numel() for p in s2_model.parameters())/1e6:.1f}M")

        t_s2 = time.time()
        s2_model = train_stage2(s2_model, train_ds2, val_ds2, device)
        logger.info(f"Stage 2 训练耗时: {format_time(time.time()-t_s2)}")
        evaluate_stage2(s2_model, test_ds2, device)

        del train_ds2, val_ds2, test_ds2, test_df2; gc.collect()
        if "df" in dir():
            del df; gc.collect()
    else:
        logger.info(f"发现 Stage2 已训练模型, 跳过训练")
        s2_model = Stage2Model(MODEL_PATH).to(device)
        s2_model.load_state_dict(torch.load(s2_best, map_location=device, weights_only=False))
        eval_path = os.path.join(CKPT_DIR, "eval_stage2.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                logger.info(f"S2 已有评估结果: {json.load(f)}")

    s2_model.eval()

    # ── 全量推理 ─────────────────────────────────────────────
    run_inference(s1_model, s2_model, tokenizer, device)

    logger.info("=" * 60)
    logger.info(f"全部完成! 总耗时: {format_time(time.time()-t_start)}")
    logger.info(f"输出文件: {OUTPUT_CSV}")
    logger.info("=" * 60)

    if AUTO_SHUTDOWN:
        logger.info("30秒后自动关机...")
        try:
            time.sleep(30); os.system("shutdown")
        except KeyboardInterrupt:
            logger.info("已取消关机")


if __name__ == "__main__":
    main()