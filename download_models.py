"""
模型下载脚本 (云服务器运行)
============================
下载 hfl/chinese-roberta-wwm-ext-large 到本地数据目录。
AutoDL 已内置 hf-mirror.com 镜像, 速度较快。

运行方式:
  cd /root/autodl-tmp/citizens-will-and-politics
  python code/download_models.py
"""

import os
import sys
import time
from datetime import timedelta

# AutoDL HuggingFace 镜像 (国内加速)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("安装 huggingface_hub ...")
    os.system(f"{sys.executable} -m pip install huggingface_hub -q")
    from huggingface_hub import snapshot_download

# ============================================================
# 配置
# ============================================================
PROJECT_DIR = "/root/autodl-tmp/citizens-will-and-politics"
DATA_DIR    = os.path.join(PROJECT_DIR, "data")

MODELS = [
    {
        "repo_id":   "hfl/chinese-roberta-wwm-ext-large",
        "local_dir": os.path.join(DATA_DIR, "chinese-roberta-wwm-ext-large"),
        "desc":      "Stage1 + Stage2 共用骨干 (330M, 中文大模型)",
    },
]


def download_model(repo_id: str, local_dir: str, desc: str):
    if os.path.exists(os.path.join(local_dir, "config.json")):
        print(f"[跳过] {desc} 已存在: {local_dir}")
        return

    os.makedirs(local_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"下载: {desc}")
    print(f"来源: {repo_id}")
    print(f"目标: {local_dir}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )
        elapsed = time.time() - t0
        print(f"✓ 下载完成: {desc}  耗时 {timedelta(seconds=int(elapsed))}")
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("  请尝试手动下载或检查网络连接")
        sys.exit(1)


def main():
    print(f"HF 镜像: {os.environ.get('HF_ENDPOINT', '默认')}")
    print(f"数据目录: {DATA_DIR}")
    os.makedirs(DATA_DIR, exist_ok=True)

    for m in MODELS:
        download_model(**m)

    print("\n全部模型下载完成！")
    print("下一步: python code/train_and_infer_purpose1_bert.py")


if __name__ == "__main__":
    main()
