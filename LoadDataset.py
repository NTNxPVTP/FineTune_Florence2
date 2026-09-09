import os

# Bật tải siêu tốc đa luồng (Rust backend) của Hugging Face Hub
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from datasets import load_dataset

ds = load_dataset(
    "Qhuy204/VQA_VN_Destination", 
    cache_dir=r"D:\Github\VQA_VN_Destination\dataset"
)

print("Tải hoàn tất:", ds)