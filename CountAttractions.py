import glob
import json
import os
from collections import defaultdict
from datasets import load_dataset

# Chỉ cần trỏ đến thư mục 'dataset' gốc
BASE_DIR = r"D:\Github\VQA_VN_Destination\dataset"

print(f"Đang tự động tìm kiếm các file .arrow trong {BASE_DIR}...")
# Tìm đệ quy toàn bộ file arrow
arrow_files = sorted(glob.glob(os.path.join(BASE_DIR, "**", "*.arrow"), recursive=True))

if not arrow_files:
    raise FileNotFoundError(f"Không tìm thấy file .arrow nào bên trong {BASE_DIR}. Vui lòng kiểm tra lại ổ D:!")

print(f"-> Đã tìm thấy {len(arrow_files)} files .arrow!")

# Nạp trực tiếp từ danh sách file đã tìm được
dataset = load_dataset("arrow", data_files=arrow_files, split="train")

location_counts = defaultdict(int)
TARGET_QUESTION = "cái nơi trong ảnh này tên là gì?"

print("Bắt đầu quét...")

for sample in dataset:
    convs = sample.get("conversations")
    if not convs:
        continue

    if isinstance(convs, str):
        try:
            convs = json.loads(convs)
        except json.JSONDecodeError:
            continue

    for i in range(len(convs) - 1):
        curr_turn = convs[i]
        next_turn = convs[i + 1]

        if (
            curr_turn.get("role") == "user"
            and TARGET_QUESTION in curr_turn.get("content", "").strip().lower()
            and next_turn.get("role") == "assistant"
        ):
            loc_name = next_turn.get("content", "").strip()
            if loc_name:
                if loc_name not in location_counts:
                    print(f"[MỚI] #{len(location_counts) + 1}: {loc_name}")
                location_counts[loc_name] += 1

OUTPUT_FILE = "destinations.jsonl"
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for loc, count in location_counts.items():
        f.write(json.dumps({"destination": loc, "sample_count": count}, ensure_ascii=False) + "\n")

print(f"\nHoàn tất! Tìm thấy {len(location_counts)} địa điểm. Đã lưu vào {OUTPUT_FILE}")