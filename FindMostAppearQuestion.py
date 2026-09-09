import glob
import json
import os
from collections import Counter
from datasets import load_dataset

BASE_DIR = r"D:\Github\VQA_VN_Destination\dataset"
OUTPUT_FILE = "list_most_appear_questions.jsonl"
MIN_APPEARANCE = 570

print(f"Đang tự động tìm kiếm các file .arrow trong {BASE_DIR}...")
arrow_files = sorted(glob.glob(os.path.join(BASE_DIR, "**", "*.arrow"), recursive=True))

if not arrow_files:
    raise FileNotFoundError(f"Không tìm thấy file .arrow nào bên trong {BASE_DIR}. Vui lòng kiểm tra lại ổ D:!")

print(f"-> Đã tìm thấy {len(arrow_files)} files .arrow!")

dataset = load_dataset("arrow", data_files=arrow_files, split="train")

question_counts = Counter()
print("Bắt đầu quét các câu hỏi của role 'user'...")

for sample in dataset:
    convs = sample.get("conversations")
    if not convs:
        continue

    if isinstance(convs, str):
        try:
            convs = json.loads(convs)
        except json.JSONDecodeError:
            continue

    for turn in convs:
        if turn.get("role") == "user":
            content = turn.get("content", "").strip()
            if content:
                question_counts[content] += 1

# Lọc các câu hỏi xuất hiện nhiều hơn 570 lần và sắp xếp giảm dần theo số lần xuất hiện
filtered_questions = [
    {"question": q, "count": cnt}
    for q, cnt in question_counts.most_common()
    if cnt > MIN_APPEARANCE
]

# Ghi ra file JSONL
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in filtered_questions:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nHoàn tất! Tìm thấy {len(filtered_questions)} câu hỏi xuất hiện > {MIN_APPEARANCE} lần.")
print(f"Kết quả đã được lưu tại: {OUTPUT_FILE}")