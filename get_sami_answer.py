import json
import requests
from tqdm import tqdm
import time
from pathlib import Path

WORK_DIR = Path(__file__).parent

DATA_DIR = WORK_DIR / "data"
OUTPUT_DIR = WORK_DIR / "output"
Q_DATASET_DIR = OUTPUT_DIR / "q_dataset"
A_DATASET_DIR = OUTPUT_DIR / "a_dataset"

input_dataset = input("Q데이터셋 입력(.json 제외): ")

dataset = input_dataset + ".json"

with open(Q_DATASET_DIR/dataset, "r", encoding="utf-8") as f:
    questions = json.load(f)

results = []

for item in tqdm(questions):
    question = item["question"]

    try:
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question},
            timeout=30
        )
        response.raise_for_status()

        answer = response.json().get("answer", "")

        results.append({
            "answer": answer,
        })

    except Exception as e:
        print(f"[오류] 질문: {question} / 오류: {e}")
        results.append({
            "answer": f"ERROR: {e}",
        })

    time.sleep(0.3)

result = "sami_" + input_dataset.split("_q_dataset")[0] + "_a_dataset.json"

print(result)

with open(A_DATASET_DIR/result, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)