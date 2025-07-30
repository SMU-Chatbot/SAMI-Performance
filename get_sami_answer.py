import json
import requests
from tqdm import tqdm
import time
from pathlib import Path
import tiktoken

WORK_DIR = Path(__file__).parent

DATA_DIR = WORK_DIR / "data"
OUTPUT_DIR = WORK_DIR / "output"
Q_DATASET_DIR = OUTPUT_DIR / "q_dataset"
A_DATASET_DIR = OUTPUT_DIR / "a_dataset"

input_dataset = input("Q데이터셋 입력(.json 제외): ")

dataset = input_dataset + ".json"

with open(Q_DATASET_DIR/dataset, "r", encoding="utf-8") as f:
    questions = json.load(f)

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

results = []

for item in tqdm(questions):
    question = item["question"]

    try:
        start_time = time.time()

        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question},
            timeout=30
        )
        response.raise_for_status()

        elapsed_time = time.time() - start_time
        answer = response.json().get("answer", "")

        num_tokens = len(encoding.encode(answer))

        results.append({
            "answer": answer,
            "response_time_sec": round(elapsed_time, 2),
            "token_amount": num_tokens
        })

    except Exception as e:
        print(f"[오류] 질문: {question} / 오류: {e}")
        results.append({
            "answer": f"ERROR: {e}",
            "response_time_sec": 0,
            "token_amount": 0
        })

    time.sleep(0.3)

result = "sami_" + input_dataset.split("_q_dataset")[0] + "_a_dataset.json"

print(result)

with open(A_DATASET_DIR/result, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)