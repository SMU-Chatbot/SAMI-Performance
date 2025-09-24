import requests
from tqdm import tqdm
import time
import tiktoken
from config.path import get_project_paths
from utils.io_utils import save_json, load_json
from utils.input_utils import *

paths = get_project_paths()

dataset = get_filename("Q데이터셋 입력(.json 제외): ")

questions = load_json(paths["Q_DATASET_DIR"]/dataset)

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

results = []

for item in tqdm(questions):
    question = item["question"]

    try:
        start_time = time.time()

        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": f"{question}\n\n간단 설명 + 예시 1개."},
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

# 평균값 계산
valid_results = [r for r in results if r["token_amount"] > 0 and r["response_time_sec"] > 0]
if valid_results:
    avg_tokens = sum(r["token_amount"] for r in valid_results) / len(valid_results)
    avg_time = sum(r["response_time_sec"] for r in valid_results) / len(valid_results)
else:
    avg_tokens, avg_time = 0, 0


result = "v6_short_with_example.json"

# JSON 저장 (평균 포함)
save_json(paths["A_DATASET_DIR"]/result, {
    "results": results,
    "average": {
        "token_amount": round(avg_tokens, 2),
        "response_time_sec": round(avg_time, 2)
    }
})
print(f"저장 완료: {result}")
