import requests
from tqdm import tqdm
import time
import tiktoken
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.path import get_project_paths
from utils.io_utils import save_json, load_json
from utils.input_utils import *

paths = get_project_paths()

dataset = get_filename("Q데이터셋 입력(.json 제외): ")

questions = load_json(paths["Q_DATASET_DIR"]/dataset)

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
cache = {}

def get_cache_key(question: str) -> str:
    return hashlib.md5(question.encode()).hexdigest()

def process_question(item):
    question = item["question"]
    key = get_cache_key(question)

    if key in cache:  # 캐싱된 결과 있으면 그대로 반환
        return cache[key]

    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question, "max_tokens": 200},
            timeout=30
        )
        response.raise_for_status()

        elapsed_time = round(time.time() - start_time, 2)
        answer = response.json().get("answer", "")
        num_tokens = len(encoding.encode(answer))

        result_item = {
            "answer": answer,
            "response_time_sec": elapsed_time,
            "token_amount": num_tokens
        }
        cache[key] = result_item
        return result_item
    except Exception as e:
        return {
            "answer": f"ERROR: {e}",
            "response_time_sec": 0,
            "token_amount": 0
        }

results = []
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_question, q) for q in questions]
    for future in tqdm(as_completed(futures), total=len(questions)):
        results.append(future.result())

# 평균값 계산
valid_results = [r for r in results if r["token_amount"] > 0 and r["response_time_sec"] > 0]
if valid_results:
    avg_tokens = sum(r["token_amount"] for r in valid_results) / len(valid_results)
    avg_time = sum(r["response_time_sec"] for r in valid_results) / len(valid_results)
else:
    avg_tokens, avg_time = 0, 0

print(f"평균 토큰량: {avg_tokens:.2f}")
print(f"평균 응답속도: {avg_time:.2f} 초")

# 결과 파일명에 Hybrid_Strategy 추가
result = "Hybrid_Strategy_" + make_sami_a_dataset_name(dataset)

# JSON 저장 (평균값 포함)
save_json(paths["A_DATASET_DIR"]/result, {
    "results": results,
    "average": {
        "token_amount": round(avg_tokens, 2),
        "response_time_sec": round(avg_time, 2)
    }
})
print(f"저장 완료: {result}")
