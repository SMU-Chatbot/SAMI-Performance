import requests
from tqdm import tqdm
import time
import tiktoken
import hashlib
from config.path import get_project_paths
from utils.io_utils import save_json, load_json
from utils.input_utils import *

paths = get_project_paths()

dataset = get_filename("Q데이터셋 입력(.json 제외): ")

questions = load_json(paths["Q_DATASET_DIR"]/dataset)

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

results = []
cache = {}  # 질문 캐시

def get_cache_key(question: str) -> str:
    return hashlib.md5(question.encode()).hexdigest()

for item in tqdm(questions):
    question = item["question"]
    key = get_cache_key(question)

    if key in cache:  # 캐싱된 답변 재사용
        results.append(cache[key])
        continue

    try:
        start_time = time.time()

        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": question, "max_tokens": 200},  # 응답 길이 제한
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
        results.append(result_item)
        cache[key] = result_item  # 캐시에 저장

    except Exception as e:
        results.append({
            "answer": f"ERROR: {e}",
            "response_time_sec": 0,
            "token_amount": 0
        })

# 평균값 계산
valid_results = [r for r in results if r["token_amount"] > 0 and r["response_time_sec"] > 0]
if valid_results:
    avg_tokens = sum(r["token_amount"] for r in valid_results) / len(valid_results)
    avg_time = sum(r["response_time_sec"] for r in valid_results) / len(valid_results)
else:
    avg_tokens, avg_time = 0, 0

print(f"평균 토큰량: {avg_tokens:.2f}")
print(f"평균 응답속도: {avg_time:.2f} 초")

# 결과 파일명에 caching_response_limit 추가
result = "caching_response_limit_" + make_sami_a_dataset_name(dataset)

# JSON 저장 (평균값 포함)
save_json(paths["A_DATASET_DIR"]/result, {
    "results": results,
    "average": {
        "token_amount": round(avg_tokens, 2),
        "response_time_sec": round(avg_time, 2)
    }
})
print(f"저장 완료: {result}")
