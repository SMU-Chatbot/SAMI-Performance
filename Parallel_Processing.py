import asyncio
import aiohttp
import time
import tiktoken
import hashlib
from tqdm.asyncio import tqdm_asyncio
from config.path import get_project_paths
from utils.io_utils import save_json, load_json
from utils.input_utils import *

paths = get_project_paths()

dataset = get_filename("Q데이터셋 입력(.json 제외): ")
questions = load_json(paths["Q_DATASET_DIR"]/dataset)
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

cache = {}  # 질문 캐시
max_concurrent_requests = 5  # 동시에 처리할 요청 수 제한
semaphore = asyncio.Semaphore(max_concurrent_requests)

def get_cache_key(question: str) -> str:
    return hashlib.md5(question.encode()).hexdigest()

async def process_question(session, item, retries=3, delay=2):
    question = item["question"]
    key = get_cache_key(question)

    # 캐시 확인
    if key in cache:
        return cache[key]

    for attempt in range(retries):
        try:
            async with semaphore:
                start_time = time.time()
                async with session.post(
                        "http://localhost:8000/ask",
                        json={"question": question, "max_tokens": 200},
                        timeout=30
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            elapsed_time = round(time.time() - start_time, 2)
            answer = data.get("answer", "")
            num_tokens = len(encoding.encode(answer))

            result_item = {
                "answer": answer,
                "response_time_sec": elapsed_time,
                "token_amount": num_tokens
            }
            cache[key] = result_item
            return result_item

        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                error_item = {
                    "answer": f"ERROR: {e}",
                    "response_time_sec": 0,
                    "token_amount": 0
                }
                cache[key] = error_item
                return error_item

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [process_question(session, q) for q in questions]
        results = await tqdm_asyncio.gather(*tasks)
    return results

# 실행
results = asyncio.run(main())

# 평균값 계산
valid_results = [r for r in results if r["token_amount"] > 0 and r["response_time_sec"] > 0]
if valid_results:
    avg_tokens = sum(r["token_amount"] for r in valid_results) / len(valid_results)
    avg_time = sum(r["response_time_sec"] for r in valid_results) / len(valid_results)
else:
    avg_tokens, avg_time = 0, 0

# 결과 파일명
result = "Async_Processing_" + make_sami_a_dataset_name(dataset)

# JSON 저장
save_json(paths["A_DATASET_DIR"]/result, {
    "results": results,
    "average": {
        "token_amount": round(avg_tokens, 2),
        "response_time_sec": round(avg_time, 2)
    }
})
print(f"저장 완료: {result}")
