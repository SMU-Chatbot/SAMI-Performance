import os
from openai import OpenAI
import json
from dotenv import load_dotenv
from tqdm import tqdm
from config.path import get_project_paths
from utils.io_utils import *
from utils.input_utils import *

paths = get_project_paths()

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")

LLM = OpenAI(api_key=api_key)

ground_truth_data = get_filename("QnA의 A 데이터셋을 입력하세요(.json 제외, 예시: student_a_dataset): ")
answer_data = get_filename("SAMI의 A 데이터셋을 입력하세요(.json 제외, 예시: sami_student_a_dataset): ")

ground_truth = load_json(paths["A_DATASET_DIR"]/ground_truth_data)
answers = load_json(paths["A_DATASET_DIR"]/answer_data)

# ✅ Ground Truth는 리스트 그대로
ground_truth_list = ground_truth

# ✅ SAMI 데이터셋은 dict 구조라면 results만 뽑기
if isinstance(answers, dict) and "results" in answers:
    answers_list = answers["results"]
else:
    answers_list = answers  # 그냥 리스트인 경우 그대로 사용

results = []

system_prompt = load_prompt(paths["WORK_DIR"] / "similarity_prompt.txt")

for ref_item, ans_item in tqdm(
        zip(ground_truth_list, answers_list),
        total=len(ground_truth_list),
        desc="LLM 기반 문장 유사도 평가 중"
):
    ref = ref_item.get("answer", "")
    ans = ans_item.get("answer", "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ref},
        {"role": "user", "content": ans}
    ]

    completion = LLM.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    similarity_text = completion.choices[0].message.content.strip()

    # 숫자로 변환 (예: "85" → 85.0)
    try:
        similarity_score = float(similarity_text)
    except ValueError:
        similarity_score = None  # 변환 실패 시 None 저장

    results.append({
        "reference": ref,
        "answer": ans,
        "similarity": similarity_score
    })

# ✅ 평균 계산
valid_scores = [r["similarity"] for r in results if r["similarity"] is not None]
avg_score = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0.0

# ✅ 결과 파일 저장
result_filename = ground_truth_data.split("_a")[0] + "_LLM_results.json"

final_output = {
    "average_similarity": avg_score,
    "results": results
}

save_json(paths["LLM_DIR"]/result_filename, final_output)

print(f"평균 유사도 점수: {avg_score:.2f} (총 {len(valid_scores)}개 문항)")


