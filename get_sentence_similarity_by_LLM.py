import os
from openai import OpenAI
import json
from dotenv import load_dotenv
from tqdm import tqdm

from config.path import get_project_paths

paths = get_project_paths()

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")

LLM = OpenAI(api_key=api_key)

input_ground_truth_data = input("QnA의 A 데이터셋을 입력하세요(.json 제외, 예시: student_a_dataset): ")
input_answer_data = input("SAMI의 A 데이터셋을 입력하세요(.json 제외, 예시: sami_student_a_dataset): ")

ground_truth_dataset = input_ground_truth_data + ".json"
answer_dataset = input_answer_data + ".json"

with open(paths["A_DATASET_DIR"]/ground_truth_dataset, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

with open(paths["A_DATASET_DIR"]/answer_dataset, "r", encoding="utf-8") as f:
    answers = json.load(f)

results = []

with open("similarity_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

for ref_item, ans_item in tqdm(zip(ground_truth, answers), total=len(ground_truth), desc="LLM 기반 문장 유사도 평가 중"):
    ref = ref_item.get("answer", "")
    ans = ans_item.get("answer", "")

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": ref},
                {"role": "user", "content": ans}]

    completion = LLM.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )

    similarity = completion.choices[0].message.content

    results.append({
        "reference": ref,
        "answer": ans,
        "similarity": similarity
    })

result = input_ground_truth_data.split("_a")[0] + "_LLM_results.json"

with open(paths["LLM_DIR"]/result, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)