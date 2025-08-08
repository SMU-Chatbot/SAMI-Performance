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

results = []

system_prompt = load_prompt(paths["WORK_DIR"] / "similarity_prompt.txt")

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

result = ground_truth_data.split("_a")[0] + "_LLM_results.json"

save_json(paths["LLM_DIR"]/result, results)