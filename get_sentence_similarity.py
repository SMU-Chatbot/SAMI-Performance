import json
from sentence_transformers import SentenceTransformer, util
from config.path import get_project_paths
from tqdm import tqdm
from utils.io_utils import save_json, load_json
from utils.input_utils import *

paths = get_project_paths()

model = SentenceTransformer('jhgan/ko-sbert-sts')

ground_truth_data = get_filename("QnA의 A 데이터셋을 입력하세요(.json 제외, 예시: student_a_dataset): ")
answer_data = get_filename("SAMI의 A 데이터셋을 입력하세요(.json 제외, 예시: sami_student_a_dataset): ")

ground_truth = load_json(paths["A_DATASET_DIR"]/ground_truth_data)
answers = load_json(paths["A_DATASET_DIR"]/answer_data)

# ✅ SAMI 데이터셋이 dict일 경우 results만 꺼냄
if isinstance(answers, dict) and "results" in answers:
    answers_list = answers["results"]
else:
    answers_list = answers

results = []

for ref_item, ans_item in tqdm(
        zip(ground_truth, answers_list),
        total=len(ground_truth),
        desc="SBERT 기반 문장 유사도 평가 중"
):
    ref = ref_item.get("answer", "")
    ans = ans_item.get("answer", "")

    embeddings = model.encode([ref, ans], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()

    results.append({
        "reference": ref,
        "answer": ans,
        "similarity": round(score, 4)
    })

# 결과 파일명 생성
result = make_sbert_results_name(ground_truth_data)

save_json(paths["SBERT_DIR"]/result, results)
