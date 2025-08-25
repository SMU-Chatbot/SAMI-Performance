from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import json
from config.path import get_project_paths
from tqdm import tqdm
from utils.io_utils import save_json, load_json
from utils.input_utils import *

paths = get_project_paths()

ref_data = get_filename("QnA의 A 데이터셋을 입력하세요(.json 제외, 예시: student_a_dataset): ")
candidates_data = get_filename("SAMI의 A 데이터셋을 입력하세요(.json 제외, 예시: sami_student_a_dataset): ")

# 1. 데이터 불러오기
references = load_json(paths["A_DATASET_DIR"]/ref_data)
candidates = load_json(paths["A_DATASET_DIR"]/candidates_data)

# ✅ candidates 구조 맞추기
if isinstance(candidates, dict) and "results" in candidates:
    candidates_list = candidates["results"]
elif isinstance(candidates, list):
    if all(isinstance(c, str) for c in candidates):
        candidates_list = [{"answer": c} for c in candidates]
    else:
        candidates_list = candidates
else:
    raise ValueError("지원하지 않는 candidates 데이터 구조")

references_list = references

# 2. BLEU와 ROUGE 점수 저장
results = []

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
smooth_fn = SmoothingFunction().method1

bleu_scores, rouge1_scores, rouge2_scores, rougel_scores = [], [], [], []

for ref_item, gen_item in tqdm(
        zip(references_list, candidates_list),
        total=len(references_list),
        desc="BLEU_ROUGE 기반 문장 유사도 평가 중"
):
    ref = ref_item.get("answer", "")
    cand = gen_item.get("answer", "")

    # BLEU
    bleu = sentence_bleu([ref.split()], cand.split(), smoothing_function=smooth_fn)

    # ROUGE
    rouge = scorer.score(ref, cand)
    rouge_1 = rouge["rouge1"].fmeasure
    rouge_2 = rouge["rouge2"].fmeasure
    rouge_l = rouge["rougeL"].fmeasure

    # 개별 결과 저장
    results.append({
        "reference": ref,
        "generated": cand,
        "BLEU": round(bleu, 4),
        "ROUGE-1": round(rouge_1, 4),
        "ROUGE-2": round(rouge_2, 4),
        "ROUGE-L": round(rouge_l, 4)
    })

    # 평균 계산용 리스트에 추가
    bleu_scores.append(bleu)
    rouge1_scores.append(rouge_1)
    rouge2_scores.append(rouge_2)
    rougel_scores.append(rouge_l)

# ✅ 평균 점수 계산 (소수점 둘째 자리까지)
avg_bleu = round(sum(bleu_scores) / len(bleu_scores), 2) if bleu_scores else 0.0
avg_rouge1 = round(sum(rouge1_scores) / len(rouge1_scores), 2) if rouge1_scores else 0.0
avg_rouge2 = round(sum(rouge2_scores) / len(rouge2_scores), 2) if rouge2_scores else 0.0
avg_rougel = round(sum(rougel_scores) / len(rougel_scores), 2) if rougel_scores else 0.0

# 결과 파일 이름
result_filename = make_bleu_rouge_results_name(ref_data)

# 3. 최종 결과 저장
final_output = {
    "average_scores": {
        "BLEU": avg_bleu,
        "ROUGE-1": avg_rouge1,
        "ROUGE-2": avg_rouge2,
        "ROUGE-L": avg_rougel
    },
    "results": results
}

save_json(paths["BLEU_ROUGE_DIR"]/result_filename, final_output)

print(f"평균 점수 (총 {len(results)}개 문항):")
print(f"  BLEU: {avg_bleu}")
print(f"  ROUGE-1: {avg_rouge1}")
print(f"  ROUGE-2: {avg_rouge2}")
print(f"  ROUGE-L: {avg_rougel}")
