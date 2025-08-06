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

# 2. BLEU와 ROUGE 점수 저장
results = []

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
smooth_fn = SmoothingFunction().method1

# in tqdm(zip(ground_truth, answers), total=len(ground_truth), desc="문장 유사도 평가 중"):
for ref_item, gen_item in tqdm(zip(references, candidates), total=len(references) ,desc="BLEU_ROUGE 기반 문장 유사도 평가 중"):
    ref = ref_item["answer"]
    cand = gen_item["answer"]

    # BLEU: 참고 문장 하나
    bleu = sentence_bleu([ref.split()], cand.split(), smoothing_function=smooth_fn)

    # ROUGE
    rouge = scorer.score(ref, cand)
    rouge_1 = rouge["rouge1"].fmeasure
    rouge_2 = rouge["rouge2"].fmeasure
    rouge_l = rouge["rougeL"].fmeasure

    results.append({
        "reference": ref,
        "generated": cand,
        "BLEU": round(bleu, 4),
        "ROUGE-1": round(rouge_1, 4),
        "ROUGE-2": round(rouge_2, 4),
        "ROUGE-L": round(rouge_l, 4)
    })

result = make_bleu_rouge_results_name(ref_data)

# 3. 결과 저장
save_json(paths["BLEU_ROUGE_DIR"]/result, results)
