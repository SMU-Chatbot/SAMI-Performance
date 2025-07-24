from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import json
from pathlib import Path

WORK_DIR = Path(__file__).parent

DATA_DIR = WORK_DIR / "data"
OUTPUT_DIR = WORK_DIR / "output"
Q_DATASET_DIR = OUTPUT_DIR / "q_dataset"
A_DATASET_DIR = OUTPUT_DIR / "a_dataset"
SIMILARITY_DIR = OUTPUT_DIR / "similarity"
BLEU_ROUGE_DIR = SIMILARITY_DIR / "BLEU_ROUGE"

input_ref_data = input("QnA의 A 데이터셋을 입력하세요(.json 제외, 예시: student_a_dataset): ")
input_candidates_data = input("SAMI의 A 데이터셋을 입력하세요(.json 제외, 예시: sami_student_a_dataset): ")

ref_data = input_ref_data + ".json"
candidates_data = input_candidates_data + ".json"

# 1. 데이터 불러오기
with open(A_DATASET_DIR/ref_data, "r", encoding="utf-8") as f:
    references = json.load(f)

with open(A_DATASET_DIR/candidates_data, "r", encoding="utf-8") as f:
    candidates = json.load(f)

# 2. BLEU와 ROUGE 점수 저장
results = []

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
smooth_fn = SmoothingFunction().method1

for ref_item, gen_item in zip(references, candidates):
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

    print(f"bleu: {bleu:.4f}")
    print(f"ROUGE-1: {rouge_1:.4f}")
    print(f"ROUGE-2: {rouge_2:.4f}")
    print(f"ROUGE-L: {rouge_l:.4f}")

result = input_ref_data.split("_a")[0] + "bleu_rouge_results.json"

# 3. 결과 저장
with open(BLEU_ROUGE_DIR/result, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
