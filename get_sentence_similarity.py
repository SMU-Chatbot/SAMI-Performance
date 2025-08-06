import json
from sentence_transformers import SentenceTransformer, util
from config.path import get_project_paths

paths = get_project_paths()

model = SentenceTransformer('jhgan/ko-sbert-sts')

input_ground_truth_data = input("QnA의 A 데이터셋을 입력하세요(.json 제외, 예시: student_a_dataset): ")
input_answer_data = input("SAMI의 A 데이터셋을 입력하세요(.json 제외, 예시: sami_student_a_dataset): ")

ground_truth_dataset = input_ground_truth_data + ".json"
answer_dataset = input_answer_data + ".json"

with open(paths["A_DATASET_DIR"]/ground_truth_dataset, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

with open(paths["A_DATASET_DIR"]/answer_dataset, "r", encoding="utf-8") as f:
    answers = json.load(f)

results = []

# ref = "상명대학교는 종로구에 있어요."
# ans = "상명대는 서울 종로구에 위치합니다."

for ref_item, ans_item in zip(ground_truth, answers):
    ref = ref_item.get("answer", "")
    ans = ans_item.get("answer", "")

    embeddings = model.encode([ref, ans], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()

    results.append({
        "reference": ref,
        "answer": ans,
        "similarity": round(score, 4)
    })
    print(f"유사도 점수: {score:.4f}")

result = input_ground_truth_data.split("_a")[0] + "_SBERT_results.json"

with open(paths["SBERT_DIR"]/result, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)