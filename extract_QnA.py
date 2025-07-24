import json
from pathlib import Path

WORK_DIR = Path(__file__).parent

DATA_DIR = WORK_DIR / "data"
OUTPUT_DIR = WORK_DIR / "output"
Q_DATASET_DIR = OUTPUT_DIR / "q_dataset"
A_DATASET_DIR = OUTPUT_DIR / "a_dataset"

input_dataset = input("데이터셋을 추출하고자하는 파일명을 입력해주세요(.json 제외): ")

dataset = input_dataset + ".json"

# 원본 Q&A 파일 경로
with open(DATA_DIR / dataset, "r", encoding="utf-8") as f:
    data = json.load(f)

# 질문만 추출
questions_only = [{"question": item["question"]} for item in data]

# 답변만 추출
answers_only = [{"answer" : item["answer"]} for item in data]

q_result = input_dataset.split("_QnA")[0] + "_q_dataset.json"
a_result = input_dataset.split("_QnA")[0] + "_a_dataset.json"

#저장 파일명 확인
print(q_result)
print(a_result)

# 결과 저장
with open(Q_DATASET_DIR/q_result, "w", encoding="utf-8") as f:
    json.dump(questions_only, f, ensure_ascii=False, indent=2)

with open(A_DATASET_DIR/a_result, "w", encoding="utf-8") as f:
    json.dump(answers_only, f, ensure_ascii=False, indent=2)