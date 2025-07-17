import json

input_dataset = input("Q데이터셋을 추출하고자하는 파일명을 입력해주세요(.json 제외): ")

dataset = input_dataset + ".json"

# 원본 Q&A 파일 경로
with open(dataset, "r", encoding="utf-8") as f:
    data = json.load(f)

# 질문만 추출
questions_only = [{"question": item["question"]} for item in data]

result = input_dataset.split("_QnA")[0] + "_q_dataset.json"

print(result)

# 결과 저장
with open(result, "w", encoding="utf-8") as f:
    json.dump(questions_only, f, ensure_ascii=False, indent=2)