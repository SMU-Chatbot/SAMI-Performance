from config.path import get_project_paths
from utils.io_utils import save_json, load_json
from utils.input_utils import *

paths = get_project_paths()

dataset = get_filename("데이터셋을 추출하고자하는 파일명을 입력해주세요(.json 제외): ")

data = load_json(paths["DATA_DIR"]/dataset)

# 질문만 추출
questions_only = [{"question": item["question"]} for item in data]

# 답변만 추출
answers_only = [{"answer" : item["answer"]} for item in data]

q_result = make_q_dataset_name(dataset)
a_result = make_a_dataset_name(dataset)

#저장 파일명 확인
print(q_result)
print(a_result)

# 결과 저장
save_json(paths["Q_DATASET_DIR"]/q_result, questions_only)
save_json(paths["A_DATASET_DIR"]/a_result, answers_only)