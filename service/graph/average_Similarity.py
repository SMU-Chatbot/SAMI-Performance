import json
import matplotlib.pyplot as plt
from config.path import get_project_paths
import numpy as np

paths = get_project_paths()

# 사용자에게 파일명 입력받기
print("데이터셋을 추출하고자 하는 파일명을 입력해주세요(.json 제외, 콤마로 구분 가능):")
input_files = input().strip().split(",")

# 데이터 추출
labels = []
avg_values = []

for dataset in input_files:
    dataset = dataset.strip()
    # 파일 경로 설정 (LLM, SBERT, BLEU_ROUGE 디렉토리 안에서 찾아봄)
    possible_dirs = ["LLM_DIR", "SBERT_DIR", "BLEU_ROUGE_DIR"]
    file_path = None
    for d in possible_dirs:
        temp_path = paths[d] / f"{dataset}.json"
        if temp_path.exists():
            file_path = temp_path
            break

    if file_path is None:
        print(f"{dataset}.json 파일을 찾을 수 없습니다.")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 평균값 추출
    if "average_similarity" in data:
        labels.append(dataset)
        avg_values.append(data["average_similarity"])
    elif "average_scores" in data:
        for metric in ["ROUGE-1", "ROUGE-2", "ROUGE-L"]:  # BLEU 제외
            labels.append(f"{dataset}_{metric}")
            avg_values.append(data["average_scores"][metric])

# 그래프 그리기
fig, ax = plt.subplots(figsize=(16,6))  # 가로 길이 16으로 늘림
x = np.arange(len(labels)) * 2.5  # 막대 간격 2.5로 늘림
bars = ax.bar(x, avg_values, color=['skyblue','lightgreen','salmon','orange','purple'], width=0.5)

ax.set_ylim(0,1)
ax.set_ylabel("평균 점수")
ax.set_title("Average Similarity Score")

# 막대 위에 값 표시
for bar, val in zip(bars, avg_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

# x축 레이블 맞추기
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)


# 이미지 저장
save_dir = paths["OUTPUT_DIR"] / "results_images"
save_dir.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(save_dir / "average_similarity_comparison.png", dpi=300)
plt.show()
