import json
import matplotlib.pyplot as plt
from utils.io_utils import load_json
from config.path import get_project_paths

paths = get_project_paths()

file_names = []
averages = []

print("여러 결과 A데이터셋의 평균 Token 수와 Response Time를 비교합니다.")
print("결과 데이터셋 입력 후, 더 이상 입력이 없으면 'q'를 입력하세요.\n")

while True:
    filename = input("A데이터셋 입력(.json 제외): ")
    if filename.lower() == "q":
        break

    path = paths["A_DATASET_DIR"] / f"{filename}.json"
    try:
        data = load_json(path)
        avg = data.get("average", {})
        token_avg = avg.get("token_amount", 0)
        time_avg = avg.get("response_time_sec", 0)

        short_name = filename.replace("_q_dataset", "")
        file_names.append(short_name)
        averages.append((token_avg, time_avg))
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {path}")
    except json.JSONDecodeError:
        print(f"❌ JSON 파싱 실패: {path}")

if not averages:
    print("⚠️ 입력된 데이터가 없습니다.")
else:
    token_values = [a[0] for a in averages]
    time_values = [a[1] for a in averages]

    x = range(len(file_names))
    width = 0.25  # 막대 폭 줄임 (이전보다 좁게)
    gap = 0.02    # 막대 사이 약간의 간격

    # 그래프 크기: 파일 개수에 따라 가로 길이 늘림
    fig, ax1 = plt.subplots(figsize=(max(14, len(file_names) * 2), 6))

    # Token Amount (왼쪽)
    bars1 = ax1.bar([i - width/2 - gap for i in x], token_values, width, label="Token Amount", color="skyblue")
    ax1.set_ylabel("Average Token Amount")
    ax1.set_xticks(x)
    ax1.set_xticklabels(file_names, rotation=0, ha="center")

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.1f}", ha='center', va='bottom', fontsize=9)

    # Response Time (오른쪽)
    ax2 = ax1.twinx()
    bars2 = ax2.bar([i + width/2 + gap for i in x], time_values, width, label="Response Time (sec)", color="salmon")
    ax2.set_ylabel("Average Response Time (sec)")

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.05, f"{height:.2f}", ha='center', va='bottom', fontsize=9)

    # y축 범위 넉넉하게
    ax1.set_ylim(0, max(token_values)*1.2)
    ax2.set_ylim(0, max(time_values)*1.5)

    # 범례
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("평균 Token Amount vs Response Time")
    plt.tight_layout()

    # 결과 이미지 저장
    save_dir = paths["OUTPUT_DIR"] / "results_images"
    save_dir.mkdir(parents=True, exist_ok=True)
    output_image_path = save_dir / "average_token_response_time.png"
    plt.savefig(output_image_path, dpi=300)
    print(f"✅ 그래프 이미지 저장 완료: {output_image_path}")

    plt.show()
