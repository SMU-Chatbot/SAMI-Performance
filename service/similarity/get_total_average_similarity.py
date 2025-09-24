import pandas as pd
from matplotlib import pyplot as plt
from config.path import get_project_paths
import numpy as np

paths = get_project_paths()

data = pd.read_excel('output/similarity/total_average_similarity.xlsx')

columns = data.columns.tolist()
values = data.iloc[0].tolist()


fig, ax = plt.subplots(figsize=(16,6))  # 가로 길이 16으로 늘림
x = np.arange(len(columns)) * 2.5  # 막대 간격 2.5로 늘림
bars = ax.bar(x, values, color=['skyblue','lightgreen','salmon','orange','purple'], width=0.5)

ax.set_ylim(0,1)
ax.set_ylabel("Average Score")
ax.set_title("Total Average of Similarity Score")

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(columns, rotation=0)

save_dir = paths["OUTPUT_DIR"] / "total_average_images"
save_dir.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(save_dir / "average_similarity_comparison.png", dpi=300)
plt.show()