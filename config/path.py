from pathlib import Path

def get_project_paths() -> dict:
    work_dir = Path(__file__).resolve().parents[1]
    output_dir = work_dir / "output"
    return {
        "WORK_DIR": work_dir,
        "DATA_DIR": work_dir/"data",
        "OUTPUT_DIR": output_dir,
        "Q_DATASET_DIR": output_dir/"q_dataset",
        "A_DATASET_DIR": output_dir/"a_dataset",
        "SBERT_DIR": output_dir/"similarity"/"SBERT",
        "BLEU_ROUGE_DIR": output_dir/"similarity"/"BLEU_ROUGE",
        "LLM_DIR": output_dir/"similarity"/"LLM",
    }