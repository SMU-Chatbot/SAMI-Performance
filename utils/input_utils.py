def get_filename(prompt: str) -> str:
    filename = input(prompt) + ".json"
    return filename

def make_q_dataset_name(filename: str) -> str:
    return filename.split("_QnA")[0] + "_q_dataset.json"

def make_a_dataset_name(filename: str) -> str:
    return filename.split("_QnA")[0] + "_a_dataset.json"

def make_sami_a_dataset_name(filename: str) -> str:
    return "sami_" + filename.split("_q_dataset")[0] + "_a_dataset.json"

def make_bleu_rouge_results_name(filename: str) -> str:
    return filename.split("_a")[0] + "_BLEU_ROUGE_results.json"

def make_sbert_results_name(filename: str) -> str:
    return filename.split("_a")[0] + "_SBERT_results.json"