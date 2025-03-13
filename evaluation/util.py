import numpy as np

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction

nltk.download('punkt')
nltk.download('punkt_tab')

def compute_bleu_representativeness(input_dataset, integrated_dataset, use_n_grams=4, smoothing_function = SmoothingFunction().method1) -> np.floating:
    """
    Compute BLEU-based representativeness score between input and integrated datasets.

    :param input_dataset: List of text entries from the input dataset
    :param integrated_dataset: List of text entries from the integrated dataset
    :param use_n_grams: Number of n-grams to use (BLEU 1, 2, 3, 4)
    :param smoothing_function: Smoothing function to use for computing BLEU
    :return: Average BLEU score as the representativeness measure
    """
    assert use_n_grams in (1,2,3,4)
    weights = np.array([0., 0., 0., 0.])
    weights[:use_n_grams] = 1.0/float(use_n_grams)

    integrated_tokens = [word_tokenize(entry.lower()) for entry in integrated_dataset]
    input_tokens = [word_tokenize(entry.lower()) for entry in input_dataset]

    bleu_scores = []

    for input_entry in input_tokens:
        item_bleu_scores = max([sentence_bleu([integrated_entry], input_entry, smoothing_function=smoothing_function, weights=weights) for integrated_entry in integrated_tokens])
        bleu_scores.append(item_bleu_scores)

    return np.average(bleu_scores)

def calc_bleu(gt: set[tuple], pred: set[tuple]):
    serialized_results = [", ".join(str(v) for v in x) for x in pred]
    serialized_ground_truth = [", ".join(str(v) for v in x) for x in gt]

    scores = {}

    if calc_bleu:
        for i in range(4):
            # if Recall == 1.0 -> BLEU is always 1
            if len(gt - pred) == 0:
                scores[f"bleu{i + 1}"] = 1.0
                continue

            try:
                scores[f"bleu{i+1}"] = compute_bleu_representativeness(serialized_ground_truth, serialized_results, use_n_grams=i+1)
            except:
                scores[f"bleu{i+1}"] = -1.0

    return scores

def calculate_metrics(gt: set[tuple], pred: set[tuple], runtime: float) -> dict:
    scores = {}
    tps, fns, fps = gt & pred, gt - pred, pred - gt
    tp, fn, fp = len(tps), len(fns), len(fps)

    scores["Precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
    scores["Recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    scores["F1 Score"] = (2 * scores["Precision"] * scores["Recall"]) / (scores["Precision"] + scores["Recall"]) \
        if (scores["Precision"] + scores["Recall"]) > 0 else 0
    scores["tp"] = tp
    scores["fn"] = fn
    scores["fp"] = fp
    scores["runtime"] = runtime
    scores["pred"] = pred

    return scores