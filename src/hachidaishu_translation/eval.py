#!/usr/bin/env python

# FIXME adapt to new code
# Experiment:
# - how many samples to translate to be able to evaluate models?
# - add RIBES?

# Contamination check:
# - check using kokindb.txt
# - check prevalance of translations online (how? search API?)
# - how many samples to check?

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.chrf_score import sentence_chrf
from collections import defaultdict
import numpy as np
import re

NUM_RUNS = 3

# Path to the CSV file containing the model outputs
csv_file = "jadh2024-waka-llm-results.csv"

# Gold standard translations
gold_translations = [
    # TODO get from gen
]


def preprocess(text, type="word"):
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    if type == "word":
        return text.lower().strip().split()
    elif type == "char":
        return [char for char in text.lower().strip()]
    else:
        raise ValueError("Invalid type")
    # Tokenize by slashes (alternative eval)
    return [line.strip().lower().split() for line in text.split("/")]


if __name__ == "__main__":
    # Load results from CSV and calculate scores
    # We use a naive CSV parser here, assuming that the CSV file is not well-formed in the result header
    results = defaultdict(list)
    header = True
    with open(csv_file, encoding="utf-8") as file:
        for line in file:
            if header:
                header = False
                continue
            fields = line.strip().split(",")
            model = fields[0]
            run = fields[1]
            result = ",".join(fields[2:])
            if result == "":  # Skip empty results (i.e. due to rate limits)
                continue
            results[model].append(result)
    for model in results.keys():
        results[model] = list(set(results[model]))[
            :NUM_RUNS
        ]  # Cap to NUM_RUNS to keep the same number as the gold translations

    # Calculate BLEU and CHRF for each model
    references_word = [
        preprocess(translation, type="word") for translation in gold_translations
    ]
    references_char = [
        preprocess(translation, type="char") for translation in gold_translations
    ]
    scores = {}
    for model, model_results in results.items():
        model_results_words = [
            preprocess(result, type="word") for result in model_results
        ]
        model_results_chars = [
            preprocess(result, type="word") for result in model_results
        ]
        # >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]
        # >>> hypotheses = [hyp1, hyp2]
        # >>> corpus_bleu(list_of_references, hypotheses)
        bleu_score = corpus_bleu(
            [references_word] * len(model_results),
            model_results_words,
        )
        # corpus_chrf([ref1, ref2, ref1, ref2], [hyp1, hyp2, hyp2, hyp1])
        # As the above does not compute macro-average CHRF score across all references, we compute it manually
        chrf_scores = []
        for hyp in model_results_chars:
            chrf_scores.append(
                np.mean([sentence_chrf(ref, hyp) for ref in references_char])
            )
        macro_chrf_score = np.mean(chrf_scores)

        scores[model] = (bleu_score, macro_chrf_score)

    # Print results
    print("Gold translations:")
    print("\n".join(gold_translations))
    for model, text in results.items():
        print(f"Model: {model}")
        for i, result in enumerate(text):
            print(f"Run {i+1}: {result}")
    for model, (bleu, chrf) in scores.items():
        print(f"Model: {model}, BLEU: {bleu:.4f}, CHRF: {chrf:.4f}")
