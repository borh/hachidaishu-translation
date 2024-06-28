#!/usr/bin/env python

# TODO
# Contamination check:
# - check prevalance of translations online
# - how many samples to check?
from hachidaishu_translation.log import logger

from nltk.translate.chrf_score import sentence_chrf
from nltk.translate.meteor_score import single_meteor_score

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.translate.ribes_score import sentence_ribes
import re


def preprocess(text: str, type: str = "word") -> list[str]:
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    if type == "word":
        return text.lower().strip().split()
    elif type == "char":
        return [char for char in text.lower().strip()]
    else:
        raise ValueError("Invalid type")
    # Tokenize by slashes (alternative eval)
    return [line.strip().lower().split() for line in text.split("/")]


def eval_translation(gold_translation: str, translation: str) -> dict[str, float]:
    logger.info(f"Evaluating translation: {gold_translation} -> {translation}")
    gt_word = preprocess(gold_translation, type="word")
    gt_char = preprocess(gold_translation, type="char")

    t_word = preprocess(translation, type="word")
    t_char = preprocess(translation, type="char")

    return {
        "chrF": sentence_chrf(gt_char, t_char),
        "METEOR": single_meteor_score(gt_word, t_word),
        # "RIBES": sentence_ribes(gt_word, t_word),
        # "BLEU": sentence_bleu(gt_word, t_word, smoothing_function=SmoothingFunction.method0),
    }


def evaluate_all_translations(translations: list[dict]) -> list[dict]:
    """
    Evaluates all translations in the database against their gold standard translations.
    """
    evaluation_results = []

    for translation in translations:
        gold_translation = translation["gold_translation"]
        generated_translation = translation["generated_translation"]

        try:
            evaluation = eval_translation(gold_translation, generated_translation)
        except Exception as e:
            logger.error(f"Failed to evaluate translation: {e}, skipping...")
            continue
        evaluation.update(
            {
                "poem_id": translation["poem_id"],
                "translation_id": translation["translation_id"],
                "gold_translation": gold_translation,
                "generated_translation": generated_translation,
                "model": translation["model"],
                "method": translation["method"],
                "gen_type": translation["gen_type"],
                "alignment": translation["alignment"],
                "original_text": translation["original_text"],
            }
        )

        evaluation_results.append(evaluation)

    return evaluation_results


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_evaluations(evaluation_results: list[dict]):
    """
    Visualizes the distribution of evaluations per model and gen type and outputs top 10 best and worst translations.

    Parameters:
    - evaluation_results: List of dictionaries containing evaluation results.

    Outputs:
    - Saves the visualization as a 600dpi PNG file.
    - Outputs tables of top 10 best and worst translations according to each score.
    """
    # Convert evaluation results to a DataFrame
    df = pd.DataFrame(evaluation_results)

    # Calculate the correlation matrix for the metrics
    metrics = ["chrF", "METEOR"]
    correlation_matrix = df[metrics].corr()

    # Melt the DataFrame to long format for seaborn
    df_melted = df.melt(
        id_vars=["model", "gen_type"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )

    # Combine model and gen_type for better distinction in the plot
    df_melted["Model/GenType"] = df_melted["model"] + " / " + df_melted["gen_type"]

    # Plot settings
    plt.figure(figsize=(14, 10), dpi=600)

    # Create a subplot for the boxplot and strip plot
    plt.subplot(2, 1, 1)
    plt.title("Distribution of Evaluations per Model and Gen Type")

    # Create the boxplot
    sns.boxplot(
        x="Model/GenType", y="Score", hue="Metric", data=df_melted, palette="Set3"
    )

    # Add stripplot to show individual data points
    sns.stripplot(
        x="Model/GenType",
        y="Score",
        hue="Metric",
        data=df_melted,
        dodge=True,
        jitter=True,
        color="black",
        alpha=0.3,
    )

    # Adjust legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[: len(metrics)], labels[: len(metrics)], title="Metric")

    # Adjust x-axis labels and rotation
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model / GenType")
    plt.ylabel("Score")

    # Create a subplot for the correlation matrix
    plt.subplot(2, 1, 2)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Evaluation Metrics")

    # Save the plot
    plt.tight_layout()
    plt.savefig("evaluation_distribution.png", dpi=600)
    plt.close()

    # Output tables of top 10 best and worst translations for each metric
    for metric in metrics:
        best_translations = df.nlargest(10, metric)[
            [
                "model",
                "gen_type",
                metric,
                "original_text",
                "generated_translation",
                "gold_translation",
            ]
        ]
        worst_translations = df.nsmallest(10, metric)[
            [
                "model",
                "gen_type",
                metric,
                "original_text",
                "generated_translation",
                "gold_translation",
            ]
        ]

        print(f"\nTop 10 Best Translations for {metric.upper()}:")
        print(best_translations)
        best_translations.to_csv(f"top_10_best_{metric}.csv", index=False)

        print(f"\nTop 10 Worst Translations for {metric.upper()}:")
        print(worst_translations)
        worst_translations.to_csv(f"top_10_worst_{metric}.csv", index=False)


if __name__ == "__main__":
    print(eval_translation("The cat sat on the mat.", "cat on mat"))
    print(eval_translation("The cat sat on the mat.", "the cat on the mat"))

    from hachidaishu_translation.db import get_translations_and_gold, get_tokens
    from hachidaishu_translation.format import (
        get_index_alignment,
        compile_latex_tikz,
        make_unique,
    )

    translations = get_translations_and_gold()
    evals = evaluate_all_translations(translations)
    visualize_evaluations(evals)
    for eval in evals:
        tokens = get_tokens(eval["poem_id"], eval["translation_id"])
        aligns = []
        for o_i, t_i in eval["alignment"]:
            aligns.append(
                (tokens["original_tokens"][o_i], tokens["translated_tokens"][t_i])
            )
        # compile_latex_tikz(
        #     # aligns,
        #     eval["alignment"],
        #     make_unique(tokens["original_tokens"]),
        #     make_unique(tokens["translated_tokens"]),
        #     f"alignment_{eval['poem_id']}_{eval['translation_id']}.pdf",
        # )
