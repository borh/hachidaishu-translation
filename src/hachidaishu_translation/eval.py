#!/usr/bin/env python
import re
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.chrf_score import sentence_chrf
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.ribes_score import sentence_ribes
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from hachidaishu_translation.log import logger

plt.rcParams["font.family"] = ["Work Sans", "Source Han Serif"]
plt.rcParams["font.weight"] = "regular"
# pgf_with_latex = {
#     "text.usetex": True,  # use LaTeX to write all text
#     "pgf.rcfonts": False,  # Ignore Matplotlibrc
#     "pgf.preamble": r"\usepackage{color}",  # xcolor for colours
# }
# plt.rcParams.update(pgf_with_latex)

PREPROCESS_REGEX = re.compile(r"[^A-Za-z0-9 ]+")


def preprocess(text: str, token_type: str = "word") -> list[str]:
    text = PREPROCESS_REGEX.sub("", text)
    if token_type == "word":
        return text.lower().strip().split()
    elif token_type == "char":
        return [char for char in text.lower().strip()]
    else:
        raise ValueError("Invalid type")
    # Tokenize by slashes (alternative eval)
    return [line.strip().lower().split() for line in text.split("/")]


def eval_translation(
    gold_translation: str,
    translation: str,
    original_text: str,
    metrics: list[str] = ["chrF", "METEOR"],
    sentence_transformers_model=None,
    japanese_model=None,
) -> dict[str, float]:
    logger.info(
        f"Evaluating translation with {metrics}: {gold_translation} -> {translation}"
    )
    gt_word = preprocess(gold_translation, token_type="word")
    gt_char = preprocess(gold_translation, token_type="char")

    t_word = preprocess(translation, token_type="word")
    t_char = preprocess(translation, token_type="char")

    metric_functions = {
        "chrF": lambda: sentence_chrf(gt_char, t_char),
        "METEOR": lambda: single_meteor_score(gt_word, t_word),
        "RIBES": lambda: sentence_ribes(gt_word, t_word),
        "BLEU": lambda: sentence_bleu(
            gt_word,
            t_word,
            smoothing_function=SmoothingFunction().method4,
            auto_reweigh=True,
        ),
        "SentenceTransformers": lambda: cosine_similarity(
            sentence_transformers_model.encode([gold_translation]),
            sentence_transformers_model.encode([translation]),
        )[0][0],
        "SentenceTransformers_vs_original": lambda: cosine_similarity(
            japanese_model.encode([original_text]),
            japanese_model.encode([translation]),
        )[0][0],
    }

    evaluation = {
        metric: func() for metric, func in metric_functions.items() if metric in metrics
    }

    return evaluation


# TODO also implement whole-corpus evaluation, not just sentence-level
def evaluate_all_translations(
    translations: list[dict],
    metrics: list[str] = ["chrF", "METEOR"],
    sentence_transformers_model: str = "llmrails/ember-v1",
    japanese_model_name: str = "pkshatech/simcse-ja-bert-base-clcmlp",
) -> list[dict]:
    """
    Evaluates all translations in the database against their gold standard translations.

    Args:
        translations (list[dict]): List of translations to evaluate.
        metrics (list[str]): List of metrics to use for evaluation. Default is ["chrF", "METEOR"].

    Returns:
        list[dict]: List of evaluation results.
    """
    evaluation_results = []

    model = (
        SentenceTransformer(sentence_transformers_model, trust_remote_code=True)
        if "SentenceTransformers" in metrics
        else None
    )

    japanese_model = (
        SentenceTransformer(japanese_model_name, trust_remote_code=True)
        if "SentenceTransformers_vs_original" in metrics
        else None
    )

    for translation in translations:
        gold_translation = translation["gold_translation"]
        generated_translation = translation["generated_translation"]
        original_text = translation["original_text"]

        try:
            evaluation = eval_translation(
                gold_translation,
                generated_translation,
                original_text,
                metrics=metrics,
                sentence_transformers_model=model,
                japanese_model=japanese_model,
            )
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
                "translation_temperature": translation["translation_temperature"],
                "original_text": translation["original_text"],
            }
        )

        evaluation_results.append(evaluation)

    return evaluation_results


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-_\. ]", "_", name)


def visualize_evaluations(
    evaluation_results: list[dict],
    metrics: list[str] = ["chrF", "METEOR"],
    sentence_transformers_model: str | None = None,
    japanese_model_name: str = "pkshatech/simcse-ja-bert-base-clcmlp",
    num_runs: int | None = None,
    fontsize: int = 18,
):
    """
    Visualizes the distribution of evaluations per model and gen type and outputs top 10 best and worst translations.

    Parameters:
    - evaluation_results: List of dictionaries containing evaluation results.

    Outputs:
    - Saves the visualization as a PDF file.
    - Outputs tables of top 10 best and worst translations according to each score.
    """
    df = pd.DataFrame(evaluation_results)

    # Rename models containing '4o' to 'gpt-4o-2024-08-06' as well as add model name to ST
    df["model"] = df["model"].apply(lambda x: "gpt-4o-2024-08-06" if "4o" in x else x)
    sf_label = (
        f"SentenceTransformers cosine similarity\n({sentence_transformers_model})"
    )
    jc_label = f"SentenceTransformers_vs_original\n({japanese_model_name})"
    correlation_matrix = (
        df[metrics]
        .corr(method="spearman")
        .rename(
            index={
                "SentenceTransformers": sf_label,
                "SentenceTransformers_vs_original": jc_label,
            }
        )
    )
    correlation_matrix = correlation_matrix.rename(
        columns={
            "SentenceTransformers": sf_label,
            "SentenceTransformers_vs_original": jc_label,
        }
    )
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

    df = df.rename(
        columns={
            "SentenceTransformers": sf_label,
            "SentenceTransformers_vs_original": jc_label,
        }
    )
    metrics = [
        m
        if m not in ["SentenceTransformers", "SentenceTransformers_vs_original"]
        else (sf_label if m == "SentenceTransformers" else jc_label)
        for m in metrics
    ]

    # Create a color map for generation methods
    # gen_methods = df["model"].unique()
    # colors = sns.color_palette("cubehelix", len(gen_methods))
    # method_color_map = dict(zip(gen_methods, colors))

    # Combine model, method, and gen_type for better distinction in the plot
    df["Model/GenType"] = df.apply(
        lambda row: f"{row['model']} / {row['method']} / {row['gen_type']} / t={format(row['translation_temperature'], '.1f')}",
        axis=1,
    )

    # Determine the number of runs if not provided
    if num_runs is None:
        num_runs = df["Model/GenType"].value_counts().max()
        logger.info(f"Automatically determined num_runs: {num_runs}")

    metric_means = df.groupby("Model/GenType")[metrics].mean()
    metric_means["GeoMean"] = metric_means.prod(axis=1) ** (1 / len(metrics))
    order = metric_means["GeoMean"].sort_values(ascending=False).index
    df_melted = df.melt(
        id_vars=["model", "method", "gen_type", "translation_temperature"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )

    # Combine model and gen_type for better distinction in the plot
    df_melted["Model/GenType"] = df_melted.apply(
        lambda row: f"{row['model']} / {row['method']} / {row['gen_type']} / t={format(row['translation_temperature'], '.1f')}",
        axis=1,
    )

    with PdfPages("evaluation_distribution.pdf") as pdf:
        # Boxplot Page
        fig, ax1 = plt.subplots(figsize=(28, 34), dpi=600)
        plt.title(
            "Distribution of evaluations per model, generation method and type ordered by geometric mean of metrics",
            loc="left",
            fontsize=fontsize,
        )

        sns.violinplot(
            ax=ax1,
            x="Score",
            y="Model/GenType",
            hue="Metric",
            data=df_melted,
            palette="colorblind",
            order=order,
            density_norm="width",
            inner="quartile",  # Shows quartiles inside the violins
        )

        # # Underline the y-axis labels with the corresponding color
        # for label in ax1.get_yticklabels():
        #     model_gen_type = label.get_text()
        #     method = model_gen_type.split(" / ")[0]
        #     color = method_color_map.get(method, "black")

        #     # Get the position of the label
        #     x, y = label.get_position()

        #     # Add a line under the label
        #     ax1.annotate(
        #         "",
        #         xy=(0, y),
        #         xycoords="data",
        #         xytext=(-0.5, y),
        #         textcoords="data",
        #         arrowprops=dict(arrowstyle="-", color=color, lw=3),
        #         annotation_clip=False,
        #     )

        evaluation_counts = df["Model/GenType"].value_counts()
        success_rates = (evaluation_counts / num_runs) * 100
        logger.info(
            f"evaluation_counts: {evaluation_counts} / num_runs: {num_runs} => success_rates: {success_rates}"
        )

        sns.stripplot(
            ax=ax1,
            x="Score",
            y="Model/GenType",
            hue="Metric",
            data=df_melted,
            dodge=True,
            jitter=True,
            palette="dark:black",
            alpha=0.5,
            size=2.5,
        )

        if num_runs is not None:
            logger.info(f"Metrics: {metrics}")
            for i, model_gen_type in enumerate(order):
                success_rate = success_rates[model_gen_type]
                logger.warning(
                    f"Success rate for {model_gen_type}: {success_rate:.0f}%"
                )
                ax1.text(
                    1.05,  # Position just outside the plot area
                    1
                    - (i + 0.5)
                    / len(order),  # Normalize y-position to middle of each box
                    f"{success_rate:.0f}%",
                    va="center",
                    ha="right",
                    fontsize=fontsize,
                    color="black",
                    transform=ax1.transAxes,  # Use axes coordinates
                )

        for i in range(1, len(order)):
            plt.axhline(y=i - 0.5, color="gray", alpha=0.5, linestyle="--")
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.title(
            "Distribution of evaluations per model, generation method and type\nordered by geometric mean of metrics",
            loc="left",
            fontsize=fontsize * 1.5,
            fontweight="bold",
        )
        plt.legend(
            handles[: len(metrics)],
            labels[: len(metrics)],
            title="Metric",
            title_fontsize=fontsize,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            fontsize=fontsize,
        )

        plt.ylabel(
            "[model / generation method / type / temperature]",
            fontsize=fontsize * 1.5,
            fontweight="bold",
        )
        plt.xlabel("Score", fontsize=fontsize * 1.5, fontweight="bold")
        ax1.tick_params(axis="x", labelsize=fontsize)
        ax1.tick_params(axis="y", labelsize=fontsize)
        plt.subplots_adjust(left=0.35, right=0.95, top=0.94, bottom=0.05)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # Correlation Matrix Page
        fig, ax2 = plt.subplots(figsize=(12, 10), dpi=600)
        plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.1)
        sns.heatmap(
            correlation_matrix,
            ax=ax2,
            mask=mask,
            annot=True,
            cmap="Blues",
            vmin=0,
            fmt=".2f",
            square=True,
            cbar_kws={"shrink": 0.5},
            annot_kws={"size": 20},
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        ax2.set_xlabel(ax2.get_xlabel(), fontsize=fontsize)
        ax2.set_ylabel(ax2.get_ylabel(), fontsize=fontsize)
        ax2.tick_params(axis="x", labelsize=fontsize)
        ax2.tick_params(axis="y", labelsize=fontsize)
        plt.title(
            "Correlation matrix of evaluation metrics (Spearman)",
            fontsize=fontsize * 1.5,
            fontweight="bold",
        )
        # plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    # Replace newlines in column names with whitespace
    # df.columns = df.columns.str.replace("\n", " ")

    # Output tables of top 10 best and worst translations for each metric
    for metric in metrics:
        model_scores = (
            df.groupby(["model", "method", "gen_type"])[metric]
            .mean()
            .sort_values(ascending=False)
        )
        print(f"\nAggregate Scores for {metric.upper()} (Best to Worst):")
        print(model_scores)

    for metric in metrics:
        best_translations = df.nlargest(10, metric)[
            [
                "model",
                "method",
                "gen_type",
                "translation_temperature",
                metric,
                "original_text",
                "generated_translation",
                "gold_translation",
            ]
        ]
        worst_translations = df.nsmallest(10, metric)[
            [
                "model",
                "method",
                "gen_type",
                "translation_temperature",
                metric,
                "original_text",
                "generated_translation",
                "gold_translation",
            ]
        ]

        print(f"\nTop 10 Best Translations for {metric.upper()}:")
        print(best_translations)
        best_translations.to_csv(
            f"top_10_best_{sanitize_filename(metric)}.csv",
            index=False,
        )

        print(f"\nTop 10 Worst Translations for {metric.upper()}:")
        print(worst_translations)
        worst_translations.to_csv(
            f"top_10_worst_{sanitize_filename(metric)}.csv",
            index=False,
        )

    # Calculate percentiles for each metric across all data
    for metric in metrics:
        df[f"{metric}_percentile"] = df[metric].rank(pct=True)

    # Calculate pairwise percentile differences between metrics
    for i, metric1 in enumerate(metrics):
        for metric2 in metrics[i + 1 :]:
            df[f"{metric1}_vs_{metric2}_percentile_diff"] = (
                df[f"{metric1}_percentile"] - df[f"{metric2}_percentile"]
            ).abs()

    # Get top 10 translations with largest pairwise percentile differences for each metric pair
    pairwise_diff_cols = [col for col in df.columns if "percentile_diff" in col]
    top_diff_translations = pd.concat(
        [df.nlargest(10, col) for col in pairwise_diff_cols]
    )

    # Select relevant columns
    columns_to_display = (
        [
            "model",
            "method",
            "gen_type",
            "translation_temperature",
            "original_text",
            "generated_translation",
            "gold_translation",
        ]
        + metrics
        + pairwise_diff_cols
        + [f"{metric}_percentile" for metric in metrics]
    )

    print("\nTop 10 Translations with Largest Metric Score Differences:")
    print(top_diff_translations[columns_to_display])
    top_diff_translations[columns_to_display].to_csv(
        "top_10_largest_diff_translations.csv", index=False
    )

    # Export all evaluation data to Excel
    df[columns_to_display].to_excel("all_evaluations.xlsx", index=False)
    print("\nAll evaluation data exported to all_evaluations.xlsx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate translations.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["chrF", "METEOR", "SentenceTransformers"],
        help="Metrics to use for evaluation.",
    )
    parser.add_argument(
        "--database",
        default="translations.db",
        help="Database name.",
    )
    parser.add_argument(
        "--merge-databases",
        action="store_true",
        help="Flag to merge databases.",
    )
    parser.add_argument(
        "--sentence-transformers-model",
        default="WhereIsAI/UAE-Large-V1",
        help="Sentence transformers model to use.",
    )
    parser.add_argument(
        "--japanese-sentence-transformers-model",
        default="pkshatech/simcse-ja-bert-base-clcmlp",
        help="Sentence transformers model to use.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=None,
        help="Override the number of runs.",
    )

    args = parser.parse_args()

    from hachidaishu_translation.db import get_translations_and_gold, create_tables

    create_tables(args.database, merge_databases=args.merge_databases)

    translations = get_translations_and_gold()
    sentence_transformers_model = args.sentence_transformers_model
    metrics = args.metrics
    evals = evaluate_all_translations(
        translations,
        metrics=metrics,
        sentence_transformers_model=sentence_transformers_model,
        japanese_model_name=args.japanese_sentence_transformers_model,
    )
    visualize_evaluations(
        evals,
        metrics=metrics,
        sentence_transformers_model=sentence_transformers_model,
        japanese_model_name=args.japanese_sentence_transformers_model,
        num_runs=args.num_runs,
    )
