import re
import argparse
from datetime import datetime
from typing import Any

import jaconv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import seaborn as sns
from prettytable import SINGLE_BORDER, PrettyTable
import pandas as pd

from hachidaishu_translation.hachidaishu import Token
from hachidaishu_translation.log import logger


plt.rcParams["font.family"] = [
    "Work Sans",
    "IBM Plex Sans JP",
]  # "Source Han Serif"]
plt.rcParams["font.weight"] = "heavy"


def transliterate(token: Token) -> str:
    return jaconv.kana2alphabet(token.kanji_reading)


def clean_bad_translation(s: str) -> str:
    # This pattern looks for a sequence of lowercase letters followed by an uppercase letter and removes everything after the first occurrence. This is a hack to remove model output degeneration.
    if re.search(r"([a-z]+[A-Z]).*", s):
        logger.warning(f"Cleaning string {s} after first upper or comma.")
        return re.sub(r"([a-z]+)([A-Z,][a-zA-Z].*)", r"\1", s)
    return s


STRIP_RE = re.compile(r"[^\w\s']")


def strip_punctuation(s: str) -> str:
    # But keep spaces and apostrophes (for possessives and plurals)
    return STRIP_RE.sub("", s)
    # return re.sub(r"[^A-Za-zぁ-ゟァ-ヿ一-鿿 ]+", "", s)


def normalize_tokens(tokens: list[str]) -> list[str]:
    normalized_tokens = [strip_punctuation(t.lower()) for t in tokens if t != "/"]
    logger.info(f"Normalizing tokens: {tokens} -> {normalized_tokens}")
    return normalized_tokens


def reformat_waka(tokens: list[Token], delim="", romaji=False) -> str:
    """Reformat a list of tokens into a waka poem with the given structure.
    This implementation allows for slight variations in mora count,
    ensuring we don't go under the expected count for each part.
    """
    waka_structure = [5, 7, 5, 7, 7]
    poem_lines = []
    mora_count = 0
    token_index = 0
    current_line = []

    for line_length in waka_structure:
        current_length = 0
        while token_index < len(tokens):
            token = tokens[token_index]
            kanji_reading_length = len(token.kanji_reading)

            # Add token if there's any space left in the current part
            if current_length < line_length or (
                current_length == 0 and kanji_reading_length > line_length
            ):
                current_line.append(
                    token.surface if not romaji else transliterate(token)
                )
                current_length += kanji_reading_length
                token_index += 1
                mora_count += kanji_reading_length
            else:
                break

        poem_lines.append(delim.join(current_line))
        current_line = []

    # Add any remaining tokens to the last line and warn
    if token_index < len(tokens):
        remaining_tokens = [
            token.surface if not romaji else transliterate(token)
            for token in tokens[token_index:]
        ]
        mora_count += sum(len(token.kanji_reading) for token in tokens[token_index:])
        poem_lines[-1] += delim + delim.join(remaining_tokens)
        logger.warning(
            f"Waka structure mismatch: {mora_count} moras found, expected {sum(waka_structure)}. Adding remaining tokens to the last line: {remaining_tokens}"
        )

    return " / ".join(poem_lines)


def format_poem(tokens: list[Token], delim="", split=False, romaji=False) -> str:
    if split:
        return reformat_waka(tokens, delim=delim, romaji=romaji)
    else:
        return delim.join(
            [t.surface if not romaji else transliterate(t) for t in tokens]
        )


def pp(s: str, lang: str = "ja") -> str:
    parts = s.split("/")
    assert len(parts) == 5
    parts = [p.replace(" ", "") if lang == "ja" else p.strip() for p in parts]
    return f"""    {parts[0]}
{parts[1]}
    {parts[2]}
{parts[3]}
{parts[4]}"""


def remove_unique_one(token: str) -> str:
    return re.sub(r"[₀₁₂₃₄₅₆₇₈₉]", "", token)


def remove_unique(tokens: list[str]) -> list[str]:
    return [remove_unique_one(token) for token in tokens]


def make_unique(names: list[str]) -> list[str]:
    name = remove_unique(names)
    counts: dict[str, int] = {}
    unique_names = []
    subscript_digits = "₀₁₂₃₄₅₆₇₈₉"

    def to_subscript(num):
        return "".join(subscript_digits[int(digit)] for digit in str(num))

    for name in names:
        counts[name] = counts.get(name, 0) + 1
        count = counts[name]
        unique_names.append(name if count == 1 else f"{name}{to_subscript(count)}")
    logger.info(f"Making unique: {names} -> {unique_names}")
    return unique_names


#############
def check_alignment_strings(
    source: list[str], target: list[str], alignment: list[tuple[str, str]]
) -> list[tuple[str, str]] | None:
    logger.info(f"Checking alignment: {source} {target} {alignment}")

    source_set = set(source)
    target_set = set(target)
    alignment_source_tokens = set(pair[0] for pair in alignment)
    alignment_target_tokens = set(pair[1] for pair in alignment)

    failed = False
    if source_set != alignment_source_tokens:
        logger.error(
            f"Source-Alignment diff: {source_set.difference(alignment_source_tokens)} {alignment_source_tokens.difference(source_set)}"
        )
        failed = True
    if target_set != alignment_target_tokens:
        logger.error(
            f"Target-Alignment diff: {target_set.difference(alignment_target_tokens)} {alignment_target_tokens.difference(target_set)}"
        )
        failed = True

    if failed:
        return None
    else:
        return alignment


#############


def get_token_alignment(
    original: list[str], translated: list[str], data
) -> dict[str, list[str]]:
    o_align: dict[str, list[str]] = {token: [] for token in original}
    t_align: dict[str, list[str]] = {token: [] for token in translated}

    if not isinstance(data, list):
        data = [[a.original_token, a.translated_token] for a in data.alignment]
        check_alignment_strings(original, translated, data)
    for o, t_maybe in data:
        try:
            o_align[o].append(t_maybe)
            t_align[t_maybe].append(o)
        except KeyError as e:
            logger.error(
                f"Alignment data {e} is not valid in {data}, {original}, {translated}, trying to split and match tokens..."
            )
            for t in t_maybe.split():  # Align multiple tokens if present separately
                try:
                    o_align[o].append(t)
                    t_align[t].append(o)
                except KeyError as e:
                    logger.error(
                        f"Splitting and searching failed for {t} in {t_maybe}."
                    )

    return t_align


def get_index_alignment(
    original: list[str], translated: list[str], token_alignments: dict[str, list[str]]
) -> list[tuple[int, int]]:
    return [
        (original.index(o), translated.index(t))
        for o, ts in token_alignments.items()
        for t in ts
    ]


def visualize_alignment(
    original: list[str], translated: list[str], data
) -> PrettyTable:
    """Visualize alignment data between original and translated tokens.
    original: list of original tokens, which must be unique
    translated: list of translated tokens, which must be unique
    """
    table = PrettyTable()
    table.set_style(SINGLE_BORDER)

    header = [""] + make_unique(original)
    table.field_names = header

    t_align = get_token_alignment(original, translated, data)

    for t in translated:
        row = [t] + ["■" if o in t_align.get(t, []) else "" for o in original]
        table.add_row(row)

    return table


def visualize_alignment_indexes(
    original, translated, data, matrix_mode=True
) -> PrettyTable:
    o_align: dict[int, list[int]] = {i: [] for i in range(len(original))}
    t_align: dict[int, list[int]] = {i: [] for i in range(len(translated))}

    try:
        for a in data.alignment:
            o, t = a.original_index, a.translated_index
            o_align[o].append(t)
            t_align[t].append(o)
    except KeyError as e:
        raise ValueError(f"Alignment data {e} is not valid in {data}.")

    if matrix_mode:
        table = PrettyTable()
        table.set_style(SINGLE_BORDER)

        header = [""] + make_unique(original)
        table.field_names = header

        for i in range(len(translated)):
            row = [translated[i]] + [
                "■" if j in t_align[i] else "" for j in range(len(original))
            ]
            table.add_row(row)

        return table
    else:
        table = PrettyTable()
        table.set_style(SINGLE_BORDER)
        table.field_names = ["Index", "Original", "→", "Translated", "←"]
        table.align = "l"

        max_len = max(len(original), len(translated))

        for i in range(max_len):
            if i < len(original):
                original_words = original[i]
            else:
                original_words = ""
            if i < len(translated):
                translated_words = translated[i]
            else:
                translated_words = ""
            table.add_row(
                [
                    i,
                    original_words,
                    ",".join(str(idx) for idx in o_align.get(i, [])),
                    translated_words,
                    ",".join(str(idx) for idx in t_align.get(i, [])),
                ]
            )

        return table


def visualize_alignment_matplotlib(
    original: list[str],
    translated: list[str],
    data: dict[str, Any],
    title: str,
    output_pdf: str | None = None,
    correct_model: str | None = None,
) -> None:
    """Visualize multiple alignments between original and translated tokens using matplotlib.

    Args:
        original: List of original tokens.
        translated: List of translated tokens.
        data: Dictionary containing alignment data.
        output_pdf: If set, saves the visualization to the specified PDF file.
        correct_model: If provided, highlights alignments in red if absent in the correct model, otherwise black.
    """
    # # Make sure original and translated tokens are unique
    # original = make_unique(original)
    # translated = make_unique(translated)

    # Sort data keys by model name
    data_keys = sorted(data.keys())
    colors = sns.color_palette("Pastel1", n_colors=len(data_keys))
    legend_handles = []

    # # Initialize o_align and t_align for all models
    # o_align: dict[str, list[str]] = {token: [] for token in original}
    # t_align: dict[str, list[str]] = {token: [] for token in translated}

    # Filter out data entries with only one value
    filtered_data = {k: v for k, v in data.items() if len(v) > 1}

    if len(filtered_data) <= 1:
        logger.info(
            f"Skipping visualization for single or no model after filtering: {filtered_data}"
        )
        return

    assert all(len(d) > 1 for d in filtered_data.values()), filtered_data
    logger.info(f"Models after filtering {len(filtered_data)}: {filtered_data}")
    data = filtered_data

    # plt.style.use("ggplot")
    _fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)
    # Aggregate alignments to ensure each token appears only once
    aggregated_alignments = {token: set() for token in original}
    for model, alignment_data in data.items():
        for o_index, t_index in alignment_data:
            # logger.warning(
            #     f"o_index={o_index}, t_index={t_index}, alignment_data={alignment_data}, original={original}, translated={translated}"
            # )
            aggregated_alignments[original[o_index]].add(t_index)

    # Precompute the status of each cell for all models
    cell_status = {
        (i, j): set() for i in range(len(original)) for j in range(len(translated))
    }
    for model, alignment_data in data.items():
        for o_index, t_index in alignment_data:
            cell_status[(o_index, t_index)].add(model)

    # Sort data keys by model name
    data_keys = sorted(data.keys())
    for idx, model in enumerate(data_keys):
        alignment_data = data[model]
        for i, _o_token in enumerate(original):
            for j, _t_token in enumerate(translated):
                if model in cell_status[(i, j)]:
                    if correct_model:
                        color = (
                            "black" if correct_model in cell_status[(i, j)] else "red"
                        )
                    else:
                        color = None

                    if color:
                        ax.add_patch(
                            Rectangle(
                                (j + 0.05, i + 0.05),
                                0.9,
                                0.9,
                                fill=False,
                                color=color,
                                alpha=1.0,
                            )
                        )
                    total_count = len(cell_status[(i, j)])
                    if total_count > 1:
                        # Create a donut plot
                        wedges, _ = ax.pie(
                            [1] * total_count,
                            colors=[
                                colors[data_keys.index(m)] for m in cell_status[(i, j)]
                            ],
                            radius=0.45,
                            center=(j + 0.5, i + 0.5),
                            wedgeprops=dict(width=0.2),  # Adjust width for donut effect
                            startangle=90,  # Rotate pie chart vertically
                        )

                        # Add text in the center of the donut
                        ax.text(
                            j + 0.5,
                            i + 0.5,
                            str(total_count),
                            horizontalalignment="center",
                            verticalalignment="center",
                            fontsize=10,
                            weight="bold",
                        )
                    else:
                        ax.pie(
                            [1] * total_count,
                            colors=[
                                colors[data_keys.index(m)] for m in cell_status[(i, j)]
                            ],
                            radius=0.45,
                            center=(j + 0.5, i + 0.5),
                            wedgeprops=dict(width=0.45),
                            startangle=90,  # Rotate pie chart vertically
                        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=model,
                markerfacecolor=colors[idx],
                markersize=10,
            )
        )

    ax.set_xlim(-0.5, len(translated) + 0.5)
    ax.set_ylim(len(original) + 0.5, -0.5)
    ax.set_xticks([i + 0.5 for i in range(len(translated))])
    ax.set_yticks([i + 0.5 for i in range(len(original))])
    ax.set_xticklabels(translated, rotation=45, weight="bold", ha="right")
    ax.set_yticklabels(original, weight="bold")
    ax.tick_params(axis="x", which="both", length=0)
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_aspect("equal", "box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="lightgray")

    # plt.subplots_adjust(left=0.2, right=0.8, bottom=0.1, top=1.0)
    ax.legend(
        handles=legend_handles,
        title="Models",
        loc="upper left",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
        fontsize="small",
        title_fontsize="small",
    )
    ax.set_title(title, fontsize=14, weight="bold")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if output_pdf:
        plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process alignment visualizations.")
    parser.add_argument(
        "--database",
        type=str,
        default="translations.db",
        help="The database file to use.",
    )
    parser.add_argument(
        "--merge-databases",
        action="store_true",
        help="Flag to merge databases.",
    )
    args = parser.parse_args()

    # original = ["猫", "が", "座っ", "た", "。"]
    # translated = ["The", "cat", "sat", "down", "."]

    # # Example alignment data
    # alignment_data = {
    #     "model_1": [
    #         ("猫", "cat"),
    #         ("座っ", "sat"),
    #         ("た", "down"),
    #     ],
    #     "model_2": [
    #         ("猫", "cat"),
    #         ("座っ", "down"),
    #         ("た", "down"),
    #         ("。", "."),
    #     ],
    #     "correct_model": [
    #         ("猫", "cat"),
    #         ("座っ", "sat"),
    #     ],
    # }

    timestamp = datetime.now().strftime("%Y-%m-%d")
    pdf_filename = f"alignments-{timestamp}.pdf"

    from matplotlib.backends.backend_pdf import PdfPages

    from hachidaishu_translation.db import (
        get_alignment_data_for_visualization,
        create_tables,
        get_translations_by_token,
    )

    create_tables(args.database, merge_databases=args.merge_databases)

    alignments = get_alignment_data_for_visualization()
    logger.info(f"Loaded alignments: {alignments}")
    from itertools import groupby
    from operator import itemgetter

    def extract_title(data):
        title_data = next(iter(data))
        temp = title_data["translation_temperature"]
        model = title_data["model"]
        method = title_data["method"]
        gen_type = title_data["gen_type"]
        assert model and method and gen_type
        if not temp:
            temp = ""
        else:
            temp = f" / t={format(temp, '.1f')}"
        title = f"{model} / {method} / {gen_type}{temp}"
        # logger.info(f"Title: {title}")
        return title

    def extract_align_label(data):
        alignment_model = data["alignment_model"]
        if not alignment_model:
            alignment_model = data["model"]
        model_temperature = data["model_temperature"]
        run_id = data["run_id"]
        if model_temperature is None:
            model_temperature = ""
        else:
            model_temperature = format(model_temperature, ".1f")

        assert alignment_model, data
        # logger.info(f"Alignment model: {alignment_model} {model_temperature} {run_id}")

        return f"{alignment_model} / {model_temperature} / {run_id}"

    alignments.sort(key=itemgetter("original_tokens", "translated_tokens"))

    collated_alignments = {}
    for (original_tokens, translated_tokens), group in groupby(
        alignments,
        key=itemgetter(
            "original_tokens",
            "translated_tokens",
        ),
    ):
        group_list = list(group)  # Convert generator to list
        for alignment in group_list:
            logger.info(f"Alignment data: {alignment}")  # Log the alignment data

        collated_alignments = {}
        for (original_tokens, translated_tokens), group in groupby(
            alignments, key=itemgetter("original_tokens", "translated_tokens")
        ):
            group_list = list(group)
            title = extract_title(group_list)
            key = (tuple(original_tokens), tuple(translated_tokens), title)
            collated_alignments[key] = {
                extract_align_label(alignment): alignment["alignment"]
                for alignment in group_list
            }

    assert len(collated_alignments) != 1, collated_alignments

    alignment_counts = []

    # Sort collated_alignments by the length of alignment_data in descending order
    sorted_alignments = sorted(
        collated_alignments.items(), key=lambda item: len(item[1]), reverse=True
    )

    with PdfPages(pdf_filename) as pdf:
        for (original, translated, title), alignment_data in sorted_alignments:
            alignment_count = len(alignment_data)
            if alignment_count == 0:
                logger.error(
                    f"Skipping empty alignment data...: {title} :: original={original}, translated={translated}, alignments={alignment_data}"
                )
                continue
            alignment_counts.append(len(alignment_data.keys()))
            logger.info(
                f"Group: {title} :: original={original}, translated={translated}, alignments={alignment_data}"
            )
            visualize_alignment_matplotlib(
                list(original),
                list(translated),
                alignment_data,
                title,
            )
            pdf.savefig()
            plt.close()

    logger.info(f"Saved all alignment visualizations to {pdf_filename}")

    from collections import Counter

    # Log histogram of alignment counts
    count_histogram = Counter(alignment_counts)
    logger.info("Histogram of Alignment Counts:")
    for count, frequency in sorted(count_histogram.items()):
        logger.info(f"Alignments: {count}, Frequency: {frequency}")

    from collections import defaultdict, Counter
    import pandas as pd

    translations = get_translations_by_token()

    # Aggregate translations
    aggregated_translations = defaultdict(lambda: defaultdict(Counter))

    for entry in translations:
        key_all = (entry["original_token"],)
        key_model = (entry["original_token"], entry["model"])

        aggregated_translations[key_all]["all"][entry["translated_token"]] += 1
        aggregated_translations[key_model][entry["model"]][
            entry["translated_token"]
        ] += 1

    # Display top 10 translations
    for key, models in aggregated_translations.items():
        for model, translations in models.items():
            token_level = "All" if model == "all" else f"Model: {model}"
            print(f"\nTop 10 translations for token '{key[0]}' at level {token_level}:")
            top_translations = translations.most_common(10)
            print(", ".join(f"{t[0]} ({t[1]})" for t in top_translations))

    # Create Excel file
    def create_excel_file(aggregated_translations):
        def sanitize_sheet_name(name):
            # Replace forward slash with underscore and limit to 31 characters
            return name.replace("/", "_")[:31]

        with pd.ExcelWriter("all_tokens.xlsx") as writer:
            # All models combined sheet
            all_data = []
            for key, models in aggregated_translations.items():
                if len(key) == 1:  # This is the 'all' key
                    original_token = key[0]
                    translations = models["all"].most_common(10)
                    row = [original_token] + [f"{t[0]} ({t[1]})" for t in translations]
                    all_data.append(row)

            df_all = pd.DataFrame(
                all_data,
                columns=["Original Token"] + [f"Translation {i+1}" for i in range(10)],
            )
            df_all.to_excel(writer, sheet_name="All Models", index=False)

            # Individual model sheets
            models = set(
                model
                for key in aggregated_translations.keys()
                if len(key) > 1
                for model in aggregated_translations[key].keys()
            )
            for model in models:
                model_data = []
                for key, translations in aggregated_translations.items():
                    if len(key) > 1 and key[1] == model:
                        original_token = key[0]
                        top_translations = translations[model].most_common(10)
                        row = [original_token] + [
                            f"{t[0]} ({t[1]})" for t in top_translations
                        ]
                        model_data.append(row)

                df_model = pd.DataFrame(
                    model_data,
                    columns=["Original Token"]
                    + [f"Translation {i+1}" for i in range(10)],
                )
                df_model.to_excel(
                    writer, sheet_name=sanitize_sheet_name(model), index=False
                )

    create_excel_file(aggregated_translations)

    # # Visualize alignment with matrix mode
    # for model, data in alignment_data.items():
    #     print(f"Model: {model}")
    #     print(visualize_alignment(original, translated, data))
