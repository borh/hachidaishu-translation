import jaconv
import re
from prettytable import PrettyTable, SINGLE_BORDER
from hachidaishu_translation.hachidaishu import Token
from hachidaishu_translation.log import logger


def transliterate(token: Token) -> str:
    return jaconv.kana2alphabet(token.kanji_reading)


def strip_punctuation(s: str) -> str:
    # But keep spaces
    return re.sub(r"[^A-Za-zぁ-ゟァ-ヿ一-鿿 ]+", "", s)


def normalize_tokens(tokens: list[str]) -> list[str]:
    return [strip_punctuation(t.lower()) for t in tokens if t != "/"]


def reformat_waka(tokens: list[Token], delim="", romaji=False) -> str:
    """Reformat a list of tokens into a waka poem with the given structure.
    Note that this is a very simple implementation that does not take into account
    the slight variations one can encounter in waka.
    """
    waka_structure = [5, 7, 5, 7, 7]
    poem_lines = []
    current_length = 0
    token_index = 0
    current_line = []

    for line_length in waka_structure:
        while current_length < line_length and token_index < len(tokens):
            token = tokens[token_index]
            kanji_reading_length = len(token.kanji_reading)
            if current_length + kanji_reading_length > line_length:
                break
            current_line.append(token.surface if not romaji else transliterate(token))
            current_length += kanji_reading_length
            token_index += 1
        poem_lines.append(delim.join(current_line))
        current_line = []
        current_length = 0

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


def remove_unique_one(token: str):
    return re.sub(r"[₀₁₂₃₄₅₆₇₈₉]", "", token)


def remove_unique(tokens: list[str]):
    return [re.sub(r"[₀₁₂₃₄₅₆₇₈₉]", "", token) for token in tokens]


def make_unique(names):
    name = remove_unique(names)
    counts = {}
    unique_names = []
    subscript_digits = "₀₁₂₃₄₅₆₇₈₉"

    def to_subscript(num):
        return "".join(subscript_digits[int(digit)] for digit in str(num))

    for name in names:
        counts[name] = counts.get(name, 0) + 1
        count = counts[name]
        unique_names.append(name if count == 1 else f"{name}{to_subscript(count)}")
    return unique_names


def get_token_alignment(
    original: list[str], translated: list[str], data
) -> dict[str, list[str]]:
    o_align = {token: [] for token in original}
    t_align = {token: [] for token in translated}

    if not isinstance(data, list):
        data = [[a.original_token, a.translated_token] for a in data.alignment]
    for o, t_maybe in data:
        for t in t_maybe.split():  # Align multiple tokens if present separately
            try:
                o_align[o].append(t)
                t_align[t].append(o)
            except KeyError as e:
                logger.error(
                    f"Alignment data {e} is not valid in {data}, {original}, {translated}, skipping..."
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
    o_align = {i: [] for i in range(len(original))}
    t_align = {i: [] for i in range(len(translated))}

    try:
        for a in data.alignment:
            o, t = a.original_index, a.translated_index
            o_align[o].append(t)
            t_align[t].append(o)
    except KeyError as e:
        raise Exception(f"Alignment data {e} is not valid in {data}.")

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


import subprocess
import os


def compile_latex_tikz(
    alignment_data, original_tokens, translated_tokens, output_file="output.pdf"
):
    def sanitize_token(token):
        """Escape special characters in LaTeX."""
        return (
            token.replace("_", "\\_")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("$", "\\$")
            .replace("#", "\\#")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("^", "\\^{}")
            .replace("~", "\\~{}")
            .replace(".", "．")
        )

    tikz_template = r"""
\documentclass{{article}}
\usepackage[a4paper,margin=0.1cm,landscape]{{geometry}}
\usepackage{{luatexja}}
\usepackage{{tikz}}
\usetikzlibrary{{tikzmark}}

\tikzset{{every tikzmarknode/.style={{inner sep = 1pt,execute at end node={{\vphantom{{bg}}}}}}}}
\begin{{document}}
    \begin{{center}}
        {}
        
        \vspace*{{1cm}}
        
        {}
    \end{{center}}
    
    \begin{{tikzpicture}}[remember picture, overlay]
{}
    \end{{tikzpicture}}
\end{{document}}
    """

    # Create unique token lists
    original_tokens_unique = make_unique(original_tokens)
    translated_tokens_unique = make_unique(translated_tokens)

    # Create original and translated nodes with O1...On and T1...Tn naming
    original_nodes = " ".join(
        [
            f"\\tikzmarknode{{O{i}}}{{{sanitize_token(remove_unique_one(token))}}}"
            for i, token in enumerate(original_tokens_unique)
        ]
    )
    translated_nodes = " ".join(
        [
            f"\\tikzmarknode{{T{i}}}{{{sanitize_token(remove_unique_one(token))}}}"
            for i, token in enumerate(translated_tokens_unique)
        ]
    )

    connections = []
    if not isinstance(alignment_data, list):
        alignment_data = [[o, t] for o, ts in alignment_data.items() for t in ts]

    for o_idx, t_idx in alignment_data:
        o_node = f"O{o_idx}"
        t_node = f"T{t_idx}"
        connections.append(f"\\draw[->] ({o_node}.south) -- ({t_node}.north);")

    tikz_code = tikz_template.format(
        original_nodes, translated_nodes, "\n".join(connections)
    )

    # Write the TikZ code to a .tex file
    tex_file = "alignment.tex"
    with open(tex_file, "w") as f:
        f.write(tikz_code)

    # Compile the LaTeX file using pdflatex
    try:
        subprocess.run(["lualatex", "--interaction=nonstopmode", tex_file], check=True)
    except subprocess.CalledProcessError as e:
        ...

    # Clean up auxiliary files generated by lualatex
    for ext in ["aux", "log"]:  # , "tex"]:
        os.remove(f"alignment.{ext}")

    # Rename the output PDF
    if os.path.exists("alignment.pdf"):
        os.rename("alignment.pdf", output_file)
        print(f"Output saved to {output_file}")
    else:
        print("Failed to generate PDF")


if __name__ == "__main__":
    original = ["猫", "が", "座っ", "た", "。"]
    translated = ["The", "cat", "sat", "down", "."]

    class Alignment:
        def __init__(self, original_index, translated_index):
            self.original_token = original_index
            self.translated_token = translated_index

    class AlignmentSchema:
        def __init__(self, alignment):
            self.alignment = alignment

    # Example alignment data
    alignment_data = [
        Alignment("猫", "cat"),
        Alignment("座っ", "sat"),
        Alignment("た", "down"),
    ]

    data = AlignmentSchema(alignment_data)

    # Visualize alignment with matrix mode
    print(visualize_alignment(original, translated, data))

    align_data = get_token_alignment(
        make_unique(original), make_unique(translated), data
    )
    compile_latex_tikz(
        align_data, make_unique(original), make_unique(translated), "word_alignment.pdf"
    )
