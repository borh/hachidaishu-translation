import jaconv
import logging
from prettytable import PrettyTable, SINGLE_BORDER
from hachidaishu_translation.hachidaishu import Token


def transliterate(token: Token) -> str:
    return jaconv.kana2alphabet(token.kanji_reading)


def remove_delimiters(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t != "/"]


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


def make_unique(names):
    counts = {}
    unique_names = []
    subscript_digits = "₀₁₂₃₄₅₆₇₈₉"

    def to_subscript(num):
        return "".join(subscript_digits[int(digit)] for digit in str(num))

    for name in names:
        if name in counts:
            counts[name] += 1
            unique_names.append(f"{name}{to_subscript(counts[name])}")
        else:
            counts[name] = 0
            unique_names.append(name)
    return unique_names


def visualize_alignment(original, translated, data, matrix_mode=True) -> PrettyTable:
    o_align = {word: [] for word in original}
    t_align = {word: [] for word in translated}

    for a in data.alignment:
        try:
            o, t = a.original_token, a.translated_token
            o_align[o].append(t)
            t_align[t].append(o)
        except KeyError as e:
            logging.error(
                f"Alignment data {e} is not valid in {data}, {original}, {translated}, skipping..."
            )

    if matrix_mode:
        table = PrettyTable()
        table.set_style(SINGLE_BORDER)

        header = [""] + make_unique(original)
        table.field_names = header

        for t in translated:
            row = [t] + ["■" if o in t_align.get(t, []) else "" for o in original]
            table.add_row(row)

        return table
    else:
        table = PrettyTable()
        table.set_style(SINGLE_BORDER)
        table.field_names = ["Index", "Original", "→", "Translated", "←"]
        table.align = "l"

        max_len = max(len(original), len(translated))

        for i in range(max_len):
            original_words = original[i] if i < len(original) else ""
            translated_words = translated[i] if i < len(translated) else ""
            table.add_row(
                [
                    i,
                    original_words,
                    ",".join(t_align.get(original_words, [])) if original_words else "",
                    translated_words,
                    ",".join(o_align.get(translated_words, []))
                    if translated_words
                    else "",
                ]
            )

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


if __name__ == "__main__":
    original = ["猫", "が", "座っ", "た", "。"]
    translated = ["The", "cat", "sat", "down", "."]

    class Alignment:
        def __init__(self, original_index, translated_index):
            self.original_index = original_index
            self.translated_index = translated_index

    class AlignmentSchema:
        def __init__(self, alignment):
            self.alignment = alignment

    # Example alignment data
    alignment_data = [
        Alignment(0, 0),
        Alignment(0, 1),
        Alignment(2, 2),
        Alignment(3, 3),
        Alignment(4, 4),
    ]

    data = AlignmentSchema(alignment_data)

    # Visualize alignment with matrix mode
    print(visualize_alignment(original, translated, data, matrix_mode=True))
