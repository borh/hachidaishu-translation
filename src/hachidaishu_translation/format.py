import jaconv
from hachidaishu_translation.hachidaishu import Token


def transliterate(token: Token) -> str:
    return jaconv.kana2alphabet(token.kanji_reading)


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


def pp(s: str, lang: str="ja") -> str:
    parts = s.split("/")
    assert len(parts) == 5
    parts = [p.replace(" ", "") if lang == "ja" else p.strip() for p in parts]
    return f"""    {parts[0]}
{parts[1]}
    {parts[2]}
{parts[3]}
{parts[4]}"""
