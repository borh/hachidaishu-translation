import duckdb
import json
from typing import Iterable
from hachidaishu_translation.log import logger

duckdb_conn = duckdb.connect("translations.db")


def create_tables():
    duckdb_conn.execute("CREATE SEQUENCE IF NOT EXISTS poem_id_sequence START 1;")
    duckdb_conn.execute(
        "CREATE SEQUENCE IF NOT EXISTS translation_id_sequence START 1;"
    )
    duckdb_conn.execute("CREATE SEQUENCE IF NOT EXISTS alignment_id_sequence START 1;")
    duckdb_conn.execute("CREATE SEQUENCE IF NOT EXISTS token_id_sequence START 1;")
    duckdb_conn.execute(
        "CREATE SEQUENCE IF NOT EXISTS translation_token_id_sequence START 1;"
    )

    duckdb_conn.execute("""
    CREATE TABLE IF NOT EXISTS poems (
        poem_id INTEGER DEFAULT nextval('poem_id_sequence') PRIMARY KEY,
        original_text TEXT
    )
    """)

    duckdb_conn.execute("""
    CREATE TABLE IF NOT EXISTS translations (
        translation_id INTEGER DEFAULT nextval('translation_id_sequence') PRIMARY KEY,
        poem_id INTEGER,
        model TEXT,
        method TEXT,
        gen_type TEXT,
        translation_text TEXT,
        FOREIGN KEY(poem_id) REFERENCES poems(poem_id)
    )
    """)

    duckdb_conn.execute("""
    CREATE TABLE IF NOT EXISTS poem_tokens (
        token_id INTEGER DEFAULT nextval('token_id_sequence') PRIMARY KEY,
        poem_id INTEGER,
        token_index INTEGER,
        token_text TEXT,
        FOREIGN KEY(poem_id) REFERENCES poems(poem_id)
    )
    """)

    duckdb_conn.execute("""
    CREATE TABLE IF NOT EXISTS translation_tokens (
        token_id INTEGER DEFAULT nextval('translation_token_id_sequence') PRIMARY KEY,
        translation_id INTEGER,
        token_index INTEGER,
        token_text TEXT,
        FOREIGN KEY(translation_id) REFERENCES translations(translation_id)
    )
    """)

    duckdb_conn.execute("""
    CREATE TABLE IF NOT EXISTS alignments (
        alignment_id INTEGER DEFAULT nextval('alignment_id_sequence') PRIMARY KEY,
        translation_id INTEGER,
        original_token_id INTEGER,
        translated_token_id INTEGER,
        FOREIGN KEY(translation_id) REFERENCES translations(translation_id),
        FOREIGN KEY(original_token_id) REFERENCES poem_tokens(token_id),
        FOREIGN KEY(translated_token_id) REFERENCES translation_tokens(token_id)
    )
    """)


def save_poem(original_text: str) -> int:
    return duckdb_conn.execute(
        "INSERT INTO poems (original_text) VALUES (?) RETURNING (poem_id)",
        (original_text,),
    ).fetchone()[0]


def save_translation(
    poem_id: int, model: str, method: str, gen_type: str, translation_text: str
) -> int:
    return duckdb_conn.execute(
        "INSERT INTO translations (poem_id, model, method, gen_type, translation_text) VALUES (?, ?, ?, ?, ?) RETURNING (translation_id)",
        (poem_id, model, method, gen_type, translation_text),
    ).fetchone()[0]


def save_poem_tokens(poem_id: int, tokens: list[str]) -> Iterable[int]:
    token_ids = []
    for index, token in enumerate(tokens):
        token_id = duckdb_conn.execute(
            "INSERT INTO poem_tokens (poem_id, token_index, token_text) VALUES (?, ?, ?) RETURNING (token_id)",
            (poem_id, index, token),
        ).fetchone()[0]
        token_ids.append(token_id)
    return token_ids


def save_translation_tokens(translation_id: int, tokens: list[str]) -> Iterable[int]:
    token_ids = []
    for index, token in enumerate(tokens):
        token_id = duckdb_conn.execute(
            "INSERT INTO translation_tokens (translation_id, token_index, token_text) VALUES (?, ?, ?) RETURNING (token_id)",
            (translation_id, index, token),
        ).fetchone()[0]
        token_ids.append(token_id)
    return token_ids


def save_alignment(
    translation_id: int,
    alignment,
    original_token_ids: dict[str, int],
    translated_token_ids: dict[str, int],
):
    """Saves the alignment between the original and translated tokens to the database.
    As the alignment is on the unique tokens, the token_ids must be provided mapping to the unique tokens and not be normalized."""
    logger.debug(f"Saving {alignment}\n{original_token_ids}\n{translated_token_ids}")
    for align in alignment.alignment:
        for translated_token in (
            align.translated_token.split()
        ):  # Split the translated token if it contains multiple tokens, and align each
            original_index = original_token_ids.get(align.original_token)
            translated_index = translated_token_ids.get(translated_token)
            # logger.debug(
            #     f"Aligning {align.original_token} -- {align.translated_token} : {original_index} -- {translated_index}"
            # )
            if not original_index or not translated_index:
                logger.error(
                    f"Failed to find token IDs ({original_index} -- {translated_index}) for alignment: {align.original_token} -- {align.translated_token}"
                )
            duckdb_conn.execute(
                "INSERT INTO alignments (translation_id, original_token_id, translated_token_id) VALUES (?, ?, ?)",
                (
                    translation_id,
                    original_index,
                    translated_index,
                ),
            )


def get_translations_and_gold() -> list[dict]:
    """
    Retrieves all translations and their corresponding gold standard translations from the database,
    including alignment information with indices starting from 0 for each translation.

    Returns:
    - A list of dictionaries containing the poem ID, model, method, gen_type, gold_translation,
      generated translation, and alignment information.
    """
    query = """
    WITH gold_translations AS (
        SELECT 
            poem_id, 
            translation_text AS gold_translation
        FROM 
            translations
        WHERE 
            model = 'gold_standard'
    ),
    alignments_combined AS (
        SELECT
            a.translation_id,
            '[' || string_agg(
                '[' || pt.token_index || ',' || tt.token_index || ']', ','
            ) || ']' AS alignment
        FROM
            alignments a
        JOIN 
            poem_tokens pt ON a.original_token_id = pt.token_id
        JOIN 
            translation_tokens tt ON a.translated_token_id = tt.token_id
        GROUP BY
            a.translation_id
    )
    SELECT 
        t.poem_id, 
        t.translation_id, 
        t.model, 
        t.method, 
        t.gen_type, 
        g.gold_translation, 
        t.translation_text AS generated_translation,
        a.alignment,
        p.original_text
    FROM 
        translations t
    JOIN 
        gold_translations g ON t.poem_id = g.poem_id
    LEFT JOIN
        alignments_combined a ON t.translation_id = a.translation_id
    LEFT JOIN
        poems p ON t.poem_id = p.poem_id
    WHERE 
        t.model != 'gold_standard';
    """
    results = duckdb_conn.execute(query).fetchall()

    translations = []
    for row in results:
        translations.append(
            {
                "poem_id": row[0],
                "translation_id": row[1],
                "model": row[2],
                "method": row[3],
                "gen_type": row[4],
                "gold_translation": row[5],
                "generated_translation": row[6],
                "alignment": json.loads(row[7]) if row[7] else [],
                "original_text": row[8],
            }
        )

    return translations


def get_tokens(poem_id: int, translation_id: int) -> dict[str, list[str]]:
    """
    Given a poem_id and translation_id, return a dictionary containing the tokens of the original and the translation.

    Parameters:
    - poem_id: The ID of the poem.
    - translation_id: The ID of the translation.

    Returns:
    - A dictionary with 'original_tokens' and 'translated_tokens' keys.
    """
    original_tokens_query = """
    SELECT token_text 
    FROM poem_tokens 
    WHERE poem_id = ? 
    ORDER BY token_index;
    """
    translated_tokens_query = """
    SELECT token_text 
    FROM translation_tokens 
    WHERE translation_id = ? 
    ORDER BY token_index;
    """

    original_tokens = duckdb_conn.execute(original_tokens_query, (poem_id,)).fetchall()
    translated_tokens = duckdb_conn.execute(
        translated_tokens_query, (translation_id,)
    ).fetchall()

    return {
        "original_tokens": [token[0] for token in original_tokens],
        "translated_tokens": [token[0] for token in translated_tokens],
    }


def query():
    return duckdb_conn.execute("""
SELECT 
    p.poem_id,
    p.original_text,
    t.translation_id,
    t.translation_text,
    pt.token_index AS original_token_index,
    pt.token_text AS original_token_text,
    tt.token_index AS translated_token_index,
    tt.token_text AS translated_token_text
FROM 
    alignments a
JOIN 
    translations t ON a.translation_id = t.translation_id
JOIN 
    poems p ON t.poem_id = p.poem_id
JOIN 
    poem_tokens pt ON a.original_token_id = pt.token_id
JOIN 
    translation_tokens tt ON a.translated_token_id = tt.token_id
ORDER BY 
    t.translation_id, pt.token_index, tt.token_index;
""")
