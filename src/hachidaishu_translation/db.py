import duckdb
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
    for align in alignment.alignment:
        original_index = original_token_ids.get(align.original_token)
        translated_index = translated_token_ids.get(align.translated_token)
        logger.warning(
            f"Aligning {align.original_token} to {align.translated_token} with indices {original_token_ids.get(align.original_token)} and {translated_token_ids.get(align.translated_token)}."
        )
        duckdb_conn.execute(
            "INSERT INTO alignments (translation_id, original_token_id, translated_token_id) VALUES (?, ?, ?)",
            (
                translation_id,
                original_index,
                translated_index,
            ),
        )


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
