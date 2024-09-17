import json
import re
import os

import duckdb

from hachidaishu_translation.log import logger
from hachidaishu_translation.models import TokenAlignment

duckdb_conn = duckdb.connect()  # NOTE Just for typing. Will be reset in create_tables.


def merge_runs(runs_dir="runs/"):
    # TODO use run_id to allow multiple translations for same model (not just alignments like now)
    logger.info("Starting database merge process.")
    db_files = [f for f in os.listdir(runs_dir) if f.endswith(".db")]

    if not db_files:
        logger.info("No database files found for merging.")
        return

    for db_file in db_files:
        logger.info(f"Merging database: {db_file}")
        try:
            source_conn = duckdb.connect(os.path.join(runs_dir, db_file))
        except duckdb.duckdb.IOException as e:
            logger.error(f"Failed to connect to {db_file}: {str(e)}")
            continue

        # Create mappings for unique (static) entries we do not want to duplicate:
        poem_id_map = {}
        translation_id_map = {}

        # Merge poems
        poems = source_conn.execute("SELECT * FROM poems").fetchall()
        for poem in poems:
            poem_id, original_text = poem
            logger.warning(f"poem_id: {poem_id}, original_text: {original_text}")
            existing_poem = duckdb_conn.execute(
                "SELECT poem_id FROM poems WHERE original_text = ?", (original_text,)
            ).fetchone()

            if existing_poem:
                poem_id_map[poem_id] = existing_poem[0]
            else:
                logger.error(f"Poem not found: {original_text}, inserting.")
                new_poem_id = duckdb_conn.execute(
                    "INSERT INTO poems (original_text) VALUES (?) RETURNING poem_id",
                    (original_text,),
                ).fetchone()[0]
                poem_id_map[poem_id] = new_poem_id

        # Merge gold_standard translations
        gold_translations = source_conn.execute(
            "SELECT * FROM translations WHERE model = 'gold_standard'"
        ).fetchall()

        for translation in gold_translations:
            (
                translation_id,
                poem_id,
                model,
                method,
                gen_type,
                translation_text,
                temperature,
            ) = translation
            mapped_poem_id = poem_id_map[poem_id]

            existing_translation = duckdb_conn.execute(
                "SELECT translation_id FROM translations WHERE poem_id = ? AND model = 'gold_standard'",
                (mapped_poem_id,),
            ).fetchone()

            if existing_translation:
                logger.info(
                    f"Gold translation already exists: {translation_id}, mapping."
                )
                translation_id_map[translation_id] = existing_translation[0]
            else:
                new_translation_id = duckdb_conn.execute(
                    "INSERT INTO translations (poem_id, model, method, gen_type, translation_text, temperature) VALUES (?, ?, ?, ?, ?, ?) RETURNING translation_id",
                    (
                        mapped_poem_id,
                        model,
                        method,
                        gen_type,
                        translation_text,
                        temperature,
                    ),
                ).fetchone()[0]
                translation_id_map[translation_id] = new_translation_id

        # Merge other translations and map their IDs
        other_translations = source_conn.execute(
            "SELECT * FROM translations WHERE model != 'gold_standard'"
        ).fetchall()

        for translation in other_translations:
            (
                translation_id,
                poem_id,
                model,
                method,
                gen_type,
                translation_text,
                temperature,
            ) = translation
            mapped_poem_id = poem_id_map[poem_id]

            new_translation_id = duckdb_conn.execute(
                "INSERT INTO translations (poem_id, model, method, gen_type, translation_text, temperature) VALUES (?, ?, ?, ?, ?, ?) RETURNING translation_id",
                (
                    mapped_poem_id,
                    model,
                    method,
                    gen_type,
                    translation_text,
                    temperature,
                ),
            ).fetchone()[0]

            # Map the new translation_id
            translation_id_map[translation_id] = new_translation_id

        # Merge poem tokens
        poem_tokens = source_conn.execute("SELECT * FROM poem_tokens").fetchall()
        for token in poem_tokens:
            token_id, poem_id, token_index, token_text = token
            mapped_poem_id = poem_id_map[poem_id]

            existing_token = duckdb_conn.execute(
                "SELECT token_id FROM poem_tokens WHERE poem_id = ? AND token_index = ? AND token_text = ?",
                (mapped_poem_id, token_index, token_text),
            ).fetchone()

            if not existing_token:
                duckdb_conn.execute(
                    "INSERT INTO poem_tokens (poem_id, token_index, token_text) VALUES (?, ?, ?)",
                    (mapped_poem_id, token_index, token_text),
                )

        # Merge translation tokens
        translation_tokens = source_conn.execute(
            "SELECT * FROM translation_tokens"
        ).fetchall()
        for token in translation_tokens:
            token_id, translation_id, token_index, token_text = token
            mapped_translation_id = translation_id_map.get(
                translation_id, translation_id
            )

            existing_token = duckdb_conn.execute(
                "SELECT token_id FROM translation_tokens WHERE translation_id = ? AND token_index = ? AND token_text = ?",
                (mapped_translation_id, token_index, token_text),
            ).fetchone()

            if not existing_token:
                duckdb_conn.execute(
                    "INSERT INTO translation_tokens (translation_id, token_index, token_text) VALUES (?, ?, ?)",
                    (mapped_translation_id, token_index, token_text),
                )

        # Create original token mapping
        original_token_map = {}
        poem_tokens = source_conn.execute("SELECT * FROM poem_tokens").fetchall()
        for token in poem_tokens:
            token_id, poem_id, token_index, token_text = token
            mapped_poem_id = poem_id_map[poem_id]

            existing_token = duckdb_conn.execute(
                "SELECT token_id FROM poem_tokens WHERE poem_id = ? AND token_index = ? AND token_text = ?",
                (mapped_poem_id, token_index, token_text),
            ).fetchone()

            if existing_token:
                original_token_map[token_id] = existing_token[0]

        # Create translated token mapping
        translated_token_map = {}
        translation_tokens = source_conn.execute(
            "SELECT * FROM translation_tokens"
        ).fetchall()
        for token in translation_tokens:
            token_id, translation_id, token_index, token_text = token
            mapped_translation_id = translation_id_map.get(
                translation_id, translation_id
            )

            existing_token = duckdb_conn.execute(
                "SELECT token_id FROM translation_tokens WHERE translation_id = ? AND token_index = ? AND token_text = ?",
                (mapped_translation_id, token_index, token_text),
            ).fetchone()

            if existing_token:
                translated_token_map[token_id] = existing_token[0]

        # Merge alignments with mapped IDs
        alignments = source_conn.execute("SELECT * FROM alignments").fetchall()
        for alignment in alignments:
            (
                alignment_id,
                translation_id,
                original_token_id,
                translated_token_id,
                alignment_model,
                temperature,
                run_id,
            ) = alignment
            mapped_translation_id = translation_id_map.get(
                translation_id, translation_id
            )
            mapped_original_token_id = original_token_map.get(
                original_token_id, original_token_id
            )
            mapped_translated_token_id = translated_token_map.get(
                translated_token_id, translated_token_id
            )

            existing_alignment = duckdb_conn.execute(
                "SELECT alignment_id FROM alignments WHERE translation_id = ? AND original_token_id = ? AND translated_token_id = ? AND run_id = ?",
                (
                    mapped_translation_id,
                    mapped_original_token_id,
                    mapped_translated_token_id,
                    run_id,
                ),
            ).fetchone()

            if not existing_alignment:
                duckdb_conn.execute(
                    "INSERT INTO alignments (translation_id, original_token_id, translated_token_id, alignment_model, temperature, run_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        mapped_translation_id,
                        mapped_original_token_id,
                        mapped_translated_token_id,
                        alignment_model,
                        temperature,
                        run_id,
                    ),
                )

        source_conn.close()

    logger.info("Database merge process completed.")


def parse_token(token: str) -> tuple[str, int]:
    """
    Parse a token, extracting the base and its order (from subscript).
    Returns a tuple of (base_token, order).
    If no subscript is present, order is 0.
    """
    # Check for subscript in parentheses first
    match = re.match(r"(.+?)(?:\((\d+)\))?$", token)
    if match:
        base_token, order = match.groups()
        return base_token, int(order) if order else 0

    # Then check for unicode subscripts
    base_token = token.rstrip("₀₁₂₃₄₅₆₇₈₉")
    subscript = token[len(base_token) :]

    if subscript:
        order = int(subscript.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")))
    else:
        order = 0

    return base_token, order


def create_tables(database: str, merge_databases: bool = False):
    global duckdb_conn
    duckdb_conn = duckdb.connect(database)

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

    # Migrations
    columns = duckdb_conn.execute("PRAGMA table_info(alignments)").fetchall()
    existing_columns = {column[1] for column in columns}

    if "alignment_model" not in existing_columns:
        duckdb_conn.execute("ALTER TABLE alignments ADD COLUMN alignment_model TEXT")
    if "temperature" not in existing_columns:
        duckdb_conn.execute("ALTER TABLE alignments ADD COLUMN temperature FLOAT")
    if "run_id" not in existing_columns:
        duckdb_conn.execute("ALTER TABLE alignments ADD COLUMN run_id TEXT")

    columns = duckdb_conn.execute("PRAGMA table_info(translations)").fetchall()
    existing_columns = {column[1] for column in columns}

    if "temperature" not in existing_columns:
        duckdb_conn.execute("ALTER TABLE translations ADD COLUMN temperature FLOAT")

    if merge_databases:
        merge_runs()


def save_poem(original_text: str) -> int:
    return duckdb_conn.execute(
        "INSERT INTO poems (original_text) VALUES (?) RETURNING (poem_id)",
        (original_text,),
    ).fetchone()[0]


def save_translation(
    poem_id: int,
    model: str,
    method: str,
    gen_type: str,
    temperature: float | None,
    translation_text: str,
) -> int:
    return duckdb_conn.execute(
        "INSERT INTO translations (poem_id, model, method, gen_type, temperature, translation_text) VALUES (?, ?, ?, ?, ?, ?) RETURNING (translation_id)",
        (poem_id, model, method, gen_type, temperature, translation_text),
    ).fetchone()[0]


def save_poem_tokens(poem_id: int, tokens: list[str]) -> list[int]:
    token_ids = []
    for index, token in enumerate(tokens):
        token_id = duckdb_conn.execute(
            "INSERT INTO poem_tokens (poem_id, token_index, token_text) VALUES (?, ?, ?) RETURNING (token_id)",
            (poem_id, index, token),
        ).fetchone()[0]
        token_ids.append(token_id)
    return token_ids


def save_translation_tokens(translation_id: int, tokens: list[str]) -> list[int]:
    token_ids = []
    for index, token in enumerate(tokens):
        token_id = duckdb_conn.execute(
            "INSERT INTO translation_tokens (translation_id, token_index, token_text) VALUES (?, ?, ?) RETURNING (token_id)",
            (translation_id, index, token),
        ).fetchone()[0]
        token_ids.append(token_id)
    return token_ids


def get_poem_by_text(original_text: str) -> dict | None:
    result = duckdb_conn.execute(
        "SELECT poem_id FROM poems WHERE original_text = ?", (original_text,)
    ).fetchone()
    return {"poem_id": result[0]} if result else None


def get_poem_token_ids(poem_id: int) -> list[int]:
    result = duckdb_conn.execute(
        "SELECT token_id FROM poem_tokens WHERE poem_id = ? ORDER BY token_index",
        (poem_id,),
    ).fetchall()
    return [row[0] for row in result]


def get_gold_translation(poem_id: int) -> dict | None:
    result = duckdb_conn.execute(
        "SELECT translation_id FROM translations WHERE poem_id = ? AND model = 'gold_standard'",
        (poem_id,),
    ).fetchone()
    return {"translation_id": result[0]} if result else None


def get_translation_token_ids(translation_id: int) -> list[int]:
    result = duckdb_conn.execute(
        "SELECT token_id FROM translation_tokens WHERE translation_id = ? ORDER BY token_index",
        (translation_id,),
    ).fetchall()
    return [row[0] for row in result]


def convert_alignment_to_indexes(
    alignments: list[tuple[str, str]],
    source_tokens: list[str],
    target_tokens: list[str],
    source_token_ids: list[int],
    target_token_ids: list[int],
) -> list[tuple[int, int]]:
    """
    Convert string-based alignments to index-based alignments.

    Parameters:
    - alignments: List of tuples containing string-based alignments.
    - source_tokens: List of source tokens.
    - target_tokens: List of target tokens.
    - source_token_ids: List of source token IDs.
    - target_token_ids: List of target token IDs.

    Returns:
    - List of tuples containing index-based alignments.
    """
    index_alignments = []

    source_token_map = {idx: db_idx for idx, db_idx in enumerate(source_token_ids)}
    target_token_map = {idx: db_idx for idx, db_idx in enumerate(target_token_ids)}

    logger.warning(f"source_token_map: {source_token_map}")
    logger.warning(f"target_token_map: {target_token_map}")
    for i, (original_token, translated_token) in enumerate(alignments):
        try:
            original_index = source_token_map[i]
            translated_index = target_token_map[i]
            index_alignments.append((original_index, translated_index))
        except KeyError:
            logger.error(
                f"Failed to find token IDs for alignment: {original_token} -- {translated_token}"
            )

    return index_alignments


def save_alignment_direct(
    translation_id: int,
    alignments: list[tuple[str, str]],
    original_token_ids: list[int],
    translated_token_ids: list[int],
    alignment_model: str,
    temperature: float,
    run_id: str,
) -> None:
    """
    Saves the alignment directly from the provided alignments.
    If the mapping is partially incorrect, it will still save the 'correct' alignments.
    """
    logger.info(
        f"Saving {alignments}, original: {original_token_ids}, translated: {translated_token_ids}"
    )
    index_alignments = convert_alignment_to_indexes(
        alignments,
        original_token_ids,
        translated_token_ids,
        original_token_ids,
        translated_token_ids,
    )

    logger.info(f"Saving index-based alignments: {index_alignments}")

    for original_index, translated_index in index_alignments:
        duckdb_conn.execute(
            "INSERT INTO alignments (translation_id, original_token_id, translated_token_id, alignment_model, temperature, run_id) VALUES (?, ?, ?, ?, ?, ?)",
            (
                translation_id,
                original_index,
                translated_index,
                alignment_model,
                temperature,
                run_id,
            ),
        )


def save_alignment(
    translation_id: int,
    alignment,
    original_tokens: list[tuple[int, str]],
    translated_tokens: list[tuple[int, str]],
    alignment_model: str,
    temperature: float,
    run_id: str,
) -> None:
    """Saves the alignment between the original and translated tokens to the database."""
    logger.info(f"Saving {alignment}\n{original_tokens}\n{translated_tokens}")

    original_dict = {}
    for t_id, token in original_tokens:
        base_token, order = parse_token(token)
        if base_token not in original_dict:
            original_dict[base_token] = []
        original_dict[base_token].append((t_id, order))

    logger.info(f"Original dict: {original_dict}")

    translated_dict = {}
    for t_id, token in translated_tokens:
        base_token, order = parse_token(token)
        if base_token not in translated_dict:
            translated_dict[base_token] = []
        translated_dict[base_token].append((t_id, order))

    logger.info(f"Translated dict: {translated_dict}")

    alignments_to_save = []

    for align in alignment.alignment:
        if not isinstance(align, TokenAlignment):
            logger.error(f"Unexpected alignment type: {type(align)}")
            continue

        original_base, original_order = parse_token(align.original_token)
        translated_base, translated_order = parse_token(align.translated_token)

        logger.info(f"Original: {original_base}, order={original_order}")
        logger.info(f"Translated: {translated_base}, order={translated_order}")
        if original_base in original_dict and translated_base in translated_dict:
            original_matches = original_dict[original_base]
            translated_matches = translated_dict[translated_base]

            original_matches.sort(key=lambda x: x[1])
            translated_matches.sort(key=lambda x: x[1])

            # Try to align the non-split token first
            original_id = next(
                (id for id, order in original_matches if order == original_order),
                None,
            )
            translated_id = next(
                (id for id, order in translated_matches if order == translated_order),
                None,
            )

            if original_id is not None and translated_id is not None:
                alignments_to_save.append(
                    (
                        original_id,
                        translated_id,
                        align.original_token,
                        align.translated_token,
                    )
                )
            else:
                # Fallback to aligning multiple tokens if present separately
                for t in align.translated_token.split():
                    translated_base, translated_order = parse_token(t)
                    translated_matches = translated_dict.get(translated_base, [])
                    translated_matches.sort(key=lambda x: x[1])

                    translated_id = next(
                        (
                            id
                            for id, order in translated_matches
                            if order == translated_order
                        ),
                        None,
                    )

                    if original_id is not None and translated_id is not None:
                        alignments_to_save.append(
                            (
                                original_id,
                                translated_id,
                                align.original_token,
                                t,
                            )
                        )
                    else:
                        if original_id is None:
                            logger.warning(
                                f"Original token not found: {align.original_token}"
                            )
                        if translated_id is None:
                            logger.warning(f"Translated token not found: {t}")

    logger.info(f"Alignments to save: {alignments_to_save}")
    # Save all valid alignments
    for (
        original_id,
        translated_id,
        original_token,
        translated_token,
    ) in alignments_to_save:
        try:
            duckdb_conn.execute(
                "INSERT INTO alignments (translation_id, original_token_id, translated_token_id, alignment_model, temperature, run_id) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    translation_id,
                    original_id,
                    translated_id,
                    alignment_model,
                    temperature,
                    run_id,
                ),
            )
        except Exception as e:
            logger.error(
                f"Failed to save alignment: {original_token} -- {translated_token}. Error: {str(e)}"
            )

    # Check for unaligned tokens
    aligned_original = set(align.original_token for align in alignment.alignment)
    aligned_translated = set(align.translated_token for align in alignment.alignment)
    unaligned_original = set(token for _, token in original_tokens) - aligned_original
    unaligned_translated = (
        set(token for _, token in translated_tokens) - aligned_translated
    )

    if unaligned_original or unaligned_translated:
        logger.warning(
            f"Not all tokens were aligned. Unaligned Original: {unaligned_original}, Unaligned Translated: {unaligned_translated}"
        )


def get_alignment_data_for_visualization() -> list[dict]:
    """
    Retrieves alignments from the database and formats them for visualization.

    Returns:
    - A list of dictionaries, each containing the original text, translated text, and alignment data.
    """
    query = """
    SELECT 
        a.translation_id,
        t.poem_id,
        t.model,
        t.method,
        t.gen_type,
        t.temperature AS translation_temperature,
        a.run_id,
        a.alignment_model,
        a.temperature AS model_temperature,
        pt.token_index AS original_index,
        tt.token_index AS translated_index
    FROM 
        alignments a
    JOIN 
        translations t ON a.translation_id = t.translation_id
    JOIN
        poem_tokens pt ON a.original_token_id = pt.token_id
    JOIN
        translation_tokens tt ON a.translated_token_id = tt.token_id
    ORDER BY
        a.run_id, t.poem_id, a.translation_id, pt.token_index, tt.token_index;
    """

    results = duckdb_conn.execute(query).fetchall()

    alignments = []
    current_translation = None
    for row in results:
        translation_id = row[0]

        if (
            current_translation is None
            or current_translation["translation_id"] != translation_id
        ):
            if current_translation is not None:
                alignments.append(current_translation)

            logger.info(f"Processing row: {row}")
            current_translation = {
                "translation_id": translation_id,
                "poem_id": row[1],
                "model": row[2],
                "method": row[3],
                "gen_type": row[4],
                "translation_temperature": row[5],
                "run_id": row[6],
                "alignment_model": row[7],
                "model_temperature": row[8],
                "alignment": [],
            }

        current_translation["alignment"].append([row[9], row[10]])

    if current_translation is not None:
        alignments.append(current_translation)

    # Now, let's fetch the tokens for each poem and translation
    for alignment in alignments:
        poem_tokens = get_poem_tokens(alignment["poem_id"])
        translation_tokens = get_translation_tokens(alignment["translation_id"])

        alignment["original_tokens"] = [token[1] for token in poem_tokens]
        alignment["translated_tokens"] = [token[1] for token in translation_tokens]

    return alignments


def get_poem_tokens(poem_id: int) -> list[tuple[int, str]]:
    """
    Retrieves tokens for a given poem.
    """
    query = """
    SELECT token_index, token_text
    FROM poem_tokens
    WHERE poem_id = ?
    ORDER BY token_index
    """
    return duckdb_conn.execute(query, [poem_id]).fetchall()


def get_translation_tokens(translation_id: int) -> list[tuple[int, str]]:
    """
    Retrieves tokens for a given translation.
    """
    query = """
    SELECT token_index, token_text
    FROM translation_tokens
    WHERE translation_id = ?
    ORDER BY token_index
    """
    return duckdb_conn.execute(query, [translation_id]).fetchall()


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
        t.temperature AS translation_temperature,
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
    seen_translations = set()
    for row in results:
        translation_key = (row[0], row[1], row[2], row[3], row[4], row[5], row[6])
        if translation_key not in seen_translations:
            seen_translations.add(translation_key)
            translations.append(
                {
                    "poem_id": row[0],
                    "translation_id": row[1],
                    "model": row[2],
                    "method": row[3],
                    "gen_type": row[4],
                    "gold_translation": row[5],
                    "generated_translation": row[6],
                    "translation_temperature": row[7],
                    "alignment": json.loads(row[8]) if row[8] else [],
                    "original_text": row[9],
                }
            )

    return translations


def get_unaligned_translations(
    poem_id: int,
    alignment_model: str | None = None,
    temperature: float | None = None,
    strict: bool = True,
) -> list[dict]:
    """
    Retrieves unaligned translations for a given poem_id.
    If strict is True, it filters out translations that are already aligned with the specific model and temperature.
    If strict is False, it returns all translations, even if they have existing alignments.
    """
    assert isinstance(poem_id, int), poem_id
    query = """
    SELECT 
        t.translation_id, 
        t.translation_text, 
        array_agg(tt.token_text ORDER BY tt.token_index) AS translated_tokens
    FROM 
        translations t
    LEFT JOIN 
        alignments a ON t.translation_id = a.translation_id
    JOIN 
        translation_tokens tt ON t.translation_id = tt.translation_id
    WHERE 
        t.poem_id = ?
    """

    params = [poem_id]

    if strict:
        if alignment_model is not None:
            query += " AND (a.alignment_model != ? OR a.alignment_model IS NULL)"
            params.append(alignment_model)

        if temperature is not None:
            query += " AND (a.temperature != ? OR a.temperature IS NULL)"
            params.append(temperature)

    query += " GROUP BY t.translation_id, t.translation_text"
    query += " ORDER BY t.translation_id;"

    results = duckdb_conn.execute(query, params).fetchall()

    translations = []
    for row in results:
        translations.append(
            {
                "translation_id": row[0],
                "translation_text": row[1],
                "translated_tokens": row[2],
            }
        )

    return translations


def get_tokens(poem_id: int, translation_id: int) -> dict[str, list[tuple[int, str]]]:
    """
    Given a poem_id and translation_id, return a dictionary containing the tokens and their IDs of the original and the translation.
    """
    original_tokens_query = """
    SELECT token_id, token_text 
    FROM poem_tokens
    WHERE poem_id = ?
    ORDER BY token_index;
    """
    translated_tokens_query = """
    SELECT token_id, token_text 
    FROM translation_tokens
    WHERE translation_id = ?
    ORDER BY token_index;
    """

    original_tokens = duckdb_conn.execute(original_tokens_query, (poem_id,)).fetchall()
    translated_tokens = duckdb_conn.execute(
        translated_tokens_query, (translation_id,)
    ).fetchall()

    return {
        "original_tokens": [(int(token[0]), token[1]) for token in original_tokens],
        "translated_tokens": [(int(token[0]), token[1]) for token in translated_tokens],
    }


def get_translations_by_token():
    """
    Returns all translations of each poem token by model, method, gen_type, alignment_model,
    alignment_model_temperature, and translation_temperature.
    """
    query = """
    SELECT 
        pt.token_text AS original_token,
        tt.token_text AS translated_token,
        t.model,
        t.method,
        t.gen_type,
        a.alignment_model,
        a.temperature AS alignment_model_temperature,
        t.temperature AS translation_temperature
    FROM 
        alignments a
    JOIN 
        translations t ON a.translation_id = t.translation_id
    JOIN 
        poem_tokens pt ON a.original_token_id = pt.token_id
    JOIN 
        translation_tokens tt ON a.translated_token_id = tt.token_id
    ORDER BY 
        pt.token_text, t.model, t.method, t.gen_type;
    """
    results = duckdb_conn.execute(query).fetchall()

    translations = []
    for row in results:
        translations.append(
            {
                "original_token": row[0],
                "translated_token": row[1],
                "model": row[2],
                "method": row[3],
                "gen_type": row[4],
                "alignment_model": row[5],
                "alignment_model_temperature": row[6],
                "translation_temperature": row[7],
            }
        )

    return translations


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
    poems p
JOIN 
    translations t ON p.poem_id = t.poem_id
LEFT JOIN 
    poem_tokens pt ON p.poem_id = pt.poem_id
LEFT JOIN 
    translation_tokens tt ON t.translation_id = tt.translation_id
ORDER BY 
    p.poem_id, t.translation_id, pt.token_index, tt.token_index;
""")


def get_translation_info(translation_id: int) -> dict:
    """
    Retrieves information about a specific translation.
    """
    query = """
    SELECT 
        model,
        method,
        gen_type,
        temperature
    FROM 
        translations
    WHERE 
        translation_id = ?
    """
    result = duckdb_conn.execute(query, (translation_id,)).fetchone()
    if result:
        return {
            "model": result[0],
            "method": result[1],
            "gen_type": result[2],
            "temperature": result[3],
        }
    else:
        return None
