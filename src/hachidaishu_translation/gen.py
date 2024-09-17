import random
import re
import uuid
from itertools import groupby, islice
from typing import Any

import outlines
from torch import t
from tqdm import tqdm

import hachidaishu_translation.db as db
from hachidaishu_translation.format import (
    clean_bad_translation,
    format_poem,
    make_unique,
    normalize_tokens,
    remove_unique,
    visualize_alignment,
)
from hachidaishu_translation.hachidaishu import HachidaishuDB
from hachidaishu_translation.log import logger
from hachidaishu_translation.models import (
    IndexedTokens,
    ModelCapability,
    ModelConfig,
    ModelType,
    TranslationType,
    align_tokens,
    build_run_configuration,
    check_alignment_strings,
    get_magentic_model,
)
from hachidaishu_translation.utils import retry


def waka_to_translations(filename="kokindb.txt") -> dict[str, str]:
    with open(filename, "r", encoding="euc-jp") as f:
        lines = f.readlines()
    mapping = {}
    entries = groupby(lines, key=lambda s: s.split(" ")[0])
    for _, lines in entries:
        original, translation = None, None
        for line in lines:
            if line.count("|") != 1:
                continue
            meta, text = line.split("|")
            code = meta[-1]
            if code == "I":
                original = re.sub(
                    r"〈[^〉]+〉", "", text.strip("＠／ \n").replace("／", "")
                )
            elif code == "N":
                translation = text.strip("＠/ \n").replace("  ", " ")

        if original and translation:
            mapping[original] = translation
        else:
            raise Exception(
                f"Missing original or translation for {original} in {lines}."
            )
    return mapping


def random_sample(lines: list[Any], n: int = 5, random_seed: int = 42) -> list[Any]:
    random.seed(random_seed)
    return random.sample(lines, n)


def extract_and_filter_poems(
    num_samples: int | None,
    limit_to_golden_translations: bool = True,
) -> tuple[dict[str, str], list[list[Any]]]:
    """
    Extracts and filters poems from the Hachidaishu database and matches them to golden translations.

    Returns:
    - A dictionary mapping original poems to their translations.
    - A list of poems.
    """
    translation_map = waka_to_translations()
    hachidaishu_db = HachidaishuDB()

    by_poem = groupby(
        hachidaishu_db.query(),
        key=lambda r: (r.anthology, r.poem),
    )
    poems = [list(poem) for _, poem in by_poem]
    logger.info(f"Extracted {len(poems)} poems.")

    if limit_to_golden_translations:
        poems = [
            poem
            for poem in poems
            if format_poem([r.token() for r in poem], delim="") in translation_map
        ]
        logger.info(f"Filtered poems to {len(poems)} with golden translations.")

    if num_samples:
        poems = random_sample(poems, n=num_samples)

    return translation_map, poems


# FIXME add type hints
def process_poem(
    poem: list[Any], translation_map: dict[str, str]
) -> tuple[int, list[int], list[str], int, list[str], list[int], list[str]]:
    tokens = [r.token() for r in poem]
    original_tokens = format_poem(tokens, delim=" ", split=True)
    original_chunks = original_tokens.split(" / ")
    gold_translation = translation_map.get(format_poem(tokens, delim=""))

    o_list = normalize_tokens(original_tokens.split())
    gt_list = normalize_tokens(gold_translation.split())

    o_list_unique = make_unique(o_list)
    gt_list_unique = make_unique(gt_list)

    logger.info(f"O\t\t{original_tokens}")
    logger.info(f"GT\t\t{gold_translation}")
    logger.debug(f"O\t\t{o_list_unique}")
    logger.debug(f"GT\t\t{gt_list_unique})")

    # Check if the poem already exists in the database
    existing_poem = db.get_poem_by_text(original_tokens)
    if existing_poem:
        poem_id = existing_poem["poem_id"]
        original_token_ids: list[int] = db.get_poem_token_ids(poem_id)
    else:
        poem_id = db.save_poem(original_tokens)
        original_token_ids: list[int] = db.save_poem_tokens(
            poem_id,
            remove_unique(o_list_unique),  # Save the non-marked-up tokens
        )

    # Check if the gold translation already exists
    existing_gold_translation = db.get_gold_translation(poem_id)
    if existing_gold_translation:
        gold_translation_id = existing_gold_translation["translation_id"]
        gold_translation_token_ids = db.get_translation_token_ids(gold_translation_id)
    else:
        gold_translation_id = db.save_translation(
            poem_id, "gold_standard", "manual", "gold", None, gold_translation
        )
        gold_translation_token_ids = list(
            db.save_translation_tokens(
                gold_translation_id,
                gt_list,  # Save the non-marked-up tokens
            )
        )

    return (
        poem_id,
        original_token_ids,
        o_list_unique,
        gold_translation_id,
        gt_list_unique,
        gold_translation_token_ids,
        original_chunks,
    )


def run_contamination_check(
    models: list[str], poems: list[list[Any]], batch_size: int
) -> None:
    for model_name in models:
        model = ModelConfig(model_name)
        formatted_poems = list(
            format_poem([r.token() for r in poem], delim="", split=True).split(" / ")
            for poem in poems
        )
        logger.info(f"Poems: {formatted_poems}")
        total_batches = (sum(1 for _ in formatted_poems) + batch_size - 1) // batch_size
        results = []
        with tqdm(
            total=total_batches, desc=f"Processing {model_name}", unit="batch"
        ) as pbar:
            for i in range(0, total_batches):
                batch = list(
                    islice(formatted_poems, i * batch_size, (i + 1) * batch_size)
                )
                batch_results = model.contamination_check(batch, batch_size=batch_size)
                results.extend(batch_results)
                pbar.update(1)
        logger.info(f"Model: {model_name}, Contamination Check: {results}")


def contains_japanese(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    return bool(re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", text))


def is_too_short(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z\s]", "", text)).count(" ") < 12


def save_translations_and_alignments(
    poem_id: int,
    model_name: str,
    model_type: ModelType,
    capability: ModelCapability,
    run_type: TranslationType,
    translation: str,
    original_tokens: list[str],
    original_token_ids: list[int],
    alignments: list[tuple[str, str]] | None,
    temperature: float,
) -> None:
    translation_id = db.save_translation(
        poem_id,
        model_name,
        model_type.name + "-" + capability.value,
        run_type.value,
        temperature,
        translation,
    )

    original_tokens = remove_unique(original_tokens)

    if alignments:
        logger.info(f"Saving translation tokens with alignments for {translation}")
        translated_tokens = [token for _, token in alignments]

        translated_token_ids = db.save_translation_tokens(
            translation_id, translated_tokens
        )

        alignments = check_alignment_strings(
            original_tokens, translated_tokens, alignments
        )
        if not alignments:
            logger.error(
                f"Failed to align translation tokens for {translation}. Skipping."
            )
            return

        run_id = str(uuid.uuid4())
        db.save_alignment_direct(
            translation_id,
            alignments,
            original_token_ids,
            translated_token_ids,
            alignment_model=model_name,
            temperature=temperature,
            run_id=run_id,
        )
    else:
        logger.info(
            f"Not saving alignments as not present for: {translation}. Needs to be separately aligned."
        )


def retry_translation(
    model_config: ModelConfig,
    original_inputs: list[list[str]] | list[str],
    capability: ModelCapability,
    run_type: TranslationType,
    batch_size: int,
    max_retries: int = 3,
) -> list[tuple[str | None, list[tuple[str, str]] | None]]:
    translations_with_alignments = model_config.translate(
        original_inputs,
        chosen_capability=capability,
        chosen_type=run_type,
        batch_size=batch_size,
    )

    logger.warning(f"Translations with alignments: {translations_with_alignments}")
    for i, (translation, _alignment) in enumerate(translations_with_alignments):
        if isinstance(translation, ValueError):
            logger.error(
                f"Cannot retry for ValueError in {translation}. Setting to None."
            )
            translations_with_alignments[i] = (None, None)
            continue
        # elif isinstance(translation, str) and alignment is not None:
        #     continue
        if translation:
            translation = clean_bad_translation(translation)

        current_translation = translation
        temperature = model_config.temperature

        not_translation = not translation
        too_short = is_too_short(translation)
        contains_ja = contains_japanese(translation)
        logger.info(
            f"current_translation: {current_translation}, not_translation: {not_translation}, too_short: {too_short}--{is_too_short(translation) if translation else 'NA'}, contains_ja: {contains_ja}--{contains_japanese(translation) if translation else 'NA'}"
        )
        if not_translation or too_short or contains_ja:
            for attempt in range(max_retries):
                temperature += 0.1
                model_config.sampler = outlines.samplers.MultinomialSampler(
                    temperature=temperature
                )
                logger.warning(
                    f"Translation rejected (not_translation={not_translation}, too_short={too_short}, contains_japanese={contains_ja}). Retrying translation '{current_translation}' with increased temperature: {temperature} (attempt {attempt + 1}/{max_retries})"
                )
                new_translation_with_alignments_batch = model_config.translate(
                    [original_inputs[i]],
                    chosen_capability=capability,
                    chosen_type=run_type,
                    batch_size=1,
                )
                if not new_translation_with_alignments_batch:
                    logger.error(
                        f"Translation batch invalid: {new_translation_with_alignments_batch}"
                    )
                    continue

                new_translation_with_alignments = new_translation_with_alignments_batch[
                    0
                ]
                new_translation = new_translation_with_alignments[0]
                current_translation = new_translation

                not_translation = not new_translation
                too_short = is_too_short(new_translation)
                contains_ja = contains_japanese(new_translation)
                if not not_translation and not too_short and not contains_ja:
                    translations_with_alignments[i] = new_translation_with_alignments
                    break
            else:
                logger.error(
                    f"Max retries reached for translation {i}. Setting to None."
                )
                model_config.sampler = model_config.default_sampler
                translations_with_alignments[i] = (None, None)

    return translations_with_alignments


def main(
    models: list[str],
    run_types: list[TranslationType],
    num_samples: int,
    align_mode: bool = False,
    temperature: float = 0.0,
    align_model_name: str | None = None,
    preferred_generation_methods: list[ModelCapability] = [
        ModelCapability.PYDANTIC,
        ModelCapability.REGEX,
    ],
    contamination_check: bool = False,
    batch_size: int = 1,
    dtype: str = "bfloat16",
    strict_alignment: bool = True,
    database: str = "translations.db",
    merge_databases: bool = False,
):
    db.create_tables(database, merge_databases)

    if contamination_check:
        translation_map, poems = extract_and_filter_poems(num_samples)
        return run_contamination_check(models, poems, batch_size)

    if align_mode:
        for poem_id in set(
            [row[0] for row in db.query().fetchall()]
        ):  # Get unique poem_ids
            unaligned_translations = db.get_unaligned_translations(
                poem_id,
                alignment_model=align_model_name,
                temperature=temperature,
                strict=strict_alignment,
            )
            for translation in unaligned_translations:
                translation_id = translation["translation_id"]

                # Check if this is a word-based translation and skip it if it is
                translation_info = db.get_translation_info(translation_id)
                if translation_info["gen_type"] == TranslationType.WORDS.value:
                    logger.info(
                        f"Skipping alignment for word-based translation (translation_id: {translation_id})"
                    )
                    continue

                tokens_data = db.get_tokens(poem_id, translation_id)
                original_tokens = [
                    token for _idx, token in tokens_data["original_tokens"]
                ]
                translated_tokens = [
                    token for _idx, token in tokens_data["translated_tokens"]
                ]

                # Save alignments on existing translations
                align_and_save(
                    source_tokens_unique=make_unique(original_tokens),
                    target_tokens_unique=make_unique(translated_tokens),
                    translation_id=translation_id,
                    poem_id=poem_id,
                    align_model_name=align_model_name,
                    temperature=temperature,
                )
        return  # Exit after finishing alignments in align_mode

    translation_map, poems = extract_and_filter_poems(num_samples)
    run_configuration = build_run_configuration(
        models,
        run_types,
        preferred_generation_methods=preferred_generation_methods,
        temperature=temperature,
        dtype=dtype,
    )
    logger.debug(run_configuration)
    for model_name, model_config in run_configuration.items():
        model_type = model_config.model_type
        try:
            for capability in model_config.prompts.keys():
                for run_type in model_config.prompts[capability].keys():
                    # Translation logic
                    for i in range(0, len(poems), batch_size):
                        batch_poems = poems[i : i + batch_size]
                        processed_poems = [
                            process_poem(poem, translation_map) for poem in batch_poems
                        ]

                        # Subscripts such as ₂ may not be in model vocabulary preventing regex generation from working so use the original chunks or tokens (?)
                        # But in any case, we should remove them at this step as not all methods require them.
                        original_inputs = [
                            original_chunks
                            if run_type == TranslationType.CHUNKS
                            else remove_unique(o_list_unique)
                            for (
                                _poem_id,
                                _original_token_ids,
                                o_list_unique,
                                _gold_translation_id,
                                _gt_list_unique,
                                _gold_translation_token_ids,
                                original_chunks,
                            ) in processed_poems
                        ]

                        translations_with_alignments: list[
                            tuple[str | None, list[tuple[str, str]] | None]
                        ] = retry_translation(
                            model_config,
                            original_inputs,
                            capability,
                            run_type,
                            batch_size,
                        )

                        # We filter out failed translations so we do not save or have to align them.
                        valid_translations = [
                            (poem, (translation, alignment))
                            for poem, (translation, alignment) in zip(
                                processed_poems, translations_with_alignments
                            )
                            if translation is not None
                        ]

                        # Now we save the translations and alignments (if available)
                        # Gold alignments are only saved in align mode, so we skip them here.
                        for (
                            poem_id,
                            original_token_ids,
                            o_list_unique,
                            _gold_translation_id,
                            _gt_list_unique,
                            _gold_translation_token_ids,
                            _original_chunks,
                        ), (translation, alignment) in valid_translations:
                            save_translations_and_alignments(
                                poem_id,
                                model_name,
                                model_type,
                                capability,
                                run_type,
                                translation,
                                o_list_unique,
                                original_token_ids,
                                alignment,
                                model_config.temperature,
                            )

        finally:
            logger.info(f"Releasing {model_name} resources...")
            model_config.gc()


def align_and_save(
    source_tokens_unique: list[str],
    target_tokens_unique: list[str],
    translation_id: int,
    poem_id: int,
    align_model_name: str,
    temperature: float,
):
    def visualize_and_save_alignment():
        with get_magentic_model(align_model_name, temperature=temperature):
            alignment = align_tokens(
                IndexedTokens(source_tokens_unique),
                IndexedTokens(target_tokens_unique),
            )
        alignment_visualization = visualize_alignment(
            source_tokens_unique,
            target_tokens_unique,
            alignment,
        )
        logger.info("\n" + alignment_visualization.get_string())

        tokens = db.get_tokens(poem_id, translation_id)
        run_id = str(uuid.uuid4())
        db.save_alignment(
            translation_id,
            alignment,
            tokens["original_tokens"],
            tokens["translated_tokens"],
            alignment_model=align_model_name,
            temperature=temperature,
            run_id=run_id,
        )

    retry(visualize_and_save_alignment)


if __name__ == "__main__":
    from statistics import median

    from hachidaishu_translation.format import normalize_tokens

    translation_map, poems = extract_and_filter_poems(None, False)

    original_word_counts = []
    golden_word_counts = []
    total_poems = len(poems)
    matched_poems = 0

    for poem in poems:
        original_tokens = normalize_tokens(
            format_poem([r.token() for r in poem], delim=" ", split=True).split()
        )
        if len(original_tokens) < 8:
            logger.error(original_tokens)
        original_word_counts.append(len(original_tokens))

        gold_translation = translation_map.get(
            format_poem([r.token() for r in poem], delim="")
        )
        if gold_translation:
            matched_poems += 1
            golden_tokens = normalize_tokens(gold_translation.split())
            golden_word_counts.append(len(golden_tokens))

    print(f"Total number of poems: {total_poems}")
    print(f"Poems matched to golden translation: {matched_poems}")
    print(f"Poems without golden translation: {total_poems - matched_poems}")

    print("\nOriginal poem statistics:")
    print(f"Min word count: {min(original_word_counts)}")
    print(f"Max word count: {max(original_word_counts)}")
    print(f"Median word count: {median(original_word_counts)}")

    if golden_word_counts:
        print("\nGolden translation statistics:")
        print(f"Min word count: {min(golden_word_counts)}")
        print(f"Max word count: {max(golden_word_counts)}")
        print(f"Median word count: {median(golden_word_counts)}")
    else:
        print("\nNo golden translations available.")

    # Print min-max length of all chunks
    chunks = [
        chunk
        for poem in poems
        for chunk in format_poem(
            [r.token() for r in poem], delim=" ", split=True
        ).split(" / ")
    ]
    chunk_lengths = [len(chunk) for chunk in chunks]
    print("\nChunk length statistics:")
    print(f"Min chunk length: {min(chunk_lengths)}")
    print(f"Max chunk length: {max(chunk_lengths)}")

    # Print 5 shortest and 5 longest chunks
    sorted_chunks = sorted(zip(chunks, chunk_lengths), key=lambda x: x[1])
    print("\n5 shortest chunks:")
    for chunk, length in sorted_chunks[:5]:
        print(f"Length {length}: {chunk}")
    print("\n5 longest chunks:")
    for chunk, length in sorted_chunks[-5:]:
        print(f"Length {length}: {chunk}")

    # Log a warning for any chunk with 0 characters
    for i, (chunk, length) in enumerate(zip(chunks, chunk_lengths)):
        if length == 0:
            poem_index = next(
                j
                for j, poem in enumerate(poems)
                if chunk
                in format_poem([r.token() for r in poem], delim=" ", split=True).split(
                    " / "
                )
            )
            full_poem = format_poem(
                [r.token() for r in poems[poem_index]], delim=" ", split=True
            )
            logger.warning(
                f"Found a chunk with 0 characters. Poem index: {poem_index}, Full poem: {full_poem}"
            )

    # Calculate and print statistics for the last chunk of each poem
    last_chunks = [
        format_poem([r.token() for r in poem], delim=" ", split=True).split(" / ")[-1]
        for poem in poems
    ]
    last_chunk_lengths = [len(chunk) for chunk in last_chunks]

    print("\nLast chunk statistics:")
    print(f"Min last chunk length: {min(last_chunk_lengths)}")
    print(f"Max last chunk length: {max(last_chunk_lengths)}")
    print(f"Median last chunk length: {median(last_chunk_lengths)}")
