import outlines
from outlines import generate

# from outlines.models.openai import OpenAIConfig
import magentic
from pydantic import BaseModel, Field
from typing import Optional, Iterable, Annotated, TypeVar
import transformers
import torch
from itertools import groupby
import re


from hachidaishu_translation.hachidaishu import HachidaishuDB
from hachidaishu_translation.format import (
    format_poem,
    remove_unique,
    visualize_alignment,
    make_unique,
    normalize_tokens,
)
import hachidaishu_translation.db as db
from hachidaishu_translation.utils import retry
from hachidaishu_translation.log import logger

import gc
import random
from tqdm import tqdm


def get_hf_model(model_name: str):
    logger.info(f"Loading model {model_name}.")
    config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.init_device = "meta"
    kwargs = {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
    }
    bab_support = (
        True
        if torch.cuda.is_available() == "cuda" and torch.version.hip is None
        else False
    )
    if bab_support:
        # This is gated to avoid loading on non-CUDA devices
        from bitsandbytes import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["quantization_config"] = quantization_config
        kwargs["load_in_4bit"] = True
        kwargs["attn_implementation"] = "flash_attention_2"

    model = outlines.models.transformers(
        model_name=model_name,
        device="cuda",
        model_kwargs=kwargs,
    )
    return model


# Nobs to tweak in instructions:
# - translate by chunks
# - translate by words
# - translate whole line (preferred)
# TODO
# - n:m or 1:1 word alignment for translations
# - literal translation vs. poetic translation
#   - focus on glossing (~literal translation w/o extra (unneeded) context)


@outlines.prompt
def translate_chunks(waka):
    """Translate the following waka poem into English. Output only the translation all in one line with / as delimiter between parts.

    {{ waka }}
    """


@outlines.prompt
def translate_words(waka):
    """Translate the following waka poem into English, word-by-word. Output the translation in one or more words, one line per word separated with a colon.
    {{ waka }}
    """


@outlines.prompt
def translate_lines(waka):
    """Translate the following waka poem into English. Output the translation as a single line.
    {{ waka }}
    """


class LinesSchema(BaseModel):
    original: str = Field(description="Original poem.")
    transliterated: Optional[str] = Field(
        description="Transliterated poem.", default=None
    )
    pos_marked: Optional[str] = Field(description="PoS-marked poem.", default=None)
    translated: str = Field(description="Translated poem.")


@magentic.prompt(
    "Translate this waka poem into English. Use at least 15 words:\n{original}"
)
def create_lines_translation(
    original: str,
    transliterated: str | None = None,
    pos_marked: str | None = None,
) -> LinesSchema: ...


# # Bitext word alignment task
# We are concerned with the alignment of tokens from the original to the translated text.
# As such, we want to capture 1:1 or 1:n alignments, and not be concerned with n:m alignments.


class TokenAlignment(BaseModel):
    original_token: str = Field(description="Original token string")
    translated_token: str = Field(description="Translated token string")


class TokenAlignmentSchema(BaseModel):
    alignment: list[TokenAlignment] = Field(
        description="List of (original_token, translated_token) alignment tuples from tokens in original to translated. Tokens are space-delimited strings and should not be further split. A token can be aligned to multiple other tokens. A token can be unaligned, in the case of tokens with predominantly grammatical function, in which case do not align it. For disambiguation purposes, a token repeated multiple times will have a suffix appended to it so you must use it indicate the correct alignment."
    )


# Each token that is aligned should consist of only one string with no whitespace. In the case of multi-token phrases, align each token of the phrase to the same token separately.
T = TypeVar("T")


class IndexedTokens(list[T]):
    def __format__(self, format_spec: str) -> str:
        # Here we ensure that the tokens passed in for alignment are normalized and unique
        actual_tokens = make_unique(normalize_tokens(self))
        return f"{' '.join(actual_tokens)} ({len(actual_tokens)} tokens)"


@magentic.prompt(
    "Align the tokens in the original and translated waka:\n\n## Original\n{original}\n\n## Translation\n{translated}"
)
def align_tokens(
    original: Annotated[
        IndexedTokens[str], Field(description="List of original tokens")
    ],
    translated: Annotated[
        IndexedTokens[str], Field(description="List of translated tokens")
    ],
) -> TokenAlignmentSchema: ...


delimiter_regex = r"([\w ]+?/){4}[\w ]+"
line_regex = r"[\w ]+"
word_regex = r"([ぁ-ゟァ-ヿ一-鿿]{1,8}: [\w ]+\n)+"

translation_matrix = {
    "lines": {
        "regex": {
            "prompt": translate_lines,
            "regex": line_regex,
        },
        "pydantic": {
            "prompt": create_lines_translation,
        },
    },
    # After experimentation, whole-line-based translation works best.
    # "chunks": {
    #     "regex": {
    #         "prompt": translate_chunks,
    #         "regex": delimiter_regex,
    #     },
    #     "pydantic": {
    #         "prompt": create_chunks_translation,
    #     },
    # },
    # "words": {
    #     "regex": {
    #         "prompt": translate_words,
    #         "regex": word_regex,
    #     },
    #     "pydantic": {
    #         "prompt": create_tokens_translation,
    #     },
    # },
}


def translate_regex(
    model,  #: outlines.models.transformers.Transformers,
    text: str,
    prompt_template,
    regex,
):
    prompt = prompt_template(text)
    generator = generate.regex(model, regex)
    answer = generator(prompt, max_tokens=100)
    return answer


# FIXME: This is a hack to avoid https://github.com/openai/openai-python/issues/1469
magentic.OpenaiChatModel._get_stream_options = lambda _: None


def translate_pydantic(
    _model,
    text: str,
    prompt_function,
):
    # with magentic.OpenaiChatModel(model, temperature=0, seed=42):
    logger.debug(prompt_function.format(text))
    answer = prompt_function(text)
    logger.info(answer)
    return answer.translated


def run(tokens: list[str], original_token_ids: list[int], poem_id: int) -> list[dict]:
    """Models are chosen based on the following criteria:
    - Score on the [Nejumi LLM Leaderboard](https://wandb.ai/wandb-japan/llm-leaderboard/reports/Nejumi-leaderboard-Neo--Vmlldzo2MTkyMTU0)
    - (Base) Models should be trained on distinct datasets
    """
    results = []
    for model_name in [
        # "microsoft/Phi-3-mini-4k-instruct",
        # "CohereForAI/aya-23-8B", # Generates gibberish
        # "augmxnt/shisa-gamma-7b-v1",
        # "Qwen/Qwen1.5-7B-Chat",
        # "TheBloke/Swallow-7B-Instruct-AWQ",
        # "tokyotech-llm/Swallow-MS-7b-instruct-v0.1",
        # "tokyotech-llm/Swallow-7b-instruct-v0.1",
        # "mistralai/Mistral-7B-Instruct-v0.3",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        "4o-global",  # 2024-05-13
    ]:
        if model_name.count("/") == 1:
            model_type = "regex"
            model = get_hf_model(model_name)
        else:
            model_type = "pydantic"
            model = model_name
        for gen_type, instructions in tqdm(
            translation_matrix.items(), position=0, desc="Parameters"
        ):
            logger.info(f"Running {model_name} {model_type} {gen_type}.")
            if model_type == "regex":
                translation = translate_regex(
                    model,
                    tokens,
                    instructions[model_type]["prompt"],
                    instructions[model_type]["regex"],
                )
            elif model_type == "pydantic":
                translation = translate_pydantic(
                    model,
                    tokens,
                    instructions[model_type]["prompt"],
                )
            else:
                raise ValueError(f"Invalid model type {model_type}.")

            # Result normalization:
            translation = translation.strip(".").replace("\n", " ")

            result = {
                "model": model_name,
                "method": model_type,
                "gen_type": gen_type,
                "translation": translation,
            }
            results.append(result)

            translation_id = db.save_translation(
                poem_id, model_name, model_type, gen_type, translation
            )
            result["translation_id"] = translation_id

            translated_tokens = normalize_tokens(
                [
                    word
                    for item in (
                        translation if isinstance(translation, list) else [translation]
                    )
                    for word in (item if isinstance(item, list) else item.split())
                ]
            )
            translated_token_ids = db.save_translation_tokens(
                translation_id, translated_tokens
            )
            result["translated_tokens"] = translated_tokens
            result["translated_token_ids"] = translated_token_ids

            # alignment = align_tokens(
            #     IndexedTokens(tokens), IndexedTokens(translated_tokens)
            # )
            # db.save_alignment(
            #     translation_id,
            #     alignment,
            #     dict(zip(make_unique(tokens), original_token_ids)),
            #     dict(zip(make_unique(translated_tokens), translated_token_ids)),
            # )
        # Free up GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return results


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


def random_sample(lines, n=5, random_seed=42):
    random.seed(random_seed)
    return random.sample(lines, n)


if __name__ == "__main__":
    db.create_tables()

    translation_map = waka_to_translations()
    hachidaishu_db = HachidaishuDB()
    by_poem = groupby(
        hachidaishu_db.query(),
        key=lambda r: (r.anthology, r.poem),
    )
    # eagerly load all poems
    poems = [list(poem) for _, poem in by_poem]
    # poem_sample = [
    #     poem
    #     for poem in poems
    #     if format_poem([r.token() for r in poem], delim="") in translation_map
    # ]
    poem_sample = random_sample(
        [
            poem
            for poem in poems
            if format_poem([r.token() for r in poem], delim="") in translation_map
        ],
        n=10,
    )

    for poem in poem_sample:
        # tokens = list(db.tokens(anthology=Anthology.Kokinshu, poem=32))
        tokens = [r.token() for r in poem]
        original_tokens = format_poem(tokens, delim=" ", split=True)
        gold_translation = translation_map[format_poem(tokens, delim="")]

        o_list = normalize_tokens(original_tokens.split())
        gt_list = normalize_tokens(gold_translation.split())

        o_list_unique = make_unique(o_list)
        gt_list_unique = make_unique(gt_list)

        logger.info(f"O\t\t{original_tokens}")
        logger.info(f"GT\t\t{gold_translation}")
        logger.info(f"O\t\t{o_list_unique}")
        logger.info(f"GT\t\t{gt_list_unique})")

        poem_id = db.save_poem(original_tokens)
        original_token_ids = db.save_poem_tokens(poem_id, remove_unique(o_list_unique))

        gold_translation_id = db.save_translation(
            poem_id, "gold_standard", "manual", "gold", gold_translation
        )
        gold_translation_token_ids = db.save_translation_tokens(
            gold_translation_id, remove_unique(gt_list_unique)
        )

        def visualize_and_save_alignment():
            alignment = align_tokens(IndexedTokens(o_list), IndexedTokens(gt_list))
            alignment_visualization = visualize_alignment(
                o_list_unique,
                gt_list_unique,
                alignment,
            )
            logger.info("\n" + alignment_visualization.get_string())
            db.save_alignment(
                gold_translation_id,
                alignment,
                dict(zip(o_list_unique, original_token_ids)),
                dict(zip(gt_list_unique, gold_translation_token_ids)),
            )

        retry(visualize_and_save_alignment)

        translations = run(o_list_unique, original_token_ids, poem_id)
        for translation in translations:
            logger.info(
                f"{translation['model']} {translation['gen_type']}\t{translation['translation']}"
            )

            if len(translation["translated_tokens"]) < 12:
                logger.warning(
                    f"Skipping short translation {translation['translation']}."
                )
                continue

            def visualize_and_save_translation_alignment():
                alignment = align_tokens(
                    IndexedTokens(o_list),
                    IndexedTokens(translation["translated_tokens"]),
                )
                alignment_visualization = visualize_alignment(
                    o_list_unique,
                    make_unique(normalize_tokens(translation["translated_tokens"])),
                    alignment,
                )
                logger.info("\n" + alignment_visualization.get_string())
                db.save_alignment(
                    translation["translation_id"],
                    alignment,
                    dict(zip(o_list_unique, original_token_ids)),
                    dict(
                        zip(
                            make_unique(translation["translated_tokens"]),
                            translation["translated_token_ids"],
                        )
                    ),
                )

            retry(visualize_and_save_translation_alignment)
