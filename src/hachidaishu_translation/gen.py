import outlines
from outlines import generate

# from outlines.models.openai import OpenAIConfig
import magentic
from pydantic import BaseModel, Field
import transformers
import torch

from hachidaishu_translation.hachidaishu import HachidaishuDB
from hachidaishu_translation.format import format_poem, pp

import logging
import gc
from tqdm import tqdm


logger = logging.getLogger(__name__)

# If we decided to use outlines for the OpenAI models again:
# def get_azure_model(model_name="gpt-4-0613", deployment_name="gpt-4-vision-japaneasy"):
#     return models.azure_openai(
#         model_name=model_name,
#         api_key=env["AZURE_API_KEY"],
#         api_version=env["AZURE_API_VERSION"],
#         azure_endpoint=env["AZURE_API_BASE"],
#         deployment_name=deployment_name,
#         config=OpenAIConfig(seed=42, temperature=0.0),
#     )
#
#
# gpt_35_turbo = get_azure_model(
#     model_name="gpt-3.5-turbo", deployment_name="gpt-35-turbo-deployment"
# )
# gpt_4_0613 = get_azure_model(
#     model_name="gpt-4-0613",
#     deployment_name="gpt-4-vision-japaneasy",
# )


def get_hf_model(model_name="mistralai/Mistral-7B-Instruct-v0.3"):
    config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.init_device = "meta"
    model = outlines.models.transformers(
        model_name=model_name,
        device="cuda",
        model_kwargs={
            "config": config,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            # Needs CUDA:
            # "load_in_8bit": True,
            # "load_in_4bit": True,
            # "attn_implementation": "flash_attention_2",
            "device_map": {"": 0},
        },
    )
    return model


# Nobs to tweak in instructions:
# - translate by chunks
# - translate by words
# - translate whole line
# - literal translation vs. poetic translation


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


delimiter_regex = r"([^/]+/){4}[^/]+"
line_regex = r"[^\n]+"
word_regex = r"([^:]+: [\w\s]+\n)+"


class LinesSchema(BaseModel):
    original: str = Field(description="Original poem.")
    transliterated: str | None = Field(description="Transliterated poem.")
    translated: str = Field(description="Translated poem.")


@magentic.prompt("Create a translation for {original}.")
def create_lines_translation(original: str) -> LinesSchema: ...


class ChunksSchema(BaseModel):
    original: list[str] = Field(description="List of original chunks.")
    transliterated: list[str] | None = Field(
        description="List of transliterated chunks."
    )
    translated: list[str] = Field(
        description="List of translations of each original chunk in same order as original."
    )


@magentic.prompt("Create a translation for {original}.")
def create_chunks_translation(original: list[str]) -> ChunksSchema: ...


@magentic.prompt(
    "Create a translation for {original} transliterated as {transliterated}."
)
def create_chunks_transliterated_translation(
    original: list[str], transliterated: list[str]
) -> ChunksSchema: ...


class TokensSchema(BaseModel):
    original: list[str] = Field(description="List of original tokens.")
    transliterated: list[str] | None = Field(
        description="List of transliterated tokens."
    )
    translated: list[str] = Field(
        description="List of translated tokens in same order as original."
    )


@magentic.prompt("Create translations for all tokens in {original}.")
def create_tokens_translation(original: str) -> TokensSchema: ...


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
    "chunks": {
        "regex": {
            "prompt": translate_chunks,
            "regex": delimiter_regex,
        },
        "pydantic": {
            "prompt": create_chunks_translation,
        },
    },
    "words": {
        "regex": {
            "prompt": translate_words,
            "regex": word_regex,
        },
        "pydantic": {
            "prompt": create_tokens_translation,
        },
    },
}


def translate_regex(
    model,  #: outlines.models.transformers.Transformers,
    text,
    prompt_template,
    regex,
):
    prompt = prompt_template(text)
    generator = generate.regex(model, regex)
    answer = generator(prompt, max_tokens=100)
    return answer


def translate_pydantic(
    model,
    text,
    prompt_function,
):
    with magentic.OpenaiChatModel(model, temperature=0, seed=42):
        answer = prompt_function(text)
        logging.info(answer)
        return answer.translated





def run(poem_text):
    """Models are chosen based on the following criteria:
    - Score on the [Nejumi LLM Leaderboard](https://wandb.ai/wandb-japan/llm-leaderboard/reports/Nejumi-leaderboard-Neo--Vmlldzo2MTkyMTU0)
    - (Base) Models should be trained on distinct datasets
    """
    results = []
    for model_name in [
        # "CohereForAI/aya-23-8B", # Generates gibberish
        # "augmxnt/shisa-gamma-7b-v1",
        # "Qwen/Qwen1.5-7B-Chat",
        # "TheBloke/Swallow-7B-Instruct-AWQ",
        # "tokyotech-llm/Swallow-MS-7b-instruct-v0.1",
        # "tokyotech-llm/Swallow-7b-instruct-v0.1",
        # "mistralai/Mistral-7B-Instruct-v0.3",
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        "gpt-4-0613",
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
            if model_type == "regex":
                translation = translate_regex(
                    model,
                    poem_text,
                    instructions[model_type]["prompt"],
                    instructions[model_type]["regex"],
                )
            elif model_type == "pydantic":
                translation = translate_pydantic(
                    model,
                    poem_text,
                    instructions[model_type]["prompt"],
                )
            else:
                raise ValueError(f"Invalid model type {model_type}.")

            result = {
                "model": model_name,
                "method": model_type,
                "gen_type": gen_type,
                "translation": translation,
            }
            results.append(result)
        # Free up GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return results


def model_judge(translations: list[str], judge_model="gpt_4_0613"):
    """Judge the quality of multiple translations using a model."""
    with magentic.OpenaiChatModel(judge_model, seed=42, temperature=0):
        judged_translations = []
        # TODO
        # for translation in translations:
        #     judged = judge_model(translation)
        #     judged_translations.append(judged)
        # return judged_translations

from itertools import groupby
import re

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
                original = re.sub(r"〈[^〉]+〉", "", text.strip("＠／ \n").replace("／", ""))
            elif code == "N":
                translation = text.strip("＠/ \n").replace("  ", " ")
        
        if original and translation:
            mapping[original] = translation
        else:
            raise Exception(f"Missing original or translation for {original} in {lines}.")
    return mapping


import random
def random_sample(lines, n=5, random_seed=42):
    random.seed(random_seed)
    return random.sample(lines, n)

if __name__ == "__main__":
    translation_map = waka_to_translations()
    db = HachidaishuDB()
    by_poem = groupby(
        db.query(),
        key=lambda r: (r.anthology, r.poem),
    )
    # eagerly load all poems
    poems = [list(poem) for _, poem in by_poem]
    poem_sample = random_sample([poem for poem in poems if format_poem([r.token() for r in poem], delim="")
                                 in translation_map], n=5)
    # found = 0
    # for _poem_info, poem in by_poem:
    #     original = re.sub(r"[〈〉]", "", format_poem([r.token() for r in poem], delim="", split=False))
    #     try:
    #         translation = translation_map[original]
    #         found += 1
    #     except KeyError:
    #         print(f"Missing translation for {original}.")
    # print(f"Found {found} translations in {len(translation_map)} map.")

    for poem in poem_sample:
        # tokens = list(db.tokens(anthology=Anthology.Kokinshu, poem=32))
        tokens = [r.token() for r in poem]
        print(format_poem(tokens, delim=" "))
        print(format_poem(tokens, delim=""))
        print(format_poem(tokens, delim=" ", split=True))
        print(format_poem(tokens, delim="", split=False))
        print(format_poem(tokens, delim=" ", romaji=True))
        print(format_poem(tokens, delim="", romaji=True))
        print(format_poem(tokens, delim=" ", split=True, romaji=True))
        print(format_poem(tokens, delim="", split=False, romaji=True))
        print(translation_map[format_poem(tokens, delim="")])
        print(pp(translation_map[format_poem(tokens, delim="")], lang="en"))
    
    # print(db.text(delimiter="", embed_metadata=True))
    # judged_translations = model_judge(translations)
    # for translation in judged_translations:
    #     print(translation)
