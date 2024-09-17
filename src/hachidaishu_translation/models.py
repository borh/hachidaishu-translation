import gc
import json
import os
import re
from enum import StrEnum, auto
from functools import partial
from itertools import islice
from typing import Annotated, Any, Callable, Dict, TypeVar

import magentic
import outlines
import pydantic_core
from annotated_types import Len
from magentic import ConfigDict
from pydantic import BaseModel, Field

from hachidaishu_translation.format import (
    clean_bad_translation,
    make_unique,
    remove_unique,
    normalize_tokens,
)
from hachidaishu_translation.log import logger

# # Bitext word alignment task
# We are concerned with the alignment of tokens from the original to the translated text.
# As such, we want to capture 1:1 or 1:n alignments, and not be concerned with n:m alignments.

# # Strategy
# We use outlines for structured generation of llama.cpp and Transformers models.
# We use magentic for structured generation of OpenAI models.
# For Chat API-only models like plamo-beta that do not support function calling/JSON output, we use magentic in chat mode (we could also use outlines).

# Nobs to tweak in instructions:
# - translate by chunks
# - translate by words
# - translate whole line (preferred)

# TODO
# - n:m or 1:1 word alignment for translations
# - literal translation vs. poetic translation
#   - focus on glossing (~literal translation w/o extra (unneeded) context)


# TODO Few-shot examples in all prompts (behind flag)

SYSTEM_PROMPT = """You are an expert Japanese-English poetry translator. Your task is to faithfully translate given Japanese waka poems into English. Ensure that each translation is between 16 and 40 words long. Provide only the translated poem without any additional notes or formatting unless ordered to do so."""

# https://manabink.com/en/2020/10/03/haiku-and-tanka-by-masaoka-shiki/
ONE_SHOT_LINE = [
    """Translate the following waka poem into English. Only output the translation as a single line.

## Japanese

くれなゐの / 二尺伸びたる / 薔薇の芽の / 針やはらかに / 春雨のふる

## English translation
""",
    """
In bright crimson red / stretching up a full two feet / the buds of roses / with their soft and tender thorns / as the rain of springtime falls
""",
]

# WORD_REGEX = r"[A-Za-z]{1,20}"
# delimiter_regex = r"([\w ]+?/){4}[\w ]+\.?"
line_regex = r"([A-Za-z;,]+ ){16,40}"  # Matches min-max (+alpha) in gold translations


def create_regex(items: list[str], type: str) -> str:
    patterns = []
    if type == "word":
        for token in items:
            pattern = (
                rf"{re.escape(token)}: ([a-zA-Z]{{1,20}} ){{1,2}}([a-zA-Z]{{1,20}})?\n"
            )
            patterns.append(pattern)
    elif type == "chunk":
        for chunk in items:
            pattern = rf"{chunk}: [a-zA-Z ]{{1,40}}\n"
            patterns.append(pattern)
    re_pattern = "".join(patterns).rstrip(r"\n")
    return re_pattern


# word_regex = None  # This will be set dynamically in the translate method
# chunk_regex = None  # This will be set dynamically in the translate method


class LinesSchema(BaseModel):
    # model_config = ConfigDict(openai_strict=True)
    translated: str = Field(
        description="Translated poem in English, around 20 to 40 words."
    )


class ChunksSchema(BaseModel):
    # model_config = ConfigDict(openai_strict=True)
    translated: tuple[str, str, str, str, str] = Field(
        description="Translated poem in 5 chunks, following original order."
    )


class ChunksSchemaOpenAI(BaseModel):
    model_config = ConfigDict(openai_strict=True)
    translated: list[str] = Field(
        description="Translated poem in 5 chunks, following original order."
    )


class WordSchema(BaseModel):
    original: str = Field(description="Original Japanese word")
    translated: str = Field(description="Translated English word")


class WordsSchema(BaseModel):
    words: Annotated[list[WordSchema], Len(min_length=9, max_length=36)] = Field(
        description="List of original-translated word pairs."
    )


class WordsSchemaOpenAI(BaseModel):
    model_config = ConfigDict(openai_strict=True)
    words: list[WordSchema] = Field(
        description="List of original-translated word pairs."
    )


# fmt: off
@outlines.prompt
def outlines_lines_prompt_fn(instructions, waka) -> LinesSchema:
    """{{ instructions }}

Translate the following waka poem into English. Only output the translation as a single line.

## Japanese

{{ waka }}

## English translation

"""


@outlines.prompt
def outlines_chunks_prompt_fn(instructions, waka) -> ChunksSchema:
    """{{ instructions }}

Translate the following waka poem into English. Only output the translation in chunks, one at a time, in their original order.

## Japanese

{{ waka }}

## English translation (chunks)

"""


# TODO: Add validation on original words, and redo up to 3 times
@outlines.prompt
def outlines_words_pydantic_prompt_fn(instructions, json_schema, waka) -> WordsSchema:
    """{{ instructions }}

Translate the following waka poem into English, word-by-word. Output the translation for each word using at most 3 English words.

You must adhere to the following JSON schema:\n<schema>\n{{ json_schema }}\n</schema>

## Japanese

{{ waka }}

## Japanese-English word pairs

"""

@outlines.prompt
def outlines_words_regex_prompt_fn(instructions, waka) -> WordsSchema:
    """{{ instructions }}

Translate the following waka poem into English, word-by-word. Output the translation for each word using at most 3 English words.

## Japanese

{{ waka }}

## Japanese-English word pairs

"""
# fmt: on

# TODO
# Contamination check:
# - check prevalance of translations online
# - how many samples to check?


class ContaminationCheck(BaseModel):
    know: bool = Field(description="Indicates whether you know the given waka poem.")
    name: str | None = Field(description="Name of the author of the waka, if known.")
    continuation: str = Field(
        description="The final 7 syllables of the given waka in Japanese."
    )


# fmt: off
@outlines.prompt
def contamination_check_prompt_fn(
    instructions: str, json_schema: str, waka: str
) -> ContaminationCheck:
    """{{ instructions }}

Complete the final 7 syllables of the given waka poem.
You must adhere to the following JSON schema:\n<schema>\n{{ json_schema }}\n</schema>

## Japanese Waka Poem (5-7-5-7 (first 24) syllables)

{{ waka }}

## Completion (Final 7 syllables)
"""
# fmt: on

# # Magentic prompts


@magentic.prompt("""{instructions}

Translate the following waka poem into English. Only output the translation as a single line.

## Japanese

{waka}

## English translation

""")
def magentic_lines_prompt_fn(instructions: str, waka: str) -> LinesSchema: ...


@magentic.prompt("""{instructions}

Translate the following waka poem into English. Only output the translation in chunks, one at a time, in their original order.

## Japanese

{waka}

## English translation (chunks)

""")
def magentic_chunks_prompt_fn(instructions: str, waka: str) -> ChunksSchemaOpenAI: ...


@magentic.prompt("""{instructions}

Translate the following waka poem into English, word-by-word. Output the translation for each word using at most 3 English words.

## Japanese

{waka}

## Japanese-English word pairs

""")
def magentic_words_prompt_fn(instructions: str, waka: str) -> WordsSchemaOpenAI: ...


# def convert_placeholders(template: str) -> str:
#     # Convert {{ variable }} to {variable}
#     return re.sub(r"\{\{\s*(\w+)\s*\}\}", r"{\1}", template)


@magentic.chatprompt(
    magentic.SystemMessage("{instructions}"),
    magentic.UserMessage(ONE_SHOT_LINE[0]),
    magentic.AssistantMessage(ONE_SHOT_LINE[1]),
    magentic.UserMessage("""Translate the following waka poem into English. Only output the translation as a single line.

## Japanese

{waka}

## English translation

"""),
)
def magentic_lines_chat_prompt_fn(
    instructions: str,
    waka: str,
) -> str: ...


@magentic.chatprompt(
    magentic.SystemMessage("{instructions}"),
    magentic.UserMessage(ONE_SHOT_LINE[0]),
    magentic.AssistantMessage(ONE_SHOT_LINE[1]),
    magentic.UserMessage("""Translate the following waka poem into English. Only output the translation in chunks, one at a time, in their original order.

## Japanese

{waka}

## English translation

"""),
)
def magentic_chunks_chat_prompt_fn(
    instructions: str,
    waka: str,
) -> str: ...


# TODO
@magentic.chatprompt(
    magentic.SystemMessage("{instructions}"),
    magentic.UserMessage("""Translate the following waka poem into English, word-by-word. Output the translation for each word using at most 3 English words, one translation per line.

## Japanese

{waka}

## Japanese-English word pairs in 'word: translation' format

"""),
)
def magentic_words_chat_prompt_fn(
    instructions: str,
    waka: str,
) -> str: ...


# # Token alignment


class TokenAlignment(BaseModel):
    model_config = magentic.ConfigDict(openai_strict=True)
    original_token: str = Field(description="A single Japanese word")
    translated_token: str = Field(description="Translation of Japanese word")


class TokenAlignmentSchema(BaseModel):
    model_config = magentic.ConfigDict(openai_strict=True)
    alignment: list[TokenAlignment] = Field(
        description="A list of (original_token, translated_token) TokenAlignment objects representing the alignment between tokens in the original and translated texts. The provided tokens are space-delimited strings and should not be further split. A token can be aligned to multiple tokens; in such cases, include multiple tuples with the same original_token paired with different translated_tokens. Unaligned tokens (typically those with predominantly grammatical function) can be omitted from the alignment list. For disambiguation, if a token appears multiple times, a suffix is appended to it; reuse this suffix to ensure correct alignment."
    )


# Each token that is aligned should consist of only one string with no whitespace. In the case of multi-token phrases, align each token of the phrase to the same token separately.
T = TypeVar("T")


class IndexedTokens(list[T]):
    def __format__(self, format_spec: str) -> str:
        # Here we ensure that the tokens passed in for alignment are normalized and unique
        actual_tokens = make_unique(normalize_tokens(list(self)))
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


# def check_alignment_index(
#     source: list[str], target: list[str], alignment: list[tuple[int, int]]
# ) -> list[tuple[int, int]] | None:
#     max_source = max(alignment, key=lambda x: x[0])[0]
#     logger.info(f"Checking alignment: {source} {target} {alignment}")
#     if max_source != len(source) - 1:
#         logger.error(f"Max source index: {max_source}, expected: {len(source) - 1}")
#     max_target = max(alignment, key=lambda x: x[1])[1]
#     if max_target != len(target) - 1:
#         logger.error(f"Max target index: {max_target}, expected: {len(target) - 1}")
#
#     return alignment


def check_alignment_strings(
    source: list[str], target: list[str], alignment: list[tuple[str, str]]
) -> list[tuple[str, str]] | None:
    logger.info(
        f"Checking alignment for hallucinated tokens: {source} {target} {alignment}"
    )

    source_set = set(source)
    target_set = set(target)
    alignment_source_tokens = set(pair[0] for pair in alignment)
    alignment_target_tokens = set(pair[1] for pair in alignment)

    failed = False
    if source_set != alignment_source_tokens:
        # Example:
        # Source-Alignment diff: {'足', '山郭公', '曵'} {'山', '足曵', '郭公'} => autumn | wear | in | hidden fin | break | is | Ashibiki | of | under mountain | resounding | deer | of | cry | probably
        # We could come up with a convoluted way to align here, but probably not worth the effort.
        logger.error(
            f"Source-Alignment diff: {source_set.difference(alignment_source_tokens)} only in original, {alignment_source_tokens.difference(source_set)} only in alignment"
        )
        failed = True
    if target_set != alignment_target_tokens:
        logger.error(
            f"Target-Alignment diff: {target_set.difference(alignment_target_tokens)} only in original,{alignment_target_tokens.difference(target_set)} only in alignment"
        )
        failed = True

    if failed:
        return None
    else:
        return alignment


class ModelType(StrEnum):
    GGUF = auto()
    HF = auto()
    MAGENTIC = auto()


class ModelCapability(StrEnum):
    REGEX = auto()
    PYDANTIC = auto()
    CHAT = auto()
    CONTAMINATION_CHECK = auto()


class TranslationType(StrEnum):
    LINES = auto()
    CHUNKS = auto()
    WORDS = auto()


class PromptConfig:
    def __init__(
        self,
        prompt_function: Callable,
        capability: ModelCapability,
        regex: str | None = None,
        return_type: Any = None,
    ):
        self.capability = capability
        self.prompt_function = prompt_function
        self.regex = regex
        self.return_type = return_type

    def __repr__(self):
        return f"PromptConfig(capability={self.capability}, prompt_function={self.prompt_function}, regex={self.regex}, return_type={self.return_type})"


class ModelConfig:
    def __init__(
        self, model_name: str, temperature: float = 0.3, dtype: str = "bfloat16"
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.dtype = dtype
        self.model_type = self._determine_model_type()
        self.capabilities = self._determine_capabilities()
        self._model = None
        self.prompts = self._build_prompts()
        self.default_sampler = (
            outlines.samplers.greedy()
            if self.temperature == 0.0
            else outlines.samplers.MultinomialSampler(temperature=self.temperature)
        )
        self.sampler = self.default_sampler

    @property
    def model(self):
        if self._model is None:
            self._model = self.load_model()
        return self._model

    def __repr__(self):
        return f"ModelConfig(model_name={self.model_name}, capabilities={self.capabilities}, prompts={self.prompts})"

    def _batch_generator(self, iterable, batch_size):
        iterator = iter(iterable)
        yield from iter(lambda: list(islice(iterator, batch_size)), [])

    def load_model(self):
        if self.model_type == ModelType.GGUF:
            return self._get_gguf_model()
        elif self.model_type == ModelType.HF:
            return self._get_hf_model()
        elif self.model_type == ModelType.MAGENTIC:
            return self._get_magentic_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def gc(self):
        self._model = self.model
        # Free up GPU memory
        if self._model is not None:
            del self._model
        gc.collect()
        if self.model_type == ModelType.HF:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass  # torch is not available, skip CUDA cleanup

    def _determine_model_type(self) -> ModelType:
        if self.model_name.endswith(".gguf"):
            return ModelType.GGUF
        elif self.model_name.count("/") == 1:
            return ModelType.HF
        else:
            return ModelType.MAGENTIC

    def _determine_capabilities(self) -> set[ModelCapability]:
        if self.model_type == ModelType.GGUF:
            return {ModelCapability.PYDANTIC, ModelCapability.REGEX}
        elif self.model_type == ModelType.HF:
            return {ModelCapability.PYDANTIC, ModelCapability.REGEX}
            # These do not support Pydantic-based structured generation (well) yet
        elif (
            "plamo" in self.model_name
            or "mistral" in self.model_name
            or "mixtral" in self.model_name
        ):
            return {ModelCapability.CHAT}
            # These can also do chat, but we prefer structured generation
        elif self.model_type == ModelType.MAGENTIC:
            return {ModelCapability.PYDANTIC}
        else:
            raise ValueError(f"Unknown model type for {self.model_name}")

    def contamination_check(self, poems: list[list[str]], batch_size: int = 1):
        json_schema = outlines.fsm.json_schema.convert_json_schema_to_str(
            ContaminationCheck
        )
        all_results = []

        for batch in self._batch_generator(poems, batch_size):
            prompts, end_strs = [], []
            for chunks in batch:
                start_str, end_str = " ".join(chunks[:-1]), chunks[-1]
                end_strs.append(end_str)
                prompt = contamination_check_prompt_fn(
                    SYSTEM_PROMPT, json_schema, start_str
                )
                logger.info(prompt)
                prompts.append(prompt)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    generator = outlines.generate.json(
                        self.model, ContaminationCheck, sampler=self.sampler
                    )
                    batch_results = generator(prompts)
                    results_contaminated = [
                        re.sub(r"\s", "", result.continuation) == end_str
                        for result, end_str in zip(batch_results, end_strs)
                    ]

                    batch_processed_results = [
                        result.model_dump(mode="python")
                        | {"is_contaminated": is_contaminated, "correct": chunks[-1]}
                        for result, is_contaminated, chunks in zip(
                            batch_results, results_contaminated, batch
                        )
                    ]
                    all_results.extend(batch_processed_results)
                    break  # If successful, break out of the retry loop
                except pydantic_core._pydantic_core.ValidationError as e:
                    if attempt < max_retries - 1:  # if it's not the last attempt
                        logger.warning(
                            f"JSONDecodeError/Pydantic validation error occurred. Retrying with different sampler (attempt {attempt + 1}/{max_retries})...: {e}"
                        )
                    else:
                        logger.error(
                            f"JSONDecodeError/Pydantic validation error persisted after {max_retries} attempts. Skipping this batch: {e}"
                        )

        contamination_ratio = sum(r["is_contaminated"] for r in all_results) / len(
            all_results
        )
        logger.warning(f"Number of contaminated poems (ratio): {contamination_ratio}")

        return all_results

    def translate(
        self,
        text: list[str],
        chosen_capability: ModelCapability,
        chosen_type: TranslationType,
        batch_size: int = 1,
    ) -> list[tuple[str | None, list[tuple[str, str]] | None]]:
        prompt_config = self.prompts.get(chosen_capability, {}).get(chosen_type)
        if not prompt_config:
            raise ValueError(
                f"No prompt configuration found for capability: {chosen_capability} and type: {chosen_type}"
            )

        # Ensure the model is loaded before translation
        _ = self.model

        # Convert text to list if it's a single string
        texts = [text] if isinstance(text, str) else text

        results = []
        for batch in self._batch_generator(texts, batch_size):
            logger.info(f"Translating batch: {batch}")
            if chosen_capability == ModelCapability.REGEX:
                if chosen_type == TranslationType.WORDS:
                    # Dynamically set the regex for word translation
                    original_tokens = (
                        batch[0] if isinstance(batch[0], list) else batch[0].split()
                    )
                    prompt_config.regex = create_regex(original_tokens, "word")
                elif chosen_type == TranslationType.CHUNKS:
                    # Dynamically set the regex for chunk translation
                    assert isinstance(batch[0], list), batch[0]
                    original_chunks = batch[0]
                    prompt_config.regex = create_regex(original_chunks, "chunk")
                batch_results = self._translate_regex(batch, prompt_config)

            elif (
                chosen_capability == ModelCapability.PYDANTIC
                and ModelCapability.REGEX in self.capabilities
            ):
                batch_results = self._translate_pydantic(batch, prompt_config)
            elif chosen_capability == ModelCapability.PYDANTIC:
                batch_results = self._translate_magentic_function(batch, prompt_config)
            elif chosen_capability == ModelCapability.CHAT:
                batch_results = self._translate_magentic_chat_function(
                    batch, prompt_config
                )
            else:
                raise ValueError(f"Unsupported capability: {chosen_capability}")

            results.extend(batch_results)

        return results  # [(result, alignment) for result, alignment in results]

    def _format_poem(
        self, tokens: list[str], return_type: LinesSchema | ChunksSchema | WordsSchema
    ) -> str:
        logger.warning(f"Formatting poem: {tokens} {return_type}")
        if return_type == LinesSchema:
            return "".join(tokens)
        elif return_type == ChunksSchema:
            return " / ".join(tokens)
        elif return_type == WordsSchema:
            return " ".join(tokens)
        else:
            raise ValueError(f"Unsupported return type: {return_type}")

    def _translate_pydantic(
        self,
        texts: list[list[str]],
        prompt_config: PromptConfig,
        sampler=None,
        temperature=None,
        max_tokens=450,  # A bit higher due to JSON overhead
        retries=3,
    ) -> list[tuple[str | None, list[tuple[str, str]] | None]]:
        """Uses Pydantic schema to translate text.
        There are three types of translations: lines, chunks, and words.

        For word translations, we obtain token alignments, but for lines and chunks, this cannot be automatic.
        Fro words, alignment is done using a mapping from token string to token(s) string, so we have to make sure we are using unique token names for the source.
        Unlike for regex-based translations, we do not need to dynamically set the regex for word/chunk translations, as the schema is not that specific. This means that we however need to verify that we are not missing or hallucinating any tokens in the translation.
        """
        if retries == 0:
            logger.error(
                f"Max retries reached for Pydantic schema using {self.model_name} / {prompt_config.capability}. Skipping this batch: {texts}"
            )
            return []

        logger.info(f"Translating {prompt_config.return_type} with Pydantic schema.")

        generator = outlines.generate.json(
            self.model,
            prompt_config.return_type,
            sampler=sampler if sampler else self.sampler,
        )
        prompts = [
            prompt_config.prompt_function(
                waka=self._format_poem(tokens, prompt_config.return_type)
            )
            for tokens in texts
        ]
        logger.info(
            f"Pydantic {self.model_name} / {prompt_config.capability} / {prompt_config.return_type} prompts: {[p for p in prompts]}"
        )
        try:
            translations = generator(prompts, max_tokens=max_tokens)
        except (
            pydantic_core._pydantic_core.ValidationError,
            json.decoder.JSONDecodeError,
        ) as e:
            temperature = temperature + 0.1 if temperature else self.temperature + 0.1
            max_tokens = max_tokens + 200
            logger.warning(
                f"Validation error on Pydantic schema using {self.model_name} / {prompt_config.capability} / {prompt_config.return_type}. Retrying with temperature={temperature} and max_tokens={max_tokens}: {e}. Retrying..."
            )
            return self._translate_pydantic(
                texts,
                prompt_config,
                sampler=outlines.samplers.MultinomialSampler(temperature=temperature),
                temperature=temperature,
                max_tokens=max_tokens,
                retries=retries - 1,
            )

        results, alignments = [], []
        for i, result in enumerate(translations):
            if isinstance(result, LinesSchema):
                results.append(result.translated)
                alignments.append(None)
            elif isinstance(result, ChunksSchema):
                results.append(" / ".join(result.translated))
                alignments.append(None)
            elif isinstance(result, WordsSchema):
                translated_tokens = remove_unique(
                    [word.translated for word in result.words]
                )
                generated_original_tokens = [word.original for word in result.words]
                # Check for extra tokens in alignment not in original (duplicates are allowed)
                extra_tokens = set(generated_original_tokens) - set(texts[i])
                if (
                    extra_tokens
                    # or check_alignment_strings(
                    #     texts[i],
                    #     translated_tokens,
                    #     [(o.original, o.translated) for o in result.words],
                    # )
                    # is False
                ):
                    logger.warning(
                        f"Token alignment mismatch: result={result}, generated_original_tokens={generated_original_tokens}, extra_tokens={extra_tokens}. Skipping translation."
                    )
                    results.append(None)
                    alignments.append(None)
                else:
                    results.append(" | ".join(translated_tokens))
                    alignments.append(
                        [(o.original, o.translated) for o in result.words]
                    )
            else:
                results.append(None)
                alignments.append(None)

        return list(zip(results, alignments))

    def _translate_regex(
        self, texts: list[list[str]], prompt_config: PromptConfig
    ) -> list[tuple[str | None | ValueError, list[tuple[str, str]] | None]]:
        logger.info(f"Translating {prompt_config.return_type} with regex.")
        results = []
        prompts = []
        regexes = []

        for text in texts:
            if prompt_config.return_type == WordsSchema:
                # Dynamically set the regex for word translation
                original_tokens = text if isinstance(text, list) else text.split()
                regex = create_regex(original_tokens, "word")
            elif prompt_config.return_type == ChunksSchema:
                # Dynamically set the regex for chunk translation
                assert isinstance(
                    text, list
                ), f"Expected list for chunks, got {type(text)}"
                regex = create_regex(text, "chunk")
            else:
                regex = prompt_config.regex

            formatted_poem = self._format_poem(text, prompt_config.return_type)
            logger.info(f"Formatted poem: {formatted_poem}")

            prompts.append(prompt_config.prompt_function(waka=formatted_poem))
            regexes.append(regex)

        logger.info(f"Prompts for regex {prompt_config}: {prompts}")

        answers = []
        # For word and chunk translations, we need to loop over the prompts as each regex is unique, effectively making this a batch of 1
        if (
            prompt_config.return_type == WordsSchema
            or prompt_config.return_type == ChunksSchema
        ):
            logger.info("Not batching for regex translations.")
            for prompt, regex in zip(prompts, regexes):
                try:
                    generator = outlines.generate.regex(
                        self.model, regex, sampler=self.sampler
                    )
                except ValueError as e:
                    logger.error(
                        f"ValueError while generating regex {regex} on model {self.model_name}: {e}. Giving up."
                    )
                    answers.append(e)
                    continue
                answer = generator(prompt, max_tokens=200)
                answers.append(answer)
        else:
            try:
                generator = outlines.generate.regex(
                    self.model, regexes[0], sampler=self.sampler
                )
            except ValueError as e:
                logger.error(
                    f"ValueError while generating regex {regexes[0]} on model {self.model_name}: {e}. Giving up."
                )
                return [(e, None)]
            answers = generator(prompts, max_tokens=200)
            answers = [clean_bad_translation(answer) for answer in answers]

        alignments = []
        for i, answer in enumerate(answers):
            if isinstance(answer, ValueError):
                results.append(answer)
                alignments.append(None)
            # Post-process the answer to extract only the translation (remove original tokens).
            elif answer and prompt_config.return_type in {ChunksSchema, WordsSchema}:
                delimiter = (
                    " / " if prompt_config.return_type == ChunksSchema else " | "
                )
                processed_lines = []
                for line in answer.split("\n"):
                    if ":" in line:
                        original, translated = line.split(":", 1)
                        processed_lines.append(
                            clean_bad_translation(translated).strip()
                        )
                    else:
                        # raise ValueError(f"Empty line in word translation: {line}")
                        logger.warning(f"Empty line in word translation: {line}")
                        processed_lines.append(
                            "()"
                        )  # TODO Placeholder for no alignment
                processed_answer = delimiter.join(processed_lines)

                generated_source_tokens = " ".join(
                    line.split(":", 1)[0].strip() if ":" in line else ""
                    for line in answer.split("\n")
                )
                text_for_comparison = (
                    " ".join(texts[i]) if isinstance(texts[i], list) else texts[i]
                )
                if generated_source_tokens != text_for_comparison:
                    logger.error(
                        f"Translation source tokens mismatch: generated={generated_source_tokens} <> original={text_for_comparison}. Not using translation or alingments."
                    )
                    results.append(None)
                    alignments.append(None)
                    continue

                results.append(processed_answer)
                if prompt_config.return_type == WordsSchema:
                    processed_alignment = []
                    for line in answer.split("\n"):
                        if ":" in line:
                            original, translated = line.split(":", 1)
                            processed_alignment.append(
                                (original, clean_bad_translation(translated).strip())
                            )
                        else:
                            logger.error(f"Empty line in word translation: {line}")
                            processed_alignment.append((line, ""))

                    assert all(
                        len(pair) == 2 for pair in processed_alignment
                    ), processed_alignment

                    alignments.append(processed_alignment)
                else:
                    # For chunk translations, alignments are done with align mode.
                    alignments.append(None)
            # Lines-based translation does not need post-processing.
            else:
                results.append(answer)
                alignments.append(None)

        return list(zip(results, alignments))

    def _translate_magentic_function(
        self, texts: list[list[str]], prompt_config: PromptConfig
    ) -> list[tuple[str | None, list[tuple[str, str]] | None]]:
        results = []
        logger.info(f"self.model: {self.model}, config: {prompt_config}")
        with self.model:
            for text in texts:
                logger.info(
                    f"Prompt function: {prompt_config.prompt_function} with return schema: {prompt_config.return_type.model_json_schema()}"
                )
                try:
                    answer = prompt_config.prompt_function(
                        SYSTEM_PROMPT,
                        self._format_poem(text, prompt_config.return_type),
                    )
                except Exception as e:
                    logger.error(
                        f"Error while generating prompt for {text} on model {self.model_name}: {e}... Giving up."
                    )
                    # TODO If we wanted to retry here we might need to provide more context to the next retry pointing out the error.
                    # It seems that models either work on first try or almost never do.
                    results.append((ValueError(e), None))
                    continue
                logger.warning(f"answer: {answer} ({type(answer)})")
                if isinstance(answer, str):
                    results.append((answer, None))
                elif isinstance(answer, ChunksSchemaOpenAI):
                    results.append((" / ".join(answer.translated), None))
                elif isinstance(answer, WordsSchemaOpenAI):
                    translation = " | ".join(word.translated for word in answer.words)
                    alignment = [
                        (word.original, word.translated) for word in answer.words
                    ]
                    results.append((translation, alignment))
                elif hasattr(answer, "translated"):
                    results.append((answer.translated, None))
                else:
                    logger.error(f"Unexpected result type: {type(answer)}")
                    results.append((None, None))
            return results

    def _translate_magentic_chat_function(
        self, texts: list[list[str]], prompt_config: PromptConfig
    ) -> list[tuple[str | None, list[tuple[str, str]] | None]]:
        results = []
        logger.info(f"self.model: {self.model}, config: {prompt_config}")
        with self.model:
            for text in texts:
                logger.info(
                    f"Prompt function: {prompt_config.prompt_function}; Prompt: {prompt_config.prompt_function.format(SYSTEM_PROMPT, text)}"
                )
                try:
                    answer = prompt_config.prompt_function(
                        SYSTEM_PROMPT,
                        self._format_poem(text, prompt_config.return_type),
                    )
                except Exception as e:
                    logger.error(
                        f"Error while generating prompt for {text} on model {self.model_name} with error: {e}... Giving up."
                    )
                    # TODO If we wanted to retry here we might need to provide more context to the next retry pointing out the error.
                    # It seems that models either work on first try or almost never do.
                    results.append((ValueError(e), None))
                    continue
                logger.warning(f"answer: {answer} ({type(answer)})")
                if isinstance(answer, str):
                    results.append((answer, None))
                else:
                    logger.error(f"Unexpected result type: {type(answer)}")
                    results.append((None, None))
            return results

    def _build_prompts(
        self,
    ) -> Dict[ModelCapability, Dict[TranslationType, PromptConfig]]:
        prompts: dict[ModelCapability, dict[TranslationType, PromptConfig]] = {
            capability: {} for capability in self.capabilities
        }

        for capability in self.capabilities:
            for run_type in TranslationType:
                prompt_fn = self._get_prompt_function(capability, run_type)
                if prompt_fn:
                    regex = (
                        line_regex
                        if run_type == TranslationType.LINES
                        and capability == ModelCapability.REGEX
                        else None  # Note: set dynamically in the translate method
                    )
                    return_type = self._get_return_type(run_type)
                    prompts[capability][run_type] = PromptConfig(
                        prompt_function=prompt_fn,
                        capability=capability,
                        regex=regex,
                        return_type=return_type,
                    )

        return prompts

    def _get_prompt_function(
        self, capability: ModelCapability, run_type: TranslationType
    ):
        match (capability, run_type, self.model_type):
            # CHAT
            case (ModelCapability.CHAT, TranslationType.LINES, _):
                return magentic_lines_chat_prompt_fn
            case (ModelCapability.CHAT, TranslationType.CHUNKS, _):
                return magentic_chunks_chat_prompt_fn
            case (ModelCapability.CHAT, TranslationType.WORDS, _):
                return None  # TODO This would need a lot of examples and extra prompting  # magentic_words_chat_prompt_fn

            # MAGENTIC structured generation
            case (ModelCapability.PYDANTIC, TranslationType.LINES, ModelType.MAGENTIC):
                return magentic_lines_prompt_fn
            case (ModelCapability.PYDANTIC, TranslationType.CHUNKS, ModelType.MAGENTIC):
                return magentic_chunks_prompt_fn
            case (ModelCapability.PYDANTIC, TranslationType.WORDS, ModelType.MAGENTIC):
                return magentic_words_prompt_fn

            # Outlines Pydantic + regex structured generation
            case (_, TranslationType.LINES, ModelType.HF):
                return partial(
                    outlines_lines_prompt_fn, instructions=SYSTEM_PROMPT
                )  # Shared between Pydantic and regex
            case (ModelCapability.PYDANTIC, TranslationType.CHUNKS, ModelType.HF):
                return partial(outlines_chunks_prompt_fn, instructions=SYSTEM_PROMPT)
            case (ModelCapability.PYDANTIC, TranslationType.WORDS, ModelType.HF):
                json_schema = outlines.fsm.json_schema.convert_json_schema_to_str(
                    WordsSchema
                )
                return partial(
                    outlines_words_pydantic_prompt_fn,
                    instructions=SYSTEM_PROMPT,
                    json_schema=json_schema,
                )
            case (ModelCapability.REGEX, TranslationType.WORDS, ModelType.HF):
                return partial(
                    outlines_words_regex_prompt_fn, instructions=SYSTEM_PROMPT
                )
            case (ModelCapability.REGEX, TranslationType.CHUNKS, ModelType.HF):
                # TODO
                # json_schema = outlines.fsm.json_schema.convert_json_schema_to_str(
                #     ChunksSchema
                # )
                return partial(outlines_chunks_prompt_fn, instructions=SYSTEM_PROMPT)
            case _:
                logger.warning(
                    f"Unsupported capability and run type combination: {capability} / {run_type} / {self.model_type}"
                )
                return None

    def _get_return_type(self, run_type: TranslationType):
        match run_type:
            case TranslationType.LINES:
                return LinesSchema
            case TranslationType.CHUNKS:
                return ChunksSchema
            case TranslationType.WORDS:
                return WordsSchema

    def _get_magentic_model(self) -> magentic.chat_model.base.ChatModel:
        return get_magentic_model(self.model_name, self.temperature)

    def _get_hf_model(self):
        # Do not load unless needed
        import torch
        import transformers

        logger.info(
            f"Loading HF model {self.model_name} ({self.dtype}) (Transformers {transformers.__version__} / PyTorch {torch.__version__})."
        )
        config, unused_config = transformers.AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            return_unused_kwargs=True,
        )
        if unused_config:
            logger.warning(f"Unused Transformers config keys: {unused_config}")

        if self.dtype == "int8" or self.dtype == "int4":
            try:
                from transformers import BitsAndBytesConfig

                logger.info(f"Using BitsAndBytesConfig for {self.dtype} quantization.")
                config.init_device = "meta"
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=(self.dtype == "int8"),
                    load_in_4bit=(self.dtype == "int4"),
                )
                kwargs = {
                    "config": config,
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                    "device_map": {"": 0},
                    "quantization_config": quantization_config,
                    "attn_implementation": "flash_attention_2",
                }
                return outlines.models.transformers(
                    model_name=self.model_name,
                    device="cuda",
                    model_kwargs=kwargs,
                )
            except ImportError as e:
                logger.info(f"BitsAndBytesConfig not available, using bfloat16: {e}")

        model = outlines.models.transformers(
            self.model_name,
            device="cuda",
            model_kwargs={
                "config": config,
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
            },
        )
        logger.debug(f"Model: {model} with config {config}")
        return model

    def _get_gguf_model(self):
        from llama_cpp import Llama

        logger.info(f"Loading GGUF model from {self.model_name}.")
        llm = Llama(self.model_name)
        model = outlines.models.LlamaCpp(llm)
        return model


def get_magentic_model(
    model_name: str, temperature: float
) -> magentic.chat_model.base.ChatModel:
    if "plamo" in model_name:
        # from magentic.chat_model.litellm_chat_model import LitellmChatModel
        # import litellm

        # litellm.set_verbose = True

        model = magentic.OpenaiChatModel(  # LitellmChatModel(
            model="plamo-beta",
            api_type="openai",
            api_key=os.environ["PLAMO_API_KEY"],
            base_url="https://platform.preferredai.jp/api/completion/v1",
            temperature=temperature,
        )
        return model
    elif "mistral" in model_name or "mixtral" in model_name:
        from magentic.chat_model.mistral_chat_model import MistralChatModel

        model = MistralChatModel(
            model=model_name,
            api_key=os.environ["MISTRAL_API_KEY"],
            temperature=temperature,
        )
        return model
    elif "claude" in model_name:
        from magentic.chat_model.anthropic_chat_model import AnthropicChatModel

        model = AnthropicChatModel(
            model=model_name,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=temperature,
        )
        return model
    elif "4o" in model_name:
        model = magentic.OpenaiChatModel(
            model=model_name,
            api_type="azure",
            api_key=os.environ["AZURE_API_KEY"],
            base_url=os.environ["AZURE_API_BASE"],
            temperature=temperature,
        )
        return model


def build_run_configuration(
    models: list[str],
    run_types: list[TranslationType],
    preferred_generation_methods: list[ModelCapability],
    temperature: float = 0.0,
    dtype: str = "bfloat16",
) -> Dict[str, ModelConfig]:
    run_configuration = {}
    for model_name in models:
        model_config = ModelConfig(
            model_name=model_name, temperature=temperature, dtype=dtype
        )
        logger.info(
            f"Model configuration: {model_config}, preferred: {preferred_generation_methods}"
        )

        filtered_prompts = {}
        for capability, run_type_prompts in model_config.prompts.items():
            logger.info(
                f"Capability: {capability}, preferred: {preferred_generation_methods}"
            )
            if capability not in preferred_generation_methods:
                logger.error(f"Filtering out capability: {capability}")
                continue
            filtered_prompts[capability] = {
                run_type: prompt
                for run_type, prompt in run_type_prompts.items()
                if run_type in run_types
            }

        if filtered_prompts:
            model_config.prompts = filtered_prompts
            run_configuration[model_name] = model_config
        else:
            logger.warning(
                f"No valid prompts for model {model_name} with given constraints"
            )

    return run_configuration

    # Classes:
    # magentic.OpenaiChatModel
    # magentic.chat_model.mistral_chat_model.MistralChatModel
    # magentic.chat_model.litellm_chat_model.LitellmChatModel
    # magentic.chat_model.anthropic_chat_model.AnthropicChatModel
    # outlines.models.LlamaCpp
    # outlines.models.Transformers
