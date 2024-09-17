# hachidaishu-translation

`hachidaishu-translation` is a project focused on translating Japanese waka poems into English and aligning tokens between them using various LLM models.
The project leverages both local and API-based models to generate structured translations and evaluates their quality using a variety of metrics.

## Features

-   **Translation Models**: Supports multiple models for translation, including local models via Transformers and models like GPT-4 via their API.
-   **Structured Output**: Utilizes structured generation techniques to ensure translations follow the waka format.
-   **Evaluation**: Implements evaluation metrics such as chrF and METEOR to assess translation quality.
-   **Alignment**: Provides token alignment between original and translated texts for detailed analysis.
-   **visualization**: Provides a novel Word alignment visualization for evaluating multiple alignment results at once.

## Installation

**Note that currently, the golden translation dataset cannot be shared, so you will not be able to run the whole program.**

This project was developed with Python 3.12, but should probably work with versions >= 3.10. Install the required dependencies using [uv](https://github.com/astral-sh/uv/):

```bash
uv sync
```

If using CUDA:

```bash
uv vevn
# Source the venv here
uv pip install torch
pip install flash-attn --no-build-isolation
uv sync --extra cuda
```

ROCM also work, but currently requires manually installing torch:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
# Or a later version
```

# Usage

The main entrypoint to the program is `src/hachidaishu_translation/cli.py`, but it is possible to specify a run configuration file like so:

```bash
python src/hachidaishu_translation/cli.py --config minimal_run_config.toml --database database-to-save-results-to.db
```

## Run configuration

The run configuration must be in toml format and can specify any of the flags provided by the CLI interface:

```
usage: cli.py [-h] [--merge-databases] [--dtype {bfloat16,int8,int4}] [--contamination-check] [--config CONFIG] [--model MODEL] [--run-types {lines,chunks,words,contamination_check} [{lines,chunks,words,contamination_check} ...]]
              [--generation-method {regex,pydantic,chat,contamination_check}] [--num-samples NUM_SAMPLES] [--align] [--no-translate] [--temperature TEMPERATURE] [--align-model-name ALIGN_MODEL_NAME] [--batch-size BATCH_SIZE] [--strict-alignment] [--database DATABASE]

Run translation models on waka poems.

options:
  -h, --help            show this help message and exit
  --merge-databases     Merge databases from the runs directory. Default is off.
  --dtype {bfloat16,int8,int4}
                        Specify the dtype for model generation. Default is 'bfloat16'.
  --contamination-check
                        Run contamination check on the models.
  --config CONFIG       Path to the configuration file (TOML format).
  --model MODEL         Specify the model to run. Can be used multiple times to specify multiple models. Default: meta-llama/Meta-Llama-3.1-8B-Instruct
  --run-types {lines,chunks,words,contamination_check} [{lines,chunks,words,contamination_check} ...]
                        Specify the types of translations to run. Default is 'lines'.
  --generation-method {regex,pydantic,chat,contamination_check}
                        Generation methods to enable for translation. Can be used multiple times.
  --num-samples NUM_SAMPLES
                        Number of samples to process.
  --align               Run only the alignments on unaligned translations (default is not to run).
  --no-translate        Do not run the translations (default is to run).
  --temperature TEMPERATURE
                        Temperature setting for the model.
  --align-model-name ALIGN_MODEL_NAME
                        Name of the model to use for alignment.
  --batch-size BATCH_SIZE
                        Batch size for processing poems. Default is 1 (no batching).
  --strict-alignment    Only align translations that don't have an alignment for the specified model and temperature.
  --database DATABASE   Specify the database file to use. Default is 'translations.db'.
```

## Translation

To translate waka poems, use the `cli.py` script.
This script processes poems from the `hachidai.db` database (available [here](https://github.com/borh/hachidaishu)) and generates translations using the specified models.

```bash
python src/hachidaishu_translation/gen.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --batch-size 5 --temperature 1.0 --dtype int8 --num-samples 10
```

The above command would use the meta-llama/Meta-Llama-3.1-8B-Instruct model via [Huggingface Transformers](https://github.com/huggingface/transformers/), quantize it to int8 using [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), and perform translations on 10 random waka poems.

## Evaluation

Evaluate the translations using the eval.py script, which compares generated translations against gold standards.

```bash
python src/hachidaishu_translation/eval.py --help
```

## Word alignment visualization and token translations

```bash
python src/hachidaishu_translation/format.py --help
```

# News

## Major Changes

-   2024-09-17: Expanded API use to Anthropic, Plamo (beta) and Mistral. More robust structured generation and prompts, as well as extensive refactoring and improved alignment visualization.
-   2024-06-28: Enhanced alignment performance and robustness.
-   2024-06-23: Added new generation and logging improvements.
-   2024-06-03: Cleaned up code and added comments for clarity.
-   2024-05-27: Introduced a new generation backend.


# Recognition

This work was accepted as an interactive session presentation at [JADH2024](https://jadh2024.l.u-tokyo.ac.jp/), held from September 18-20, 2024.


# Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.


# License

This project is licensed under the MIT License. See the LICENSE file for details.
