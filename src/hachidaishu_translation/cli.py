import argparse

import tomllib

from hachidaishu_translation.log import logger
from hachidaishu_translation.models import (
    ModelCapability,
    TranslationType,
    contamination_check_prompt_fn,
)

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run translation models on waka poems."
    )

    parser.add_argument(
        "--merge-databases",
        action="store_true",
        help="Merge databases from the runs directory. Default is off.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "int8", "int4"],
        default=None,
        help="Specify the dtype for model generation. Default is 'bfloat16'.",
    )

    parser.add_argument(
        "--contamination-check",
        action="store_true",
        help="Run contamination check on the models.",
    )

    parser.add_argument(
        "--config",
        required=False,
        default=None,
        help="Path to the configuration file (TOML format).",
    )

    parser.add_argument(
        "--model",
        action="append",
        required=False,
        default=[],
        help=f"Specify the model to run. Can be used multiple times to specify multiple models. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--run-types",
        nargs="+",
        required=False,
        default=["lines"],
        choices=[t.value for t in TranslationType] + ["contamination_check"],
        help="Specify the types of translations to run. Default is 'lines'.",
    )
    parser.add_argument(
        "--generation-method",
        action="append",
        choices=[c.value for c in ModelCapability],
        default=[],
        help="Generation methods to enable for translation. Can be used multiple times.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process.",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Run only the alignments on unaligned translations (default is not to run).",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Do not run the translations (default is to run).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature setting for the model.",
    )
    parser.add_argument(
        "--align-model-name",
        type=str,
        default="4o-global",
        help="Name of the model to use for alignment.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for processing poems. Default is 1 (no batching).",
    )

    parser.add_argument(
        "--strict-alignment",
        action="store_true",
        help="Only align translations that don't have an alignment for the specified model and temperature.",
    )

    parser.add_argument(
        "--database",
        type=str,
        default="translations.db",
        help="Specify the database file to use. Default is 'translations.db'.",
    )

    args = parser.parse_args()

    # If a config file is specified, read it and merge with command-line arguments
    if args.config:
        with open(args.config, "rb") as config_file:
            config = tomllib.load(config_file)

        # Merge config file values with command-line arguments
        for key, value in config.items():
            if key == "model":
                args.model = value if isinstance(value, list) else [value]
            elif key == "run_types":
                args.run_types = [TranslationType(t) for t in value]
            elif key == "generation_method":
                args.generation_method = [ModelCapability(method) for method in value]
            elif hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    # If no models are specified, use the default model
    if not args.model:
        args.model = [DEFAULT_MODEL]
    if not args.dtype:
        args.dtype = "bfloat16"

    # Convert run_types and generation_method to enums if they're not already
    if args.run_types and not isinstance(args.run_types[0], TranslationType):
        args.run_types = [TranslationType(t) for t in args.run_types]
    if args.generation_method:
        args.generation_method = [ModelCapability(m) for m in args.generation_method]
    else:
        args.generation_method = [ModelCapability.PYDANTIC, ModelCapability.REGEX]

    return args


def main():
    args = parse_args()
    # If no models are specified, use the default model
    if not args.model:
        args.model = [DEFAULT_MODEL]
    if args.run_types:
        args.run_types = [TranslationType(t) for t in args.run_types]
    if args.generation_method:
        args.generation_method = [ModelCapability(m) for m in args.generation_method]

    logger.info(args)

    import hachidaishu_translation.gen

    logger.info(f"Running with models: {args.model}")
    hachidaishu_translation.gen.main(
        args.model,
        args.run_types,
        args.num_samples,
        align_mode=args.align,
        temperature=args.temperature,
        align_model_name=args.align_model_name,
        preferred_generation_methods=args.generation_method,
        contamination_check=args.contamination_check,
        batch_size=args.batch_size,
        dtype=args.dtype,
        strict_alignment=args.strict_alignment,
        database=args.database,
        merge_databases=args.merge_databases,
    )


if __name__ == "__main__":
    main()
