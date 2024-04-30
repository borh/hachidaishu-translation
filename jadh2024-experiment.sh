#!/usr/bin/env bash

# Requirements: https://github.com/simonw/llm
# Plugins and API keys for all model types used.

input_file="jadh2024-waka-sample.txt"
output_csv="jadh2024-waka-llm-results.csv"

models=("gpt-4" "gpt-4-turbo-preview" "gemini-1.5-pro-latest" "claude-3-opus-20240229" "mistral/mistral-large-latest")

echo "model,run,result" > "$output_csv"

for MODEL in "${models[@]}"; do
    if [[ "$MODEL" == gpt-* ]]; then
        extra_args="-o seed 42"
    else
        extra_args=""
    fi

    # Run each model 10 times
    for (( i=1; i<=10; i++ )); do
        OUTPUT=$(cat "$input_file" | llm -m "$MODEL" $extra_args 'Translate the following waka poem into English. Output only translation in one line with / as delimiter between parts.')

        echo "$MODEL,$i,$OUTPUT" >> "$output_csv"
    done
done

