{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/basics/

  # https://devenv.sh/packages/
  packages = [ pkgs.git ];

  # https://devenv.sh/scripts/

  enterShell = ''
    export OPENAI_API_KEY=$(cat /run/agenix/openai-api)
    export ANTHROPIC_API_KEY=$(cat /run/agenix/anthropic-key)
    export AZURE_API_KEY=$(cat /run/agenix/azure-openai-key)
    export AZURE_API_BASE=$(cat /run/agenix/azure-openai-base)
    export AZURE_API_VERSION="2024-02-15-preview"
    export GEMINI_API_KEY=$(cat /run/agenix/gemini-vertex-key)
    export MISTRAL_API_KEY=$(cat /run/agenix/mistral-key)
  '';

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep "2.42.0"
  '';

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    uv.enable = true;
    venv.enable = true;
    venv.requirements = ''
      litellm
      google-generativeai # gemini

      numpy
      nltk # BLEU and CHRF eval
    '';
  };

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # https://devenv.sh/processes/
  # processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
}
