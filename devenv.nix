{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  cuda = false;
  rocm = true;
in {
  # https://devenv.sh/basics/

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.openblas
    pkgs.cmake
    pkgs.stdenv.cc.cc
    pkgs.stdenv.cc.cc.lib
    pkgs.gcc
    pkgs.binutils
    pkgs.zlib
  ];

  # https://devenv.sh/scripts/
  env.LD_LIBRARY_PATH = ".devenv/profile/lib";

  enterShell = ''
    export OPENAI_API_KEY=$(cat /run/agenix/openai-api)
    export ANTHROPIC_API_KEY=$(cat /run/agenix/anthropic-key)
    export AZURE_API_KEY=$(cat /run/agenix/azure-openai-key)
    export AZURE_API_BASE=$(cat /run/agenix/azure-openai-base)
    export AZURE_API_VERSION="2024-02-15-preview"
    export GEMINI_API_KEY=$(cat /run/agenix/gemini-vertex-key)
    export MISTRAL_API_KEY=$(cat /run/agenix/mistral-key)
    export HF_TOKEN=$(cat /run/agenix/hf-token)

    export OPENAI_API_VERSION="2024-02-15-preview"
    export MAGENTIC_OPENAI_API_KEY=$(cat /run/agenix/azure-openai-key)
    export MAGENTIC_OPENAI_API_TYPE=azure
    export MAGENTIC_OPENAI_BASE_URL=$(cat /run/agenix/azure-openai-base)
    export MAGENTIC_OPENAI_SEED=42
    export MAGENTIC_OPENAI_TEMPERATURE=0.0

    # TODO fix llama-cpp-python build
    export CMAKE_ARGS="-DLLAMA_BUILD=OFF"
    export LLAMA_CPP_LIB=$(readlink -f ~/.local/state/home-manager/gcroots/current-home/home-path/lib/libllama.so)
    # export CMAKE_ARGS="-DLLAMA_BLAS=ON;-DLLAMA_BLAS_VENDOR=OpenBLAS${
      if cuda
      then ";-DLLAMA_CUDA=on"
      else
        (
          if rocm
          then ";-DLLAMA_HIPBLAS=on"
          else ""
        )
    }"
    ${
      if rocm
      then "uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.0"
      else ""
    }
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
    venv = {
      enable = true;
      requirements =
        (builtins.readFile ./requirements.txt)
        + (
          if rocm
          then "--extra-index-url https://download.pytorch.org/whl/rocm6.0\ntorch"
          else "torch\n"
        );
    };
  };

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # https://devenv.sh/processes/
  # processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
}
