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
  packages =
    [
      pkgs.git
      pkgs.openblas
      pkgs.cmake
      pkgs.ninja
      pkgs.zlib
      pkgs.stdenv.cc.cc
      pkgs.stdenv.cc.cc.lib
      pkgs.gcc
      pkgs.binutils

      pkgs.duckdb

      pkgs.libxml2
      pkgs.nodejs
    ]
    ++ lib.optionals cuda [
      pkgs.linuxPackages_latest.nvidia_x11
    ];

  # https://devenv.sh/scripts/
  # env.LD_LIBRARY_PATH = ".devenv/profile/lib";
  env.LD_LIBRARY_PATH = ".devenv/profile/lib:.devenv/profile/lib64${
    if cuda
    then ":${pkgs.cudatoolkit}/lib:${pkgs.cudatoolkit}/lib64:${pkgs.cudatoolkit}/host-linux-x64/Mesa"
    else ""
  }";

  enterShell = ''
    export OPENAI_API_KEY=$(cat /run/agenix/openai-api)
    export ANTHROPIC_API_KEY=$(cat /run/agenix/anthropic-key)
    export AZURE_API_KEY=$(cat /run/agenix/azure-us-west3-openai-key)
    export AZURE_API_BASE=$(cat /run/agenix/azure-us-west3-openai-base)
    export AZURE_DEPLOYMENT_NAME=$(cat /run/agenix/azure-us-west3-deployment-name)
    export AZURE_API_VERSION="2024-02-15-preview"
    export GEMINI_API_KEY=$(cat /run/agenix/gemini-vertex-key)
    export MISTRAL_API_KEY=$(cat /run/agenix/mistral-key)
    export HF_TOKEN=$(cat /run/agenix/hf-token)

    export OPENAI_API_VERSION="2024-02-15-preview"
    export MAGENTIC_OPENAI_API_KEY=$(cat /run/agenix/azure-us-west3-openai-key)
    export MAGENTIC_OPENAI_API_TYPE=azure
    # NOTE: baseurl needs openai added
    export MAGENTIC_OPENAI_BASE_URL=$(cat /run/agenix/azure-us-west3-openai-base)openai
    export MAGENTIC_OPENAI_SEED=420
    export MAGENTIC_OPENAI_TEMPERATURE=0.0
    export MAGENTIC_OPENAI_MODEL=4o-global

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
      else "uv pip install torch"
    }
    ${
      if cuda
      then "export CUDA_HOME=${pkgs.cudatoolkit}"
      else ""
    }

    python -m nltk.downloader wordnet
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
        )
        + (
          if cuda
          then "packaging\nsetuptools\nwheel\ntorch\nbitsandbytes\n"
          else ""
        );
    };
  };

  languages.javascript.enable = true;

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # https://devenv.sh/processes/
  # processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
}
