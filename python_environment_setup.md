## Python Environment Setup

**Environment setup is performed using the [uv package manager](https://docs.astral.sh/uv/) which will need to be installed to run the tutorial code.**

The uv package manager allows for the installation of specific package versions to maintain reproducibility. The list of packages and their versions is included in the `pyproject.toml` file.

**This code has been tested on Windows 11 and Ubuntu 24.04.2**

1. Install uv package manager by following the instructions at https://docs.astral.sh/uv/getting-started/installation. This typically involves running the following command in your terminal:

    ```
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2. Open a terminal and check that uv is installed by running 

    ```
    uv --help
    ```

3. Ensure uv is up-to-date by running

    ```
    uv self update
    ```

4. Clone GitHub repository

    ```
    git clone https://github.com/Brent-Murray/DeepLearningEFI
    cd DeepLearningEFI
    ```

5. Intialize the uv environment and install dependencies.

    ```
    uv sync
    ```
