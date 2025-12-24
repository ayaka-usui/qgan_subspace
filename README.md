# qgan_subspace

This code has been working on based on the repository "qWGAN" of yiminghwang.

## License

Distributed under the MIT License. See LICENSE for more information.

## Usage

This section covers how to set up a local development environment for qgan_subspace and run the tests.

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ayaka-usui/qgan_subspace.git
   cd qgan_subspace
   ```

2. **Sync dependencies**:

   - We maintain a list with all the needed dependencies in `requirements.txt`.
   - To create a local environment using `venv`, and install the necessary dependencies, run:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate 
    pip install -r requirements.txt   
    ```

    *(Exact command can vary depending on your shell and OS.)*

    Now you can run all necessary commands (run, tests, etc.) within this environment.

3. **Running**:

    - The only file you need to edit, for changing experiments, is `config.py`.
    - Then, after the `config.py` has been set, execute `main.py`:

    ```bash
    .venv/bin/python src/main.py
    ```

    - If you want to replot, some `generated_data` after editing data (remove runs, etc..), edit and use `replot.py`:

    ```bash
    .venv/bin/python src/replot.py
    ```
