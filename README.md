# qgan_subspace

This code has been working on based on the repository "qWGAN" of yiminghwang.

## License

Distributed under the MIT License. See LICENSE for more information.

## Usage

This section covers how to set up a local development environment for qgan_subspace and configure and run experiments.

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

    Now you can run all next necessary commands within this environment.

3. **Running**:

    - The only file you need to edit, for changing experiments, is `src/config.py`.
    - Then, after the configuration has been set, execute `src/main.py`:

        ```bash
        .venv/bin/python src/main.py
        ```

    - If you want to replot some `generated_data`, after editing data (remove runs, etc..), edit and execute `src/replot.py`:

        ```bash
        .venv/bin/python src/replot.py
        ```
