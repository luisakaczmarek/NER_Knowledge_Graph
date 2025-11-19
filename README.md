# **Part 1 — Named Entity Recognition (NER) Knowledge Graph**

This repository contains a Jupyter notebook for experimenting with Named Entity Recognition (NER). 
It walks through data loading, preprocessing, model training/evaluation, and visualization steps (see the notebook for details).

## Contents

- `Part_1_NER.ipynb` — the main tutorial/workbook

- `requirements.txt` — minimal list of Python packages inferred from the notebook imports (unpinned to reduce conflicts)

## Quick Start

### 1) Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Run Jupyter

```bash
python -m pip install jupyter
jupyter notebook  # or: jupyter lab
```

## Fixing the `externally-managed-environment` error (macOS/Homebrew)

macOS with Homebrew ships Python as an **externally managed environment** (see [PEP 668](https://peps.python.org/pep-0668/)). 
If you see the error, do **one** of the following:

1. **Use a virtualenv** (preferred):

   ```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
   ```

2. **Use Conda (alternative)**:

   ```bash
conda create -n ner python=3.11 -y
conda activate ner
pip install -r requirements.txt
   ```

3. **Use pipx for CLI apps** (not typical for libraries used in notebooks):

   ```bash
brew install pipx
pipx install some-cli
   ```

4. **(Not recommended)** Override with `--break-system-packages`:

   ```bash
pip install --break-system-packages -r requirements.txt
   ```

## Notes & Recommendations

- Avoid `!pip install ...` inside notebooks on macOS/Homebrew because it triggers PEP 668 issues. Prefer installing into a virtualenv **before** running the notebook.

- If using spaCy models (e.g., `en_core_web_sm`), install them explicitly:

  ```bash
python -m spacy download en_core_web_sm
  ```

- If you use Hugging Face models, first `pip install transformers datasets accelerate evaluate` and ensure you have a compatible PyTorch build for your platform.

- To export results reproducibly, consider pinning versions in `requirements.txt` (e.g., `pandas==2.2.*`). Start unpinned while prototyping, then pin once stable.


