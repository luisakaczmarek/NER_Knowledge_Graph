# **Named Entity Recognition (NER) & Knowledge Graph Construction**

This repository contains a Jupyter notebook that explores **Named Entity Recognition (NER)** and the subsequent **construction of a Knowledge Graph** from the extracted entities and relationships.
It walks through the full pipeline — from **data loading and preprocessing**, through **NER model training and evaluation**, to **visualizing entities and their interconnections** within a knowledge graph.
The goal is to demonstrate how unstructured text can be transformed into a **structured, interpretable network of knowledge**.

---

## **Contents**

* `Part_1_NER.ipynb` — main notebook with the complete NER & Knowledge Graph workflow
* `requirements.txt` — list of Python dependencies required to run the notebook

---

## **Quick Start**

### 1️⃣ Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2️⃣ Launch Jupyter Notebook or Lab

```bash
python -m pip install jupyter
jupyter notebook   # or: jupyter lab
```

Then open **`Part_1_NER.ipynb`** and run all cells sequentially.

---

## **Notes**

* All dependencies are now managed via `requirements.txt`.
  There are **no `!pip install` commands** inside the notebook.
* Using a virtual environment ensures compatibility with systems like macOS/Homebrew that use **externally managed Python environments** (see [PEP 668](https://peps.python.org/pep-0668/)).
* If you use additional NLP models (e.g., spaCy models), install them explicitly, e.g.:

  ```bash
  python -m spacy download en_core_web_sm
  ```

---
