# **Part 1: NER_Knowledge_Graph & Q&A**
xx (to be edited)

## **Part 2: LLM Grader Chatbot**


This project is a Streamlit web app that evaluates student answers against reference answers using NLP techniques (TF-IDF, token overlap, sentiment, etc.).

Functionalities of the streamlit app:

1.  Loads questions and target (“standard”) answers from Q&A_db_practice.json.
2.  Shows one question at a time.
3.	Lets the user (student) type an answer.
4.	Scores it by comparing it to the reference answer using text similarity metrics (e.g., ROUGE-like F1, TF-IDF cosine).
5.	Explains differences using either rule-based text comparison or a small LLM model.
6.	Optionally analyzes user feedback sentiment (via VADER).
7.	Logs results and allows CSV download.

### **Approach followed and rationale**

To successfully design an automated system that can evaluate responses to open-ended questions in ML we created a streamlit-based chatbot that compares answers to a curated dataset of “standard” answers from the *Q&A_db_practice.json* dataset. Instead of relying on deep learning models or external APIs, we used deterministic text similarity metrics for transparent evaluation. Specifically, three complementary methods were combined:
1. Token F1 overlap (to capture exact keyword matching)
2. ROUGE-L F1 (to measure shared sequence structure and phrasing)
3. TF-IDF cosine similarity (to detect overall semantic similarity)

Each metric contributes to a weighted composite score ranging from 0 to 100, giving us a balanced assessment between lexical precision and contextual relevance. Also, we used a rule-based feedback function that highlights missing key terms and phrasing differences to justify why answers received a particular score. We also included sentiment analysis using the VADER lexicon, enabling qualitative feedback assessment, and stores all results for later analysis.

This approach prioritizes interpretability and reliability which is important for an academic grading assistant that must run consistently without GPU dependencies. However, it does not capture deeper semantic relationships or rephrased explanations as effectively as transformer-based models. To improve this, we could fine-tune or integrate a LLM such as FLAN-T5, BERTScore, or GPT-style evaluation frameworks to get contextual scoring and natural language feedback. Another improvement would be supervised learning on annotated student responses, where a regression or classification model learns to predict human-assigned scores. Also, methods like embedding-based similarity (e.g., Sentence-BERT) or semantic entailment detection could make scoring more reasonable by understanding meaning rather than surface-level similarity. To successfully do this we would need labeled data, more computational resources, and robust validation to ensure fairness and explainability.

---

## 🚀 Getting Started

Follow these steps to set up and run the app locally.

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

