import json, random, difflib, re
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st

# ------------------ NLP resources ------------------
@st.cache_resource(show_spinner=False)
def _nlp_resources():
    try: nltk.data.find("corpora/stopwords")
    except LookupError: nltk.download("stopwords")
    try: nltk.data.find("corpora/wordnet")
    except LookupError: nltk.download("wordnet")
    return set(stopwords.words("english")), WordNetLemmatizer()

def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-zA-Z0-9\s]", " ", (t or "")).lower().split())

def _tokens(t: str, lenient: bool = True) -> List[str]:
    toks = _norm(t).split()
    if not lenient: return toks
    stops, lem = _nlp_resources()
    return [lem.lemmatize(w) for w in toks if w not in stops]

# ------------------ Metrics ------------------
def token_f1(s: str, t: str) -> float:
    S, T = set(_tokens(s)), set(_tokens(t))
    if not S or not T: return 0.0
    tp = len(S & T); prec = tp/len(S); rec = tp/len(T)
    return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

def rouge_l_f1(s: str, t: str) -> float:
    s_tok, t_tok = _tokens(s), _tokens(t)
    if not s_tok or not t_tok: return 0.0
    sm = difflib.SequenceMatcher(None, t_tok, s_tok)
    lcs = sum(m.size for m in sm.get_matching_blocks())
    prec = lcs/len(s_tok); rec = lcs/len(t_tok)
    return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

def tfidf_cosine(s: str, t: str) -> float:
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform([" ".join(_tokens(t)), " ".join(_tokens(s))])
    return float(cosine_similarity(X[0], X[1])[0,0])

def composite_score(s: str, t: str) -> Tuple[float, Dict[str,float]]:
    f1 = token_f1(s, t)
    rl = rouge_l_f1(s, t)
    tf = tfidf_cosine(s, t)
    score = 100 * (0.34*f1 + 0.33*rl + 0.33*tf)
    return round(score,1), {"token_f1": f1, "rougeL_f1": rl, "tfidf_cosine": tf}

def diff_explanation(s: str, t: str) -> str:
    S = set(_tokens(s)); T = _tokens(t)
    missing, seen = [], set()
    for w in T:
        if w not in S and w not in seen:
            missing.append(w); seen.add(w)
    missing = missing[:6]
    sm = difflib.SequenceMatcher(None, t, s)
    edits = [op for op in sm.get_opcodes() if op[0]!="equal"]
    msg = ""
    if missing: msg += "Missing key terms: " + ", ".join(missing) + ". "
    if edits: msg += "Some phrasing differs. "
    if not msg: msg = "Your answer closely matches the reference. "
    return msg + "Include precise keywords and be concise."

# ------------------ Dataset ------------------
def load_qa():
    p = Path("Q&A_db_practice.json")
    if p.exists():
        return json.loads(p.read_text()), "Loaded local Q&A_db_practice.json"
    st.sidebar.warning("Q&A_db_practice.json not found. Upload it here.")
    up = st.sidebar.file_uploader("Upload Q&A_db_practice.json", type=["json"])
    if up: return json.loads(up.getvalue().decode("utf-8")), "Loaded from upload"
    st.stop()

# ------------------ UI / State ------------------
st.set_page_config(page_title="Assignment 11 – Part 2", layout="wide")
st.title("Assignment 11 – Part 2: LLM Grader Chatbot")

qa_items, load_msg = load_qa()
st.caption(load_msg)

randomize = st.sidebar.checkbox("Randomize next question", value=True)

if "current_idx" not in st.session_state:
    st.session_state.current_idx = random.randrange(len(qa_items)) if randomize else 0
if "history" not in st.session_state:
    st.session_state.history = []

qa = qa_items[st.session_state.current_idx]

st.subheader("Question")
st.write(qa["question"])

ans = st.text_area("Your answer", height=160, placeholder="Type your response…")

col1, col2 = st.columns([1,1])
with col1:  do_eval = st.button("Evaluate")
with col2:  next_q  = st.button("Next question")

# ---- Evaluate (score + base logging; feedback captured in a second step) ----
if do_eval:
    if not ans.strip():
        st.warning("Please write an answer first.")
    else:
        score, parts = composite_score(ans, qa["answer"])
        st.metric("Score (0–100)", f"{score:.1f}")

        st.subheader("Feedback")
        st.write(diff_explanation(ans, qa["answer"]))

        with st.expander("Reference answer"):
            st.write(qa["answer"])
        with st.expander("Metric details"):
            st.json({k: round(v, 3) for k, v in parts.items()})

        base_row = {
            "question": qa["question"],
            "student_answer": ans,
            "reference_answer": qa["answer"],
            "score": float(score),
            **{f"metric_{k}": float(v) for k, v in parts.items()},
            "feedback_tags": "",
            "comment": "",
        }
        st.session_state.history.append(base_row)
        st.session_state.last_eval_idx = len(st.session_state.history) - 1
        st.info("Add your evaluation of the assessment below and click **Save feedback**.")

# ---- Feedback form updates the last evaluated row ----
if "last_eval_idx" in st.session_state:
    i = st.session_state.last_eval_idx
    if 0 <= i < len(st.session_state.history):
        st.markdown("---")
        st.subheader("Your evaluation of the assessment")
        with st.form(key=f"fb_form_{i}"):
            tags = st.multiselect(
                "How was this assessment?",
                ["useful", "rigorous", "clear", "relevant", "instructive", "unrelated"],
                default=[]
            )
            comment = st.text_input("Optional comment about the assessment")
            submit_fb = st.form_submit_button("Save feedback")
        if submit_fb:
            st.session_state.history[i]["feedback_tags"] = ", ".join(tags) if tags else ""
            st.session_state.history[i]["comment"] = comment.strip()

            # optional sentiment on comment
            if comment.strip():
                try:
                    sia = SentimentIntensityAnalyzer()
                except LookupError:
                    nltk.download("vader_lexicon"); sia = SentimentIntensityAnalyzer()
                sent = sia.polarity_scores(comment)
                st.session_state.history[i].update({
                    "sent_pos": sent["pos"],
                    "sent_neu": sent["neu"],
                    "sent_neg": sent["neg"],
                    "sent_compound": sent["compound"],
                })
            st.success("Feedback saved.")

# ---- Next question ----
if next_q:
    st.session_state.current_idx = (random.randrange(len(qa_items)) if randomize
                                   else (st.session_state.current_idx + 1) % len(qa_items))
    try: st.rerun()
    except Exception: st.experimental_rerun()

# ---- Results table + CSV ----
st.markdown("---")
st.subheader("Session results")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    desired = ["question","student_answer","reference_answer","score",
               "metric_token_f1","metric_rougeL_f1","metric_tfidf_cosine",
               "feedback_tags","comment","sent_pos","sent_neu","sent_neg","sent_compound"]
    df = df[[c for c in desired if c in df.columns]]
    st.dataframe(df, use_container_width=True)
    st.download_button("Download results (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name="part2_results.csv", mime="text/csv")
else:
    st.info("No evaluations yet. Submit an answer above.")
