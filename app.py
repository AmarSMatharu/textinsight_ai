import logging
from collections import Counter
from collections.abc import Callable
from typing import Any

import pandas as pd
import spacy
import streamlit as st
from textblob import TextBlob

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="TextInsight AI", layout="wide")

# Hard cap to prevent DoS via large inputs blocking the main thread.
# spaCy processes text in O(n) CPU and memory; 10,000 chars covers typical
# clinical notes and research abstracts comfortably.
MAX_INPUT_CHARS = 10_000


@st.cache_resource
def load_general_nlp() -> spacy.language.Language:
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_medical_nlp() -> spacy.language.Language | None:
    try:
        return spacy.load("en_core_sci_sm")
    except OSError:
        # Model not installed — expected in environments without scispacy model.
        return None
    except Exception:
        logger.exception("Unexpected error loading medical NLP model")
        return None


def get_nlp_model(mode: str) -> spacy.language.Language:
    if mode in ["Clinical Notes", "Research Abstracts"]:
        medical_nlp = load_medical_nlp()
        if medical_nlp is not None:
            return medical_nlp
    return load_general_nlp()


def sumy_summarizer(text: str, sentence_count: int = 2) -> str:
    if not text.strip():
        return ""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, sentence_count)
        return " ".join(str(sentence) for sentence in summary)
    except Exception:
        logger.exception("Summarization failed")
        return ""


def extract_tokens(text: str, mode: str) -> pd.DataFrame:
    nlp = get_nlp_model(mode)
    doc = nlp(text)
    rows = [
        {"Token": token.text, "Lemma": token.lemma_, "POS": token.pos_}
        for token in doc
    ]
    return pd.DataFrame(rows)


def extract_entities(text: str, mode: str) -> pd.DataFrame:
    nlp = get_nlp_model(mode)
    doc = nlp(text)
    rows = [
        {"Entity": ent.text, "Label": ent.label_}
        for ent in doc.ents
    ]
    return pd.DataFrame(rows)


def extract_keywords(text: str, mode: str, top_n: int = 10) -> pd.DataFrame:
    nlp = get_nlp_model(mode)
    doc = nlp(text.lower())

    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.is_alpha
        and len(token.text) > 2
    ]

    most_common = Counter(lemmas).most_common(top_n)
    return pd.DataFrame(most_common, columns=["Keyword", "Frequency"])


def run_nlp_safely(fn: Callable[..., Any], *args: Any, error_label: str = "Analysis") -> Any | None:
    """Wrap any NLP call and surface a user-friendly Streamlit error on failure."""
    try:
        return fn(*args)
    except MemoryError:
        st.error(f"{error_label} failed: input too large for available memory.")
        logger.exception("%s hit MemoryError", error_label)
    except Exception:
        st.error(f"{error_label} failed due to an unexpected error. Check the app logs.")
        logger.exception("%s failed", error_label)
    return None


def enforce_input_limit(text: str) -> str | None:
    """
    Return the text unchanged if within limit, otherwise show an error
    and return None to halt processing.
    """
    if len(text) > MAX_INPUT_CHARS:
        st.error(
            f"Input exceeds the {MAX_INPUT_CHARS:,}-character limit "
            f"({len(text):,} characters submitted). "
            "Please shorten your text and try again."
        )
        return None
    return text


def main() -> None:
    st.title("TextInsight AI")
    st.subheader("Understand your text more clearly")
    st.markdown("Paste in text to explore summaries, keywords, entities, and other useful insights.")

    mode: str = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["General Text", "Clinical Notes", "Research Abstracts"]
    )

    st.sidebar.markdown("### Mode Guide")
    if mode == "General Text":
        st.sidebar.info("General-purpose NLP for everyday text.")
    elif mode == "Clinical Notes":
        st.sidebar.info("Uses biomedical NLP when available for clinical text.")
    elif mode == "Research Abstracts":
        st.sidebar.info("Uses biomedical NLP when available for abstracts and research text.")

    medical_model_loaded = load_medical_nlp() is not None
    if mode in ["Clinical Notes", "Research Abstracts"] and not medical_model_loaded:
        st.warning("Medical model not found. Falling back to the general NLP model.")

    message: str = st.text_area(
        "Enter Text",
        height=220,
        help=f"Maximum {MAX_INPUT_CHARS:,} characters.",
        max_chars=MAX_INPUT_CHARS,
    )

    if not message.strip():
        st.info("Paste text to begin.")
        return

    safe_message = enforce_input_limit(message)
    if safe_message is None:
        return

    if mode == "General Text":
        option: str = st.radio(
            "Choose NLP Task",
            ["Tokenization", "Named Entity Recognition", "Sentiment Analysis", "Summarization", "Keyword Extraction"]
        )

        if option == "Tokenization":
            st.subheader("Token Analysis")
            if st.button("Run Token Analysis"):
                result = run_nlp_safely(extract_tokens, safe_message, mode, error_label="Tokenization")
                if result is not None:
                    st.dataframe(result, use_container_width=True)

        elif option == "Named Entity Recognition":
            st.subheader("Named Entity Recognition")
            if st.button("Extract Entities"):
                result = run_nlp_safely(extract_entities, safe_message, mode, error_label="Entity Extraction")
                if result is not None:
                    if result.empty:
                        st.warning("No named entities found.")
                    else:
                        st.dataframe(result, use_container_width=True)

        elif option == "Sentiment Analysis":
            st.subheader("Sentiment Analysis")
            if st.button("Analyze Sentiment"):
                try:
                    sentiment = TextBlob(safe_message).sentiment
                    col1, col2 = st.columns(2)
                    col1.metric("Polarity", round(sentiment.polarity, 3))
                    col2.metric("Subjectivity", round(sentiment.subjectivity, 3))
                except Exception:
                    st.error("Sentiment analysis failed due to an unexpected error.")
                    logger.exception("Sentiment analysis failed")

        elif option == "Summarization":
            st.subheader("Summarization")
            if st.button("Generate Summary"):
                summary = run_nlp_safely(sumy_summarizer, safe_message, 2, error_label="Summarization")
                if summary is not None:
                    st.success(summary if summary else "No summary could be generated.")

        elif option == "Keyword Extraction":
            st.subheader("Top Keywords")
            if st.button("Extract Keywords"):
                result = run_nlp_safely(extract_keywords, safe_message, mode, 10, error_label="Keyword Extraction")
                if result is not None:
                    if result.empty:
                        st.warning("No keywords found.")
                    else:
                        st.dataframe(result, use_container_width=True)
                        st.bar_chart(result.set_index("Keyword"))

    elif mode == "Clinical Notes":
        st.subheader("Clinical Note Analysis")
        if st.button("Analyze Clinical Note"):
            summary = run_nlp_safely(sumy_summarizer, safe_message, 3, error_label="Clinical Summarization")
            entities_df = run_nlp_safely(extract_entities, safe_message, mode, error_label="Clinical Entity Extraction")
            keywords_df = run_nlp_safely(extract_keywords, safe_message, mode, 10, error_label="Clinical Keyword Extraction")

            st.markdown("### Clinical Summary")
            if summary is not None:
                st.success(summary if summary else "No summary could be generated.")

            st.markdown("### Medical / Clinical Entities")
            if entities_df is not None:
                if entities_df.empty:
                    st.warning("No entities found.")
                else:
                    st.dataframe(entities_df, use_container_width=True)

            st.markdown("### Key Terms")
            if keywords_df is not None:
                if keywords_df.empty:
                    st.warning("No keywords found.")
                else:
                    st.dataframe(keywords_df, use_container_width=True)

    elif mode == "Research Abstracts":
        st.subheader("Research Abstract Analysis")
        if st.button("Analyze Abstract"):
            summary = run_nlp_safely(sumy_summarizer, safe_message, 3, error_label="Abstract Summarization")
            entities_df = run_nlp_safely(extract_entities, safe_message, mode, error_label="Abstract Entity Extraction")
            keywords_df = run_nlp_safely(extract_keywords, safe_message, mode, 12, error_label="Abstract Keyword Extraction")

            st.markdown("### Abstract Summary")
            if summary is not None:
                st.success(summary if summary else "No summary could be generated.")

            st.markdown("### Biomedical / Research Entities")
            if entities_df is not None:
                if entities_df.empty:
                    st.warning("No entities found.")
                else:
                    st.dataframe(entities_df, use_container_width=True)

            st.markdown("### Top Keywords")
            if keywords_df is not None:
                if keywords_df.empty:
                    st.warning("No keywords found.")
                else:
                    st.dataframe(keywords_df, use_container_width=True)


if __name__ == "__main__":
    main()
