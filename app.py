import streamlit as st
from textblob import TextBlob
import spacy
import pandas as pd

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

st.set_page_config(page_title="TextInsight AI", layout="wide")


@st.cache_resource
def load_general_nlp():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_medical_nlp():
    try:
        return spacy.load("en_core_sci_sm")
    except Exception:
        return None


def get_nlp_model(mode):
    if mode in ["Clinical Notes", "Research Abstracts"]:
        medical_nlp = load_medical_nlp()
        if medical_nlp is not None:
            return medical_nlp
    return load_general_nlp()


def sumy_summarizer(text, sentence_count=2):
    if not text.strip():
        return ""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)


def compute_text_statistics(text):
    words = text.split()
    sentence_count = text.count(".") + text.count("!") + text.count("?")
    unique_words = len(set(word.lower() for word in words))
    avg_word_length = round(sum(len(word) for word in words) / len(words), 2) if words else 0

    return {
        "Words": len(words),
        "Sentences": sentence_count,
        "Unique Words": unique_words,
        "Avg Word Length": avg_word_length
    }


def extract_tokens(text, mode):
    nlp = get_nlp_model(mode)
    doc = nlp(text)
    rows = []
    for token in doc:
        rows.append({
            "Token": token.text,
            "Lemma": token.lemma_,
            "POS": token.pos_
        })
    return pd.DataFrame(rows)


def extract_entities(text, mode):
    nlp = get_nlp_model(mode)
    doc = nlp(text)
    rows = []
    for ent in doc.ents:
        rows.append({
            "Entity": ent.text,
            "Label": ent.label_
        })
    return pd.DataFrame(rows)


def extract_keywords(text, mode, top_n=10):
    nlp = get_nlp_model(mode)
    doc = nlp(text.lower())

    words = [
        token.lemma_ for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.is_alpha
        and len(token.text) > 2
    ]

    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(sorted_words, columns=["Keyword", "Frequency"])



def main():
    st.title("TextInsight AI")
    st.subheader("Understand your text more clearly")
    st.markdown("Paste in text to explore summaries, keywords, entities, and other useful insights.")

    mode = st.sidebar.selectbox(
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

    message = st.text_area("Enter Text", height=220)

    if not message.strip():
        st.info("Paste text to begin.")
        return

    if mode == "General Text":
        option = st.radio(
            "Choose NLP Task",
            ["Tokenization", "Named Entity Recognition", "Sentiment Analysis", "Summarization", "Keyword Extraction"]
        )

        if option == "Tokenization":
            st.subheader("Token Analysis")
            if st.button("Run Token Analysis"):
                st.dataframe(extract_tokens(message, mode), use_container_width=True)

        elif option == "Named Entity Recognition":
            st.subheader("Named Entity Recognition")
            if st.button("Extract Entities"):
                entities_df = extract_entities(message, mode)
                if entities_df.empty:
                    st.warning("No named entities found.")
                else:
                    st.dataframe(entities_df, use_container_width=True)

        elif option == "Sentiment Analysis":
            st.subheader("Sentiment Analysis")
            if st.button("Analyze Sentiment"):
                sentiment = TextBlob(message).sentiment
                col1, col2 = st.columns(2)
                col1.metric("Polarity", round(sentiment.polarity, 3))
                col2.metric("Subjectivity", round(sentiment.subjectivity, 3))

        elif option == "Summarization":
            st.subheader("Summarization")
            if st.button("Generate Summary"):
                summary = sumy_summarizer(message, sentence_count=2)
                st.success(summary if summary else "No summary could be generated.")

        elif option == "Keyword Extraction":
            st.subheader("Top Keywords")
            if st.button("Extract Keywords"):
                keywords_df = extract_keywords(message, mode, top_n=10)
                if keywords_df.empty:
                    st.warning("No keywords found.")
                else:
                    st.dataframe(keywords_df, use_container_width=True)
                    st.bar_chart(keywords_df.set_index("Keyword"))

    elif mode == "Clinical Notes":
        st.subheader("Clinical Note Analysis")
        if st.button("Analyze Clinical Note"):
            summary = sumy_summarizer(message, sentence_count=3)
            entities_df = extract_entities(message, mode)
            keywords_df = extract_keywords(message, mode, top_n=10)

            st.markdown("### Clinical Summary")
            st.success(summary if summary else "No summary could be generated.")

            st.markdown("### Medical / Clinical Entities")
            if entities_df.empty:
                st.warning("No entities found.")
            else:
                st.dataframe(entities_df, use_container_width=True)

            st.markdown("### Key Terms")
            if keywords_df.empty:
                st.warning("No keywords found.")
            else:
                st.dataframe(keywords_df, use_container_width=True)

    elif mode == "Research Abstracts":
        st.subheader("Research Abstract Analysis")
        if st.button("Analyze Abstract"):
            summary = sumy_summarizer(message, sentence_count=3)
            entities_df = extract_entities(message, mode)
            keywords_df = extract_keywords(message, mode, top_n=12)

            st.markdown("### Abstract Summary")
            st.success(summary if summary else "No summary could be generated.")

            st.markdown("### Biomedical / Research Entities")
            if entities_df.empty:
                st.warning("No entities found.")
            else:
                st.dataframe(entities_df, use_container_width=True)

            st.markdown("### Top Keywords")
            if keywords_df.empty:
                st.warning("No keywords found.")
            else:
                st.dataframe(keywords_df, use_container_width=True)



if __name__ == "__main__":
    main()