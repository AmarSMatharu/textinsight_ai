# TextInsight AI
### NLP-powered text analysis for general, clinical, and research content

TextInsight AI is a Streamlit app that lets you paste in any text and instantly extract meaningful insights — summaries, keywords, named entities, sentiment, and token analysis. It supports three analysis modes including a biomedical mode for clinical notes and research abstracts.

---

## Features

- **Sentiment Analysis** — Measures polarity and subjectivity using TextBlob
- **Summarization** — Extracts key sentences using the LexRank algorithm via Sumy
- **Keyword Extraction** — Identifies the most frequent meaningful terms with frequency chart
- **Named Entity Recognition** — Detects people, organizations, locations, and more
- **Tokenization** — Breaks text into tokens with lemmas and part-of-speech tags
- **Three Analysis Modes:**
  - **General Text** — Everyday NLP for any kind of text
  - **Clinical Notes** — Biomedical NLP optimized for clinical documentation
  - **Research Abstracts** — Biomedical NLP for scientific and academic text

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| NLP | spaCy (`en_core_web_sm`) |
| Sentiment | TextBlob |
| Summarization | Sumy (LexRank) |
| Data | Pandas |

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/AmarSMatharu/textinsight_ai.git
cd textinsight_ai
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## Usage

1. **Select a mode** from the sidebar — General Text, Clinical Notes, or Research Abstracts
2. **Paste your text** into the input box
3. **Choose an NLP task** (General Text mode) or click Analyze (Clinical/Research modes)
4. View results as tables, metrics, and charts