# 📄 NLP Resume Screener

An AI-powered resume screener that scores how well a resume matches a job description, using semantic search (SBERT) combined with rule-based ATS scoring.

**🔗 Live demo:** https://nlp-resume-screener-daqzybueybdkrmmjswufer.streamlit.app/

## What it does

- Extracts text from PDF resumes
- Detects resume sections (Experience, Skills, Education, Projects, etc.)
- Extracts technical skills using a custom NLP pipeline (spaCy EntityRuler + fuzzy matching)
- Compares resume vs. job description using:
  - **SBERT** (semantic/meaning-based similarity)
  - **TF-IDF** (keyword-based baseline)
- Computes a 0–100 ATS score across 5 weighted factors:
  - Skill Match (40 pts)
  - Semantic Relevance (25 pts)
  - Experience Relevance (20 pts) — rewards skills actually demonstrated in Experience/Projects, not just listed
  - Resume Structure (10 pts)
  - Keyword Coverage (5 pts)
- Visualizes matched vs. missing skills as a bar chart

## Evaluation: SBERT vs TF-IDF

Benchmarked on a hand-labeled dataset of job descriptions + ranked resumes:

| Metric | SBERT | TF-IDF |
|---|---|---|
| Precision@1 | 100% | 66.7% |
| MRR | 1.000 | 0.833 |
| NDCG@3 | 1.000 | 0.879 |

SBERT's top-ranked resume matched the human-labeled best fit in every test case; TF-IDF missed once, because keyword overlap alone can't distinguish "ML Engineer with PyTorch" from "Software Engineer who happens to mention Python."

## Tech stack

Python · spaCy · Sentence-Transformers (SBERT) · scikit-learn · Streamlit · pdfplumber · rapidfuzz

## Run locally

\`\`\`bash
git clone https://github.com/lokeshahlawat338/nlp-resume-screener.git
cd nlp-resume-screener
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
\`\`\`

## Run the evaluation benchmark

\`\`\`bash
python -m evaluation.evaluate
\`\`\`