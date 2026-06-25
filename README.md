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

## UI Features

- **Multi-resume upload** — analyze one resume (job seeker mode) or several at once (recruiter mode)
- **Candidate leaderboard** — resumes ranked by final match score when multiple are uploaded
- **Skill-gap bar chart** — visual matched vs. missing skills (custom enhancement on top of the original design)
- **Per-resume improvement suggestions** — specific, actionable tips for each missing skill
- **Downloadable ATS report** — full breakdown exportable as a `.txt` file per resume
- **Live benchmark widget** — sidebar shows real SBERT vs TF-IDF precision stats pulled from the evaluation harness

## Evaluation: SBERT vs TF-IDF

Benchmarked on a hand-labeled dataset of job descriptions + ranked resumes:

| Metric | SBERT | TF-IDF |
|---|---|---|
| Precision@1 | 100% | 66.7% |
| MRR | 1.000 | 0.833 |
| NDCG@3 | 1.000 | 0.879 |

SBERT's top-ranked resume matched the human-labeled best fit in every test case; TF-IDF missed once, because keyword overlap alone can't distinguish "ML Engineer with PyTorch" from "Software Engineer who happens to mention Python."

## Tech stack

Python · spaCy · Sentence-Transformers (SBERT) · scikit-learn · Streamlit · pdfplumber · rapidfuzz · matplotlib

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

## Project structure

\`\`\`
src/
  extractor.py        # PDF text extraction
  preprocessing.py     # Dual-mode text cleaning (SBERT vs TF-IDF)
  section_parser.py     # Resume section detection
  ner_extractor.py      # Skill extraction (EntityRuler + fuzzy matching)
  skill_patterns.json    # Skill taxonomy
  embedding.py          # SBERT semantic matcher
  tfidf_matcher.py        # TF-IDF baseline matcher
  ats_scorer.py           # 5-factor ATS scoring engine
  ranking_engine.py       # Multi-resume ranking
evaluation/
  labeled_dataset.py      # Hand-labeled JD/resume pairs
  evaluate.py             # SBERT vs TF-IDF benchmark (P@1, MRR, NDCG@3)
main.py                   # CLI entry point
app.py                    # Streamlit web UI
\`\`\`