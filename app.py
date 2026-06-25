"""
NLP Resume Screener — Streamlit UI.
"""

import streamlit as st
import matplotlib.pyplot as plt

from src.extractor import extract_pdf_text
from src.preprocessing import TextPreprocessor
from src.section_parser import parse_sections, locate_skills_in_sections
from src.ner_extractor import SkillExtractor
from src.embedding import SemanticMatcher
from src.tfidf_matcher import TFIDFMatcher
from src.ats_scorer import compute_ats_score, ats_grade


st.set_page_config(page_title="NLP Resume Screener", page_icon="📄", layout="wide")


# Cache heavy objects so they load ONCE per session, not on every interaction.
@st.cache_resource
def load_pipeline_components():
    return {
        "preprocessor": TextPreprocessor(),
        "skill_extractor": SkillExtractor(),
        "sbert_matcher": SemanticMatcher(),
        "tfidf_matcher": TFIDFMatcher(),
    }


def screen_resume(job_description: str, resume_text: str, components: dict) -> dict:
    pre = components["preprocessor"]
    skill_extractor = components["skill_extractor"]

    sections = parse_sections(resume_text)
    detected_sections = list(sections.keys())

    jd_skills = skill_extractor.extract_skills(job_description)
    resume_skills = skill_extractor.extract_skills(resume_text)

    skill_locations_raw = locate_skills_in_sections(resume_skills, sections)
    skill_locations = {
        skill: (locs[0] if locs else "")
        for skill, locs in skill_locations_raw.items()
    }

    sbert_score = components["sbert_matcher"].similarity(
        pre.clean_only(job_description), pre.clean_only(resume_text)
    )
    tfidf_score = components["tfidf_matcher"].similarity(
        pre.full(job_description), pre.full(resume_text)
    )

    return compute_ats_score(
        jd_skills=jd_skills,
        resume_skills=resume_skills,
        sbert_score=sbert_score,
        tfidf_score=tfidf_score,
        skill_locations=skill_locations,
        detected_sections=detected_sections,
    )


# ── UI ────────────────────────────────────────────────────────────────────

st.title("📄 NLP Resume Screener")
st.caption("AI-powered resume screening using SBERT semantic matching + ATS scoring")

components = load_pipeline_components()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the job description here",
        height=250,
        placeholder="e.g. We are looking for a Backend Engineer with experience in Python, Django, AWS...",
    )

with col2:
    st.subheader("Resume")
    uploaded_file = st.file_uploader("Upload a resume (PDF)", type=["pdf"])

analyze_clicked = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

if analyze_clicked:
    if not job_description.strip():
        st.warning("Please paste a job description.")
    elif not uploaded_file:
        st.warning("Please upload a resume PDF.")
    else:
        with st.spinner("Analyzing resume..."):
            resume_text = extract_pdf_text(uploaded_file)
            result = screen_resume(job_description, resume_text, components)
            grade, label, color = ats_grade(result["total"])

        st.divider()

        # ── Score header ──
        score_col, grade_col = st.columns([2, 1])
        with score_col:
            st.metric("ATS Score", f"{result['total']}/100")
            st.progress(result["total"] / 100)
        with grade_col:
            st.markdown(f"### Grade: **{grade}**")
            st.markdown(f"*{label}*")

        st.divider()

        # ── Factor breakdown ──
        st.subheader("Score Breakdown")
        b1, b2, b3, b4, b5 = st.columns(5)
        b1.metric("Skill Match", f"{result['skill_match']}/40")
        b2.metric("Semantic", f"{result['semantic_relevance']}/25")
        b3.metric("Experience", f"{result['experience_relevance']}/20")
        b4.metric("Structure", f"{result['resume_structure']}/10")
        b5.metric("Keywords", f"{result['keyword_coverage']}/5")

        st.divider()

        # ── Matched / Missing skills: visual chart + text ──
        breakdown = result["breakdown"]
        matched = breakdown["matched_skills"]
        missing = breakdown["missing_skills"]

        st.subheader("📊 Skill Gap Analysis")

        if matched or missing:
            skill_labels = matched + missing
            skill_colors = ["#2ecc71"] * len(matched) + ["#e74c3c"] * len(missing)
            skill_values = [1] * len(skill_labels)  # uniform bar length; color is the signal

            fig, ax = plt.subplots(figsize=(8, max(2, len(skill_labels) * 0.4)))
            bars = ax.barh(skill_labels, skill_values, color=skill_colors)
            ax.set_xlim(0, 1.3)
            ax.set_xticks([])
            ax.invert_yaxis()  # first skill at top
            ax.spines[["top", "right", "bottom"]].set_visible(False)

            for bar, label in zip(bars, ["Matched"] * len(matched) + ["Missing"] * len(missing)):
                ax.text(
                    bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=9, color="#555"
                )

            st.pyplot(fig, use_container_width=True)
        else:
            st.write("No skills detected to compare.")

        skill_col1, skill_col2 = st.columns(2)
        with skill_col1:
            st.markdown("**✅ Matched Skills**")
            st.write(", ".join(matched) if matched else "None")
        with skill_col2:
            st.markdown("**❌ Missing Skills**")
            st.write(", ".join(missing) if missing else "None — great match!")

        st.divider()

        # ── Resume structure ──
        st.subheader("📋 Resume Structure")
        st.write(f"**Sections found:** {', '.join(breakdown['sections_found']) or 'None'}")
        if breakdown["sections_missing"]:
            st.write(f"**Sections missing:** {', '.join(breakdown['sections_missing'])}")