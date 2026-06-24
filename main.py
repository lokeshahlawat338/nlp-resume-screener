"""
NLP Resume Screener — CLI entry point.

Usage:
    python main.py
(Edit JD_TEXT and RESUME_PATH below, or extend this with argparse later.)
"""

from src.extractor import extract_pdf_text
from src.preprocessing import TextPreprocessor
from src.section_parser import parse_sections, locate_skills_in_sections
from src.ner_extractor import SkillExtractor
from src.embedding import SemanticMatcher
from src.tfidf_matcher import TFIDFMatcher
from src.ats_scorer import compute_ats_score, ats_grade


def screen_resume(job_description: str, resume_path: str) -> dict:
    """
    Run the full screening pipeline on one resume against one JD.
    Returns the ATS score result dict.
    """
    pre = TextPreprocessor()
    skill_extractor = SkillExtractor()
    sbert_matcher = SemanticMatcher()
    tfidf_matcher = TFIDFMatcher()

    resume_text = extract_pdf_text(resume_path)

    sections = parse_sections(resume_text)
    detected_sections = list(sections.keys())

    jd_skills = skill_extractor.extract_skills(job_description)
    resume_skills = skill_extractor.extract_skills(resume_text)

    skill_locations_raw = locate_skills_in_sections(resume_skills, sections)
    skill_locations = {
        skill: (locs[0] if locs else "")
        for skill, locs in skill_locations_raw.items()
    }

    sbert_score = sbert_matcher.similarity(
        pre.clean_only(job_description), pre.clean_only(resume_text)
    )
    tfidf_score = tfidf_matcher.similarity(
        pre.full(job_description), pre.full(resume_text)
    )

    result = compute_ats_score(
        jd_skills=jd_skills,
        resume_skills=resume_skills,
        sbert_score=sbert_score,
        tfidf_score=tfidf_score,
        skill_locations=skill_locations,
        detected_sections=detected_sections,
    )
    return result


def print_report(result: dict) -> None:
    grade, label, color = ats_grade(result["total"])
    print("\n" + "=" * 50)
    print(f"  ATS SCORE: {result['total']}/100   Grade: {grade} ({label})")
    print("=" * 50)
    print(f"  Skill Match:          {result['skill_match']:>3}/40")
    print(f"  Semantic Relevance:   {result['semantic_relevance']:>3}/25")
    print(f"  Experience Relevance: {result['experience_relevance']:>3}/20")
    print(f"  Resume Structure:     {result['resume_structure']:>3}/10")
    print(f"  Keyword Coverage:     {result['keyword_coverage']:>3}/5")
    print("-" * 50)
    b = result["breakdown"]
    print(f"  Matched skills:  {', '.join(b['matched_skills']) or 'None'}")
    print(f"  Missing skills:  {', '.join(b['missing_skills']) or 'None'}")
    print(f"  Sections found:  {', '.join(b['sections_found'])}")
    print(f"  Sections missing:{', '.join(b['sections_missing']) or ' None'}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    JOB_DESCRIPTION = """
    We are looking for a Backend Engineer with strong experience in Python
    and Django, building REST APIs. Experience with AWS, Docker, and
    PostgreSQL is required. Familiarity with CI/CD pipelines is a plus.
    """
    RESUME_PATH = "data/resumes/sample_resume.pdf"

    print("Screening resume... (loading models, may take a few seconds)")
    result = screen_resume(JOB_DESCRIPTION, RESUME_PATH)
    print_report(result)