"""
Resume section parser.

Detects common resume sections (Experience, Education, Skills, Projects,
etc.) so other modules can tell whether a skill appears in a strong
context (e.g. Experience) vs. a weak one (e.g. just listed in Skills).
"""

import re

# Maps a "canonical" section name to the heading variants we should
# recognize in real resumes (case-insensitive).
SECTION_PATTERNS = {
    "Summary": [r"summary", r"profile", r"objective"],
    "Skills": [r"skills", r"technical skills", r"core competencies"],
    "Experience": [r"experience", r"work experience", r"employment history"],
    "Internship": [r"internship", r"internships"],
    "Projects": [r"projects", r"personal projects", r"academic projects"],
    "Education": [r"education", r"academic background"],
    "Certifications": [r"certifications", r"certificates", r"licenses"],
    "Research": [r"research", r"publications"],
}

# Build a single regex that matches any heading line, e.g. a short
# line that is just "Experience" or "Work Experience" with nothing else.
_ALL_PATTERNS = []
for canonical, variants in SECTION_PATTERNS.items():
    for v in variants:
        _ALL_PATTERNS.append((canonical, re.compile(rf"^\s*{v}\s*$", re.IGNORECASE)))


def parse_sections(text: str) -> dict:
    """
    Split resume text into sections.

    Returns:
        dict mapping canonical section name -> section text (str).
        Any text before the first recognized heading is stored under
        "Header" (typically name/contact info).
    """
    lines = text.split("\n")
    sections = {}
    current_section = "Header"
    buffer = []

    for line in lines:
        matched_section = None
        for canonical, pattern in _ALL_PATTERNS:
            if pattern.match(line):
                matched_section = canonical
                break

        if matched_section:
            # Save what we've collected so far under the previous section
            if buffer:
                sections[current_section] = sections.get(current_section, "") + "\n".join(buffer) + "\n"
            current_section = matched_section
            buffer = []
        else:
            buffer.append(line)

    # Don't forget the last section's content
    if buffer:
        sections[current_section] = sections.get(current_section, "") + "\n".join(buffer) + "\n"

    return sections


def locate_skills_in_sections(skills: list, sections: dict) -> dict:
    """
    For each skill, find which sections it appears in.

    Args:
        skills: list of skill strings (e.g. ["Python", "Docker"])
        sections: output of parse_sections()

    Returns:
        dict mapping skill -> list of section names it was found in.
    """
    result = {}
    for skill in skills:
        found_in = []
        skill_lower = skill.lower()
        for section_name, section_text in sections.items():
            if skill_lower in section_text.lower():
                found_in.append(section_name)
        result[skill] = found_in
    return result