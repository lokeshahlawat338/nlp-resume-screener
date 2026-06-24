"""
Skill extraction module.

Uses spaCy's EntityRuler for exact phrase matching against a skill
taxonomy (src/skill_patterns.json), then falls back to rapidfuzz for
fuzzy matching to catch variant spellings the EntityRuler missed
(e.g. "Postgres" should resolve to "PostgreSQL").
"""

import json
import os
import spacy
from rapidfuzz import fuzz, process

_PATTERNS_PATH = os.path.join(os.path.dirname(__file__), "skill_patterns.json")


class SkillExtractor:
    def __init__(self, fuzzy_threshold: int = 85):
        with open(_PATTERNS_PATH, "r", encoding="utf-8") as f:
            self.skill_taxonomy = json.load(f)  # canonical -> [variants]

        # Map every variant (lowercased) back to its canonical skill name,
        # so once the EntityRuler matches text, we know what to call it.
        self.variant_to_canonical = {}
        for canonical, variants in self.skill_taxonomy.items():
            for v in variants:
                self.variant_to_canonical[v.lower()] = canonical

        self.fuzzy_threshold = fuzzy_threshold
        self.all_variants = list(self.variant_to_canonical.keys())

        # Blank English pipeline — we only need tokenization + EntityRuler,
        # not the full en_core_web_sm model (faster for this specific job).
        self.nlp = spacy.blank("en")
        ruler = self.nlp.add_pipe("entity_ruler")
        patterns = [
            {"label": "SKILL", "pattern": variant}
            for variant in self.all_variants
        ]
        ruler.add_patterns(patterns)

    def extract_skills(self, text: str) -> list:
        """
        Extract a de-duplicated list of canonical skill names found in
        the given text, using exact match first, fuzzy match as fallback.
        """
        found = set()

        # Layer 1: exact EntityRuler match
        doc = self.nlp(text.lower())
        for ent in doc.ents:
            if ent.label_ == "SKILL":
                canonical = self.variant_to_canonical.get(ent.text.lower())
                if canonical:
                    found.add(canonical)

        # Layer 2: fuzzy fallback — catch near-misses the exact ruler missed
        # by checking each word/short phrase against our known variants.
        words = text.lower().split()
        for word in set(words):
            if len(word) < 3:
                continue
            match = process.extractOne(
                word, self.all_variants, scorer=fuzz.ratio,
                score_cutoff=self.fuzzy_threshold
            )
            if match:
                variant_matched = match[0]
                canonical = self.variant_to_canonical[variant_matched]
                found.add(canonical)

        return sorted(found)