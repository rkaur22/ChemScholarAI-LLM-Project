"""
Computational chemistry paper filtering.

This module contains all domain-specific logic for determining whether
an arXiv paper is relevant to computational chemistry.

Stage 1:
- Keyword-based filtering
- Relevance score
- Matched keywords
"""

from typing import List

# ---------------------------------------------------------------------
# Computational Chemistry Keywords
# ---------------------------------------------------------------------

COMPCHEM_KEYWORDS = {
    # Quantum Chemistry Methods
    "dft",
    "density functional theory",
    "hartree-fock",
    "post-hartree-fock",
    "ab initio",
    "first principles",
    "electronic structure",

    # Wavefunction Methods
    "mp2",
    "mp3",
    "ccsd",
    "ccsd(t)",
    "coupled cluster",
    "configuration interaction",
    "casscf",
    "mrci",

    # Software Packages
    "gaussian",
    "orca",
    "psi4",
    "q-chem",
    "nwchem",
    "cp2k",
    "gamess",
    "molpro",

    # Molecular Simulation
    "molecular dynamics",
    "ab initio molecular dynamics",
    "monte carlo simulation",
    "rpmd",

    # Electronic Structure
    "basis set",
    "wavefunction",
    "electron correlation",
    "potential energy surface",

    # General
    "quantum chemistry",
    "computational chemistry",
}


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def matched_keywords(text: str) -> List[str]:
    text = normalize_text(text)
    matches = [
        keyword
        for keyword in COMPCHEM_KEYWORDS
        if keyword in text
    ]
    return sorted(matches)


def computational_chemistry_score(title: str,
                                  abstract: str,
                                  categories: List[str]) -> tuple[int, List[str]]:
    """
    Calculate a keyword score.

    Parameters
    title : str
    abstract : str
    categories : List[str]
    Returns     (score, matched_keywords)
    """
    text = " ".join([
        title,
        abstract,
        " ".join(categories)
    ])
    matches = matched_keywords(text)
    score = len(matches)
    return score, matches


def is_computational_chemistry(title: str,
                               abstract: str,
                               categories: List[str],
                               min_score: int = 2) -> tuple[bool, int, List[str]]:
    score, matches = computational_chemistry_score(
        title,
        abstract,
        categories,
    )

    return (
        score >= min_score,
        score,
        matches,
    )