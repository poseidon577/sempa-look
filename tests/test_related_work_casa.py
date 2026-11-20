"""
The literature check behind the CASA write-up must cite and contrast the
closest prior work (docs/related_work.md).
"""

from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "related_work.md"

REQUIRED = [
    "2504.19591",  # SemPA-GBeam
    "2506.19451",  # SemPA-Look
    "2608.08698",  # video tokens over block fading (closest)
    "2511.16983",  # random unit grouping against bursts
    "2502.12096",  # TokCom random scrambling
    "2604.12931",  # TokCode round-robin interleaving
]


def test_related_work_cites_closest_prior_art():
    text = DOC.read_text()
    missing = [arxiv_id for arxiv_id in REQUIRED if arxiv_id not in text]
    assert not missing, f"docs/related_work.md must cite {missing}"


def test_related_work_states_a_verdict_and_claim():
    text = DOC.read_text()
    assert "## Verdict for CASA" in text
    assert "joint loss law" in text
