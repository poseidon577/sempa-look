"""
Consistency checks between the README, docs, configs and the repository.

No model or dataset is loaded, so these run in well under a second
(`make check-docs`).
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()

LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def section(text, heading):
    """Body of the top-level `# heading` section, up to the next `# `."""
    start = text.index(f"\n# {heading}\n")
    end = text.find("\n# ", start + 1)
    return text[start : end if end != -1 else None]


def local_links(path):
    text = path.read_text()
    for target in LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        yield target.split("#", 1)[0]


@pytest.mark.parametrize("doc", ["README.md", *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs").glob("*.md"))])
def test_local_links_resolve(doc):
    path = ROOT / doc
    missing = [t for t in local_links(path) if not (path.parent / t).exists()]
    assert not missing, f"{doc} links to missing files: {missing}"


def test_summary_at_top_covers_reproduced_new_and_gaps():
    summary = README[: README.index("\n# Research Question")]
    assert "## Summary" in summary
    for needle in [
        "all-mpnet-base-v2",
        "**Reproduced.**",
        "**New in this repository.**",
        "CASA",
        "PR #8",
        "PR #10",
        "**Not matched.**",
        "Random PA at 6 dB",
        "Fig. 3",
        "Fig. 13",
        "**Not attempted.**",
    ]:
        assert needle in summary, needle


def test_summary_numbers_match_reproduction_table():
    # Each "ours" value quoted in the summary must appear in the generated table.
    table = (ROOT / "docs" / "reproduction_table.md").read_text()
    for value in ["0.6316", "0.8778", "0.9705", "0.8658", "0.8766", "0.8828", "0.8871"]:
        assert value in table, value


def test_current_encoder_is_mpnet():
    body = section(README, "Current Encoder")
    first_block = re.search(r"```text\n(.*?)```", body, re.S).group(1)
    assert "all-mpnet-base-v2" in first_block
    assert "MiniLM" not in first_block


def test_every_reported_config_and_result_uses_mpnet():
    paths = [
        *(ROOT / "configs" / "repro").glob("*.yaml"),
        *(ROOT / "configs" / "channel").glob("*.yaml"),
        *(ROOT / "results" / "stage4").glob("*/config.yaml"),
        *(ROOT / "results" / "channel").glob("*/config.yaml"),
    ]
    assert len(paths) > 40
    wrong = [p for p in paths if yaml.safe_load(p.read_text()).get("encoder") != "mpnet"]
    assert not wrong, wrong


def test_repo_tree_has_no_planned_markers_and_lists_real_dirs():
    body = section(README, "Repository Structure")
    assert "planned" not in body.lower()
    for name in ["configs/", "results/", "docs/", "scripts/", "tests/", "Makefile"]:
        assert name in body, name
        assert (ROOT / name.rstrip("/")).exists(), name
    for sub in ["repro/", "channel/", "stage2/", "stage4/"]:
        assert sub in body, sub


def test_stage2_checklist_is_complete():
    body = section(README, "Stage 2 — Algorithmic Validation")
    checklist = re.search(r"```text\n(.*?)```", body, re.S).group(1)
    assert "[ ]" not in checklist
    assert "[x] SemPA-GBeam" in checklist and "[x] MCTS-UCT" in checklist
    for module in ["sempa_gbeam.py", "mcts_uct.py"]:
        assert (ROOT / "src" / "sempalab" / "algorithms" / module).exists()


def test_results_tree_paths_exist():
    body = section(README, "Results")
    tree = re.search(r"```text\n(results/.*?)```", body, re.S).group(1)
    for path in [
        "results/stage2/acceptance.csv",
        "results/stage2/messages.json",
        "results/stage2/summary.json",
        "results/stage4/eval_sets",
        "results/stage4/paper_reference.yaml",
        "results/stage4/paper_curves.yaml",
        "results/stage4/comparison.csv",
        "results/stage4/figures/fig3_tss_vs_rss.png",
        "results/stage4/figures/fig6_ats_wts.png",
        "results/stage4/coco_k12/summary.csv",
        "results/channel/coco_k12/results.csv.gz",
        "results/channel/coco_k12/summary_minilm.csv",
        "results/channel/coco_k12/summary_bge-small.csv",
        "results/channel/figures/coco_k12_N6_gain_vs_p.png",
    ]:
        assert path.rsplit("/", 1)[-1] in tree, path
        assert (ROOT / path).exists(), path
    assert "reproduction/" not in tree


def test_related_work_cites_required_papers():
    text = (ROOT / "docs" / "related_work.md").read_text()
    for arxiv_id in ["2506.19451", "2504.19591", "2511.16983", "2506.08980"]:
        assert arxiv_id in text, arxiv_id


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
@pytest.mark.parametrize(
    "target, expected",
    [
        ("test", ["pytest -q"]),
        ("check-docs", ["tests/test_repo_docs.py"]),
        ("figures", ["scripts/make_figures.py", "scripts/make_channel_figures.py", "scripts/compare_paper.py"]),
        ("repro", ["sempalab.repro", "configs/repro/coco_k12.yaml", "configs/repro/wikihow_p.yaml"]),
        ("channel", ["sempalab.channel_experiment", "configs/channel/coco_k12.yaml"]),
    ],
)
def test_makefile_targets(target, expected):
    out = subprocess.run(["make", "-n", target], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for needle in expected:
        assert needle in out, (target, needle)


def test_makefile_scripts_and_modules_exist():
    for script in ["make_figures.py", "make_channel_figures.py", "compare_paper.py"]:
        assert (ROOT / "scripts" / script).exists()
    for module in ["repro.py", "channel_experiment.py"]:
        assert (ROOT / "src" / "sempalab" / module).exists()


def test_readme_code_fences_are_balanced():
    assert README.count("```") % 2 == 0
