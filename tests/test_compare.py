from pathlib import Path

import pytest

from sempalab.compare import compare, load_reference, lookup, to_markdown


SUMMARIES = {
    "exp": [
        {"algorithm": "sempa_look", "snr_db": 6, "M": 2, "mean_ats": 0.60, "mean_wts_L1": 0.8, "mean_logical_calls": 190.0},
        {"algorithm": "random_pa", "snr_db": 6, "M": 2, "mean_ats": 0.50, "mean_wts_L1": 0.7, "mean_logical_calls": 0.0},
        {"algorithm": "sempa_look", "snr_db": 7, "M": 2, "mean_ats": 0.80, "mean_wts_L1": 0.8, "mean_logical_calls": 190.0},
    ]
}


def test_lookup_filters_rows():
    spec = {"experiment": "exp", "algorithm": "sempa_look", "metric": "ats", "where": {"snr_db": 7}}
    assert lookup(spec, SUMMARIES) == 0.80

    assert lookup({**spec, "experiment": "missing"}, SUMMARIES) is None
    assert lookup({**spec, "where": {"snr_db": 9}}, SUMMARIES) is None


def test_lookup_rejects_ambiguous_spec():
    with pytest.raises(ValueError):
        lookup({"experiment": "exp", "algorithm": "sempa_look", "metric": "ats"}, SUMMARIES)


def test_compare_gain_value_and_missing():
    entries = [
        {
            "id": "gain", "figure": "F", "quantity": "q", "setting": "s", "paper": 14.0, "unit": "%", "source": "text",
            "ours": {"gain": {
                "a": {"experiment": "exp", "algorithm": "sempa_look", "metric": "ats", "where": {"snr_db": 6}},
                "b": {"experiment": "exp", "algorithm": "random_pa", "metric": "ats", "where": {"snr_db": 6}},
            }},
        },
        {
            "id": "calls", "figure": "T", "quantity": "q", "setting": "s", "paper": 250, "source": "text",
            "ours": {"value": {"experiment": "exp", "algorithm": "sempa_look", "metric": "logical_calls", "where": {"snr_db": 6}}},
        },
        {
            "id": "missing", "figure": "T", "quantity": "q", "setting": "s", "paper": 0.5, "source": "plot",
            "ours": {"value": {"experiment": "nope", "algorithm": "sempa_look", "metric": "ats"}},
        },
    ]

    rows = {row["id"]: row for row in compare(entries, SUMMARIES)}

    assert rows["gain"]["ours"] == pytest.approx(20.0)
    assert rows["gain"]["diff"] == pytest.approx(6.0)  # percentage points
    assert rows["calls"]["diff"] == pytest.approx((190 - 250) / 250 * 100)
    assert rows["missing"]["ours"] is None

    table = to_markdown(list(rows.values()))
    assert "+20.00%" in table and "not run" in table and "read off plot" in table


def test_reference_file_is_well_formed():
    path = Path(__file__).resolve().parents[1] / "results" / "stage4" / "paper_reference.yaml"
    entries = load_reference(path)

    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids))

    for entry in entries:
        assert entry["source"] in {"text", "plot"}
        assert ("value" in entry["ours"]) != ("gain" in entry["ours"])

        # An unquoted comma inside a YAML flow mapping silently splits a
        # field into a stray null key and truncates the text.
        assert set(entry) <= {"id", "figure", "quantity", "setting", "paper", "source", "ours", "note", "unit"}
        assert all(value is not None for value in entry.values())
