from pathlib import Path


def test_conformer_search_baseline_docs_describe_reproducible_pipeline():
    document = Path(__file__).parents[1] / "docs" / "conformer-search-baseline.md"
    text = document.read_text(encoding="utf-8")

    for phrase in (
        "GFNFF",
        "LangevinConfig",
        "energy_stratified_fps",
        "iRMSD",
        "distance-fingerprint",
        "PAM-SSW",
        "target_uphill_energy_eV",
        "g-xTB",
    ):
        assert phrase in text
