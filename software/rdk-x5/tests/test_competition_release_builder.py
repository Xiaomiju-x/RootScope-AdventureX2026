from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "tools/release/build_rootscope_competition_runtime_v2.py"


def test_release_packages_live_launcher_used_by_runtime() -> None:
    text = BUILDER.read_text(encoding="utf-8")
    assert '"rootscope/tools/start_x5_competition_runtime_v2.sh"' in text
    assert '"rootscope/tools/start_x5_competition_live_vision_v2.sh"' in text
    assert '"rootscope/tools/x5_competition_live_vision_v2.py"' in text
    assert '"rootscope/configs/omega/field_knowledge.v1.md"' in text
    assert '"rootscope/evidence/physical_laptop_batch_20260723T131242Z/output/summary.json"' in text
    assert '"rootscope/tools/x5_competition_live_vision.py"' in text
    assert '"rootscope/OMEGA_V3_X5_FINAL_HANDOFF_20260723.md"' in text
    assert '"tools/release/build_rootscope_competition_runtime_v2.py"' in text
