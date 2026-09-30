"""End-to-end test of the NMIMS export tool in partial mode (never touches the course repository)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TITLE = ("Can an autonomous last-mile ground AED delivery vehicle simulated in MuJoCo reduce time-to-first-shock "
         "below urban ambulance congestion delays (15-20 minutes), given that sudden cardiac arrest survival drops "
         "7-10% for every minute without defibrillation?")


@pytest.mark.slow
def test_partial_export_is_complete_and_compliant(tmp_path):
    out = tmp_path / "nmims"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_nmims.py"), "--out", str(out),
                           "--allow-partial", "--accept-stale", "--skip-runtime-checks"],
                          capture_output=True, text=True, cwd=ROOT, timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "EXPORT OK" in proc.stdout
    group = out / "Group_03_Kashish_Vaishnavi"
    for rel in ("README.md", "RESEARCH_AND_IMPLEMENTATION_GUIDE.md", "docs/TEAM_ROSTER.json",
                "docs/LITERATURE_REVIEW_AND_FOUNDATIONAL_PAPERS.md", "docs/RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md",
                "models/aed_delivery_amr.xml", "src/test_env.py", "src/aed_navigation_controller.py",
                "analytics/aed_delivery_benchmark.csv", "docs/figures/figure1_system_architecture.png"):
        assert (group / rel).exists(), rel
    assert TITLE in (group / "README.md").read_text(encoding="utf-8")
    roster = json.loads((group / "docs" / "TEAM_ROSTER.json").read_text(encoding="utf-8"))
    assert len(roster["foundational_papers"]) == 6
    assert [m["roll_no"] for m in roster["members"]] == ["E026", "E046"]
    assert roster["project_showcase"]["linkedin_url"] == "PENDING_SUBMISSION"
