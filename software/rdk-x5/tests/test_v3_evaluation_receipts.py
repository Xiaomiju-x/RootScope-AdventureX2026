from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_generated_v3_pc_evaluations_match_frozen_schemas(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    adventurex = root.parent
    validation_python = adventurex / ".ai_curation_venv" / "Scripts" / "python.exe"
    assert validation_python.is_file()
    result = subprocess.run(
        [
            str(validation_python),
            str(root / "tools" / "generate_v3_pc_evaluations.py"),
            "--output-root",
            str(tmp_path),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    v3 = adventurex / "rootscope_v3"
    pairs = (
        (
            "resource_evaluation.schema.json",
            "resource_pc_contract_20260724.json",
        ),
        (
            "physical_loop_evaluation.schema.json",
            "physical_loop_pc_simulation_20260724.json",
        ),
    )
    for schema_name, receipt_name in pairs:
        schema_path = v3 / "schemas" / "evaluation" / schema_name
        receipt_path = tmp_path / receipt_name
        validation = subprocess.run(
            [
                str(validation_python),
                "-c",
                (
                    "import json,jsonschema,sys;"
                    "s=json.load(open(sys.argv[1],encoding='utf-8'));"
                    "r=json.load(open(sys.argv[2],encoding='utf-8'));"
                    "jsonschema.Draft202012Validator(s).validate(r)"
                ),
                str(schema_path),
                str(receipt_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert validation.returncode == 0, validation.stderr
    physical = json.loads(
        (
            tmp_path / "physical_loop_pc_simulation_20260724.json"
        ).read_text(encoding="utf-8")
    )
    assert physical["execution_mode"] == "SIMULATION"
    assert physical["actuation"]["serial_write_count"] == 0
    assert physical["actuation"]["pump_energized"] is False
