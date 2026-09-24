"""Check that every project file is present (useful after uploading or downloading via the GitHub website,
which can silently drop hidden or empty files). Standard library only.

    python3 verify_install.py
"""
import sys
from pathlib import Path

EXPECTED = [
    ".dockerignore",
    ".env.example",
    ".github/workflows/tests.yml",
    ".gitignore",
    ".streamlit/config.toml",
    "Dockerfile",
    "Makefile",
    "README.md",
    "app/streamlit_app.py",
    "cockpit/__init__.py",
    "cockpit/agents/__init__.py",
    "cockpit/agents/adk_runtime.py",
    "cockpit/agents/assessment_agent.py",
    "cockpit/agents/copilot.py",
    "cockpit/agents/grounding.py",
    "cockpit/agents/prompts.py",
    "cockpit/agents/tools.py",
    "cockpit/batch.py",
    "cockpit/check_provider.py",
    "cockpit/config.py",
    "cockpit/data_loader.py",
    "cockpit/db.py",
    "cockpit/evaluation_suite.py",
    "cockpit/knowledge/billing_pattern.md",
    "cockpit/knowledge/data_quality.md",
    "cockpit/knowledge/duplicate_billing.md",
    "cockpit/knowledge/escalation.md",
    "cockpit/knowledge/false_positives.md",
    "cockpit/knowledge/service_overlap.md",
    "cockpit/knowledge/shared_contact.md",
    "cockpit/knowledge/signal_dictionary.md",
    "cockpit/knowledge/utilization.md",
    "cockpit/knowledge_base.py",
    "cockpit/llm/__init__.py",
    "cockpit/llm/providers.py",
    "cockpit/mcp_server.py",
    "cockpit/models/__init__.py",
    "cockpit/models/anomaly_model.py",
    "cockpit/models/base.py",
    "cockpit/models/rules_model.py",
    "cockpit/models/supervised_model.py",
    "cockpit/models/synthetic.py",
    "cockpit/peer.py",
    "cockpit/rules/ruleset_v1.json",
    "cockpit/schemas.py",
    "cockpit/service.py",
    "data/sample_cases_synthetic.csv",
    "docker-compose.yml",
    "docs/DEPLOY.md",
    "docs/Fraud_Investigator_Tool.pptx",
    "docs/MODEL_COMPARISON.md",
    "pyproject.toml",
    "requirements-optional.txt",
    "requirements.txt",
    "run.sh",
    "run_windows.bat",
    "tests/__init__.py",
    "tests/smoke_ui.py",
    "tests/test_app_streamlit.py",
    "tests/test_cockpit.py",
    "verify_install.py",
]


def main() -> int:
    root = Path(__file__).resolve().parent
    missing = [f for f in EXPECTED if not (root / f).is_file()]
    empty = [f for f in EXPECTED if (root / f).is_file() and (root / f).stat().st_size == 0]
    if not missing and not empty:
        print(f"All {len(EXPECTED)} project files present.")
        return 0
    if missing:
        print(f"Missing {len(missing)} of {len(EXPECTED)} files:")
        for f in missing:
            print(f"  - {f}")
        hidden = [f for f in missing if any(part.startswith(".") for part in Path(f).parts)]
        if hidden:
            print("Files starting with a dot are hidden on macOS: press Cmd+Shift+. in Finder before uploading.")
    if empty:
        print("Empty files (should have content): " + ", ".join(empty))
    print("Re-download the full project zip, or upload the missing files to the same folder paths.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
