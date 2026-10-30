from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "project_audit.json"

REQUIRED_FILES = [
    "README.md", "requirements.txt", ".gitignore", ".env.example",
    "run_ingestion.py", "validate_chunks.py", "run_api.py", "api/main.py",
    "ui/streamlit_app.py", "rag/config.py", "rag/logging_config.py",
    "rag/chains/rag_chain.py", "rag/retrieval/retriever.py",
    "rag/vectorstore/chroma_store.py", "rag/llm/ollama_client.py",
    "rag/agent/agent_controller.py", "rag/agent/memory.py",
    "rag/agent/tools/calculator_tool.py",
    "rag/agent/tools/document_stats_tool.py",
]

FEATURE_CHECKS: dict[str, dict[str, Any]] = {
    "FastAPI Agent endpoint": {
        "file": "api/main.py",
        "markers": ["/agent", "run_agent"],
    },
    "File upload endpoint": {
        "file": "api/main.py",
        "markers": ["/upload", "UploadFile"],
    },
    "Document listing endpoint": {
        "file": "api/main.py",
        "markers": ["/documents"],
    },
    "Knowledge-base rebuilding": {
        "file": "api/main.py",
        "markers": ["run_reindex_pipeline", "rag.vectorstore.chroma_store"],
    },
    "Streamlit Agent connection": {
        "file": "ui/streamlit_app.py",
        "markers": ["/agent", "session_id", "use_memory"],
    },
    "Calculator Tool": {
        "file": "rag/agent/tools/calculator_tool.py",
        "markers": ["run_calculator_tool", "ast.parse"],
    },
    "Document Statistics Tool": {
        "file": "rag/agent/tools/document_stats_tool.py",
        "markers": ["run_document_stats_tool", "build_document_statistics"],
    },
    "Agent routing": {
        "file": "rag/agent/agent_controller.py",
        "markers": ["run_agent", "tool_used", "agent_route"],
    },
    "Session memory": {
        "file": "rag/agent/memory.py",
        "markers": ["ConversationMemory", "session_id", "deque"],
    },
    "Central configuration": {
        "file": "rag/config.py",
        "markers": ["class Settings", "load_dotenv", "get_settings"],
    },
    "Application logging": {
        "file": "rag/logging_config.py",
        "markers": ["RotatingFileHandler", "get_logger"],
    },
}

EXPECTED_SCREENSHOTS = [
    "01_streamlit_home.png", "02_upload_success.png",
    "03_rag_answer_sources.png", "04_fastapi_docs.png",
    "05_integration_tests.png",
]

IGNORED_SCAN_DIRECTORIES = {
    ".git", ".venv", "venv", "__pycache__", "chroma_db", "chromadb",
    "data", "logs", "models", "embeddings",
}

SCAN_EXTENSIONS = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".txt"}

SECRET_PATTERNS = {
    "OpenAI-style API key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "Hard-coded password": re.compile(
        r'''(?ix)\bpassword\s*=\s*["'](?!your-|example|placeholder)[^"']{6,}["']'''
    ),
    "Hard-coded token": re.compile(
        r'''(?ix)\btoken\s*=\s*["'](?!your-|example|placeholder)[^"']{8,}["']'''
    ),
    "Hard-coded API key": re.compile(
        r'''(?ix)\bapi[_-]?key\s*=\s*["'](?!your-|example|placeholder)[^"']{8,}["']'''
    ),
}


def read_text_safely(path: Path) -> str:
    """Read UTF-8/UTF-16/GB18030 text safely, including Windows BOM files."""
    try:
        raw = path.read_bytes()
    except OSError:
        return ""

    if not raw:
        return ""

    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass

    if raw.startswith(b"\xef\xbb\xbf"):
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            pass

    for encoding in ("utf-8", "utf-8-sig", "utf-16", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return ""


def check_required_files() -> list[dict[str, Any]]:
    return [
        {"path": relative_path, "exists": (PROJECT_ROOT / relative_path).is_file()}
        for relative_path in REQUIRED_FILES
    ]


def check_features() -> list[dict[str, Any]]:
    results = []
    for feature_name, feature_info in FEATURE_CHECKS.items():
        relative_path = feature_info["file"]
        path = PROJECT_ROOT / relative_path
        markers = feature_info["markers"]

        if not path.is_file():
            results.append({
                "feature": feature_name,
                "file": relative_path,
                "passed": False,
                "missing_markers": markers,
                "reason": "File does not exist.",
            })
            continue

        content = read_text_safely(path)
        missing_markers = [marker for marker in markers if marker not in content]
        results.append({
            "feature": feature_name,
            "file": relative_path,
            "passed": not missing_markers,
            "missing_markers": missing_markers,
        })
    return results


def find_screenshot(filename: str) -> str | None:
    for path in PROJECT_ROOT.rglob(filename):
        if path.is_file():
            try:
                return str(path.relative_to(PROJECT_ROOT))
            except ValueError:
                return str(path)
    return None


def check_screenshots() -> list[dict[str, Any]]:
    return [
        {"filename": filename, "found_at": find_screenshot(filename)}
        for filename in EXPECTED_SCREENSHOTS
    ]


def should_scan_file(path: Path) -> bool:
    try:
        relative_parts = path.relative_to(PROJECT_ROOT).parts
    except ValueError:
        return False

    if any(part in IGNORED_SCAN_DIRECTORIES for part in relative_parts):
        return False
    if path.name == ".env":
        return False
    return path.suffix.lower() in SCAN_EXTENSIONS


def scan_possible_secrets() -> list[dict[str, Any]]:
    findings = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or not should_scan_file(path):
            continue
        content = read_text_safely(path)
        if not content:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            for pattern_name, pattern in SECRET_PATTERNS.items():
                if pattern.search(line):
                    findings.append({
                        "type": pattern_name,
                        "file": str(path.relative_to(PROJECT_ROOT)),
                        "line": line_number,
                        "preview": line.strip()[:160],
                    })
    return findings


def check_gitignore() -> dict[str, Any]:
    gitignore_path = PROJECT_ROOT / ".gitignore"
    required_rules = [".env", "chroma_db/", "data/raw/*", "__pycache__/", "*.log"]

    if not gitignore_path.is_file():
        return {"exists": False, "missing_rules": required_rules, "passed": False}

    content = read_text_safely(gitignore_path)
    lines = {
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    missing_rules = [rule for rule in required_rules if rule not in lines]
    return {"exists": True, "missing_rules": missing_rules, "passed": not missing_rules}


def normalize_requirement_name(line: str) -> str | None:
    """Normalize requirements lines such as uvicorn[standard] or fastapi>=0.1."""
    cleaned_line = line.strip()
    if not cleaned_line or cleaned_line.startswith("#"):
        return None

    cleaned_line = cleaned_line.split("#", 1)[0].strip()
    if not cleaned_line or cleaned_line.startswith("-"):
        return None

    package_name = re.split(
        r"(?:===|==|~=|>=|<=|!=|>|<|@)",
        cleaned_line,
        maxsplit=1,
    )[0].strip()

    if "[" in package_name:
        package_name = package_name.split("[", 1)[0].strip()

    package_name = package_name.lower()
    return package_name or None


def check_requirements() -> dict[str, Any]:
    requirements_path = PROJECT_ROOT / "requirements.txt"
    required_packages = {
        "chromadb", "fastapi", "pydantic", "pypdf", "python-dotenv",
        "python-multipart", "requests", "sentence-transformers",
        "streamlit", "uvicorn",
    }

    if not requirements_path.is_file():
        return {
            "exists": False,
            "installed_packages": [],
            "missing_packages": sorted(required_packages),
            "passed": False,
            "error": "requirements.txt does not exist.",
        }

    content = read_text_safely(requirements_path)
    if not content:
        return {
            "exists": True,
            "installed_packages": [],
            "missing_packages": sorted(required_packages),
            "passed": False,
            "error": "requirements.txt could not be decoded or is empty.",
        }

    installed_names: set[str] = set()
    for line in content.splitlines():
        package_name = normalize_requirement_name(line)
        if package_name:
            installed_names.add(package_name)

    missing_packages = sorted(required_packages - installed_names)
    return {
        "exists": True,
        "installed_packages": sorted(installed_names),
        "missing_packages": missing_packages,
        "passed": not missing_packages,
    }


def build_summary(audit: dict[str, Any]) -> dict[str, Any]:
    required_file_results = audit["required_files"]
    feature_results = audit["features"]
    screenshot_results = audit["screenshots"]

    files_passed = sum(result["exists"] for result in required_file_results)
    features_passed = sum(result["passed"] for result in feature_results)
    screenshots_found = sum(
        result["found_at"] is not None for result in screenshot_results
    )

    gitignore_passed = audit["gitignore"].get("passed", False)
    requirements_passed = audit["requirements"].get("passed", False)

    ready_for_manual_review = (
        files_passed == len(required_file_results)
        and features_passed == len(feature_results)
        and gitignore_passed
        and requirements_passed
    )

    return {
        "required_files": f"{files_passed}/{len(required_file_results)}",
        "feature_checks": f"{features_passed}/{len(feature_results)}",
        "screenshots": f"{screenshots_found}/{len(screenshot_results)}",
        "possible_secret_findings": len(audit["possible_secrets"]),
        "gitignore_passed": gitignore_passed,
        "requirements_passed": requirements_passed,
        "ready_for_manual_review": ready_for_manual_review,
    }


def print_diagnostics(audit: dict[str, Any]) -> None:
    gitignore = audit["gitignore"]
    requirements = audit["requirements"]

    if not gitignore.get("passed", False):
        print("\nGitignore check failed:")
        for rule in gitignore.get("missing_rules", []):
            print(f"- Missing rule: {rule}")

    if not requirements.get("passed", False):
        print("\nRequirements check failed:")
        if requirements.get("error"):
            print("- " + requirements["error"])
        installed_packages = requirements.get("installed_packages", [])
        print(
            "- Parsed packages: "
            + (", ".join(installed_packages) if installed_packages else "none")
        )
        for package in requirements.get("missing_packages", []):
            print(f"- Missing package: {package}")


def main() -> None:
    audit: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "required_files": check_required_files(),
        "features": check_features(),
        "screenshots": check_screenshots(),
        "gitignore": check_gitignore(),
        "requirements": check_requirements(),
        "possible_secrets": scan_possible_secrets(),
    }

    audit["summary"] = build_summary(audit)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = audit["summary"]
    print("=" * 60)
    print("Day 29 Project Audit")
    print("=" * 60)
    print(f"Required files: {summary['required_files']}")
    print(f"Feature checks: {summary['feature_checks']}")
    print(f"Screenshots: {summary['screenshots']}")
    print(f"Possible secret findings: {summary['possible_secret_findings']}")
    print(f"Gitignore passed: {summary['gitignore_passed']}")
    print(f"Requirements passed: {summary['requirements_passed']}")
    print(f"Ready for manual review: {summary['ready_for_manual_review']}")
    print(f"\nDetailed report: {OUTPUT_PATH}")

    if audit["possible_secrets"]:
        print("\nWarning: possible secrets were found.")
        print("Review project_audit.json manually.")

    missing_files = [
        result["path"] for result in audit["required_files"] if not result["exists"]
    ]
    if missing_files:
        print("\nMissing files:")
        for path in missing_files:
            print(f"- {path}")

    failed_features = [result for result in audit["features"] if not result["passed"]]
    if failed_features:
        print("\nFailed feature checks:")
        for result in failed_features:
            print(f"- {result['feature']}: {result['missing_markers']}")

    print_diagnostics(audit)


if __name__ == "__main__":
    main()
