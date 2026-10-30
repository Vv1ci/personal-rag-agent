import json
from pathlib import Path
from typing import Callable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

API_BASE_URL = "http://127.0.0.1:8000"

HEALTH_URL = f"{API_BASE_URL}/health"
UPLOAD_URL = f"{API_BASE_URL}/upload"
DOCUMENTS_URL = f"{API_BASE_URL}/documents"
ASK_URL = f"{API_BASE_URL}/ask"

TEST_DATA_DIR = PROJECT_ROOT / "scripts" / "test_data"
TEST_FILE_PATH = TEST_DATA_DIR / "day22_upload_test.txt"
INVALID_FILE_PATH = TEST_DATA_DIR / "day22_invalid_test.csv"


def print_response(response: requests.Response) -> None:
    """
    Print the HTTP status code and response body.

    response.json() converts returned JSON into a Python object.
    If the response is not valid JSON, response.text is printed instead.
    """
    print(f"HTTP status: {response.status_code}")

    try:
        data = response.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)


def require_status(
    response: requests.Response,
    expected_status: int,
) -> None:
    """
    Check whether the response has the expected HTTP status code.

    Raises:
        AssertionError: If the actual status is different.
    """
    if response.status_code != expected_status:
        print_response(response)

        raise AssertionError(
            f"Expected HTTP {expected_status}, "
            f"but received HTTP {response.status_code}."
        )


def prepare_test_files() -> None:
    """
    Create files used by the integration tests.
    """
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    TEST_FILE_PATH.write_text(
        (
            "Day 22 Integration Test Document\n\n"
            "The internal project codename is Aurora-17.\n\n"
            "The knowledge base refresh trigger is called Blue Lantern.\n\n"
            "This document exists only for testing the Personal RAG Agent "
            "upload, reindex, retrieval, and answer-generation workflow.\n"
        ),
        encoding="utf-8",
    )

    INVALID_FILE_PATH.write_text(
        "This CSV file should be rejected by the upload endpoint.",
        encoding="utf-8",
    )


def test_health() -> None:
    """
    Test whether the FastAPI backend is online.
    """
    response = requests.get(
        HEALTH_URL,
        timeout=5,
    )

    require_status(response, 200)

    data = response.json()

    if data.get("status") != "ok":
        raise AssertionError("The backend did not return status='ok'.")


def test_upload_supported_file() -> None:
    """
    Upload a supported .txt file and rebuild the knowledge base.
    """
    with TEST_FILE_PATH.open("rb") as file_handle:
        files = {
            "file": (
                TEST_FILE_PATH.name,
                file_handle,
                "text/plain",
            )
        }

        response = requests.post(
            UPLOAD_URL,
            files=files,
            params={"rebuild": "true"},
            timeout=600,
        )

    require_status(response, 200)

    data = response.json()

    if data.get("message") != "File uploaded successfully.":
        raise AssertionError("The upload endpoint returned an unexpected message.")


def test_documents_contains_uploaded_file() -> None:
    """
    Confirm that /documents lists the uploaded test document.
    """
    response = requests.get(
        DOCUMENTS_URL,
        timeout=10,
    )

    require_status(response, 200)

    data = response.json()
    documents = data.get("documents", [])

    filenames = [
        document.get("filename", "")
        for document in documents
    ]

    matched_files = [
        filename
        for filename in filenames
        if filename.startswith("day22_upload_test")
    ]

    if not matched_files:
        raise AssertionError(
            "The uploaded Day 22 test file was not found in /documents."
        )


def test_relevant_question() -> None:
    """
    Ask a question whose answer exists in the uploaded document.
    """
    payload = {
        "question": (
            "What is the internal project codename "
            "in the Day 22 integration test document?"
        ),
        "top_k": 5,
        "min_relevance_score": 0.20,
        "language": "en",
        "model": None,
        "debug": True,
    }

    response = requests.post(
        ASK_URL,
        json=payload,
        timeout=180,
    )

    require_status(response, 200)

    data = response.json()

    answer = str(data.get("answer", ""))
    sources = data.get("sources", [])

    if not answer.strip():
        raise AssertionError("The RAG API returned an empty answer.")

    if not sources:
        raise AssertionError(
            "The relevant question returned no sources. "
            "Check ingestion, ChromaDB, or the relevance threshold."
        )

    if "Aurora-17".lower() not in answer.lower():
        print(
            "Warning: The answer did not contain the exact text 'Aurora-17'. "
            "The LLM may have paraphrased or retrieval may need inspection."
        )


def test_invalid_top_k() -> None:
    """
    Confirm that FastAPI rejects an invalid top_k value.
    """
    payload = {
        "question": "What is RAG?",
        "top_k": 0,
        "min_relevance_score": 0.42,
        "language": "en",
        "model": None,
        "debug": False,
    }

    response = requests.post(
        ASK_URL,
        json=payload,
        timeout=10,
    )

    require_status(response, 422)


def test_unsupported_file_type() -> None:
    """
    Confirm that the backend rejects unsupported file extensions.
    """
    with INVALID_FILE_PATH.open("rb") as file_handle:
        files = {
            "file": (
                INVALID_FILE_PATH.name,
                file_handle,
                "text/csv",
            )
        }

        response = requests.post(
            UPLOAD_URL,
            files=files,
            params={"rebuild": "false"},
            timeout=30,
        )

    require_status(response, 400)


def test_unrelated_question() -> None:
    """
    Ask an unrelated question using a strict relevance threshold.

    The expected result is:
    - HTTP 200: the API itself worked
    - no sources: no relevant knowledge was found
    """
    payload = {
        "question": (
            "What is the current ticket price "
            "for Mars Metro Line Nine?"
        ),
        "top_k": 3,
        "min_relevance_score": 1.0,
        "language": "en",
        "model": None,
        "debug": True,
    }

    response = requests.post(
        ASK_URL,
        json=payload,
        timeout=180,
    )

    require_status(response, 200)

    data = response.json()
    sources = data.get("sources", [])

    if sources:
        raise AssertionError(
            "The unrelated question still returned sources "
            "under a strict relevance threshold."
        )


def run_test(
    name: str,
    test_function: Callable[[], None],
) -> bool:
    """
    Run one test and print PASS or FAIL.

    Callable[[], None] means:
    - test_function is callable
    - it takes no arguments
    - it returns None
    """
    print()
    print("=" * 70)
    print(f"Running: {name}")
    print("=" * 70)

    try:
        test_function()

    except Exception as error:
        print(f"[FAIL] {name}")
        print(f"{type(error).__name__}: {error}")
        return False

    print(f"[PASS] {name}")
    return True


def main() -> None:
    prepare_test_files()

    tests = [
        ("FastAPI health check", test_health),
        ("Supported file upload and rebuild", test_upload_supported_file),
        ("Uploaded file appears in documents", test_documents_contains_uploaded_file),
        ("Relevant RAG question", test_relevant_question),
        ("Invalid top_k returns 422", test_invalid_top_k),
        ("Unsupported file type returns 400", test_unsupported_file_type),
        ("Unrelated question returns no sources", test_unrelated_question),
    ]

    passed_count = 0

    for name, test_function in tests:
        if run_test(name, test_function):
            passed_count += 1

    total_count = len(tests)

    print()
    print("=" * 70)
    print("Day 22 Integration Test Summary")
    print("=" * 70)
    print(f"Passed: {passed_count}/{total_count}")
    print(f"Failed: {total_count - passed_count}/{total_count}")

    if passed_count == total_count:
        print("All integration tests passed.")
    else:
        print("Some tests failed. Review the error messages above.")


if __name__ == "__main__":
    main()