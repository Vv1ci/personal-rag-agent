import json
import os
from datetime import datetime
from pathlib import Path

from rag.chains.rag_chain import answer_with_rag, DEFAULT_MIN_RELEVANCE_SCORE
from rag.llm.ollama_client import DEFAULT_MODEL


HISTORY_FILE = "data/processed/chat_history.jsonl"


def print_banner() -> None:
    """
    Print CLI welcome message.
    """
    print("=" * 80)
    print("Personal RAG Agent")
    print("=" * 80)
    print("Type your question and press Enter.")
    print("Type /help to see available commands.")
    print("Type /exit to stop.")
    print()


def print_help() -> None:
    """
    Print available CLI commands.
    """
    print("\n===== COMMANDS =====")
    print("/help                  Show available commands")
    print("/settings              Show current settings")
    print("/topk <number>         Set top-k retrieval, example: /topk 5")
    print("/threshold <number>    Set minimum relevance score, example: /threshold 0.43")
    print("/debug on              Enable debug output")
    print("/debug off             Disable debug output")
    print("/lang en               Answer in English")
    print("/lang zh               Answer in Chinese")
    print("/model <name>          Set Ollama model, example: /model qwen2.5:1.5b")
    print("/history               Show questions asked in this session")
    print("/clear                 Clear the terminal screen")
    print("/exit                  Exit the program")
    print()


def print_settings(settings: dict) -> None:
    """
    Print current CLI settings.
    """
    print("\n===== CURRENT SETTINGS =====")
    print(f"Language: {settings['language']}")
    print(f"Top-k: {settings['top_k']}")
    print(f"Minimum relevance score: {settings['min_relevance_score']}")
    print(f"Debug: {settings['debug']}")
    print(f"Model: {settings['model']}")
    print()


def print_sources(sources: list[dict]) -> None:
    """
    Print retrieved sources in a readable format.
    """
    if not sources:
        print("No sources found.")
        return

    for source in sources:
        page_number = source.get("page_number", -1)

        if page_number == -1:
            page_text = "no page"
        else:
            page_text = f"page {page_number}"

        print(
            f"- rank {source['rank']} | "
            f"{source['source_file']} | "
            f"{page_text} | "
            f"score {source['relevance_score']:.4f}"
        )


def print_retrieved_debug(result: dict) -> None:
    """
    Print all retrieved results and filtered results for debugging.
    """
    print("\n===== ALL RETRIEVED RESULTS BEFORE FILTERING =====")
    for item in result["retrieved_results"]:
        metadata = item["metadata"]
        print(
            f"- rank {item['rank']} | "
            f"{item['id']} | "
            f"{metadata.get('source_file', 'unknown')} | "
            f"score {item['relevance_score']:.4f}"
        )

    print("\n===== FILTERED RESULTS AFTER THRESHOLD =====")
    if not result["filtered_results"]:
        print("No results passed the relevance threshold.")
    else:
        for item in result["filtered_results"]:
            metadata = item["metadata"]
            print(
                f"- rank {item['rank']} | "
                f"{item['id']} | "
                f"{metadata.get('source_file', 'unknown')} | "
                f"score {item['relevance_score']:.4f}"
            )

    print("\n===== DEBUG PROMPT =====")
    print(result["prompt"])


def save_history_record(record: dict) -> None:
    """
    Save one question-answer record to a JSONL history file.

    JSONL means each line is one JSON object.
    """
    output_path = Path(HISTORY_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def print_session_history(session_history: list[dict]) -> None:
    """
    Print questions asked during the current CLI session.
    """
    print("\n===== SESSION HISTORY =====")

    if not session_history:
        print("No questions asked yet.")
        return

    for index, item in enumerate(session_history, start=1):
        print(f"{index}. {item['query']}")


def ask_once(query: str, settings: dict) -> dict:
    """
    Ask one question using current settings.
    """
    result = answer_with_rag(
        query=query,
        top_k=settings["top_k"],
        language=settings["language"],
        model=settings["model"],
        min_relevance_score=settings["min_relevance_score"]
    )

    print("\n===== ANSWER =====")
    print(result["answer"])

    print("\n===== SOURCES =====")
    print_sources(result["sources"])

    if settings["debug"]:
        print_retrieved_debug(result)

    return result


def handle_command(
    command: str,
    settings: dict,
    session_history: list[dict]
) -> bool:
    """
    Handle a CLI command.

    Returns:
        True if the program should continue.
        False if the program should exit.
    """
    parts = command.strip().split()

    if not parts:
        return True

    command_name = parts[0].lower()

    if command_name in ["/exit", "/quit", "/q"]:
        print("Goodbye.")
        return False

    if command_name == "/help":
        print_help()
        return True

    if command_name == "/settings":
        print_settings(settings)
        return True

    if command_name == "/history":
        print_session_history(session_history)
        return True

    if command_name == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
        return True

    if command_name == "/topk":
        if len(parts) != 2:
            print("Usage: /topk <number>")
            return True

        try:
            top_k = int(parts[1])

            if top_k <= 0:
                print("top-k must be greater than 0.")
                return True

            settings["top_k"] = top_k
            print(f"Top-k updated to {top_k}.")

        except ValueError:
            print("Invalid top-k. Please enter an integer.")

        return True

    if command_name == "/threshold":
        if len(parts) != 2:
            print("Usage: /threshold <number>")
            return True

        try:
            threshold = float(parts[1])

            if threshold < 0:
                print("Minimum relevance score cannot be negative.")
                return True

            settings["min_relevance_score"] = threshold
            print(f"Minimum relevance score updated to {threshold}.")

        except ValueError:
            print("Invalid threshold. Please enter a number.")

        return True

    if command_name == "/debug":
        if len(parts) != 2:
            print("Usage: /debug on OR /debug off")
            return True

        value = parts[1].lower()

        if value == "on":
            settings["debug"] = True
            print("Debug enabled.")
        elif value == "off":
            settings["debug"] = False
            print("Debug disabled.")
        else:
            print("Usage: /debug on OR /debug off")

        return True

    if command_name == "/lang":
        if len(parts) != 2:
            print("Usage: /lang en OR /lang zh")
            return True

        value = parts[1].lower()

        if value not in ["en", "zh"]:
            print("Unsupported language. Use /lang en or /lang zh.")
            return True

        settings["language"] = value
        print(f"Language updated to {value}.")
        return True

    if command_name == "/model":
        if len(parts) < 2:
            print("Usage: /model <ollama_model_name>")
            return True

        model_name = " ".join(parts[1:])
        settings["model"] = model_name
        print(f"Model updated to {model_name}.")
        return True

    print(f"Unknown command: {command_name}")
    print("Type /help to see available commands.")
    return True


def main() -> None:
    """
    Interactive command-line RAG assistant.
    """
    settings = {
        "language": "en",
        "top_k": 3,
        "min_relevance_score": DEFAULT_MIN_RELEVANCE_SCORE,
        "debug": False,
        "model": DEFAULT_MODEL
    }

    session_history = []

    print_banner()
    print_settings(settings)

    while True:
        user_input = input("\nAsk your question or type a command:\n> ").strip()

        if not user_input:
            print("Input cannot be empty.")
            continue

        if user_input.startswith("/"):
            should_continue = handle_command(
                command=user_input,
                settings=settings,
                session_history=session_history
            )

            if not should_continue:
                break

            continue

        try:
            result = ask_once(
                query=user_input,
                settings=settings
            )

            history_record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "query": user_input,
                "answer": result["answer"],
                "sources": result["sources"],
                "settings": {
                    "language": settings["language"],
                    "top_k": settings["top_k"],
                    "min_relevance_score": settings["min_relevance_score"],
                    "debug": settings["debug"],
                    "model": settings["model"]
                }
            }

            session_history.append(history_record)
            save_history_record(history_record)

        except Exception as error:
            print("\nAn error occurred:")
            print(error)


if __name__ == "__main__":
    main()