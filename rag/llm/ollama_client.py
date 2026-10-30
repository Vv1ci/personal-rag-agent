from rag.config import settings
import requests


DEFAULT_OLLAMA_GENERATE_URL = settings.ollama_base_url.rstrip("/") + "/api/generate"
DEFAULT_MODEL = settings.ollama_model

def generate_with_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    url: str = DEFAULT_OLLAMA_GENERATE_URL,
    temperature: float = 0.2,
    num_predict: int = 512,
    timeout: int = 120
) -> str:
    """
    Generate an answer using a local Ollama model.

    Args:
        prompt: the full prompt sent to the LLM
        model: Ollama model name
        url: Ollama generate API endpoint
        temperature: lower means more stable and less random
        num_predict: maximum number of tokens to generate
        timeout: request timeout in seconds

    Returns:
        Generated answer text.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict
        }
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout
        )

        response.raise_for_status()

    except requests.exceptions.ConnectionError as error:
        raise ConnectionError(
            "Cannot connect to Ollama. Please make sure Ollama is running."
        ) from error

    except requests.exceptions.Timeout as error:
        raise TimeoutError(
            "Ollama request timed out. Try a smaller model or increase timeout."
        ) from error

    data = response.json()

    if "response" not in data:
        raise ValueError(f"Unexpected Ollama response format: {data}")

    return data["response"].strip()


if __name__ == "__main__":
    test_prompt = "Explain RAG in one short paragraph."

    answer = generate_with_ollama(test_prompt)

    print("===== OLLAMA ANSWER =====")
    print(answer)