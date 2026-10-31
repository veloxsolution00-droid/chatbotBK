"""Ollama client with actionable diagnostics for local model failures."""
import requests
from threading import BoundedSemaphore
from app.core.config import settings

# Keep local inference bounded on low-memory machines. Ollama remains responsible
# for model lifetime; this semaphore prevents this API from flooding it with work.
_MODEL_SLOT = BoundedSemaphore(value=1)


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama or a configured local model cannot be used."""


def _model_names():
    try:
        response = requests.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        return [item.get("name", "") for item in response.json().get("models", [])]
    except requests.RequestException as exc:
        raise OllamaUnavailableError(
            f"Cannot connect to Ollama at {settings.ollama_base_url}. "
            "Start Ollama and verify OLLAMA_BASE_URL. Original error: "
            f"{exc}"
        ) from exc
    except (ValueError, AttributeError) as exc:
        raise OllamaUnavailableError(
            "Ollama responded with an unexpected /api/tags response."
        ) from exc


def _has_model(names, configured):
    # Ollama's /api/tags normally returns names including the tag, e.g. gemma2:2b.
    return configured in names or f"{configured}:latest" in names


def status():
    try:
        names = _model_names()
        return {
            "available": True,
            "base_url": settings.ollama_base_url,
            "installed_models": names,
            "generation_model": settings.generation_model,
            "generation_model_available": _has_model(names, settings.generation_model),
            "embedding_model": settings.embedding_model,
            "embedding_model_available": _has_model(names, settings.embedding_model),
        }
    except OllamaUnavailableError as exc:
        return {
            "available": False,
            "base_url": settings.ollama_base_url,
            "generation_model": settings.generation_model,
            "generation_model_available": False,
            "embedding_model": settings.embedding_model,
            "embedding_model_available": False,
            "error": str(exc),
        }


def _request_model(model, endpoint, payload, timeout):
    try:
        response = requests.post(
            f"{settings.ollama_base_url.rstrip('/')}{endpoint}",
            json=payload,
            timeout=timeout,
        )
        if response.status_code == 404:
            raise OllamaUnavailableError(
                f"Ollama could not find model '{model}'. Install it in a terminal with "
                f"`ollama pull {model}`, then verify with `ollama list`. "
                f"Endpoint response: {response.text[:500]}"
            )
        response.raise_for_status()
        return response.json()
    except OllamaUnavailableError:
        raise
    except requests.ConnectionError as exc:
        raise OllamaUnavailableError(
            f"Cannot connect to Ollama at {settings.ollama_base_url}. "
            "Make sure Ollama is running. Original error: " + str(exc)
        ) from exc
    except requests.Timeout as exc:
        raise OllamaUnavailableError(
            f"Ollama timed out while using model '{model}' after {timeout} seconds. "
            "Try a smaller model, increase TIMEOUT in .env, or check available RAM. "
            f"Original error: {exc}"
        ) from exc
    except requests.HTTPError as exc:
        detail = exc.response.text[:700] if exc.response is not None else str(exc)
        raise OllamaUnavailableError(
            f"Ollama returned an HTTP error for model '{model}': {detail}"
        ) from exc
    except (ValueError, KeyError) as exc:
        raise OllamaUnavailableError(
            f"Ollama returned an invalid response for model '{model}': {exc}"
        ) from exc


def embed(texts):
    """Embed a batch in one Ollama request when supported, with legacy fallback.

    Ollama's /api/embed accepts an input list and avoids one HTTP/model call per
    chunk. Older Ollama versions fall back to /api/embeddings one text at a time.
    """
    if not texts:
        return []
    with _MODEL_SLOT:
        url = f"{settings.ollama_base_url.rstrip('/')}/api/embed"
        try:
            response = requests.post(
                url,
                json={"model": settings.embedding_model, "input": texts},
                timeout=settings.timeout,
            )
            if response.status_code not in (404, 405):
                response.raise_for_status()
                payload = response.json()
                vectors = payload.get("embeddings")
                if not isinstance(vectors, list) or len(vectors) != len(texts):
                    raise OllamaUnavailableError(
                        "Ollama /api/embed returned an unexpected number of embeddings."
                    )
                return vectors
        except requests.ConnectionError as exc:
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {settings.ollama_base_url}: {exc}"
            ) from exc
        except requests.Timeout as exc:
            raise OllamaUnavailableError(
                f"Ollama embedding timed out after {settings.timeout} seconds: {exc}"
            ) from exc
        except requests.HTTPError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            raise OllamaUnavailableError(f"Ollama embedding failed: {detail}") from exc

        # Compatibility path for Ollama versions without /api/embed.
        out = []
        for item in texts:
            result = _request_model(
                settings.embedding_model,
                "/api/embeddings",
                {"model": settings.embedding_model, "prompt": item},
                settings.timeout,
            )
            if "embedding" not in result:
                raise OllamaUnavailableError(
                    f"Ollama returned no embedding for '{settings.embedding_model}'."
                )
            out.append(result["embedding"])
        return out



def embed_query(text):
    """Return a cached query embedding, computing it with Ollama on a miss.

    Document/chunk embeddings still use ``embed`` directly so a large corpus
    cannot fill the query cache with every indexed chunk.
    """
    from app.cache.manager import get_embedding, put_embedding, normalize_query
    cache_text = normalize_query(text)
    cached = get_embedding(cache_text, settings.embedding_model)
    if cached is not None:
        return cached
    vector = embed([cache_text])[0]
    put_embedding(cache_text, settings.embedding_model, vector)
    return vector

def generate(prompt):
    with _MODEL_SLOT:
        result = _request_model(
            settings.generation_model,
            "/api/generate",
            {
                "model": settings.generation_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": settings.temperature, "num_ctx": 4096},
            },
            settings.timeout,
        )
    if "response" not in result:
        raise OllamaUnavailableError(
            f"Ollama returned no generated response for '{settings.generation_model}'."
        )
    return result["response"].strip()
