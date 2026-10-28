"""Conexión con el modelo de lenguaje (LLM) usando Ollama local y CUDA."""
import json
import os
from pathlib import Path
from typing import Any, Generator
import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

# Modelos recomendados para RTX 4060 (8GB VRAM con CUDA):
# 'llama3.1:8b' (~4.9 GB) o 'qwen2.5:7b' (~4.7 GB)
MODELO_DEFAULT = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def stream_llm(
    sistema: str,
    mensajes: list[dict[str, Any]],
    modelo: str = MODELO_DEFAULT,
    temperatura: float = 0.2,
) -> Generator[str, None, None]:
    """Transmite en tiempo real (streaming) la respuesta generada por Ollama."""
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    payload_messages = []
    if sistema:
        payload_messages.append({"role": "system", "content": sistema})
    payload_messages.extend(mensajes)

    payload = {
        "model": modelo,
        "messages": payload_messages,
        "stream": True,
        "options": {
            "temperature": temperatura,
        },
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as respuesta:
            respuesta.raise_for_status()
            for linea in respuesta.iter_lines():
                if linea:
                    datos = json.loads(linea.decode("utf-8"))
                    chunk = datos.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if datos.get("done", False):
                        break
    except requests.exceptions.ConnectionError:
        yield (
            f"\n[Error de conexión]: No fue posible conectar con Ollama en {OLLAMA_HOST}.\n"
            "Asegúrese de que el servicio de Ollama esté ejecutándose (ej. 'ollama serve').\n"
        )
    except Exception as e:
        yield f"\n[Error en Ollama]: {e}\n"


def llamar_llm(
    sistema: str,
    mensajes: list[dict[str, Any]],
    max_tokens: int = 1024,
    modelo: str = MODELO_DEFAULT,
    temperatura: float = 0.2,
) -> str:
    """Envía prompt de sistema y mensajes a Ollama local y devuelve el texto completo."""
    if not mensajes:
        raise ValueError("La lista de mensajes no puede estar vacía.")

    # 1. Intentar llamar a Ollama local
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    payload_messages = []
    if sistema:
        payload_messages.append({"role": "system", "content": sistema})
    payload_messages.extend(mensajes)

    payload = {
        "model": modelo,
        "messages": payload_messages,
        "stream": False,
        "options": {
            "temperature": temperatura,
            "num_predict": max_tokens,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("message", {}).get("content", "")
    except requests.exceptions.ConnectionError:
        # Si Ollama no está activo y se tiene clave Anthropic, usar fallback opcional
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                import anthropic

                cliente = anthropic.Anthropic()
                respuesta = cliente.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=max_tokens,
                    system=sistema,
                    messages=mensajes,
                )
                textos = [b.text for b in getattr(respuesta, "content", []) if hasattr(b, "text")]
                if textos:
                    return "".join(textos)
            except Exception:
                pass

        return (
            "[Error]: No se pudo conectar a Ollama en "
            f"{OLLAMA_HOST}. Inicie Ollama con 'ollama serve'."
        )
    except Exception as e:
        return f"[Error LLM]: {e}"

    return ""
