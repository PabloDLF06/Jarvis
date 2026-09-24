"""
core/ollama_client.py — Puerta de JARVIS hacia Ollama.
================================================================================
Ollama es el motor que ejecuta los modelos de lenguaje en el propio ordenador
de Pablo. Este módulo habla con su API local (HTTP en 127.0.0.1:11434) y ofrece
lo que el resto de JARVIS necesita:

* `health()`      — ¿está Ollama despierto?
* `chat()`        — conversación con flujo por fragmentos (baja latencia).
* `generate()`    — generación simple (visión, utilidades).
* `ps()`          — qué modelos están cargados AHORA y cuánta VRAM ocupan.
* `unload()`      — expulsar un modelo de la GPU (`keep_alive: 0`).
* `pull()`        — descargar un modelo que falte, con progreso.

Nada de esto sale del ordenador: la dirección es siempre local, nunca hay
claves de API y ningún dato viaja a internet.
================================================================================
"""

from __future__ import annotations

import base64
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import requests

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import config as cfg  # noqa: E402
from core.utils import retry, truncate  # noqa: E402

logger = logging.getLogger("jarvis.ollama")


# =============================================================================
# ERRORES
# =============================================================================

class OllamaError(RuntimeError):
    """Error genérico de comunicación con Ollama."""


class OllamaNotRunningError(OllamaError):
    """Ollama no está en marcha o no responde en el puerto local."""

    def __init__(self, host: str = cfg.OLLAMA_HOST) -> None:
        super().__init__(
            f"No consigo hablar con Ollama en {host}.\n"
            "  · Comprueba que la aplicación Ollama esté abierta (bandeja del sistema).\n"
            "  · Si no la tienes instalada: https://ollama.com/download\n"
            "  · En una terminal:  ollama serve"
        )


class OllamaModelMissingError(OllamaError):
    """El modelo solicitado no está descargado."""

    def __init__(self, model: str) -> None:
        super().__init__(
            f"El modelo '{model}' no está descargado.\n"
            f"  · Descárgalo con:  ollama pull {model}\n"
            "  · O ejecuta de nuevo install.bat / python install.py"
        )


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass
class LoadedModel:
    """Un modelo residente en memoria (y en la GPU, si procede)."""

    name: str
    size_bytes: int = 0
    size_vram_bytes: int = 0
    expires_at: str = ""
    details: dict = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        return self.size_bytes / (1024 ** 3)

    @property
    def vram_gb(self) -> float:
        return self.size_vram_bytes / (1024 ** 3)

    def describe(self) -> str:
        return f"{self.name} ({self.size_gb:.2f} GB, VRAM {self.vram_gb:.2f} GB)"


@dataclass
class ChatResult:
    """Respuesta completa de una inferencia."""

    text: str = ""
    model: str = ""
    done_reason: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    raw: dict = field(default_factory=dict)
    aborted: bool = False

    @property
    def tokens_per_second(self) -> float:
        if self.duration_s <= 0 or self.output_tokens <= 0:
            return 0.0
        return self.output_tokens / self.duration_s

    def describe(self) -> str:
        return (
            f"{self.model}: {self.output_tokens} tokens en {self.duration_s:.2f} s "
            f"({self.tokens_per_second:.1f} tok/s)"
        )


# =============================================================================
# UTILIDADES DE IMAGEN
# =============================================================================

def encode_image_b64(path: str | Path) -> str:
    """Codifica una imagen en Base64 tal y como la espera llama3.2-vision."""
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


# =============================================================================
# CLIENTE
# =============================================================================

class OllamaClient:
    """Cliente HTTP sincrónico de la API local de Ollama.

    Es seguro para usar desde varios hilos: cada petición crea su propia
    conexión a través de una `requests.Session` con grupo de conexiones.
    """

    def __init__(
        self,
        host: str = cfg.OLLAMA_HOST,
        session: requests.Session | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.cancel_check = cancel_check
        self._model_cache: tuple[float, list[dict]] = (0.0, [])

    # --- Infraestructura ---------------------------------------------------

    def _url(self, endpoint: str) -> str:
        return f"{self.host}{endpoint}"

    def _should_cancel(self) -> bool:
        """¿Debe abortarse la petición en curso (kill-switch o parada)?"""
        return bool(self.cancel_check and self.cancel_check())

    def _post(self, endpoint: str, payload: dict, timeout: float | None = None, stream: bool = False):
        try:
            return self.session.post(
                self._url(endpoint),
                json=payload,
                timeout=timeout or cfg.OLLAMA_CHAT_TIMEOUT,
                stream=stream,
            )
        except requests.exceptions.ConnectionError as exc:
            raise OllamaNotRunningError(self.host) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaError(
                f"Ollama no respondió a tiempo ({timeout or cfg.OLLAMA_CHAT_TIMEOUT:.0f} s). "
                "El modelo puede ser demasiado grande para esta tarjeta gráfica."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaError(f"Error de red hablando con Ollama: {exc}") from exc

    def _get(self, endpoint: str, timeout: float | None = None):
        try:
            return self.session.get(self._url(endpoint), timeout=timeout or cfg.OLLAMA_SHORT_TIMEOUT)
        except requests.exceptions.ConnectionError as exc:
            raise OllamaNotRunningError(self.host) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaError(f"Error de red hablando con Ollama: {exc}") from exc

    @staticmethod
    def _iter_stream(response) -> Iterable[dict]:
        """Itera las líneas JSON del flujo de Ollama (una respuesta por línea)."""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.debug("Línea no JSON de Ollama: %s", truncate(line, 120))
                continue

    # --- Estado del servicio ----------------------------------------------

    def health(self, timeout: float | None = None) -> bool:
        """True si Ollama responde en el puerto local."""
        try:
            response = self._get("/api/tags", timeout=timeout or cfg.OLLAMA_HEALTH_TIMEOUT)
            return response.status_code == 200
        except OllamaError:
            return False

    def require_running(self) -> None:
        """Lanza `OllamaNotRunningError` si el servicio no está disponible."""
        if not self.health():
            raise OllamaNotRunningError(self.host)

    def version(self) -> str | None:
        """Versión de Ollama instalada (o None si no responde)."""
        try:
            response = self._get("/api/version")
            if response.status_code == 200:
                return str(response.json().get("version", ""))
        except (OllamaError, ValueError):
            return None
        return None

    # --- Catálogo de modelos ----------------------------------------------

    def list_models(self, use_cache: bool = True, ttl_s: float = 20.0) -> list[dict]:
        """Modelos descargados en el disco (nombre, tamaño, fecha)."""
        now = time.monotonic()
        cached_at, cached = self._model_cache
        if use_cache and cached and (now - cached_at) < ttl_s:
            return cached
        response = self._get("/api/tags")
        if response.status_code != 200:
            raise OllamaError(f"Ollama devolvió {response.status_code} al listar modelos.")
        models = list(response.json().get("models", []))
        self._model_cache = (now, models)
        return models

    def has_model(self, model: str) -> bool:
        """¿Está descargado el modelo? Acepta coincidencia por nombre base.

        Ollama trata `llama3.1` y `llama3.1:latest` como el mismo modelo, así
        que la comparación se hace con y sin etiqueta.
        """
        target = model.strip().lower()
        bare = target.split(":")[0]
        try:
            names = {str(m.get("name", "")).lower() for m in self.list_models()}
        except OllamaError:
            return False
        if target in names or f"{target}:latest" in names:
            return True
        return any(name.split(":")[0] == bare for name in names)

    def resolve_model_name(self, model: str) -> str:
        """Devuelve el nombre real registrado en Ollama para `model`."""
        try:
            for entry in self.list_models():
                name = str(entry.get("name", ""))
                if name.lower() == model.lower() or name.split(":")[0].lower() == model.split(":")[0].lower():
                    return name
        except OllamaError:
            pass
        return model

    def model_info(self, model: str) -> dict:
        """Ficha técnica del modelo (`/api/show`): familia, parámetros, plantilla."""
        response = self._post("/api/show", {"model": model}, timeout=cfg.OLLAMA_SHORT_TIMEOUT)
        if response.status_code == 404:
            raise OllamaModelMissingError(model)
        if response.status_code != 200:
            raise OllamaError(f"Ollama devolvió {response.status_code} para /api/show.")
        try:
            return response.json()
        except ValueError as exc:
            raise OllamaError("Respuesta ilegible de /api/show.") from exc

    def missing_models(self, models: Sequence[str] = cfg.MODELS_TO_AUTO_PULL) -> list[str]:
        """De una lista, los que NO están descargados todavía."""
        return [m for m in models if not self.has_model(m)]

    # --- Modelos cargados en memoria / VRAM --------------------------------

    def ps(self) -> list[LoadedModel]:
        """Modelos cargados ahora mismo, con su consumo de VRAM."""
        try:
            response = self._get("/api/ps")
        except OllamaError:
            return []
        if response.status_code != 200:
            return []
        try:
            payload = response.json()
        except ValueError:
            return []
        loaded: list[LoadedModel] = []
        for entry in payload.get("models", []):
            loaded.append(
                LoadedModel(
                    name=str(entry.get("name") or entry.get("model") or ""),
                    size_bytes=int(entry.get("size") or 0),
                    size_vram_bytes=int(entry.get("size_vram") or 0),
                    expires_at=str(entry.get("expires_at") or ""),
                    details=dict(entry.get("details") or {}),
                )
            )
        return loaded

    def loaded_names(self) -> list[str]:
        return [m.name for m in self.ps()]

    def vram_in_use_gb(self) -> float:
        """Suma de VRAM ocupada por los modelos cargados (según Ollama)."""
        return sum(m.vram_gb for m in self.ps())

    def unload(self, model: str, wait: bool = True) -> bool:
        """Expulsa un modelo de la memoria y de la GPU.

        Se hace con `keep_alive: 0` (y un prompt vacío), que es la forma
        documentada de pedirle a Ollama que libere el modelo inmediatamente.
        Este es el ladrillo con el que se construye la ley de monogamia de VRAM.
        """
        if not model:
            return False
        if model not in self.loaded_names():
            logger.debug("Nada que descargar: '%s' no estaba cargado.", model)
            return True
        logger.info("Descargando modelo de la VRAM: %s", model)
        try:
            response = self._post(
                "/api/generate",
                {"model": model, "prompt": "", "keep_alive": 0},
                timeout=cfg.OLLAMA_SHORT_TIMEOUT,
            )
            ok = response.status_code in (200, 404)
        except OllamaError as exc:
            logger.warning("No se pudo descargar '%s': %s", model, exc)
            ok = False

        if wait and ok:
            self.wait_until_unloaded(model)
        return ok

    def wait_until_unloaded(self, model: str, timeout: float = cfg.VRAM_UNLOAD_WAIT_S) -> bool:
        """Espera activamente a que Ollama confirme que el modelo salió de VRAM."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if model not in self.loaded_names():
                    return True
            except OllamaError:
                return False
            time.sleep(cfg.VRAM_UNLOAD_POLL_S)
        logger.warning("'%s' seguía cargado tras %.1f s de espera.", model, timeout)
        return False

    def unload_all(self, keep: str | None = None) -> list[str]:
        """Expulsa TODOS los modelos residentes (salvo `keep`, si se indica).

        Se usa antes de cargar un modelo pesado (por ejemplo el de visión) para
        garantizar que solo hay un LLM en la GPU.
        """
        freed: list[str] = []
        for loaded in self.ps():
            if keep and loaded.name == keep:
                continue
            if self.unload(loaded.name):
                freed.append(loaded.name)
        return freed

    # --- Descarga de modelos ----------------------------------------------

    def pull(
        self,
        model: str,
        progress_cb: Callable[[str, int, int], None] | None = None,
    ) -> bool:
        """Descarga un modelo con Ollama, informando del progreso.

        `progress_cb(estado, completado, total)` permite pintar una barra en
        `install.py` sin que este módulo sepa nada de interfaces.
        """
        logger.info("Descargando modelo '%s' (puede tardar varios minutos)...", model)
        try:
            response = self._post(
                "/api/pull",
                {"model": model, "stream": True},
                timeout=cfg.OLLAMA_PULL_TIMEOUT,
                stream=True,
            )
        except OllamaError as exc:
            logger.error("No se pudo descargar '%s': %s", model, exc)
            return False

        if response.status_code != 200:
            logger.error("Ollama devolvió %d al descargar '%s'.", response.status_code, model)
            return False

        last_status = ""
        try:
            for chunk in self._iter_stream(response):
                if "error" in chunk:
                    logger.error("Error de Ollama descargando '%s': %s", model, chunk["error"])
                    return False
                status = str(chunk.get("status", ""))
                completed = int(chunk.get("completed") or 0)
                total = int(chunk.get("total") or 0)
                if status != last_status:
                    logger.info("  %s: %s", model, status)
                    last_status = status
                if progress_cb is not None:
                    try:
                        progress_cb(status, completed, total)
                    except Exception:  # la interfaz no puede romper la descarga
                        pass
        except requests.exceptions.RequestException as exc:
            logger.error("Conexión interrumpida descargando '%s': %s", model, exc)
            return False

        self._model_cache = (0.0, [])  # Forzar recarga del catálogo.
        logger.info("Modelo '%s' disponible.", model)
        return True

    def pull_via_cli(self, model: str, timeout: float = cfg.OLLAMA_PULL_TIMEOUT) -> bool:
        """Alternativa: descarga mediante el comando `ollama pull`."""
        from safety.command_guard import CommandGuardError, run_guarded

        try:
            result = run_guarded(
                [cfg.OLLAMA_BIN, "pull", model],
                timeout=timeout,
                trust="internal",
            )
        except (CommandGuardError, OSError) as exc:
            logger.error("No se pudo ejecutar 'ollama pull %s': %s", model, exc)
            return False
        if result.returncode != 0:
            logger.error("'ollama pull %s' falló: %s", model, truncate(result.stderr, 300))
            return False
        self._model_cache = (0.0, [])
        return True

    # --- Inferencia --------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: Sequence[dict],
        options: dict | None = None,
        keep_alive: int | str | None = None,
        stream: bool = True,
        on_token: Callable[[str], None] | None = None,
        timeout: float | None = None,
        format_json: bool = False,
        tools: Sequence[dict] | None = None,
    ) -> ChatResult:
        """Conversación con el modelo. Devuelve el texto completo.

        Con ``stream=True`` (por defecto) el texto llega fragmento a fragmento,
        lo que permite que el TTS empiece a hablar antes de que el modelo
        termine de pensar. Entre fragmentos se comprueba el kill-switch: si
        Pablo lo activa, la inferencia se aborta de inmediato.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "stream": stream,
            "options": dict(options or {}),
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if format_json:
            payload["format"] = "json"
        if tools:
            payload["tools"] = list(tools)

        started = time.perf_counter()
        result = ChatResult(model=model)

        try:
            response = self._post(
                "/api/chat",
                payload,
                timeout=timeout or cfg.OLLAMA_CHAT_TIMEOUT,
                stream=stream,
            )
        except OllamaError as exc:
            logger.error("Fallo de inferencia con %s: %s", model, exc)
            result.done_reason = "error"
            result.raw = {"error": str(exc)}
            return result

        if response.status_code == 404:
            raise OllamaModelMissingError(model)
        if response.status_code != 200:
            message = truncate(response.text, 300)
            raise OllamaError(f"Ollama devolvió {response.status_code}: {message}")

        if not stream:
            try:
                payload_json = response.json()
            except ValueError as exc:
                raise OllamaError("Respuesta ilegible de /api/chat.") from exc
            message = payload_json.get("message") or {}
            result.text = str(message.get("content", ""))
            result.done_reason = str(payload_json.get("done_reason", ""))
            result.prompt_tokens = int(payload_json.get("prompt_eval_count") or 0)
            result.output_tokens = int(payload_json.get("eval_count") or 0)
            result.raw = payload_json
            result.duration_s = time.perf_counter() - started
            return result

        pieces: list[str] = []
        try:
            for chunk in self._iter_stream(response):
                if self._should_cancel():
                    result.aborted = True
                    result.done_reason = "abortado"
                    logger.warning("Inferencia con %s abortada (kill-switch).", model)
                    break
                if chunk.get("error"):
                    raise OllamaError(f"Ollama informó de un error: {chunk['error']}")
                message = chunk.get("message") or {}
                token = str(message.get("content", ""))
                if token:
                    pieces.append(token)
                    if on_token is not None:
                        try:
                            on_token(token)
                        except Exception as exc:  # el consumidor no puede romper la inferencia
                            logger.debug("on_token lanzó una excepción: %s", exc)
                if chunk.get("done"):
                    result.done_reason = str(chunk.get("done_reason", ""))
                    result.prompt_tokens = int(chunk.get("prompt_eval_count") or 0)
                    result.output_tokens = int(chunk.get("eval_count") or 0)
                    result.raw = chunk
                    break
        except requests.exceptions.RequestException as exc:
            raise OllamaError(f"Conexión interrumpida durante la inferencia: {exc}") from exc
        finally:
            response.close()

        result.text = "".join(pieces)
        if not result.output_tokens and pieces:
            result.output_tokens = len(result.text.split())
        result.duration_s = time.perf_counter() - started
        logger.debug("Inferencia %s", result.describe())
        return result

    def generate(
        self,
        model: str,
        prompt: str,
        images: Sequence[str | Path] | None = None,
        options: dict | None = None,
        keep_alive: int | str | None = None,
        stream: bool = False,
        system: str | None = None,
        timeout: float | None = None,
        format_json: bool = False,
    ) -> ChatResult:
        """Generación simple (la vía natural para la visión por imágenes)."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": dict(options or {}),
        }
        if system:
            payload["system"] = system
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if format_json:
            payload["format"] = "json"
        if images:
            payload["images"] = [
                encode_image_b64(image) if not str(image).startswith(("iVBOR", "/9j/")) else str(image)
                for image in images
            ]

        started = time.perf_counter()
        result = ChatResult(model=model)
        try:
            response = self._post(
                "/api/generate",
                payload,
                timeout=timeout or cfg.OLLAMA_VISION_TIMEOUT,
                stream=stream,
            )
        except OllamaError as exc:
            logger.error("Fallo de generación con %s: %s", model, exc)
            result.raw = {"error": str(exc)}
            return result

        if response.status_code == 404:
            raise OllamaModelMissingError(model)
        if response.status_code != 200:
            raise OllamaError(f"Ollama devolvió {response.status_code}: {truncate(response.text, 300)}")

        if stream:
            pieces: list[str] = []
            try:
                for chunk in self._iter_stream(response):
                    if self._should_cancel():
                        result.aborted = True
                        break
                    pieces.append(str(chunk.get("response", "")))
                    if chunk.get("done"):
                        result.output_tokens = int(chunk.get("eval_count") or 0)
                        result.prompt_tokens = int(chunk.get("prompt_eval_count") or 0)
                        break
            finally:
                response.close()
            result.text = "".join(pieces)
        else:
            try:
                payload_json = response.json()
            except ValueError as exc:
                raise OllamaError("Respuesta ilegible de /api/generate.") from exc
            result.text = str(payload_json.get("response", ""))
            result.output_tokens = int(payload_json.get("eval_count") or 0)
            result.prompt_tokens = int(payload_json.get("prompt_eval_count") or 0)
            result.done_reason = str(payload_json.get("done_reason", ""))
            result.raw = payload_json

        result.duration_s = time.perf_counter() - started
        return result

    # --- Utilidades --------------------------------------------------------

    def wait_until_ready(self, attempts: int = 30, delay: float = 1.0) -> bool:
        """Espera a que Ollama arranque (útil justo después de lanzarlo)."""
        for _ in range(max(1, attempts)):
            if self.health():
                return True
            time.sleep(delay)
        return False

    def summary(self) -> dict:
        """Resumen del estado del servicio (diagnóstico)."""
        running = self.health()
        info: dict[str, Any] = {
            "servicio": "activo" if running else "parado",
            "host": self.host,
            "version": self.version() if running else None,
            "modelos_descargados": len(self.list_models()) if running else 0,
            "modelos_cargados": [],
        }
        if running:
            info["modelos_cargados"] = [
                {"nombre": m.name, "vram_gb": round(m.vram_gb, 2)} for m in self.ps()
            ]
        return info

    @retry(attempts=cfg.OLLAMA_RETRY_ATTEMPTS, delay=cfg.OLLAMA_RETRY_BACKOFF_S)
    def warmup(self, model: str) -> bool:
        """Carga un modelo en memoria con una petición mínima (calentamiento).

        Cargar un modelo de 8B tarda unos segundos; hacerlo antes de la primera
        frase real evita que Pablo espere dos veces.
        """
        response = self._post(
            "/api/generate",
            {"model": model, "prompt": "ok", "stream": False, "options": {"num_predict": 1}},
            timeout=cfg.OLLAMA_CHAT_TIMEOUT,
        )
        return response.status_code == 200

    def close(self) -> None:
        """Cierra el grupo de conexiones HTTP."""
        try:
            self.session.close()
        except Exception:
            pass
