"""
core/utils.py — Utilidades transversales de JARVIS.
================================================================================
Contiene las funciones pequeñas que usan todos los módulos: reintentos con
espera exponencial, escritura atómica de archivos, búsqueda de ejecutables,
limpieza de texto para voz, troceado de frases y un cronómetro.

Regla de diseño: **este módulo no importa nada del resto del proyecto**, de
modo que cualquier módulo pueda usarlo sin crear dependencias circulares.
================================================================================
"""

from __future__ import annotations

import functools
import json
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

logger = logging.getLogger("jarvis.utils")

T = TypeVar("T")


# =============================================================================
# TIEMPO
# =============================================================================

def now_iso() -> str:
    """Marca de tiempo local en formato ISO-8601 (segundos)."""
    import datetime as _dt

    return _dt.datetime.now().replace(microsecond=0).isoformat()


def timestamp_slug() -> str:
    """Marca de tiempo apta para nombres de rama, carpeta o archivo.

    >>> len(timestamp_slug())
    15
    """
    import datetime as _dt

    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def format_clock(seconds: float) -> str:
    """Convierte segundos en un texto legible en español (para el HUD/log)."""
    seconds = max(0.0, float(seconds))
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {int(rest)} s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(minutes)} min"


@contextmanager
def stopwatch(label: str = "", log: logging.Logger | None = None) -> Iterator[dict]:
    """Cronómetro reutilizable.

    >>> with stopwatch() as crono:
    ...     pass
    >>> "segundos" in crono
    True
    """
    result: dict[str, Any] = {"label": label, "segundos": 0.0, "inicio": time.perf_counter()}
    try:
        yield result
    finally:
        result["segundos"] = time.perf_counter() - result["inicio"]
        if label:
            (log or logger).debug("%s: %.3f s", label, result["segundos"])


def sleep_until(deadline: float, step: float = 0.01) -> None:
    """Duerme hasta un instante `time.monotonic()` concreto, sin sobrepasarse."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(step, remaining))


# =============================================================================
# REINTENTOS
# =============================================================================

def retry(
    attempts: int = 3,
    delay: float = 0.5,
    backoff: float = 2.0,
    jitter: float = 0.1,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorador de reintentos con espera exponencial y algo de aleatoriedad.

    Se usa sobre todo con la API de Ollama: si el servicio se reinicia o
    tarda en responder, JARVIS no debe darse por vencido a la primera.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: BaseException | None = None
            for attempt in range(1, max(1, attempts) + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # pragma: no cover - depende de la red
                    last_exc = exc
                    if attempt >= attempts:
                        break
                    wait = delay * (backoff ** (attempt - 1))
                    wait += random.uniform(0, max(0.0, jitter))
                    if on_retry is not None:
                        on_retry(attempt, exc)
                    logger.warning(
                        "%s falló (intento %d/%d): %s. Reintento en %.1f s.",
                        func.__name__, attempt, attempts, exc, wait,
                    )
                    time.sleep(wait)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


# =============================================================================
# ARCHIVOS
# =============================================================================

def atomic_write_text(path: str | os.PathLike[str], content: str, encoding: str = "utf-8") -> Path:
    """Escribe un archivo de forma atómica (temporal + reemplazo).

    Evita que un corte de luz o un kill-switch a mitad de escritura deje un
    archivo corrupto: o está el contenido nuevo entero, o sigue el anterior.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def atomic_write_bytes(path: str | os.PathLike[str], content: bytes) -> Path:
    """Como `atomic_write_text` pero para contenido binario."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def load_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    """Lee un JSON tolerante a fallos: si algo va mal, devuelve `default`."""
    path = Path(path)
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("No se pudo leer %s: %s", path, exc)
        return default


def save_json(path: str | os.PathLike[str], data: Any, indent: int = 2) -> Path:
    """Guarda JSON con escritura atómica (y acentos legibles, sin escapes)."""
    return atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=indent))


def append_jsonl(path: str | os.PathLike[str], record: dict) -> None:
    """Añade una línea JSON (historial de conversación)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike[str], limit: int | None = None) -> list[dict]:
    """Lee un archivo JSONL saltándose las líneas corruptas."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                clean = line.strip()
                if not clean:
                    continue
                try:
                    records.append(json.loads(clean))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        logger.warning("No se pudo leer %s: %s", path, exc)
        return []
    return records[-limit:] if limit else records


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Crea la carpeta (y sus padres) si no existe y la devuelve."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# EJECUTABLES Y APLICACIONES
# =============================================================================

def find_executable(
    candidates: Sequence[str | os.PathLike[str]],
    verify: bool = False,
) -> str | None:
    """Devuelve el primer ejecutable que exista de una lista de candidatos.

    Acepta rutas completas (se comprueba que el archivo exista) y nombres
    sueltos (se buscan en el PATH con `shutil.which`). El objetivo es que
    "localizar Comet" o "localizar Spotify" sea una única llamada.

    Parameters
    ----------
    verify:
        Si es True, solo se devuelven rutas que además sean archivos normales.
    """
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if not text:
            continue
        expanded = os.path.expandvars(os.path.expanduser(text))
        if os.path.isabs(expanded) or os.sep in expanded:
            if os.path.isfile(expanded):
                return os.path.normpath(expanded)
            continue
        found = shutil.which(expanded)
        if found:
            return found
        if not verify and re.fullmatch(r"[\w.\-]+", expanded):
            # Comando desnudo: puede ser un protocolo (ms-settings:) o resolverse
            # al ejecutarlo. Lo aceptamos como último recurso legítimo.
            continue
    return None


def which_any(names: Iterable[str]) -> str | None:
    """Primer nombre de la lista presente en el PATH."""
    return find_executable(tuple(names))


def human_bytes(size: float) -> str:
    """Tamaño legible en español (KB/MB/GB con coma decimal)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            text = f"{size:.1f}".rstrip("0").rstrip(".")
            return f"{text} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


# =============================================================================
# TEXTO (para voz y para el modelo)
# =============================================================================

_ACCENTS = str.maketrans(
    "áàäâãéèëêíìïîóòöôõúùüûñçÁÀÄÂÃÉÈËÊÍÌÏÎÓÒÖÔÕÚÙÜÛÑÇ",
    "aaaaaeeeeiiiiooooouuuuncAAAAAEEEEIIIIOOOOOUUUUNC",
)


def strip_accents(text: str) -> str:
    """Quita tildes y diéresis conservando el resto de caracteres.

    >>> strip_accents("¿Qué tal, señor?")
    '¿Que tal, senor?'
    """
    return str(text).translate(_ACCENTS)


def normalize_text(text: str) -> str:
    """Normaliza para comparar órdenes: minúsculas, sin tildes ni puntuación.

    >>> normalize_text("  ¡JARVIS, ABRE   Spotify! ")
    'jarvis abre spotify'
    """
    text = strip_accents(str(text)).lower()
    text = re.sub(r"[^a-z0-9ñ\s]", " ", text)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text_map(text: str) -> tuple[str, list[int]]:
    """Igual que `normalize_text`, pero devuelve además el mapa de posiciones.

    El mapa dice de qué carácter del texto **original** viene cada carácter
    normalizado. Es lo que permite localizar una palabra clave ignorando
    tildes y mayúsculas sin perder las tildes del fragmento que se extrae
    después (por ejemplo: "el botón azul" en lugar de "el boton azul").
    """
    raw = str(text)
    plain_chars: list[str] = []
    plain_map: list[int] = []
    for index, char in enumerate(raw):
        for sub in strip_accents(char).lower():
            plain_chars.append(sub)
            plain_map.append(index)
    plain = "".join(plain_chars)

    filtered_chars: list[str] = []
    filtered_map: list[int] = []
    for position, char in enumerate(plain):
        if re.fullmatch(r"[a-z0-9ñ\s]", char):
            filtered_chars.append(char)
            filtered_map.append(plain_map[position])

    out: list[str] = []
    out_map: list[int] = []
    pending_space = False
    for position, char in enumerate(filtered_chars):
        if char.isspace():
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            out_map.append(filtered_map[position])
            pending_space = False
        out.append(char)
        out_map.append(filtered_map[position])
    return "".join(out), out_map


def contains_any(haystack: str, needles: Iterable[str]) -> bool:
    """¿El texto normalizado contiene alguna de las palabras clave?"""
    normalized = normalize_text(haystack)
    return any(normalize_text(n) in normalized for n in needles)


def strip_markdown(text: str) -> str:
    """Limpia markdown para que el TTS lea algo natural (nada de asteriscos)."""
    clean = str(text)
    clean = re.sub(r"```.*?```", " ", clean, flags=re.DOTALL)
    clean = re.sub(r"`([^`]*)`", r"\1", clean)
    clean = re.sub(r"^\s{0,3}#{1,6}\s*", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
    clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
    clean = re.sub(r"__([^_]+)__", r"\1", clean)
    clean = re.sub(r"~~([^~]+)~~", r"\1", clean)
    clean = re.sub(r"^\s*[-*+•]\s+", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"^\s*\d+[.)]\s+", "", clean, flags=re.MULTILINE)
    clean = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", clean)
    clean = re.sub(r"https?://\S+", "", clean)
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    return re.sub(r"\n{3,}", "\n\n", clean).strip()


def split_sentences(text: str, max_chars: int = 180) -> list[str]:
    """Trocea un texto en frases listas para el TTS.

    Respeta la puntuación fuerte (.!?…;) y, si una frase sigue siendo larga,
    la parte por comas. Es la clave de la latencia percibida baja: JARVIS
    empieza a hablar con la primera frase mientras sintetiza el resto.

    >>> split_sentences("Hola, Pablo. ¿Qué tal?")[:2]
    ['Hola, Pablo.', '¿Qué tal?']
    """
    text = strip_markdown(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    raw = re.split(r"(?<=[.!?…])\s+(?=[¿¡A-ZÁÉÍÓÚÑ0-9\"'])", text)
    chunks: list[str] = []
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= max_chars:
            chunks.append(piece)
            continue
        # Frase demasiado larga: se parte por comas, puntos y coma o dos puntos.
        buffer = ""
        for part in re.split(r"(?<=[,;:])\s+", piece):
            if not buffer:
                buffer = part
            elif len(buffer) + len(part) + 1 <= max_chars:
                buffer = f"{buffer} {part}"
            else:
                chunks.append(buffer.strip())
                buffer = part
            while len(buffer) > max_chars:
                cut = buffer.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 2 else max_chars
                chunks.append(buffer[:cut].strip())
                buffer = buffer[cut:].strip()
        if buffer.strip():
            chunks.append(buffer.strip())
    return [c for c in chunks if c]


def truncate(text: str, limit: int = 400, suffix: str = "…") -> str:
    """Recorta un texto largo para los registros (nunca para el usuario)."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def safe_int(value: Any, default: int = 0) -> int:
    """Conversión a entero tolerante a fallos (respuestas de un LLM)."""
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        match = re.search(r"-?\d+", str(value))
        return int(match.group(0)) if match else default
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Conversión a decimal tolerante a fallos (admite coma decimal)."""
    try:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).replace(",", ".").strip()
        match = re.search(r"-?\d+(\.\d+)?", text)
        return float(match.group(0)) if match else default
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    """Limita un valor al intervalo [low, high] (útil con coordenadas)."""
    return max(low, min(high, value))


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Interpreta de forma humana un booleano que puede venir como texto."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "si", "sí", "yes", "y", "verdadero", "on"}


def extract_json(text: str) -> Any:
    """Extrae el primer objeto/array JSON válido de una respuesta de un modelo.

    Los LLM suelen envolver el JSON en bloques de código o añadir texto antes
    y después; esta función rescata la parte útil o devuelve None.

    >>> extract_json('Claro: {"a": 1} ¡listo!')
    {'a': 1}
    """
    if not text:
        return None
    raw = str(text).strip()
    candidates: list[str] = []

    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(raw)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = candidate.find(opener)
            end = candidate.rfind(closer)
            if start != -1 and end > start:
                fragment = candidate[start : end + 1]
                for fixed in (fragment, _repair_json(fragment)):
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        continue
    return None


def _repair_json(fragment: str) -> str:
    """Arreglos mínimos de JSON malformado por un modelo (comas y comillas)."""
    fixed = re.sub(r",\s*([}\]])", r"\1", fragment)
    fixed = re.sub(r"(?<![\\])'([^']*)'(?=\s*:)", r'"\1"', fixed)
    return fixed


# =============================================================================
# CONCURRENCIA
# =============================================================================

class RateLimiter:
    """Limitador de frecuencia por hilo (por ejemplo, la captura de pantalla)."""

    def __init__(self, min_interval_s: float) -> None:
        self.min_interval_s = float(min_interval_s)
        self._last = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """Devuelve True si ha pasado suficiente tiempo desde la última vez."""
        with self._lock:
            now = time.monotonic()
            if now - self._last >= self.min_interval_s:
                self._last = now
                return True
            return False

    def wait(self) -> None:
        """Bloquea el tiempo necesario y deja pasar la siguiente llamada."""
        with self._lock:
            now = time.monotonic()
            remaining = self.min_interval_s - (now - self._last)
            if remaining > 0:
                time.sleep(remaining)
            self._last = time.monotonic()


class Adder:
    """Contador atómico sencillo (métricas de ejecución para el HUD)."""

    def __init__(self, start: int = 0) -> None:
        self._value = start
        self._lock = threading.Lock()

    def inc(self, amount: int = 1) -> int:
        with self._lock:
            self._value += amount
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


def run_in_thread(target: Callable[..., Any], *args: Any, name: str = "", daemon: bool = True, **kwargs: Any) -> threading.Thread:
    """Lanza una función en un hilo con nombre (para que los logs sean legibles)."""
    thread = threading.Thread(target=target, args=args, kwargs=kwargs, name=name or "jarvis-hilo", daemon=daemon)
    thread.start()
    return thread


def platform_tag() -> str:
    """Etiqueta corta de plataforma (para el diagnóstico y los registros)."""
    return f"{sys.platform}-{os.name}"
