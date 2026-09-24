"""
core/model_router.py — Enrutador dinámico de modelos y LEY DE MONOGAMIA DE VRAM.
================================================================================
JARVIS tiene tres cerebros y una sola tarjeta gráfica de 8 GB. Este módulo es
el que decide, en cada momento, cuál de ellos está despierto y garantiza que
NUNCA haya dos modelos grandes ocupando la GPU a la vez.

MATRIZ DE DISTRIBUCIÓN (lo que pide Pablo -> lo que se carga)
-------------------------------------------------------------
| Orden de Pablo                                    | Modelo                |
|---------------------------------------------------|-----------------------|
| Conversar, planificar, decidir qué habilidad usar | llama3.1:8b           |
| Mirar la pantalla, encontrar botones y campos     | llama3.2-vision:latest|
| Programar y construir aplicaciones o webs         | qwen2.5-coder:7b      |
| Despertar con la palabra "Jarvis"                 | openWakeWord (CPU)    |
| Transcribir y hablar                              | faster-whisper + Kokoro |

CÓMO SE CUMPLE LA LEY DE MONOGAMIA
----------------------------------
1. Todas las inferencias pasan por un **cerrojo único** (`_gpu_lock`): nunca
   hay dos generaciones simultáneas, aunque dos módulos las pidan a la vez.
2. Antes de cargar un modelo, se ejecuta `client.unload_all()`: se expulsa lo
   que hubiera con `keep_alive: 0` y **se espera** a que Ollama lo confirme.
3. Los modelos de visión son **efímeros**: se cargan, se usan y se descargan
   en la misma operación (`keep_alive: 0`), devolviendo la VRAM al instante.
4. Tras cargar, se mide la VRAM ocupada. Si supera el presupuesto
   (`VRAM_SAFE_BUDGET_GB`), se descarga TODO y la tarea se rechaza con un
   mensaje claro en vez de arrastrar al sistema a un intercambio en disco.
5. La transcripción (Whisper) y la voz (Kokoro) viven en la CPU: la VRAM es
   un recurso exclusivo de los modelos de lenguaje.
================================================================================
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import config as cfg  # noqa: E402
from core.ollama_client import (  # noqa: E402
    ChatResult,
    OllamaClient,
    OllamaError,
    OllamaModelMissingError,
    OllamaNotRunningError,
)
from core.utils import extract_json, truncate  # noqa: E402

logger = logging.getLogger("jarvis.router")


# =============================================================================
# ERRORES ESPECÍFICOS
# =============================================================================

class RouterError(RuntimeError):
    """Error genérico del enrutador."""


class VRAMOverflowError(RouterError):
    """Cargar el modelo superaría el presupuesto de VRAM seguro."""

    def __init__(self, model: str, used_gb: float, budget_gb: float) -> None:
        super().__init__(
            f"Monogamia de VRAM: '{model}' ocupa {used_gb:.2f} GB y el presupuesto "
            f"seguro es {budget_gb:.2f} GB. Se ha descargado todo para proteger el equipo.\n"
            "  · Prueba con un modelo más pequeño (por ejemplo llama3.2:3b) o\n"
            "  · reduce `num_ctx` en config.py (menos contexto = menos VRAM)."
        )
        self.model = model
        self.used_gb = used_gb
        self.budget_gb = budget_gb


# =============================================================================
# INFORMES
# =============================================================================

@dataclass
class SwapReport:
    """Resultado del cambio de modelo (lo que se descargó y lo que se cargó)."""

    model: str = ""
    swapped: bool = False
    unloaded: list[str] = field(default_factory=list)
    wait_s: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = cfg.VRAM_TOTAL_GB
    ephemeral: bool = False
    note: str = ""

    @property
    def headroom_gb(self) -> float:
        return max(0.0, self.vram_total_gb - self.vram_used_gb)

    def describe(self) -> str:
        if not self.swapped:
            return f"Sin cambio de modelo ({self.model} ya estaba listo)."
        freed = ", ".join(self.unloaded) if self.unloaded else "nada"
        extra = f" [{self.note}]" if self.note else ""
        return (
            f"Cambio de modelo -> {self.model} | descargado: {freed} | "
            f"VRAM ocupada: {self.vram_used_gb:.2f} GB en {self.wait_s:.1f} s{extra}"
        )


@dataclass
class InferenceReport:
    """Todo lo que ha pasado en una inferencia (texto, modelo y coste)."""

    task: str = ""
    model: str = ""
    text: str = ""
    swap: SwapReport = field(default_factory=SwapReport)
    duration_s: float = 0.0
    output_tokens: int = 0
    aborted: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and not self.aborted

    def describe(self) -> str:
        return (
            f"[{self.task}] {self.model} | {self.output_tokens} tokens | "
            f"{self.duration_s:.2f} s | VRAM {self.swap.vram_used_gb:.2f} GB"
        )


# =============================================================================
# ENRUTADOR
# =============================================================================

class ModelRouter:
    """Decide qué modelo responde a cada orden y respeta la monogamia de VRAM.

    Parameters
    ----------
    client:
        Cliente de Ollama. Si no se pasa, se crea uno.
    killswitch:
        Interruptor de emergencia. Se consulta entre fragmentos de inferencia.
    notifier:
        Función `(estado, detalle)` para el HUD ("swapping", "thinking"...).
    auto_pull:
        Si un modelo falta, intentar descargarlo automáticamente (la primera
        vez tarda, pero deja el sistema listo sin intervención de Pablo).
    """

    def __init__(
        self,
        client: OllamaClient | None = None,
        killswitch: Any | None = None,
        notifier: Callable[[str, str], None] | None = None,
        auto_pull: bool = True,
    ) -> None:
        self.killswitch = killswitch
        self.notifier = notifier
        self.auto_pull = auto_pull
        self._gpu_lock = threading.RLock()          # El corazón de la monogamia.
        self.client = client or OllamaClient(
            cancel_check=self._cancelled,
        )
        self._stats: dict[str, dict[str, float]] = {}
        self._current_model: str = ""
        self._swaps = 0

    # --- Infraestructura ---------------------------------------------------

    def _cancelled(self) -> bool:
        return bool(self.killswitch is not None and self.killswitch.is_tripped())

    def _notify(self, state: str, detail: str = "") -> None:
        if self.notifier is None:
            return
        try:
            self.notifier(state, detail)
        except Exception as exc:  # la interfaz nunca tumba al enrutador
            logger.debug("Notificador del HUD falló: %s", exc)

    def _record(self, model: str, seconds: float, tokens: int) -> None:
        entry = self._stats.setdefault(model, {"llamadas": 0, "segundos": 0.0, "tokens": 0})
        entry["llamadas"] += 1
        entry["segundos"] += seconds
        entry["tokens"] += tokens

    # --- Enrutado ----------------------------------------------------------

    @staticmethod
    def model_for_task(task: str) -> str:
        """Modelo asignado a una tarea lógica de la matriz de distribución."""
        return cfg.TASK_ROUTES.get(str(task).strip().lower(), cfg.MODEL_CHAT)

    def resolve(self, task: str | None = None, model: str | None = None) -> str:
        """Nombre final del modelo a usar (tarea o modelo explícito)."""
        if model:
            return self.client.resolve_model_name(model)
        return self.client.resolve_model_name(self.model_for_task(task or cfg.TASK_CHAT))

    @property
    def current_model(self) -> str:
        """Modelo residente en la GPU según el propio Ollama."""
        loaded = self.client.loaded_names()
        self._current_model = loaded[0] if loaded else ""
        return self._current_model

    def status(self) -> dict:
        """Estado completo para el HUD y el diagnóstico."""
        try:
            loaded = self.client.ps()
        except OllamaError:
            loaded = []
        return {
            "modelo_actual": loaded[0].name if loaded else "",
            "cargados": [
                {"nombre": m.name, "vram_gb": round(m.vram_gb, 2)} for m in loaded
            ],
            "vram_ocupada_gb": round(self.client.vram_in_use_gb(), 2),
            "vram_presupuesto_gb": cfg.VRAM_SAFE_BUDGET_GB,
            "cambios_de_modelo": self._swaps,
            "estadisticas": self._stats,
            "matriz": {
                "conversacion": cfg.MODEL_CHAT,
                "vision": cfg.MODEL_VISION,
                "programacion": cfg.MODEL_CODER,
            },
        }

    # --- Ley de monogamia de VRAM -----------------------------------------

    def available_models(self) -> list[str]:
        """Modelos descargados en el disco."""
        try:
            return [str(m.get("name", "")) for m in self.client.list_models()]
        except OllamaError:
            return []

    def ensure_model(
        self,
        task: str | None = None,
        model: str | None = None,
        ephemeral: bool | None = None,
    ) -> SwapReport:
        """Deja la GPU con EXACTAMENTE un modelo cargado: el que toca ahora.

        Pasos (siempre en este orden):
          1. ¿Está Ollama despierto?
          2. ¿Está descargado el modelo? Si no, se descarga (si `auto_pull`).
          3. ¿Qué hay cargado ahora? Si ya es solo el modelo objetivo, se sale.
          4. Se descargan TODOS los modelos y se espera confirmación de Ollama.
          5. Se carga el objetivo (calentamiento) y se mide la VRAM.
          6. Si se pasa del presupuesto seguro, se descarga todo y se avisa.
        """
        target = self.resolve(task, model)
        is_ephemeral = (target in cfg.EPHEMERAL_MODELS) if ephemeral is None else bool(ephemeral)
        report = SwapReport(model=target, ephemeral=is_ephemeral)

        if not self.client.health():
            raise OllamaNotRunningError(self.client.host)

        if not self.client.has_model(target):
            if not self.auto_pull:
                raise OllamaModelMissingError(target)
            logger.warning("El modelo '%s' no está descargado. Descargándolo ahora...", target)
            self._notify("swapping", f"Descargando {target}")
            if not self.client.pull(target):
                raise OllamaModelMissingError(target)

        started = time.perf_counter()

        # --- ¿Hace falta cambiar algo? -------------------------------------
        loaded = self.client.ps()
        loaded_names = [m.name for m in loaded]
        if loaded_names == [target]:
            report.vram_used_gb = sum(m.vram_gb for m in loaded)
            report.note = "ya estaba cargado"
            self._current_model = target
            logger.debug("VRAM: '%s' ya era el único modelo residente.", target)
            return report

        if loaded_names:
            logger.info("Monogamia de VRAM: liberando %s", ", ".join(loaded_names))
            self._notify("swapping", "Liberando VRAM")
        report.unloaded = self.client.unload_all()
        if report.unloaded:
            report.swapped = True

        # --- Carga del modelo objetivo -------------------------------------
        if cfg.ENFORCE_VRAM_MONOGAMY and is_ephemeral:
            logger.info("Carga efímera de '%s': se descargará al terminar la inferencia.", target)

        try:
            self.client.warmup(target)
        except OllamaError as exc:
            logger.error("No se pudo cargar '%s': %s", target, exc)
            report.note = f"error al cargar: {exc}"
            raise

        # Confirma que no quedó ningún intruso en la GPU.
        leftovers = [name for name in self.client.loaded_names() if name != target]
        if leftovers:
            logger.warning("Monogamia de VRAM: quedaban %s; se expulsan.", leftovers)
            for name in leftovers:
                self.client.unload(name)
            report.unloaded.extend(leftovers)

        report.swapped = True
        report.vram_used_gb = self.client.vram_in_use_gb()
        report.wait_s = time.perf_counter() - started
        self._current_model = target
        self._swaps += 1
        logger.info(report.describe())

        if report.vram_used_gb > cfg.VRAM_SAFE_BUDGET_GB:
            self.client.unload_all()
            raise VRAMOverflowError(target, report.vram_used_gb, cfg.VRAM_SAFE_BUDGET_GB)
        return report

    def unload_all(self) -> list[str]:
        """Descarga todos los modelos (modo reposo: VRAM al mínimo)."""
        with self._gpu_lock:
            freed = self.client.unload_all()
            if freed:
                logger.info("VRAM liberada por completo: %s", ", ".join(freed))
            self._current_model = ""
            return freed

    @contextmanager
    def exclusive_gpu(self, task: str | None = None, model: str | None = None) -> Iterator[SwapReport]:
        """Bloquea la GPU para una tarea y garantiza la limpieza al salir.

        > with router.exclusive_gpu(cfg.TASK_CHAT) as report:
        >     ...usar el modelo sabiendo que nadie más tocará la GPU...
        """
        with self._gpu_lock:
            report = self.ensure_model(task, model)
            try:
                yield report
            finally:
                if report.ephemeral:
                    self.client.unload(report.model)

    # --- Inferencia --------------------------------------------------------

    def infer(
        self,
        task: str | None = None,
        model: str | None = None,
        prompt: str | None = None,
        messages: Sequence[dict] | None = None,
        images: Sequence[str | Path] | None = None,
        system: str | None = None,
        json_mode: bool = False,
        on_token: Callable[[str], None] | None = None,
        stream: bool = False,
        ephemeral: bool | None = None,
        extra_options: dict | None = None,
        keep_alive: int | str | None = None,
    ) -> InferenceReport:
        """Ejecuta una inferencia completa respetando la monogamia de VRAM.

        Es el único punto de entrada al modelo: se encarga del cambio de modelo,
        de las opciones correctas según la tarea, del kill-switch y de descargar
        los modelos efímeros al terminar.
        """
        task_key = (task or cfg.TASK_CHAT).strip().lower()
        target = self.resolve(task_key, model)
        is_ephemeral = (
            (target in cfg.EPHEMERAL_MODELS) if ephemeral is None else bool(ephemeral)
        )
        options = dict(cfg.TASK_OPTIONS.get(task_key, cfg.TASK_OPTIONS[cfg.TASK_CHAT]))
        if extra_options:
            options.update(extra_options)
        if keep_alive is None:
            keep_alive = cfg.KEEP_ALIVE_VISION if is_ephemeral else self._default_keep_alive(target)

        if self._cancelled():
            return InferenceReport(
                task=task_key, model=target, aborted=True, error="kill-switch activado"
            )

        self._notify("swapping" if is_ephemeral else "thinking", target)
        started = time.perf_counter()
        report = InferenceReport(task=task_key, model=target)

        with self._gpu_lock:
            try:
                report.swap = self.ensure_model(task_key, target, ephemeral=is_ephemeral)
            except (OllamaError, RouterError) as exc:
                report.error = str(exc)
                logger.error("Enrutador: %s", exc)
                return report

            self._notify("thinking", target)
            try:
                if images:
                    result: ChatResult = self.client.generate(
                        model=target,
                        prompt=prompt or "",
                        images=images,
                        options=options,
                        keep_alive=keep_alive,
                        system=system,
                        format_json=json_mode,
                        stream=stream,
                    )
                else:
                    conversation = list(messages) if messages else [
                        {"role": "user", "content": prompt or ""}
                    ]
                    if system:
                        conversation = [{"role": "system", "content": system}] + conversation
                    result = self.client.chat(
                        model=target,
                        messages=conversation,
                        options=options,
                        keep_alive=keep_alive,
                        format_json=json_mode,
                        stream=stream,
                        on_token=on_token,
                    )
            except OllamaModelMissingError as exc:
                report.error = str(exc)
                return report
            except OllamaError as exc:
                report.error = str(exc)
                logger.error("Fallo de inferencia (%s): %s", task_key, exc)
                return report
            finally:
                if is_ephemeral:
                    self._release_ephemeral(target)

        report.text = result.text
        report.aborted = result.aborted
        report.output_tokens = result.output_tokens
        report.duration_s = time.perf_counter() - started
        self._record(target, report.duration_s, result.output_tokens)
        return report

    def _default_keep_alive(self, model: str) -> int | str:
        """Tiempo que el modelo recién cargado se queda en memoria."""
        if model == cfg.MODEL_CHAT:
            return cfg.KEEP_ALIVE_CHAT
        if model == cfg.MODEL_CODER:
            return cfg.KEEP_ALIVE_CODER
        return cfg.KEEP_ALIVE_CHAT

    def _release_ephemeral(self, model: str) -> None:
        """Devuelve la VRAM inmediatamente tras usar un modelo efímero."""
        logger.debug("Liberando modelo efímero '%s'.", model)
        self.client.unload(model, wait=True)
        self._notify("idle", "VRAM libre")

    # --- Atajos de alto nivel ---------------------------------------------

    def chat(
        self,
        prompt: str,
        system: str | None = cfg.SYSTEM_PROMPT_CHAT,
        history: Sequence[dict] | None = None,
        on_token: Callable[[str], None] | None = None,
        stream: bool = True,
    ) -> str:
        """Conversación normal (llama3.1:8b). Devuelve el texto de la respuesta."""
        messages = list(history or []) + [{"role": "user", "content": prompt}]
        report = self.infer(
            task=cfg.TASK_CHAT,
            messages=messages,
            system=system,
            on_token=on_token,
            stream=stream,
        )
        return report.text

    def classify_intent(self, text: str) -> dict:
        """Clasifica una orden hablada en JSON (modelo de routing, temperatura 0)."""
        prompt = (
            "Clasifica la orden del usuario y responde SOLO con JSON válido con las claves:\n"
            '{"intencion": "<conversar|abrir_app|abrir_web|buscar|escribir|automatizar|'
            'musica|hora|fecha|crear_app|crear_web|programar|sistema|desconocida>", '
            '"parametros": {...}, "confianza": 0.0}\n\n'
            f"Orden: {text}"
        )
        report = self.infer(task=cfg.TASK_ROUTING, prompt=prompt, json_mode=True)
        parsed = extract_json(report.text) if report.ok else None
        if isinstance(parsed, dict):
            parsed.setdefault("intencion", "desconocida")
            parsed.setdefault("confianza", 0.0)
            return parsed
        return {"intencion": "desconocida", "parametros": {}, "confianza": 0.0, "crudo": report.text}

    def plan(self, order: str, screen_context: str = "") -> dict:
        """Convierte una orden en pasos ejecutables (llama3.1:8b, JSON estricto)."""
        context = f"\n\nContexto de la pantalla: {screen_context}" if screen_context else ""
        report = self.infer(
            task=cfg.TASK_PLANNING,
            prompt=f"Orden de Pablo: {order}{context}",
            system=cfg.SYSTEM_PROMPT_PLANNER,
            json_mode=True,
        )
        parsed = extract_json(report.text) if report.ok else None
        if isinstance(parsed, dict):
            parsed.setdefault("objetivo", order)
            parsed.setdefault("pasos", [])
            parsed.setdefault("confianza", 0.5)
            return parsed
        logger.warning("El plan no era JSON válido: %s", truncate(report.text, 200))
        return {"objetivo": order, "pasos": [], "confianza": 0.0, "crudo": report.text}

    def look_at_screen(self, image_path: str | Path, question: str = "") -> str:
        """Describe lo que hay en pantalla con llama3.2-vision (carga efímera)."""
        prompt = question or (
            "Describe qué aplicación está abierta, qué se ve y cualquier detalle "
            "relevante para ayudar a Pablo. Responde en español y en pocas frases."
        )
        report = self.infer(
            task=cfg.TASK_SCREEN_PERCEPTION,
            prompt=prompt,
            images=[image_path],
            system=cfg.SYSTEM_PROMPT_VISION_DESCRIBE,
            ephemeral=True,
        )
        return report.text if report.ok else ""

    def find_ui_elements(self, image_path: str | Path, description: str) -> dict:
        """Localiza elementos de interfaz en una captura (JSON normalizado 0-1000)."""
        prompt = (
            "Localiza en la captura los elementos descritos a continuación y devuelve "
            f"su posición en la rejilla 0-1000.\nElementos buscados: {description}"
        )
        report = self.infer(
            task=cfg.TASK_UI_DETECTION,
            prompt=prompt,
            images=[image_path],
            system=cfg.SYSTEM_PROMPT_VISION_UI,
            json_mode=True,
            ephemeral=True,
        )
        parsed = extract_json(report.text) if report.ok else None
        if isinstance(parsed, dict):
            elements = parsed.get("elementos")
            if isinstance(elements, list):
                parsed["elementos"] = elements[: cfg.VISION_MAX_ELEMENTS]
            else:
                parsed["elementos"] = []
            return parsed
        logger.warning("La detección de UI no devolvió JSON: %s", truncate(report.text, 200))
        return {"elementos": [], "encontrado": False, "crudo": report.text}

    def write_code(self, instruction: str, context: str = "", json_mode: bool = False) -> str:
        """Genera código con qwen2.5-coder:7b."""
        prompt = instruction
        if context:
            prompt = f"{instruction}\n\nContexto del proyecto:\n{context}"
        report = self.infer(
            task=cfg.TASK_CODE,
            prompt=prompt,
            system=cfg.SYSTEM_PROMPT_CODER,
            json_mode=json_mode,
        )
        return report.text

    def summarize(self, text: str, max_words: int = 60) -> str:
        """Resume un texto largo (para registros y memoria de conversación)."""
        report = self.infer(
            task=cfg.TASK_SUMMARY,
            prompt=f"Resume en español, en menos de {max_words} palabras:\n\n{text}",
        )
        return report.text

    # --- Diagnóstico -------------------------------------------------------

    def preflight(self) -> dict:
        """Comprobaciones previas al arranque, con mensajes para Pablo."""
        info: dict[str, Any] = {
            "ollama_activo": False,
            "version_ollama": None,
            "modelos_disponibles": [],
            "modelos_que_faltan": [],
            "vram_ocupada_gb": 0.0,
            "presupuesto_vram_gb": cfg.VRAM_SAFE_BUDGET_GB,
            "avisos": [],
        }
        if not self.client.health():
            info["avisos"].append(
                "Ollama no responde. Ábrelo (o ejecuta 'ollama serve') y vuelve a intentarlo."
            )
            return info

        info["ollama_activo"] = True
        info["version_ollama"] = self.client.version()
        info["modelos_disponibles"] = self.available_models()
        info["modelos_que_faltan"] = self.client.missing_models()
        info["vram_ocupada_gb"] = round(self.client.vram_in_use_gb(), 2)

        for missing in info["modelos_que_faltan"]:
            info["avisos"].append(f"Falta el modelo '{missing}'. Se descargará automáticamente.")
        if len(self.client.ps()) > 1:
            info["avisos"].append(
                "Hay varios modelos cargados a la vez: se descargarán para cumplir la monogamia de VRAM."
            )
        return info

    def enforce_monogamy_now(self) -> list[str]:
        """Aplica la ley ahora mismo: deja la GPU con un único modelo (o ninguna).

        Se llama al arrancar y tras cualquier operación dudosa.
        """
        with self._gpu_lock:
            loaded = self.client.ps()
            if len(loaded) <= 1:
                return []
            keep = loaded[0].name if loaded else None
            freed = self.client.unload_all(keep=keep)
            logger.info("Monogamia forzada: %s descargado(s).", len(freed))
            return freed

    def close(self) -> None:
        """Suelta la GPU y cierra el cliente HTTP."""
        try:
            self.unload_all()
        finally:
            self.client.close()
