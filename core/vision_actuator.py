"""
core/vision_actuator.py — Ojos en la pantalla y manos en el ratón.
================================================================================
JARVIS necesita dos cosas para "usar el ordenador": VER (capturar la pantalla y
entenderla con `llama3.2-vision`) y ACTUAR (mover el ratón, hacer clic y
escribir). Este módulo hace las dos y, sobre todo, **obedece**:

* Antes de cada paso comprueba el **kill-switch**. Si Pablo lo activa, la
  automatización se detiene en seco (`AutomationInterrupted`).
* Antes de cada paso consulta la **puerta de barge-in**. Si Pablo habla, el
  plan se congela y JARVIS espera la corrección.
* Con `ACTUATOR_SAFE_FORESHOT` activo, si el elemento a pulsar ya no está donde
  decía el plan, se vuelve a mirar la pantalla antes de hacer clic.

COORDENADAS
-----------
El modelo de visión devuelve posiciones en una **rejilla normalizada de 0 a
1000** (x=0 izquierda, x=1000 derecha, y=0 arriba, y=1000 abajo). Es mucho más
robusto que pedirle píxeles exactos. Aquí se traducen a píxeles reales de la
pantalla, teniendo en cuenta que la imagen se envía escalada para ir rápido.
================================================================================
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import config as cfg  # noqa: E402
from core.barge_in import AutomationGate, AutomationInterrupted  # noqa: E402
from core.utils import RateLimiter, clamp, ensure_dir, safe_int, timestamp_slug, truncate  # noqa: E402

logger = logging.getLogger("jarvis.vision")


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass
class ScreenShot:
    """Una captura de pantalla lista para enviar al modelo de visión."""

    path: Path
    width: int = 0
    height: int = 0
    scale: float = 1.0
    monitor: int = 0
    timestamp: float = 0.0

    @property
    def real_width(self) -> int:
        return int(round(self.width / self.scale)) if self.scale else self.width

    @property
    def real_height(self) -> int:
        return int(round(self.height / self.scale)) if self.scale else self.height

    def describe(self) -> str:
        return (
            f"captura {self.width}x{self.height} "
            f"(escala {self.scale:.2f}, pantalla real {self.real_width}x{self.real_height})"
        )


@dataclass
class UIElement:
    """Un elemento de la interfaz localizado en la pantalla."""

    nombre: str = ""
    descripcion: str = ""
    x_norm: float = 0.0
    y_norm: float = 0.0
    ancho_norm: float = 0.0
    alto_norm: float = 0.0
    confianza: float = 0.0
    x: int = 0          # Píxeles reales del centro.
    y: int = 0
    ancho: int = 0
    alto: int = 0

    def describe(self) -> str:
        return (
            f"{self.nombre or self.descripcion} en ({self.x}, {self.y}) "
            f"[{self.ancho}x{self.alto}] confianza {self.confianza:.2f}"
        )

    def as_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "descripcion": self.descripcion,
            "x": self.x,
            "y": self.y,
            "ancho": self.ancho,
            "alto": self.alto,
            "confianza": self.confianza,
        }


@dataclass
class StepResult:
    """Resultado de un paso de automatización."""

    accion: str = ""
    exito: bool = False
    detalle: str = ""
    screenshot: str = ""
    duration_s: float = 0.0


@dataclass
class TaskOutcome:
    """Resultado completo de un plan ejecutado."""

    objetivo: str = ""
    exito: bool = False
    motivo: str = ""
    pasos_ejecutados: int = 0
    total_pasos: int = 0
    resultados: list[StepResult] = field(default_factory=list)
    rectificado: bool = False

    def summary(self) -> str:
        estado = "completada" if self.exito else "no completada"
        return (
            f"Tarea {estado}: {self.objetivo} "
            f"({self.pasos_ejecutados}/{self.total_pasos} pasos) {self.motivo}"
        )


# =============================================================================
# 1. CAPTURA DE PANTALLA
# =============================================================================

class ScreenCapture:
    """Captura la pantalla y la prepara para el modelo (rápida y ligera).

    Usa `mss` si está disponible (muy rápido, soporta varios monitores) y cae a
    `PIL.ImageGrab` como reserva. La imagen se escala a `VISION_SCALE_MAX_WIDTH`
    y se guarda en JPEG para que viaje al modelo en milisegundos, no en segundos.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = ensure_dir(output_dir or cfg.VISION_SCREENSHOTS_DIR)
        self._limiter = RateLimiter(0.25)  # Máximo 4 capturas por segundo.
        self.stats = {"capturas": 0, "segundos": 0.0}

    def capture(self, monitor: int = cfg.VISION_MONITOR_INDEX, force: bool = False) -> ScreenShot | None:
        """Captura la pantalla. Devuelve None si no se puede (con aviso en el log)."""
        if not force and not self._limiter.allow():
            logger.debug("Captura limitada por frecuencia; se reutiliza la última.")
        started = time.perf_counter()
        raw = self._grab(monitor)
        if raw is None:
            return None

        image, width, height = raw
        image, scale = self._resize(image, width)
        path = self.output_dir / f"pantalla_{timestamp_slug()}_{int(time.time() * 1000) % 1000:03d}.jpg"
        try:
            image.save(path, format="JPEG", quality=cfg.VISION_JPEG_QUALITY, optimize=True)
        except Exception as exc:
            logger.error("No he podido guardar la captura: %s", exc)
            return None
        width_out, height_out = image.width, image.height

        elapsed = time.perf_counter() - started
        self.stats["capturas"] += 1
        self.stats["segundos"] += elapsed
        shot = ScreenShot(
            path=path,
            width=width_out,
            height=height_out,
            scale=scale,
            monitor=monitor,
            timestamp=time.time(),
        )
        logger.debug("Captura lista: %s en %.3f s", shot.describe(), elapsed)
        self._cleanup_old()
        return shot

    def _grab(self, monitor: int):
        """Obtiene una imagen PIL de la pantalla (mss -> ImageGrab)."""
        try:
            import mss  # type: ignore
            from PIL import Image  # type: ignore

            with mss.mss() as grabber:
                monitors = grabber.monitors
                index = monitor if 0 <= monitor < len(monitors) else 0
                raw = grabber.grab(monitors[index])
                return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX"), raw.width, raw.height
        except ImportError:
            logger.debug("mss no está disponible; se usa PIL.ImageGrab.")
        except Exception as exc:
            logger.warning("La captura con mss falló (%s); se intenta con PIL.", exc)

        try:
            from PIL import ImageGrab  # type: ignore

            image = ImageGrab.grab(all_screens=(monitor == 0))
            return image, image.width, image.height
        except Exception as exc:
            logger.error(
                "No he podido capturar la pantalla: %s\n"
                "  · Comprueba que no estés en una sesión remota o bloqueada.",
                exc,
            )
            return None

    @staticmethod
    def _resize(image, width: int):
        """Escala la imagen si supera el ancho máximo configurado."""
        max_width = cfg.VISION_SCALE_MAX_WIDTH
        if width <= max_width or max_width <= 0:
            return image, 1.0
        scale = max_width / float(width)
        try:
            from PIL import Image  # type: ignore

            new_size = (max_width, max(1, int(image.height * scale)))
            return image.resize(new_size, Image.LANCZOS), scale
        except Exception as exc:  # pragma: no cover
            logger.debug("No se pudo escalar la captura: %s", exc)
            return image, 1.0

    def _cleanup_old(self) -> None:
        """Borra las capturas antiguas para no llenar el disco."""
        keep = cfg.VISION_KEEP_LAST_SCREENSHOTS
        try:
            files = sorted(self.output_dir.glob("pantalla_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in files[keep:]:
                old.unlink(missing_ok=True)
        except OSError:
            pass


# =============================================================================
# 2. PERCEPCIÓN (VISIÓN) — LO QUE JARVIS VE
# =============================================================================

class ScreenPerception:
    """Interpreta la pantalla con `llama3.2-vision` (carga efímera de VRAM)."""

    def __init__(
        self,
        router: Any,
        capture: ScreenCapture | None = None,
        notifier: Callable[[str, str], None] | None = None,
    ) -> None:
        self.router = router
        self.capture = capture or ScreenCapture()
        self.notifier = notifier
        self._last: ScreenShot | None = None
        self.stats = {"descripciones": 0, "localizaciones": 0, "elementos_encontrados": 0}

    def _notify(self, state: str, detail: str = "") -> None:
        if self.notifier is None:
            return
        try:
            self.notifier(state, detail)
        except Exception:  # pragma: no cover
            pass

    @property
    def last_shot(self) -> ScreenShot | None:
        """Última captura realizada (útil para reutilizarla en el mismo turno)."""
        return self._last

    def look(self, monitor: int = cfg.VISION_MONITOR_INDEX) -> ScreenShot | None:
        """Hace una captura y avisa al HUD (se ilumina el marco de neón)."""
        self._notify("vision", "Mirando la pantalla")
        shot = self.capture.capture(monitor)
        if shot is None:
            self._notify("error", "No puedo ver la pantalla")
            return None
        self._last = shot
        return shot

    def describe(self, question: str = "", reuse: bool = False) -> str:
        """Describe lo que hay en pantalla en lenguaje natural (español)."""
        if not cfg.VISION_ENABLED:
            return "La visión está desactivada en la configuración."
        shot = self._last if (reuse and self._last is not None) else self.look()
        if shot is None:
            return ""
        text = self.router.look_at_screen(shot.path, question)
        self.stats["descripciones"] += 1
        self._notify("acting", "")
        return text

    def locate(self, description: str, min_confidence: float = 0.25, reuse: bool = False) -> list[UIElement]:
        """Localiza elementos de interfaz y devuelve sus coordenadas REALES."""
        shot = self._last if (reuse and self._last is not None) else self.look()
        if shot is None:
            return []
        payload = self.router.find_ui_elements(shot.path, description)
        self.stats["localizaciones"] += 1
        elements = [
            element
            for element in (self._to_element(raw, shot) for raw in payload.get("elementos", []))
            if element.confianza >= min_confidence
        ]
        elements.sort(key=lambda e: e.confianza, reverse=True)
        self.stats["elementos_encontrados"] += len(elements)
        if not elements:
            logger.info("No he encontrado en pantalla: %s", truncate(description, 120))
        else:
            logger.info("Elementos encontrados: %s", " | ".join(e.describe() for e in elements[:3]))
        return elements

    @staticmethod
    def _to_element(raw: dict, shot: ScreenShot) -> UIElement:
        """Convierte la respuesta del modelo (rejilla 0-1000) a píxeles reales."""
        x_norm = float(safe_int(raw.get("x", raw.get("centro_x", 0))))
        y_norm = float(safe_int(raw.get("y", raw.get("centro_y", 0))))
        w_norm = float(safe_int(raw.get("ancho", raw.get("width", 0))))
        h_norm = float(safe_int(raw.get("alto", raw.get("height", 0))))

        # Defensa: si el modelo devolvió píxeles de la imagen en lugar de 0-1000.
        if x_norm > 1000 or y_norm > 1000:
            x_px = clamp(x_norm / max(1e-6, shot.scale), 0, shot.real_width - 1)
            y_px = clamp(y_norm / max(1e-6, shot.scale), 0, shot.real_height - 1)
            w_px = w_norm / max(1e-6, shot.scale)
            h_px = h_norm / max(1e-6, shot.scale)
        else:
            x_px = clamp(x_norm / 1000.0 * shot.real_width, 0, shot.real_width - 1)
            y_px = clamp(y_norm / 1000.0 * shot.real_height, 0, shot.real_height - 1)
            w_px = w_norm / 1000.0 * shot.real_width
            h_px = h_norm / 1000.0 * shot.real_height

        return UIElement(
            nombre=str(raw.get("nombre", "")),
            descripcion=str(raw.get("descripcion", "")),
            x_norm=x_norm,
            y_norm=y_norm,
            ancho_norm=w_norm,
            alto_norm=h_norm,
            confianza=clamp(float(raw.get("confianza", 0.0) or 0.0), 0.0, 1.0),
            x=int(round(x_px)),
            y=int(round(y_px)),
            ancho=max(0, int(round(w_px))),
            alto=max(0, int(round(h_px))),
        )

    def status(self) -> dict:
        return {
            "vision_activa": cfg.VISION_ENABLED,
            "ultima_captura": self._last.describe() if self._last else "",
            **self.capture.stats,
            **self.stats,
        }


# =============================================================================
# 3. ACTUADOR — LAS MANOS DE JARVIS
# =============================================================================

class ScreenActuator:
    """Mueve el ratón, hace clic y escribe, siempre con permiso de Pablo.

    Todas las primitivas pasan por tres filtros antes de tocar el sistema:
    el kill-switch, la puerta de barge-in y el aviso al HUD (para que el marco
    de neón se ilumine mientras JARVIS "toca" el ordenador).
    """

    def __init__(
        self,
        gate: AutomationGate | None = None,
        killswitch: Any | None = None,
        notifier: Callable[[str, str], None] | None = None,
        enabled: bool = cfg.ACTUATOR_ENABLED,
    ) -> None:
        self.gate = gate or AutomationGate()
        self.killswitch = killswitch
        self.notifier = notifier
        self.enabled = enabled
        self._pyautogui: Any | None = None
        self.stats = {"clics": 0, "teclas": 0, "movimientos": 0, "pasos_abortados": 0}
        self._foreshot_capture: ScreenCapture | None = None

    # --- Infraestructura ---------------------------------------------------

    @property
    def gui(self):
        """Importa pyautogui de forma perezosa (solo cuando hace falta mover algo)."""
        if self._pyautogui is None:
            try:
                import pyautogui  # type: ignore

                pyautogui.FAILSAFE = cfg.PYAUTOGUI_FAILSAFE
                pyautogui.PAUSE = 0.02
                self._pyautogui = pyautogui
            except ImportError as exc:
                # Se trata como interrupción controlada: JARVIS no puede tocar
                # el ratón, pero el sistema sigue funcionando y lo explica.
                raise AutomationInterrupted(
                    "Falta la librería 'pyautogui' para controlar el ratón y el teclado. "
                    "Ejecuta install.bat."
                ) from exc
        return self._pyautogui

    def _notify(self, state: str, detail: str = "") -> None:
        if self.notifier is None:
            return
        try:
            self.notifier(state, detail)
        except Exception:  # pragma: no cover
            pass

    def _check_permissions(self) -> None:
        """Comprueba que JARVIS tiene permiso para tocar el ratón y el teclado."""
        if not self.enabled:
            raise AutomationInterrupted("La automatización está desactivada en config.py.")
        if self.killswitch is not None:
            if self.killswitch.is_tripped():
                self.stats["pasos_abortados"] += 1
                raise AutomationInterrupted("Kill-switch activado: control devuelto a Pablo.")
        self.gate.wait_before_step(timeout=cfg.ACTUATOR_STEP_TIMEOUT_S)
        if self.killswitch is not None:
            self.killswitch.notify_activity()  # Ignora los movimientos que hacemos nosotros.

    # --- Primitivas --------------------------------------------------------

    def move_to(self, x: int, y: int, duration: float | None = None) -> None:
        """Mueve el ratón con un gesto humano y suave."""
        self._check_permissions()
        self._notify("acting", "Moviendo el ratón")
        self.gui.moveTo(x, y, duration=cfg.ACTUATOR_MOVE_DURATION if duration is None else duration,
                        tween=cfg.ACTUATOR_MOVE_TWEEN)
        self.stats["movimientos"] += 1

    def click(self, x: int | None = None, y: int | None = None, button: str = "left", clicks: int = 1) -> None:
        """Hace clic (moviéndose antes si se indican coordenadas)."""
        self._check_permissions()
        if x is not None and y is not None:
            self.move_to(x, y)
        time.sleep(cfg.ACTUATOR_CLICK_DELAY)
        self.gui.click(button=button, clicks=clicks)
        self.stats["clics"] += 1
        self._notify("acting", "Clic")
        time.sleep(cfg.ACTUATOR_POST_CLICK_DELAY)

    def double_click(self, x: int | None = None, y: int | None = None) -> None:
        self.click(x, y, clicks=2)

    def right_click(self, x: int | None = None, y: int | None = None) -> None:
        self.click(x, y, button="right")

    def write(self, text: str, interval: float | None = None) -> None:
        """Escribe texto. Si tiene acentos o caracteres raros, usa el portapapeles."""
        self._check_permissions()
        self._notify("acting", "Escribiendo")
        interval = cfg.ACTUATOR_TYPE_INTERVAL if interval is None else interval
        if any(ord(ch) > 126 for ch in text):
            if self._write_via_clipboard(text):
                self.stats["teclas"] += len(text)
                return
        self.gui.write(text, interval=interval)
        self.stats["teclas"] += len(text)

    def _write_via_clipboard(self, text: str) -> bool:
        """Alternativa para acentos y símbolos: copiar, pegar y restaurar."""
        try:
            import pyperclip  # type: ignore

            previous = ""
            try:
                previous = pyperclip.paste()
            except Exception:
                previous = ""
            pyperclip.copy(text)
            time.sleep(0.05)
            self.gui.hotkey("ctrl", "v")
            time.sleep(0.1)
            if previous:
                try:
                    pyperclip.copy(previous)
                except Exception:
                    pass
            return True
        except Exception as exc:
            logger.debug("Portapapeles no disponible (%s); se escribe directamente.", exc)
            return False

    def press(self, keys: str | Sequence[str]) -> None:
        """Pulsa una tecla o una combinación (p. ej. "ctrl+s" o ["alt", "f4"])."""
        self._check_permissions()
        self._notify("acting", "Pulsando teclas")
        if isinstance(keys, str) and ("+" in keys or "," in keys):
            parts = [p.strip() for p in keys.replace(",", "+").split("+") if p.strip()]
            self.gui.hotkey(*parts)
            self.stats["teclas"] += len(parts)
        elif isinstance(keys, (list, tuple)):
            self.gui.hotkey(*[str(k) for k in keys])
            self.stats["teclas"] += len(keys)
        else:
            self.gui.press(str(keys))
            self.stats["teclas"] += 1

    def scroll(self, amount: int = cfg.ACTUATOR_SCROLL_CLICKS, x: int | None = None, y: int | None = None) -> None:
        """Desplaza la rueda del ratón (positivo = arriba)."""
        self._check_permissions()
        if x is not None and y is not None:
            self.move_to(x, y)
        self.gui.scroll(int(amount))
        self._notify("acting", "Desplazando")

    def screenshot(self) -> ScreenShot | None:
        """Captura la pantalla (ilumina el marco del HUD mientras lo hace)."""
        self._notify("vision", "Capturando pantalla")
        if self._foreshot_capture is None:
            self._foreshot_capture = ScreenCapture()
        return self._foreshot_capture.capture(force=True)

    def position(self) -> tuple[int, int]:
        """Posición actual del ratón."""
        try:
            x, y = self.gui.position()
            return int(x), int(y)
        except Exception:
            return (0, 0)

    def status(self) -> dict:
        return {"activo": self.enabled, "congelado": self.gate.frozen, **self.stats}


# =============================================================================
# 4. EJECUTOR DE PLANES (CON RECTIFICACIÓN EN VUELO)
# =============================================================================

class VisionExecutor:
    """Ejecuta los planes de JARVIS sobre la pantalla, paso a paso.

    Se encarga de traducir el JSON del planificador a acciones reales y de
    aplicar la **rectificación en vuelo**: si Pablo interrumpe ("no, detente,
    haz clic en el otro botón"), el plan actual se cancela y se genera uno
    nuevo con la corrección y una mirada fresca a la pantalla.
    """

    SUPPORTED_ACTIONS = {
        "escribir_texto",
        "pulsar_teclas",
        "clic_elemento",
        "mover_raton",
        "desplazar",
        "captura_pantalla",
        "describir_pantalla",
    }

    def __init__(
        self,
        perception: ScreenPerception,
        actuator: ScreenActuator,
        router: Any | None = None,
        voice: Any | None = None,
        notifier: Callable[[str, str], None] | None = None,
    ) -> None:
        self.perception = perception
        self.actuator = actuator
        self.router = router or getattr(perception, "router", None)
        self.voice = voice
        self.notifier = notifier
        self.current_plan: dict | None = None
        self.last_outcome: TaskOutcome | None = None

    # --- Utilidades --------------------------------------------------------

    def _notify(self, state: str, detail: str = "") -> None:
        if self.notifier is None:
            return
        try:
            self.notifier(state, detail)
        except Exception:  # pragma: no cover
            pass

    def _note_automation(self, active: bool) -> None:
        """Avisa al motor de voz de que empieza o acaba una automatización."""
        if self.voice is not None:
            try:
                self.voice.note_automation(active)
            except Exception:  # pragma: no cover
                pass

    # --- Ejecución ---------------------------------------------------------

    def execute(self, plan: dict, speak: Callable[[str], None] | None = None) -> TaskOutcome:
        """Ejecuta un plan completo (lista de pasos con acción y argumentos)."""
        steps = list(plan.get("pasos") or [])
        outcome = TaskOutcome(
            objetivo=str(plan.get("objetivo", "")),
            total_pasos=len(steps),
        )
        self.current_plan = plan

        if not steps:
            outcome.exito = True
            outcome.motivo = "No había nada que ejecutar."
            self.last_outcome = outcome
            return outcome

        self.actuator.gate.clear()
        self._note_automation(True)
        self._notify("acting", outcome.objetivo or "Ejecutando")

        try:
            for index, step in enumerate(steps, start=1):
                if index > cfg.ACTUATOR_MAX_STEPS_PER_TASK:
                    outcome.motivo = f"Demasiados pasos (máximo {cfg.ACTUATOR_MAX_STEPS_PER_TASK})."
                    break
                action = str(step.get("accion", "")).strip().lower()
                args = step.get("argumentos") or {}
                if action not in self.SUPPORTED_ACTIONS:
                    outcome.resultados.append(
                        StepResult(accion=action, exito=False, detalle="Acción no soportada por el actuador; la gestiona el cerebro.")
                    )
                    outcome.pasos_ejecutados += 1
                    continue

                result = self._run_step(action, args, index)
                outcome.resultados.append(result)
                outcome.pasos_ejecutados += 1
                if not result.exito:
                    outcome.motivo = result.detalle
                    break
            else:
                outcome.exito = True
                outcome.motivo = "Todos los pasos se ejecutaron."
        except AutomationInterrupted as exc:
            outcome.motivo = str(exc)
            logger.warning("Automatización interrumpida: %s", exc)
            if speak is not None:
                speak("De acuerdo, Pablo. Me detengo.")
        except Exception as exc:  # pragma: no cover - depende del sistema
            outcome.motivo = f"Error inesperado: {exc}"
            logger.exception("Error ejecutando el plan")
        finally:
            self._note_automation(False)
            self._notify("idle", "")
            self.last_outcome = outcome

        logger.info(outcome.summary())
        return outcome

    def _run_step(self, action: str, args: dict, index: int) -> StepResult:
        """Ejecuta un paso concreto con su tiempo y su resultado."""
        started = time.perf_counter()
        result = StepResult(accion=action)
        try:
            if action == "escribir_texto":
                text = str(args.get("texto", args.get("text", "")))
                if not text:
                    raise ValueError("No hay texto que escribir.")
                self.actuator.write(text)

            elif action == "pulsar_teclas":
                keys = args.get("teclas", args.get("keys", ""))
                if not keys:
                    raise ValueError("No hay teclas que pulsar.")
                self.actuator.press(keys)

            elif action == "mover_raton":
                self.actuator.move_to(safe_int(args.get("x", 0)), safe_int(args.get("y", 0)))

            elif action == "desplazar":
                self.actuator.scroll(safe_int(args.get("cantidad", args.get("amount", cfg.ACTUATOR_SCROLL_CLICKS))))

            elif action == "captura_pantalla":
                shot = self.actuator.screenshot()
                result.screenshot = str(shot.path) if shot else ""
                if shot is None:
                    raise RuntimeError("No he podido capturar la pantalla.")

            elif action == "describir_pantalla":
                question = str(args.get("pregunta", args.get("question", "")))
                result.detalle = self.perception.describe(question)

            elif action == "clic_elemento":
                description = str(
                    args.get("descripcion", args.get("elemento", args.get("objetivo", "")))
                ).strip()
                if not description:
                    raise ValueError("No me has dicho en qué elemento hacer clic.")
                elements = self.perception.locate(description)
                if not elements:
                    raise RuntimeError(f"No encuentro '{description}' en la pantalla.")
                target = elements[0]
                if cfg.ACTUATOR_SAFE_FORESHOT:
                    target = self._verify_before_click(description, target)
                self.actuator.click(target.x, target.y, clicks=safe_int(args.get("clics", 1)) or 1)
                result.detalle = target.describe()

            result.exito = True
            if not result.detalle:
                result.detalle = f"{action} completado"
            logger.info("Paso %d (%s): %s", index, action, result.detalle)
        except AutomationInterrupted:
            raise
        except Exception as exc:
            result.exito = False
            result.detalle = f"{action}: {exc}"
            logger.warning("Paso %d (%s) falló: %s", index, action, exc)
        result.duration_s = time.perf_counter() - started
        return result

    def _verify_before_click(self, description: str, target: UIElement) -> UIElement:
        """Re-verifica sobre una captura nueva que el elemento sigue ahí.

        Es la "garantía de tiro": antes de pulsar, JARVIS vuelve a mirar. Si la
        ventana se ha movido, se corrige el punto de clic; si el elemento ha
        desaparecido, se usa la posición anterior (probablemente la interfaz
        simplemente tardó en repintarse).
        """
        fresh = self.perception.locate(description, reuse=False)
        if not fresh:
            return target
        best = fresh[0]
        moved = abs(best.x - target.x) + abs(best.y - target.y)
        if moved > 6:
            logger.debug("El elemento se ha movido %d px; se usa la posición nueva.", moved)
        return best

    # --- Rectificación en vuelo -------------------------------------------

    def rectify(self, correction: str, speak: Callable[[str], None] | None = None) -> TaskOutcome:
        """Corrige el plan en curso con la orden de voz de Pablo.

        1. Congela la automatización (si no lo estaba ya).
        2. Mira la pantalla de nuevo.
        3. Pide un plan nuevo al modelo con la corrección y el contexto visual.
        4. Descarta el plan viejo y ejecuta el nuevo.
        """
        if self.router is None:
            return TaskOutcome(objetivo=correction, exito=False, motivo="No hay enrutador de modelos disponible.")

        self.actuator.gate.freeze("corrección por voz")
        logger.info("Rectificando el plan en vuelo: %s", truncate(correction, 160))

        context = ""
        try:
            shot = self.perception.look()
            if shot is not None:
                context = self.perception.describe(
                    "Describe la aplicación activa y los elementos de interfaz visibles "
                    "que sean relevantes para esta corrección que acaba de pedir el usuario: "
                    f"{correction}",
                    reuse=True,
                )
        except Exception as exc:
            logger.debug("No se pudo refrescar el contexto visual: %s", exc)

        objetivo_original = str((self.current_plan or {}).get("objetivo", ""))
        order = (
            f"Nuevo plan que sustituye al anterior.\n"
            f"Objetivo original: {objetivo_original}\n"
            f"Corrección de Pablo: {correction}"
        )
        new_plan = self.router.plan(order, screen_context=context)
        self.actuator.gate.thaw()
        outcome = self.execute(new_plan, speak=speak)
        outcome.rectificado = True
        return outcome

    def describe_screen(self, question: str = "") -> str:
        """Azúcar sintáctico: mira y describe (lo usa el cerebro para responder)."""
        return self.perception.describe(question)

    def find(self, description: str) -> list[UIElement]:
        """Azúcar sintáctico: localiza elementos y los devuelve."""
        return self.perception.locate(description)

    def status(self) -> dict:
        return {
            "percepcion": self.perception.status(),
            "actuador": self.actuator.status(),
            "plan_actual": truncate(str((self.current_plan or {}).get("objetivo", "")), 80),
            "interrumpible": self.actuator.gate.status(),
        }
