"""
core/barge_in.py — Interrupción en vuelo (BARGE-IN).
================================================================================
Pablo nunca debe esperar a que JARVIS termine de hablar para corregirle. Este
módulo implementa las dos formas de interrupción:

1. **CORTE DE VOZ (TTS Barge-In)**
   Mientras JARVIS habla, un detector de actividad de voz sigue escuchando el
   micrófono. En el milisegundo en que Pablo empieza a hablar:
   * se corta el audio de salida (el stream se rellena con silencio, sin
     chasquidos y sin esperar a que termine la frase);
   * el HUD pasa a modo "escuchando" (cian eléctrico);
   * lo que diga Pablo se transcribe y se procesa como una orden nueva.

2. **RECTIFICACIÓN DE ACCIONES EN VUELO**
   Si JARVIS está moviendo el ratón o escribiendo, la voz de Pablo **congela la
   cola de ejecución** al instante (`AutomationGate`). La corrección ("no,
   detente, haz clic en el otro botón") se transcribe, se re-evalúa la pantalla
   con el módulo de visión y se reescribe el plan sin perder el contexto.

UMBRALES ADAPTATIVOS
--------------------
Durante la reproducción, el micrófono oye la propia voz de JARVIS. Por eso el
umbral de energía se eleva mientras habla (`SPEAKING_THRESHOLD_MULTIPLIER`) y
se exige un poco más de voz continua antes de considerar que Pablo le ha
interrumpido de verdad. Así JARVIS no se calla solo por el eco de su voz.
================================================================================
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import config as cfg  # noqa: E402

logger = logging.getLogger("jarvis.bargein")

# Motivos de interrupción.
BARGE_IN_SPEECH = "voz"
BARGE_IN_KILLSWITCH = "killswitch"
BARGE_IN_MANUAL = "manual"


# =============================================================================
# EXCEPCIONES DE CONTROL DE FLUJO
# =============================================================================

class AutomationInterrupted(RuntimeError):
    """La automatización fue interrumpida por la voz de Pablo o el kill-switch."""


# =============================================================================
# EVENTOS
# =============================================================================

@dataclass
class BargeInEvent:
    """Instante en el que Pablo ha tomado la palabra."""

    reason: str = BARGE_IN_SPEECH
    timestamp: float = 0.0
    level: float = 0.0
    during_speech: bool = False
    during_automation: bool = False
    speech_ms: float = 0.0

    def describe(self) -> str:
        parts = [f"motivo={self.reason}"]
        if self.during_speech:
            parts.append("interrumpiendo la voz de JARVIS")
        if self.during_automation:
            parts.append("interrumpiendo una automatización")
        parts.append(f"nivel={self.level:.3f}")
        return "Interrupción (" + ", ".join(parts) + ")"


# =============================================================================
# DETECTOR DSP
# =============================================================================

class BargeInDetector:
    """Detecta el inicio de una frase con umbral adaptativo al ruido y al eco.

    Se alimenta de las mismas tramas de 20 ms que el resto del sistema. Cuando
    detecta `speech_trigger_ms` milisegundos consecutivos de voz por encima del
    umbral (más exigente si JARVIS está hablando), lanza un `BargeInEvent`.
    """

    def __init__(
        self,
        speech_trigger_ms: float | None = None,
        speaking_multiplier: float = 2.2,
        enabled: bool = True,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.speech_trigger_ms = float(speech_trigger_ms or cfg.VAD_SPEECH_TRIGGER_MS)
        self.speaking_multiplier = float(speaking_multiplier)
        self.enabled = enabled
        # Reloj inyectable: los tests simulan el paso del tiempo sin esperarlo.
        self._clock = clock or time.monotonic
        self.noise_floor = float(cfg.VAD_NOISE_FLOOR_INIT)
        self._voice_frames = 0
        self._last_event = 0.0
        self.stats = {"tramas": 0, "interrupciones": 0, "falsos_positivos": 0}

    # --- Cálculo -----------------------------------------------------------

    def _threshold(self, jarviss_speaking: bool) -> float:
        """Umbral de energía efectivo, elevado mientras JARVIS habla."""
        base = max(cfg.VAD_ENERGY_THRESHOLD, self.noise_floor * cfg.VAD_NOISE_MARGIN)
        return base * (self.speaking_multiplier if jarviss_speaking else 1.0)

    def _update_noise(self, rms: float, threshold: float) -> None:
        if rms < threshold:
            alpha = cfg.VAD_NOISE_FLOOR_ALPHA
            self.noise_floor = max(0.0005, alpha * self.noise_floor + (1.0 - alpha) * rms)

    def feed(
        self,
        frame: np.ndarray,
        jarviss_speaking: bool = False,
        automation_active: bool = False,
    ) -> BargeInEvent | None:
        """Analiza una trama y decide si Pablo ha tomado la palabra."""
        if not self.enabled or frame is None or len(frame) == 0:
            return None

        self.stats["tramas"] += 1
        samples = frame.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples)))
        threshold = self._threshold(jarviss_speaking)
        self._update_noise(rms, threshold)

        now = self._clock()
        if rms >= threshold:
            self._voice_frames += 1
        else:
            self._voice_frames = 0
            return None

        speech_ms = self._voice_frames * cfg.AUDIO_FRAME_MS
        if speech_ms < self.speech_trigger_ms:
            return None

        # Antirrebote: como mucho una interrupción cada 400 ms.
        if now - self._last_event < 0.4:
            return None

        self._last_event = now
        self._voice_frames = 0
        self.stats["interrupciones"] += 1
        event = BargeInEvent(
            reason=BARGE_IN_SPEECH,
            timestamp=time.time(),
            level=rms,
            during_speech=jarviss_speaking,
            during_automation=automation_active,
            speech_ms=speech_ms,
        )
        logger.info("%s", event.describe())
        return event

    def reset(self) -> None:
        """Limpia el contador de voz (tras una orden procesada)."""
        self._voice_frames = 0


# =============================================================================
# PUERTA DE AUTOMATIZACIÓN (congelar la cola de ejecución)
# =============================================================================

class AutomationGate:
    """Permite congelar y cancelar la automatización de ratón y teclado.

    El actuador consulta `wait_before_step()` antes de CADA paso. Si la puerta
    está congelada, el actuador se queda esperando (la ejecución se detiene en
    seco, sin perder el plan); si se cancela, lanza `AutomationInterrupted`.
    """

    def __init__(self) -> None:
        self._frozen = threading.Event()
        self._cancelled = threading.Event()
        self._reason = ""
        self._freeze_count = 0
        self._total_frozen_s = 0.0
        self._frozen_since = 0.0
        self._lock = threading.Lock()

    @property
    def frozen(self) -> bool:
        return self._frozen.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def freeze(self, reason: str = "corrección por voz") -> None:
        """Congela la automatización inmediatamente."""
        with self._lock:
            if not self._frozen.is_set():
                self._frozen_since = time.monotonic()
                self._freeze_count += 1
            self._reason = reason
            self._frozen.set()
        logger.warning("Automatización CONGELADA: %s", reason)

    def thaw(self) -> None:
        """Descongela y permite continuar por donde se quedó."""
        with self._lock:
            if self._frozen.is_set():
                self._total_frozen_s += time.monotonic() - self._frozen_since
            self._frozen.clear()
            self._reason = ""
        logger.info("Automatización reanudada.")

    def cancel(self, reason: str = "cancelado") -> None:
        """Cancela la automatización por completo (se abandona el plan)."""
        with self._lock:
            self._reason = reason
            self._frozen.clear()
            self._cancelled.set()
        logger.warning("Automatización CANCELADA: %s", reason)

    def clear(self) -> None:
        """Prepara la puerta para el siguiente plan."""
        with self._lock:
            self._frozen.clear()
            self._cancelled.clear()
            self._reason = ""

    def wait_before_step(self, poll_s: float = 0.05, timeout: float | None = None) -> None:
        """Se llama antes de cada paso de automatización.

        * Si está cancelada -> lanza `AutomationInterrupted`.
        * Si está congelada -> espera (sin consumir CPU) a que se descongele.
        * Si el kill-switch está activado -> lanza `AutomationInterrupted`.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._cancelled.is_set():
                raise AutomationInterrupted(f"Automatización cancelada: {self._reason}")
            if not self._frozen.is_set():
                return
            if deadline is not None and time.monotonic() >= deadline:
                # Se abandona este paso, pero la puerta no se cancela sola: quien
                # manda (el ejecutor de visión) decide si reanuda o lo deja todo.
                logger.warning(
                    "La automatización lleva %.2f s congelada: se abandona este paso.", timeout
                )
                raise AutomationInterrupted("Se agotó el tiempo de espera congelado.")
            time.sleep(poll_s)

    def status(self) -> dict:
        return {
            "congelada": self.frozen,
            "cancelada": self.cancelled,
            "motivo": self._reason,
            "congelaciones": self._freeze_count,
            "tiempo_congelado_s": round(self._total_frozen_s, 2),
        }


# =============================================================================
# CONTROLADOR
# =============================================================================

class BargeInController:
    """Une el detector, el motor de voz y el actuador.

    Parameters
    ----------
    on_interrupt:
        Función que se llama al detectar la interrupción. El motor de voz la usa
        para cortar el TTS y pasar a escuchar. Recibe el `BargeInEvent`.
    is_speaking:
        Devuelve si JARVIS está hablando ahora mismo.
    is_automating:
        Devuelve si hay una automatización en curso.
    killswitch:
        Interruptor de emergencia: al activarse, también corta la voz.
    notifier:
        Función `(estado, detalle)` para el HUD.
    """

    def __init__(
        self,
        on_interrupt: Callable[[BargeInEvent], None] | None = None,
        is_speaking: Callable[[], bool] | None = None,
        is_automating: Callable[[], bool] | None = None,
        killswitch: Any | None = None,
        notifier: Callable[[str, str], None] | None = None,
        detector: BargeInDetector | None = None,
    ) -> None:
        self.detector = detector or BargeInDetector()
        self.gate = AutomationGate()
        self.on_interrupt = on_interrupt
        self.is_speaking = is_speaking or (lambda: False)
        self.is_automating = is_automating or (lambda: False)
        self.killswitch = killswitch
        self.notifier = notifier
        self._armed = threading.Event()
        self._history: deque[BargeInEvent] = deque(maxlen=20)

    # --- Enganche al audio -------------------------------------------------

    def arm(self) -> None:
        """Activa la escucha de interrupciones."""
        self._armed.set()
        if self.killswitch is not None:
            self.killswitch.register("barge_in", self._on_killswitch)
        logger.debug("Barge-in armado.")

    def disarm(self) -> None:
        self._armed.clear()
        if self.killswitch is not None:
            self.killswitch.unregister("barge_in")
        logger.debug("Barge-in desarmado.")

    @property
    def armed(self) -> bool:
        return self._armed.is_set()

    def on_frame(self, frame: np.ndarray) -> BargeInEvent | None:
        """Recibe audio del motor (llamado en el hilo de captura, sin bloqueos)."""
        if not self._armed.is_set():
            return None
        event = self.detector.feed(
            frame,
            jarviss_speaking=self._speaking(),
            automation_active=self._automating(),
        )
        if event is None:
            return None
        self._history.append(event)
        self._handle(event)
        return event

    def _speaking(self) -> bool:
        try:
            return bool(self.is_speaking())
        except Exception:
            return False

    def _automating(self) -> bool:
        try:
            return bool(self.is_automating())
        except Exception:
            return False

    # --- Reacción ----------------------------------------------------------

    def _handle(self, event: BargeInEvent) -> None:
        """Corta lo que esté pasando y avisa a quien corresponda."""
        if event.during_automation:
            # 1) Congelar la cola de ejecución AL INSTANTE.
            self.gate.freeze(event.reason)
            self._notify("listening", "Corrección")
        if event.during_speech:
            self._notify("listening", "Le escucho")
        elif not event.during_automation:
            self._notify("listening", "Le escucho")

        if self.on_interrupt is not None:
            try:
                self.on_interrupt(event)
            except Exception as exc:  # nunca se puede romper el bucle de audio
                logger.error("El manejador de interrupción falló: %s", exc)

    def _on_killswitch(self, event: Any) -> None:
        """El kill-switch también manda callar a JARVIS y cancela la cola."""
        self.gate.cancel("kill-switch")
        self.detector.stats["interrupciones"] += 1
        self._notify("killswitch", "Control liberado")
        if self.on_interrupt is not None:
            try:
                self.on_interrupt(
                    BargeInEvent(reason=BARGE_IN_KILLSWITCH, timestamp=time.time())
                )
            except Exception as exc:  # pragma: no cover
                logger.debug("Interrupción por kill-switch ignorada: %s", exc)

    def _notify(self, state: str, detail: str = "") -> None:
        if self.notifier is None:
            return
        try:
            self.notifier(state, detail)
        except Exception:  # pragma: no cover
            pass

    # --- Utilidades --------------------------------------------------------

    def manual_interrupt(self) -> BargeInEvent:
        """Interrupción provocada por código (botón, orden de texto, pruebas)."""
        event = BargeInEvent(reason=BARGE_IN_MANUAL, timestamp=time.time())
        self._history.append(event)
        self._handle(event)
        return event

    def recent(self) -> list[BargeInEvent]:
        return list(self._history)

    def status(self) -> dict:
        return {
            "armado": self.armed,
            "detector": dict(self.detector.stats),
            "puerta": self.gate.status(),
            "interrupciones_recientes": [e.describe() for e in list(self._history)[-3:]],
        }
