"""
core/acoustic_detector.py — Detector de doble palmada.
================================================================================
Pablo no siempre quiere hablar: a veces solo quiere que suene la música. Este
módulo escucha el ambiente en busca de DOS PALMADAS SECAS separadas entre 200 y
750 milisegundos y, cuando las reconoce, lanza la rutina de arranque:

    1. Saludo según la hora del reloj  ->  "Buenos días Pablo" / "Buenas tardes
       Pablo" / "Buenas noches Pablo".
    2. "Loser" de Tame Impala          ->  Spotify local y, si falla, Comet
       (navegador de Perplexity) en YouTube.

CÓMO SE RECONOCE UNA PALMADA (y no otra cosa)
---------------------------------------------
Una palmada es un **transitorio de banda ancha**: sube de golpe (< 20 ms), dura
poco (< 60 ms) y reparte su energía por todo el espectro. Eso permite
distinguirla de una voz, de un portazo sordo o de un tono agudo:

* **Volumen:** tiene que superar el ruido ambiental por un factor claro.
* **Ataque:** el salto respecto a la trama anterior debe ser brusco.
* **Duración:** si el estruendo dura más de 60 ms, es un golpe, no una palmada.
* **Aplanamiento espectral:** el ruido blanco de una palmada tiene un espectro
  "plano"; un silbido o una nota musical tienen picos estrechos y se descartan.
* **Centroide:** se exige energía en frecuencias medias-altas (> 900 Hz) para
  ignorar golpes sordos sobre la mesa.

El detector es puro cálculo (recibe tramas de audio y devuelve eventos), así que
se puede probar sin micrófono. `AcousticTrigger` añade la capa de reproducción
y `run_standalone()` permite probarlo a mano desde la consola.
================================================================================
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import config as cfg  # noqa: E402
from core.utils import run_in_thread  # noqa: E402

logger = logging.getLogger("jarvis.palmada")


# =============================================================================
# EVENTO
# =============================================================================

@dataclass
class ClapEvent:
    """Dos palmadas reconocidas."""

    timestamp: float = 0.0
    gap_ms: float = 0.0
    level: float = 0.0
    confidence: float = 0.0

    def describe(self) -> str:
        return (
            f"Doble palmada (separación {self.gap_ms:.0f} ms, "
            f"nivel {self.level:.3f}, confianza {self.confidence:.2f})"
        )


@dataclass
class DetectorStats:
    """Contadores para el diagnóstico y las pruebas."""

    tramas: int = 0
    picos: int = 0
    descartes: dict[str, int] = field(default_factory=dict)
    disparos: int = 0

    def reject(self, reason: str) -> None:
        self.descartes[reason] = self.descartes.get(reason, 0) + 1


# =============================================================================
# DETECTOR (DSP PURO)
# =============================================================================

class ClapDetector:
    """Analiza tramas de audio y avisa cuando suenan dos palmadas seguidas.

    Todas las tramas deben ser `numpy.ndarray` de enteros de 16 bits, mono, a
    `cfg.AUDIO_SAMPLE_RATE` Hz y de `cfg.AUDIO_FRAME_MS` milisegundos. Es la
    misma forma que entrega el motor de voz, así que el detector se engancha a
    su flujo sin conversiones.
    """

    def __init__(
        self,
        min_gap_ms: float = cfg.CLAP_MIN_GAP_MS,
        max_gap_ms: float = cfg.CLAP_MAX_GAP_MS,
        refractory_ms: float = cfg.CLAP_REFRACTORY_MS,
        enabled: bool = cfg.CLAP_ENABLED,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.min_gap_ms = float(min_gap_ms)
        self.max_gap_ms = float(max_gap_ms)
        self.refractory_ms = float(refractory_ms)
        self.enabled = bool(enabled)
        # Reloj inyectable: permite probar la separación entre palmadas sin
        # depender del tiempo real (los tests pasan un reloj falso).
        self._clock = clock or time.monotonic

        self.noise_floor = float(cfg.VAD_NOISE_FLOOR_INIT)
        self.stats = DetectorStats()
        self._peaks: deque[float] = deque(maxlen=8)       # Instantes de palmada.
        self._pending = 0.0                               # Segunda palmada a la espera.
        self._pending_level = 0.0
        self._last_trigger = 0.0
        self._above_frames = 0                            # Tramas seguidas con energía.
        self._suspended_until = 0.0                       # Silencio obligatorio.
        self._window = np.hanning(cfg.AUDIO_FRAME_SAMPLES).astype(np.float32)
        self._frequencies = np.fft.rfftfreq(cfg.AUDIO_FRAME_SAMPLES, d=1.0 / cfg.AUDIO_SAMPLE_RATE)
        self._previous_frame: np.ndarray | None = None

    # --- Análisis de una trama --------------------------------------------

    @staticmethod
    def frame_rms(frame: np.ndarray) -> float:
        """Volumen de la trama normalizado a 0..1 (RMS sobre la escala de 16 bits)."""
        if frame is None or len(frame) == 0:
            return 0.0
        samples = frame.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(samples * samples)))

    def _spectral_features(self, frame: np.ndarray) -> tuple[float, float]:
        """Devuelve `(aplanamiento, centroide_hz)` de la trama."""
        samples = (frame.astype(np.float32) / 32768.0) * self._window
        spectrum = np.abs(np.fft.rfft(samples))
        total = float(np.sum(spectrum))
        if total <= 1e-9:
            return 0.0, 0.0
        # Aplanamiento espectral (media geométrica / media aritmética).
        flatness = float(np.exp(np.mean(np.log(spectrum + 1e-10))) / (np.mean(spectrum) + 1e-10))
        centroid = float(np.sum(self._frequencies * spectrum) / total)
        return flatness, centroid

    def _update_noise_floor(self, rms: float) -> None:
        """Aprende el ruido de la habitación sin dejarse engañar por las palmadas."""
        if rms > self.noise_floor * cfg.CLAP_NOISE_MULTIPLIER:
            return  # Probable palmada o voz: no contamina el suelo de ruido.
        alpha = cfg.VAD_NOISE_FLOOR_ALPHA
        self.noise_floor = max(0.0005, alpha * self.noise_floor + (1.0 - alpha) * rms)

    def _is_peak(self, rms: float, previous_rms: float) -> tuple[bool, str]:
        """Decide si la trama contiene el ataque de una palmada."""
        threshold = max(cfg.CLAP_MIN_PEAK_RMS, self.noise_floor * cfg.CLAP_NOISE_MULTIPLIER)
        if rms < threshold:
            return False, "sin_volumen"
        if rms < previous_rms * cfg.CLAP_ATTACK_RATIO:
            return False, "sin_ataque"
        return True, ""

    def feed(self, frame: np.ndarray) -> ClapEvent | None:
        """Procesa una trama. Devuelve un `ClapEvent` si hay doble palmada.

        Es una función sin estado global: se puede llamar mil veces por segundo
        desde el hilo de audio sin bloquear nada.
        """
        if not self.enabled or frame is None or len(frame) < 8:
            return None

        now = self._clock()
        self.stats.tramas += 1
        rms = self.frame_rms(frame)
        previous_rms = self.frame_rms(self._previous_frame) if self._previous_frame is not None else 0.0

        # Periodo refractario: después de una orden, JARVIS "aparta el oído".
        if now < self._suspended_until:
            self._previous_frame = frame
            self._update_noise_floor(rms)
            return None

        peak, _reason = self._is_peak(rms, previous_rms)

        if peak:
            self._above_frames += 1
            flatness, centroid = self._spectral_features(frame)

            # Un estruendo largo no es una palmada (portazo, grito, cristal).
            if self._above_frames * (cfg.AUDIO_FRAME_MS / 1000.0) > (cfg.CLAP_MAX_CLAP_DURATION_MS / 1000.0):
                if self._peaks:
                    self._peaks.pop()  # El pico anterior era en realidad un golpe.
                self.stats.reject("duracion_excesiva")
                self._previous_frame = frame
                return None

            if not (cfg.CLAP_FLATNESS_MIN <= flatness <= cfg.CLAP_FLATNESS_MAX):
                self.stats.reject("espectro_tonal")
                self._previous_frame = frame
                return None
            if centroid < cfg.CLAP_SPECTRAL_CENTROID_MIN_HZ:
                self.stats.reject("golpe_sordo")
                self._previous_frame = frame
                return None

            self.stats.picos += 1
            self._peaks.append(now)
            event = self._register_peak(now, rms)
            self._previous_frame = frame
            return event

        # Tramas sin pico: se reinicia el contador de duración.
        self._above_frames = 0
        self._update_noise_floor(rms)
        self._previous_frame = frame

        # Si había una segunda palmada esperando confirmación, se comprueba.
        event = self._maybe_confirm(now)

        # Limpia picos demasiado antiguos (evita falsos positivos al azar).
        while self._peaks and (now - self._peaks[0]) * 1000.0 > self.max_gap_ms * 4:
            self._peaks.popleft()
        return event

    def _register_peak(self, now: float, rms: float) -> ClapEvent | None:
        """Registra una palmada nueva y decide si forma un doble golpe válido.

        La confirmación es deliberadamente perezosa: cuando suenan dos palmadas
        con una separación válida, JARVIS espera `CLAP_CONFIRM_MS` antes de
        actuar. Si en ese margen llega una tercera, es un aplauso y no una
        orden: se descarta todo.
        """
        if len(self._peaks) >= 3:
            self.stats.reject("tres_o_mas")
            self._peaks.clear()
            self._pending = 0.0
            return None

        if len(self._peaks) < 2:
            return None

        first, second = self._peaks[-2], self._peaks[-1]
        gap_ms = (second - first) * 1000.0

        if gap_ms < self.min_gap_ms:
            self.stats.reject("demasiado_juntas")
            self._peaks.pop()          # La segunda palmada no valía: se descarta.
            return None
        if gap_ms > self.max_gap_ms:
            self.stats.reject("demasiado_separadas")
            self._peaks.popleft()      # La primera era demasiado antigua.
            return None

        # Separación correcta: queda pendiente de confirmación.
        self._pending = second
        self._pending_level = rms
        return None

    def _maybe_confirm(self, now: float) -> ClapEvent | None:
        """Confirma el doble golpe si no ha llegado una tercera palmada a tiempo."""
        if not self._pending:
            return None
        if (now - self._pending) * 1000.0 < cfg.CLAP_CONFIRM_MS:
            return None

        if len(self._peaks) != 2:
            self.stats.reject("tres_o_mas")
            self._peaks.clear()
            self._pending = 0.0
            return None

        first, second = self._peaks
        gap_ms = (second - first) * 1000.0
        level = max(self._pending_level, self.noise_floor)

        # Confianza: cuánto se acerca la separación al centro del rango válido.
        center = (self.min_gap_ms + self.max_gap_ms) / 2.0
        half_range = (self.max_gap_ms - self.min_gap_ms) / 2.0
        confidence = 1.0 - min(1.0, abs(gap_ms - center) / max(1.0, half_range))
        confidence *= min(1.0, level / max(1e-6, cfg.CLAP_MIN_PEAK_RMS))

        event = ClapEvent(
            timestamp=time.time(),
            gap_ms=gap_ms,
            level=level,
            confidence=round(float(confidence), 3),
        )
        self._peaks.clear()
        self._pending = 0.0
        self._last_trigger = now
        self._suspended_until = now + self.refractory_ms / 1000.0
        self.stats.disparos += 1
        logger.info("¡%s!", event.describe())
        return event

    # --- Control -----------------------------------------------------------

    def suspend(self, seconds: float) -> None:
        """Silencia el detector un tiempo (por ejemplo, mientras JARVIS habla)."""
        self._suspended_until = max(self._suspended_until, self._clock() + float(seconds))

    def reset(self) -> None:
        """Reinicia el estado interno (cambio de dispositivo de audio, etc.)."""
        self._peaks.clear()
        self._pending = 0.0
        self._above_frames = 0
        self._previous_frame = None
        self._suspended_until = 0.0

    def status(self) -> dict:
        """Informe para el HUD y el diagnóstico."""
        return {
            "activo": self.enabled,
            "ruido_ambiente": round(self.noise_floor, 5),
            "umbral_actual": round(
                max(cfg.CLAP_MIN_PEAK_RMS, self.noise_floor * cfg.CLAP_NOISE_MULTIPLIER), 5
            ),
            "tramas": self.stats.tramas,
            "picos": self.stats.picos,
            "disparos": self.stats.disparos,
            "descartes": dict(self.stats.descartes),
        }


# =============================================================================
# ORQUESTADOR: DETECCIÓN -> ACCIÓN
# =============================================================================

class AcousticTrigger:
    """Conecta la detección de palmadas con la acción de arranque.

    Se alimenta de dos maneras:

    * **Enganchado al motor de voz** (`on_frame`): el motor ya tiene el
      micrófono abierto, así que este módulo no abre un segundo flujo de audio.
    * **En solitario** (`run_standalone`): abre su propio micrófono, útil para
      probar la doble palmada sin arrancar todo JARVIS.
    """

    def __init__(
        self,
        action: Callable[[ClapEvent], Any] | None = None,
        detector: ClapDetector | None = None,
        notifier: Callable[[str, str], None] | None = None,
    ) -> None:
        self.detector = detector or ClapDetector()
        self.action = action
        self.notifier = notifier
        self._busy = threading.Lock()
        self._stream: Any | None = None
        self._running = threading.Event()

    # --- Enganche al flujo de audio del motor de voz -----------------------

    def on_frame(self, frame: np.ndarray) -> ClapEvent | None:
        """Recibe una trama del motor de voz. Devuelve el evento si lo hubo."""
        event = self.detector.feed(frame)
        if event is not None:
            self._fire(event)
        return event

    def _fire(self, event: ClapEvent) -> None:
        """Ejecuta la acción en un hilo aparte: el audio nunca se bloquea."""
        if self.action is None:
            logger.info("Doble palmada detectada, pero no hay acción configurada.")
            return
        if self._busy.locked():  # Ya se está ejecutando la rutina.
            logger.debug("Rutina de palmada ya en curso: se ignora la repetición.")
            return
        run_in_thread(self._run_action, event, name="jarvis-accion-palmada")

    def _run_action(self, event: ClapEvent) -> None:
        with self._busy:
            if self.notifier is not None:
                with suppress(Exception):
                    self.notifier("acting", "Doble palmada")
            try:
                self.action(event)
            except Exception as exc:  # pragma: no cover - depende del sistema
                logger.error("La rutina de la doble palmada falló: %s", exc)
            finally:
                if self.notifier is not None:
                    with suppress(Exception):
                        self.notifier("idle", "")

    # --- Modo autónomo (micrófono propio) ---------------------------------

    def start(self) -> bool:
        """Abre el micrófono y escucha en segundo plano. Devuelve si lo logró."""
        if self._running.is_set():
            return True
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            logger.error(
                "Falta la librería de audio (sounddevice). Ejecuta install.bat "
                "para instalarla."
            )
            return False

        def callback(indata, _frames, _time_info, status) -> None:  # noqa: ANN001
            if status:
                logger.debug("Estado del micrófono: %s", status)
            self.on_frame(np.frombuffer(bytes(indata), dtype=np.int16))

        try:
            self._stream = sd.InputStream(
                samplerate=cfg.AUDIO_SAMPLE_RATE,
                blocksize=cfg.AUDIO_FRAME_SAMPLES,
                channels=cfg.AUDIO_CHANNELS,
                dtype=cfg.AUDIO_DTYPE,
                device=cfg.AUDIO_INPUT_DEVICE,
                callback=callback,
            )
            self._stream.start()
            self._running.set()
            logger.info("Escuchando en busca de dobles palmadas...")
            return True
        except Exception as exc:
            logger.error("No se pudo abrir el micrófono para la doble palmada: %s", exc)
            return False

    def stop(self) -> None:
        """Cierra el micrófono en modo autónomo."""
        self._running.clear()
        if self._stream is not None:
            with suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None
        logger.debug("Detector de palmadas detenido.")


def run_standalone(seconds: float = 60.0) -> None:  # pragma: no cover - manual
    """Prueba manual: `python -m core.acoustic_detector` y da dos palmadas."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    def demo_action(event: ClapEvent) -> None:
        saludo = cfg.greeting_for_hour()
        print("=" * 62)
        print(f"  ¡{event.describe()}!")
        print(f"  JARVIS diría: {saludo}")
        print(f"  Y pondría: {cfg.CLAP_TRACK_QUERY} en Spotify / YouTube")
        print("=" * 62)

    trigger = AcousticTrigger(action=demo_action, notifier=lambda s, d: logger.info("estado: %s %s", s, d))
    if not trigger.start():
        print("No he podido abrir el micrófono. Revisa que no lo use otra aplicación.")
        return
    print(f"Escuchando {int(seconds)} segundos... da DOS palmadas seguidas.")
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        pass
    finally:
        trigger.stop()
        print("Estadísticas:", trigger.detector.status())


if __name__ == "__main__":  # pragma: no cover - manual
    run_standalone()
