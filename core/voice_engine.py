"""
core/voice_engine.py — Oídos y boca de JARVIS (más el interruptor de escucha).
================================================================================
Este módulo concentra TODA la voz del sistema, con un único micrófono
compartido y una cadena de reserva en cada etapa para que JARVIS nunca se
quede mudo ni sordo:

    Micrófono (16 kHz, 20 ms por trama)
        |
        +--> openWakeWord  -> "Jarvis" ......... despierta a JARVIS
        +--> VAD adaptativo -> frases .......... delimita lo que dice Pablo
        +--> Detector de doble palmada ......... rutina de música (otro módulo)
        +--> Barge-in ......................... corta el TTS si Pablo interrumpe
        |
        v
    faster-whisper (CPU, int8) --> texto
        |
        v
    Enrutador de modelos (llama3.1:8b / qwen2.5-coder:7b / llama3.2-vision)
        |
        v
    Kokoro-82M --> Piper --> voz SAPI de Windows --> silencio (CPU, sin VRAM)

DECISIONES DE RENDIMIENTO
-------------------------
* **STT y TTS en la CPU.** La GPU (8 GB) es exclusiva del modelo de lenguaje:
  así la ley de monogamia de VRAM nunca se rompe por culpa de la voz.
* **Sin ecos ni solapamientos:** mientras habla, el umbral de voz sube y el
  detector de palmadas se suspende.
* **Respuesta por frases:** se sintetiza la primera frase mientras el modelo
  sigue generando las demás, de modo que JARVIS contesta en décimas de segundo.
* **Modo conversación:** tras despertarle, JARVIS sigue escuchando 6 segundos
  sin exigir que Pablo repita "Jarvis" en cada frase.
================================================================================
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import config as cfg  # noqa: E402
from core.barge_in import AutomationGate, BargeInController, BargeInEvent  # noqa: E402
from core.utils import normalize_text, run_in_thread, split_sentences, truncate  # noqa: E402

logger = logging.getLogger("jarvis.voz")


# =============================================================================
# ESTADOS DE JARVIS (los dibuja el HUD)
# =============================================================================

STATE_SLEEPING = "sleeping"
STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_TRANSCRIBING = "transcribing"
STATE_THINKING = "thinking"
STATE_SPEAKING = "speaking"
STATE_ACTING = "acting"
STATE_VISION = "vision"
STATE_SWAPPING = "swapping"
STATE_ERROR = "error"
STATE_KILLSWITCH = "killswitch"

VALID_STATES = frozenset(cfg.HUD_STATUS_TEXT.keys())


# =============================================================================
# 1. CENTRO DE AUDIO (UN SOLO MICRÓFONO PARA TODO JARVIS)
# =============================================================================

class AudioHub:
    """Abre el micrófono UNA vez y reparte las tramas a todos los consumidores.

    Windows no permite abrir el mismo micrófono desde varios sitios y
    comportarse bien a la vez, así que aquí hay un único flujo de entrada que
    alimenta al detector de palmadas, al VAD, al wake word y al barge-in.
    """

    def __init__(
        self,
        on_frame: Callable[[np.ndarray], None] | None = None,
        sample_rate: int = cfg.AUDIO_SAMPLE_RATE,
        frame_samples: int = cfg.AUDIO_FRAME_SAMPLES,
        device: Any = cfg.AUDIO_INPUT_DEVICE,
    ) -> None:
        self.on_frame = on_frame
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        self._stream: Any | None = None
        self._running = threading.Event()
        self.drop_count = 0

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def start(self) -> bool:
        """Abre el micrófono. Devuelve False (con explicación) si no se puede."""
        if self.running:
            return True
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            logger.error(
                "Falta 'sounddevice' (la librería del micrófono). "
                "Ejecuta install.bat para instalarla."
            )
            return False

        def callback(indata, _frames, _time_info, status) -> None:  # noqa: ANN001
            if status:
                logger.debug("Aviso del micrófono: %s", status)
            frame = np.frombuffer(bytes(indata), dtype=np.int16)
            if self.on_frame is None:
                return
            try:
                self.on_frame(frame)
            except Exception as exc:  # el callback de audio NUNCA debe lanzar
                logger.error("Error procesando una trama de audio: %s", exc)

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.frame_samples,
                channels=cfg.AUDIO_CHANNELS,
                dtype=cfg.AUDIO_DTYPE,
                device=self.device,
                callback=callback,
            )
            self._stream.start()
            self._running.set()
            info = self.describe_device(sd)
            logger.info("Micrófono abierto: %s", info)
            return True
        except Exception as exc:
            logger.error(
                "No he podido abrir el micrófono: %s\n"
                "  · Comprueba que Windows permite el acceso al micrófono\n"
                "    (Configuración > Privacidad > Micrófono).\n"
                "  · Y que ninguna otra aplicación lo tenga en exclusiva.",
                exc,
            )
            return False

    @staticmethod
    def describe_device(sd_module: Any) -> str:  # pragma: no cover - depende del SO
        """Nombre legible del dispositivo de entrada en uso."""
        try:
            device = sd_module.query_devices(kind="input")
            return f"{device.get('name', 'desconocido')} ({device.get('default_samplerate', 0):.0f} Hz)"
        except Exception:
            return "dispositivo predeterminado"

    def stop(self) -> None:
        """Cierra el micrófono."""
        self._running.clear()
        if self._stream is not None:
            with suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None
        logger.debug("Micrófono cerrado.")

    def status(self) -> dict:
        return {"activo": self.running, "frecuencia": self.sample_rate, "tramas_perdidas": self.drop_count}


# =============================================================================
# 2. VAD (DETECCIÓN DE VOZ) ADAPTATIVO
# =============================================================================

class VoiceActivityDetector:
    """Decide si una trama contiene voz humana, aprendiendo el ruido de la sala.

    Usa energía RMS con suelo de ruido adaptativo y, si la librería
    `webrtcvad` está instalada, combina ambos criterios para reducir falsos
    positivos con ventiladores o teclados.
    """

    def __init__(self) -> None:
        self.noise_floor = float(cfg.VAD_NOISE_FLOOR_INIT)
        self._webrtc = None
        if cfg.VAD_USE_WEBRTCVAD:
            try:
                import webrtcvad  # type: ignore

                self._webrtc = webrtcvad.Vad(cfg.VAD_WEBRTC_AGGRESSIVENESS)
                logger.debug("VAD de WebRTC disponible (agresividad %d).", cfg.VAD_WEBRTC_AGGRESSIVENESS)
            except ImportError:
                logger.debug("webrtcvad no instalado: se usa solo energía RMS.")

    def rms(self, frame: np.ndarray) -> float:
        samples = frame.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(samples * samples)))

    def threshold(self, boost: float = 1.0) -> float:
        """Umbral efectivo de voz (con margen sobre el ruido de fondo)."""
        return max(cfg.VAD_ENERGY_THRESHOLD, self.noise_floor * cfg.VAD_NOISE_MARGIN) * boost

    def update_noise(self, rms: float, threshold: float) -> None:
        """Aprende el ruido ambiente solo con las tramas silenciosas."""
        if rms < threshold:
            alpha = cfg.VAD_NOISE_FLOOR_ALPHA
            self.noise_floor = max(0.0005, alpha * self.noise_floor + (1.0 - alpha) * rms)

    def is_speech(self, frame: np.ndarray, boost: float = 1.0) -> bool:
        """¿Contiene voz esta trama?"""
        rms = self.rms(frame)
        threshold = self.threshold(boost)
        self.update_noise(rms, threshold)
        if rms < threshold:
            return False
        if self._webrtc is not None and len(frame) in (160, 320, 480):
            try:
                pcm = frame.astype(np.int16).tobytes()
                if not self._webrtc.is_speech(pcm, cfg.AUDIO_SAMPLE_RATE):
                    return False
            except Exception:
                pass
        return True


# =============================================================================
# 3. STT — FASTER-WHISPER EN LA CPU
# =============================================================================

class SpeechRecognizer:
    """Transcribe voz a texto con faster-whisper (local, sin internet).

    El modelo `base` en CPU con cuantización int8 ocupa ~150 MB de RAM y tarda
    del orden de una décima de segundo por segundo de audio en un 7840HS: más
    que suficiente para conversar y sin tocar la VRAM.
    """

    def __init__(
        self,
        model_name: str = cfg.STT_MODEL,
        device: str = cfg.STT_DEVICE,
        compute_type: str = cfg.STT_COMPUTE_TYPE,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._unavailable_reason = ""
        self.stats = {"transcripciones": 0, "segundos_audio": 0.0, "segundos_calculo": 0.0}

    @property
    def available(self) -> bool:
        """¿Se puede usar el reconocimiento de voz?"""
        return self._model is not None or not self._unavailable_reason

    def preload(self) -> bool:
        """Carga el modelo en memoria (la primera vez descarga ~150 MB)."""
        return self._load() is not None

    def _load(self) -> Any | None:
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except ImportError:
                self._unavailable_reason = "falta la librería faster-whisper"
                logger.error(
                    "No puedo transcribir voz: %s. Ejecuta install.bat.",
                    self._unavailable_reason,
                )
                return None
            try:
                logger.info(
                    "Cargando el modelo de voz '%s' (%s/%s)...",
                    self.model_name, self.device, self.compute_type,
                )
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=cfg.STT_DOWNLOAD_ROOT,
                    cpu_threads=cfg.STT_CPU_THREADS,
                    num_workers=cfg.STT_NUM_WORKERS,
                )
                logger.info("Modelo de voz listo.")
                return self._model
            except Exception as exc:
                self._unavailable_reason = str(exc)
                logger.error("No he podido cargar el modelo de voz: %s", exc)
                return None

    # --- Conversión de audio ----------------------------------------------

    @staticmethod
    def to_float32(audio: np.ndarray | bytes) -> np.ndarray:
        """Convierte int16 (lo que da el micrófono) a float32 (-1..1)."""
        if isinstance(audio, bytes):
            array = np.frombuffer(audio, dtype=np.int16)
        else:
            array = np.asarray(audio)
        if array.dtype == np.int16:
            return array.astype(np.float32) / 32768.0
        return array.astype(np.float32)

    def transcribe(self, audio: np.ndarray | bytes, language: str = cfg.STT_LANGUAGE) -> str:
        """Devuelve el texto dicho en el audio (cadena vacía si no se entiende)."""
        model = self._load()
        if model is None:
            return ""

        samples = self.to_float32(audio)
        seconds = len(samples) / cfg.AUDIO_SAMPLE_RATE
        if seconds < cfg.STT_MIN_AUDIO_S:
            return ""
        if seconds > cfg.STT_MAX_AUDIO_S:
            samples = samples[: int(cfg.STT_MAX_AUDIO_S * cfg.AUDIO_SAMPLE_RATE)]

        started = time.perf_counter()
        try:
            segments, _info = model.transcribe(
                samples,
                language=language,
                beam_size=cfg.STT_BEAM_SIZE,
                best_of=cfg.STT_BEST_OF,
                temperature=cfg.STT_TEMPERATURE,
                condition_on_previous_text=cfg.STT_CONDITION_ON_PREVIOUS,
                vad_filter=cfg.STT_VAD_FILTER,
                initial_prompt=cfg.STT_INITIAL_PROMPT,
            )
            pieces = [segment.text.strip() for segment in segments if segment.text.strip()]
        except Exception as exc:
            logger.error("La transcripción falló: %s", exc)
            return ""

        elapsed = time.perf_counter() - started
        self.stats["transcripciones"] += 1
        self.stats["segundos_audio"] += seconds
        self.stats["segundos_calculo"] += elapsed

        text = " ".join(pieces).strip()
        text = self._clean(text)
        logger.info(
            "Transcripción (%.2f s de audio en %.2f s): %s",
            seconds, elapsed, truncate(text, 160) or "<vacío>",
        )
        return text

    @staticmethod
    def _clean(text: str) -> str:
        """Descarta las alucinaciones típicas de Whisper con ruido o silencio."""
        clean = " ".join(str(text).split())
        low = clean.lower()
        if not clean or len(clean) < 2:
            return ""
        for junk in cfg.STT_HALLUCINATION_BLACKLIST:
            if low.strip(" .!¡¿?") == junk.strip(" .!¡¿?"):
                logger.debug("Transcripción descartada (alucinación): %s", clean)
                return ""
        if low in {"ya", "eh", "mmm", "hmm", "ah", "oh"}:  # ruidos sueltos
            return ""
        return clean

    def status(self) -> dict:
        return {
            "modelo": self.model_name,
            "dispositivo": self.device,
            "disponible": self._model is not None,
            "motivo": self._unavailable_reason,
            **self.stats,
        }


# =============================================================================
# 4. TTS — KOKORO / PIPER / SAPI (CADENA DE RESERVA)
# =============================================================================

class TextToSpeech:
    """Voz neural en español con tres motores de reserva.

    Orden de preferencia (el primero disponible gana):

    1. **Kokoro-82M** (`kokoro-onnx`, voz ``ef_dora``): la más natural.
    2. **Piper** (`piper-tts`, voces ``es_ES``): excelente y muy ligera.
    3. **SAPI de Windows** (`pyttsx3`): siempre está, aunque suene robótica.
    4. **Silencio**: si no hay ninguna, JARVIS escribe en el registro.

    `stop()` corta el audio en milisegundos: es la pieza que hace posible que
    Pablo interrumpa a JARVIS a media frase.
    """

    def __init__(self, notifier: Callable[[str, str], None] | None = None) -> None:
        self.notifier = notifier
        self._backend: str = ""
        self._kokoro: Any | None = None
        self._piper: Any | None = None
        self._sapi: Any | None = None
        self._speaking = threading.Event()
        self._stop_event = threading.Event()
        self._closed = False
        self._synth_queue: queue.Queue = queue.Queue(maxsize=8)
        self._audio_queue: queue.Queue = queue.Queue(maxsize=64)
        self._threads: list[threading.Thread] = []
        self.voice_name = cfg.TTS_KOKORO_VOICE
        self.stats = {"frases": 0, "caracteres": 0, "segundos": 0.0, "interrupciones": 0}
        self._started = False

    # --- Ciclo de vida -----------------------------------------------------

    def start(self) -> bool:
        """Elige el mejor motor disponible y arranca los hilos de audio.

        Devuelve True solo si hay una voz real: si no hubiera ninguna, el
        llamante debe poder saberlo (y avisar) también en llamadas repetidas.
        """
        if self._started:
            return self._backend not in ("", "null")
        self._select_backend()
        self._threads = [
            run_in_thread(self._synth_loop, name="jarvis-tts-sintesis"),
            run_in_thread(self._player_loop, name="jarvis-tts-reproduccion"),
        ]
        self._started = True
        return self._backend not in ("", "null")

    def shutdown(self) -> None:
        """Detiene el audio y libera los motores."""
        self._closed = True
        self.stop()
        with suppress(Exception):
            self._synth_queue.put_nowait(None)
            self._audio_queue.put_nowait(None)
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()

    def _select_backend(self) -> None:
        """Prueba los motores en orden y se queda con el primero que funcione."""
        for candidate in cfg.TTS_ENGINE_ORDER:
            if candidate == "kokoro" and self._try_kokoro():
                self._backend = "kokoro"
                break
            if candidate == "piper" and self._try_piper():
                self._backend = "piper"
                break
            if candidate == "sapi" and self._try_sapi():
                self._backend = "sapi"
                break
            if candidate == "null":
                self._backend = "null"
                break
        logger.info("Motor de voz seleccionado: %s", self._backend or "ninguno")

    def _try_kokoro(self) -> bool:
        model_path = Path(cfg.TTS_MODELS_DIR) / cfg.TTS_KOKORO_MODEL_FILE
        voices_path = Path(cfg.TTS_MODELS_DIR) / cfg.TTS_KOKORO_VOICES_FILE
        if not (model_path.exists() and voices_path.exists()):
            logger.debug("Kokoro no está descargado todavía (%s).", model_path.name)
            return False
        try:
            from kokoro_onnx import Kokoro  # type: ignore

            self._kokoro = Kokoro(str(model_path), str(voices_path))
            logger.info(
                "Kokoro-82M listo (voz '%s', modelo %s).", self.voice_name, model_path.name
            )
            return True
        except ImportError:
            logger.debug("kokoro-onnx no está instalado.")
        except Exception as exc:
            logger.warning("Kokoro no pudo cargarse: %s", exc)
        return False

    def _try_piper(self) -> bool:
        try:
            from piper.voice import PiperVoice  # type: ignore
        except ImportError:
            logger.debug("piper-tts no está instalado.")
            return False

        voices_dir = Path(cfg.TTS_PIPER_VOICES_DIR)
        candidates = sorted(voices_dir.glob("*.onnx")) if voices_dir.exists() else []
        if not candidates:
            logger.debug("No hay voces de Piper descargadas en %s.", voices_dir)
            return False
        try:
            self._piper = PiperVoice.load(str(candidates[0]))
            self.voice_name = candidates[0].stem
            logger.info("Piper listo (voz '%s').", self.voice_name)
            return True
        except Exception as exc:
            logger.warning("Piper no pudo cargarse: %s", exc)
            return False

    def _try_sapi(self) -> bool:
        try:
            import pyttsx3  # type: ignore
        except ImportError:
            logger.debug("pyttsx3 no está instalado (voz SAPI no disponible).")
            return False
        try:
            engine = pyttsx3.init()
            voice_id = self._pick_sapi_voice(engine)
            if voice_id:
                engine.setProperty("voice", voice_id)
            engine.setProperty("rate", 175 + cfg.TTS_SAPI_RATE * 12)
            engine.setProperty("volume", cfg.TTS_SAPI_VOLUME)
            self._sapi = engine
            logger.info("Voz SAPI de Windows lista.")
            return True
        except Exception as exc:
            logger.warning("La voz SAPI no pudo iniciarse: %s", exc)
            return False

    @staticmethod
    def _pick_sapi_voice(engine: Any) -> str | None:
        """Busca una voz en español entre las instaladas en Windows."""
        try:
            for voice in engine.getProperty("voices"):
                label = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
                if any(hint in label for hint in cfg.TTS_SAPI_VOICE_HINTS):
                    return voice.id
        except Exception:
            return None
        return None

    # --- API pública -------------------------------------------------------

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def backend(self) -> str:
        return self._backend

    def say(self, text: str, blocking: bool = False) -> float:
        """Dice un texto. Devuelve la duración estimada en segundos.

        El texto se trocea en frases: la primera suena mientras se sintetizan
        las siguientes, que es lo que hace que JARVIS responda rápido.
        """
        clean = " ".join(str(text or "").split())
        if not clean or self._backend in ("", "null"):
            if clean:
                logger.info("(sin voz) JARVIS diría: %s", truncate(clean, 200))
            return 0.0

        sentences = split_sentences(clean, cfg.TTS_CHUNK_MAX_CHARS)
        if not sentences:
            return 0.0

        self.stats["frases"] += len(sentences)
        self.stats["caracteres"] += len(clean)
        self._stop_event.clear()
        self._set_state(STATE_SPEAKING)

        for sentence in sentences:
            if self._stop_event.is_set():
                break
            try:
                self._synth_queue.put(sentence, timeout=5.0)
            except queue.Full:  # pragma: no cover - solo si el motor va muy lento
                logger.debug("Cola de síntesis llena: se descarta una frase.")

        start = time.perf_counter()
        if blocking:
            self.wait_until_done()
        return time.perf_counter() - start

    def wait_until_done(self, timeout: float | None = None) -> bool:
        """Espera a que termine de sonar (o al cortocircuito por interrupción)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.speaking or not self._audio_queue.empty():
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.02)
        return True

    def stop(self) -> None:
        """Corta la voz inmediatamente (barge-in, kill-switch o cierre)."""
        was_speaking = self.speaking
        self._stop_event.set()
        for channel in (self._synth_queue, self._audio_queue):
            while True:
                try:
                    channel.get_nowait()
                except queue.Empty:
                    break
        if self._backend == "sapi" and self._sapi is not None:
            with suppress(Exception):
                self._sapi.stop()
        self._speaking.clear()
        if was_speaking:
            self.stats["interrupciones"] += 1
            logger.info("Voz de JARVIS interrumpida.")
        self._set_state(STATE_LISTENING if was_speaking else STATE_IDLE)

    # --- Hilos internos ----------------------------------------------------

    def _synth_loop(self) -> None:
        """Sintetiza frases en serie y las deja listas para sonar."""
        while not self._closed:
            try:
                sentence = self._synth_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if sentence is None:
                break
            if self._stop_event.is_set():
                continue
            if self._backend == "sapi":
                with suppress(Exception):
                    self._sapi.say(sentence)  # type: ignore[union-attr]
                    if self._sapi is not None:
                        self._sapi.runAndWait()
                self._mark_ended()
                continue
            audio = self._synthesize(sentence)
            if audio is None:
                self._mark_ended()
                continue
            try:
                self._audio_queue.put(audio, timeout=10.0)
            except queue.Full:  # pragma: no cover
                logger.debug("Cola de audio llena: se descarta una frase.")

    def _synthesize(self, sentence: str) -> tuple[np.ndarray, int] | None:
        """Genera el audio de una frase con el motor activo."""
        if self._backend == "kokoro" and self._kokoro is not None:
            try:
                samples, sample_rate = self._kokoro.create(
                    sentence,
                    voice=self.voice_name,
                    speed=cfg.TTS_KOKORO_SPEED,
                    lang=cfg.TTS_KOKORO_LANG,
                )
                return np.asarray(samples, dtype=np.float32), int(sample_rate)
            except Exception as exc:
                logger.warning("Kokoro falló con esta frase (%s); se reintenta con Piper.", exc)
                if not self._try_piper():
                    return None
                self._backend = "piper"

        if self._backend == "piper" and self._piper is not None:
            try:
                chunks: list[np.ndarray] = []
                sample_rate = cfg.TTS_SAMPLE_RATE
                for chunk in self._piper.synthesize(sentence):
                    sample_rate = getattr(chunk, "sample_rate", sample_rate)
                    raw = getattr(chunk, "audio_int16_bytes", None)
                    if raw is None:
                        raw = getattr(chunk, "audio", b"")
                    chunks.append(np.frombuffer(bytes(raw), dtype=np.int16))
                if not chunks:
                    return None
                audio = np.concatenate(chunks).astype(np.float32) / 32768.0
                return audio, int(sample_rate)
            except Exception as exc:
                logger.error("Piper falló: %s", exc)
                return None
        return None

    def _player_loop(self) -> None:
        """Reproduce el audio generado, en trozos cortos para poder cortar ya."""
        while not self._closed:
            try:
                item = self._audio_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if item is None:
                break
            if self._stop_event.is_set():
                continue
            self._speaking.set()
            self._play_block(item)
            if self._audio_queue.empty():
                self._mark_ended()

    def _play_block(self, item: tuple[np.ndarray, int]) -> None:
        """Escribe el audio en la tarjeta de sonido, troceado en 80 ms."""
        audio, sample_rate = item
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            logger.debug("sounddevice no disponible: no se puede reproducir voz.")
            return

        chunk = max(1, int(sample_rate * 0.08))
        try:
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=cfg.AUDIO_OUTPUT_DEVICE,
            ) as stream:
                for start in range(0, len(audio), chunk):
                    if self._stop_event.is_set() or self._closed:
                        break
                    block = audio[start : start + chunk]
                    stream.write(block.reshape(-1, 1))
        except Exception as exc:
            logger.warning("No he podido reproducir la voz: %s", exc)

    def _mark_ended(self) -> None:
        """Marca el final de una locución (si no queda nada más en cola)."""
        if self._synth_queue.empty() and self._audio_queue.empty():
            self._speaking.clear()
            if not self._stop_event.is_set():
                self._set_state(STATE_IDLE)

    def _set_state(self, state: str) -> None:
        if self.notifier is not None:
            with suppress(Exception):
                self.notifier(state, "")

    def status(self) -> dict:
        return {
            "motor": self._backend,
            "voz": self.voice_name,
            "hablando": self.speaking,
            **self.stats,
        }


# =============================================================================
# 5. WAKE WORD — "JARVIS" (OPENWAKEWORD, <1% DE CPU)
# =============================================================================

class WakeWordEngine:
    """Escucha la palabra "Jarvis" con openWakeWord (modelo `hey_jarvis`).

    Consume tramas de 80 ms (1280 muestras a 16 kHz), que es exactamente lo que
    exige el modelo. Su coste en CPU es inferior al 1% en un Ryzen 7840HS, así
    que puede estar escuchando todo el día.
    """

    def __init__(self, threshold: float = cfg.WAKE_WORD_THRESHOLD, model_name: str = cfg.WAKE_WORD_MODEL) -> None:
        self.threshold = float(threshold)
        self.model_name = model_name
        self._model: Any | None = None
        self._buffer = np.zeros(0, dtype=np.int16)
        self._unavailable_reason = ""
        self.stats = {"tramas": 0, "detecciones": 0, "ultima_puntuacion": 0.0}
        self._last_detection = 0.0

    @property
    def available(self) -> bool:
        """¿Se ha podido cargar el modelo de la palabra clave?"""
        return self._model is not None

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    def load(self) -> bool:
        """Carga (y si hace falta descarga) el modelo de openWakeWord."""
        if self._model is not None:
            return True
        try:
            from openwakeword.model import Model  # type: ignore
        except ImportError:
            self._unavailable_reason = "falta la librería openwakeword"
            logger.warning(
                "Palabra clave desactivada: %s. JARVIS escuchará de forma continua.",
                self._unavailable_reason,
            )
            return False

        model_path = self._ensure_model_file()
        try:
            if model_path and Path(model_path).exists():
                self._model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
                logger.info("Palabra clave 'Jarvis' cargada desde %s.", Path(model_path).name)
            else:
                self._model = Model(inference_framework="onnx")
                logger.info("Palabra clave 'Jarvis' cargada (modelos integrados).")
            return True
        except Exception as exc:
            self._unavailable_reason = str(exc)
            logger.warning("No he podido cargar la palabra clave: %s", exc)
            return False

    @staticmethod
    def _ensure_model_file() -> Path | None:
        """Comprueba (y descarga la primera vez) el modelo `hey_jarvis`."""
        target = Path(cfg.WAKE_MODELS_DIR) / cfg.WAKE_WORD_MODEL_FILE
        if target.exists():
            return target
        try:
            from openwakeword.utils import download_models  # type: ignore

            logger.info("Descargando el modelo de la palabra clave (una sola vez)...")
            download_models(model_names=[cfg.WAKE_WORD_MODEL], target_directory=str(cfg.WAKE_MODELS_DIR))
        except Exception as exc:
            logger.debug("No se pudo descargar el modelo de wake word: %s", exc)
            return None
        return target if target.exists() else None

    def feed(self, frame: np.ndarray) -> bool:
        """Añade audio y devuelve True si acaba de oírse "Jarvis"."""
        if self._model is None:
            return False
        self.stats["tramas"] += 1
        self._buffer = np.concatenate([self._buffer, frame.astype(np.int16)])

        detected = False
        while len(self._buffer) >= cfg.WAKE_WORD_CHUNK_SAMPLES:
            chunk = self._buffer[: cfg.WAKE_WORD_CHUNK_SAMPLES]
            self._buffer = self._buffer[cfg.WAKE_WORD_CHUNK_SAMPLES :]
            try:
                predictions = self._model.predict(chunk)
            except Exception as exc:
                logger.debug("Error del modelo de wake word: %s", exc)
                continue
            best = 0.0
            for _name, score in (predictions or {}).items():
                best = max(best, float(score))
            self.stats["ultima_puntuacion"] = round(best, 3)
            now = time.monotonic()
            if best >= self.threshold and (now - self._last_detection) > 1.5:
                self._last_detection = now
                self.stats["detecciones"] += 1
                logger.info("Palabra clave detectada (%.2f).", best)
                detected = True
        return detected

    def status(self) -> dict:
        return {
            "modelo": self.model_name,
            "umbral": self.threshold,
            "disponible": self.available,
            "motivo": self._unavailable_reason,
            **self.stats,
        }


# =============================================================================
# 6. MOTOR DE VOZ (ORQUESTADOR)
# =============================================================================

@dataclass
class Utterance:
    """Una frase de Pablo lista para transcribir."""

    audio: np.ndarray
    duration_s: float
    started_at: float
    reason: str = ""


class VoiceEngine:
    """Une micrófono, VAD, wake word, STT, TTS y barge-in en un solo bucle.

    Parameters
    ----------
    on_command:
        Función que recibe el texto de Pablo y devuelve la respuesta que JARVIS
        debe decir (o None para no decir nada). Aquí se conecta `main.py`.
    on_state:
        Función `(estado, detalle)` con la que el HUD se pinta.
    killswitch:
        Interruptor de emergencia.
    clap_trigger:
        Objeto con `on_frame(trama)` para la doble palmada (puede ser None).
    """

    def __init__(
        self,
        on_command: Callable[[str], str | None] | None = None,
        on_state: Callable[[str, str], None] | None = None,
        killswitch: Any | None = None,
        clap_trigger: Any | None = None,
        on_wake: Callable[[], None] | None = None,
        allow_wake_word: bool = cfg.WAKE_WORD_ENABLED,
        allow_barge_in: bool = True,
    ) -> None:
        self.on_command = on_command
        self.on_state_cb = on_state
        self.on_wake = on_wake
        self.killswitch = killswitch
        self.clap_trigger = clap_trigger

        self.stt = SpeechRecognizer()
        self.tts = TextToSpeech(notifier=self._on_tts_state)
        self.vad = VoiceActivityDetector()
        self.hub = AudioHub(on_frame=self._on_frame)
        self.barge_in = BargeInController(
            on_interrupt=self._on_barge_in,
            is_speaking=lambda: self.tts.speaking,
            is_automating=lambda: self._automating,
            killswitch=killswitch,
            notifier=self._on_tts_state,
        )
        self.wake = WakeWordEngine()
        self._allow_wake_word = allow_wake_word
        self._allow_barge_in = allow_barge_in

        self.state: str = STATE_SLEEPING if allow_wake_word else STATE_IDLE
        self.state_detail = ""
        self._state_lock = threading.Lock()
        self._automating = False
        self._running = threading.Event()
        self._shutdown = threading.Event()
        self._capture: list[np.ndarray] = []
        self._capturing = False
        self._silence_ms = 0.0
        self._voiced_ms = 0.0
        self._capture_started = 0.0
        self._last_activity = time.monotonic()
        self._listen_thread: threading.Thread | None = None
        self._process_lock = threading.Lock()
        self.history: list[dict] = []

        # El kill-switch debe poder callar a JARVIS desde el primer segundo,
        # incluso antes de que la escucha esté activa.
        if killswitch is not None:
            killswitch.register("voice_engine", self._on_killswitch)

    # --- Estados -----------------------------------------------------------

    def set_state(self, state: str, detail: str = "") -> None:
        """Cambia el estado visible de JARVIS (lo pinta el HUD)."""
        if state not in VALID_STATES:
            logger.debug("Estado desconocido ignorado: %s", state)
            return
        with self._state_lock:
            if state == self.state and detail == self.state_detail:
                return
            self.state = state
            self.state_detail = detail
        logger.debug("Estado: %s %s", state, f"({detail})" if detail else "")
        if self.on_state_cb is not None:
            with suppress(Exception):
                self.on_state_cb(state, detail)

    def _on_tts_state(self, state: str, detail: str = "") -> None:
        """El TTS avisa de que empieza o acaba de hablar."""
        if state == STATE_SPEAKING:
            self.set_state(STATE_SPEAKING, detail)
        elif self.state == STATE_SPEAKING and state in (STATE_IDLE, STATE_LISTENING):
            self.set_state(STATE_LISTENING if self._allow_wake_word else STATE_IDLE, detail)

    @property
    def speaking(self) -> bool:
        return self.tts.speaking

    @property
    def automating(self) -> bool:
        return self._automating

    def note_automation(self, active: bool) -> None:
        """El actuador avisa de que empieza o termina una automatización."""
        self._automating = active
        if not active and self.barge_in.gate.frozen:
            logger.debug("La automatización terminó congelada: se espera a Pablo.")

    # --- Ciclo de vida -----------------------------------------------------

    def start(self) -> bool:
        """Arranca voz y oídos. Devuelve False si no hay ningún audio posible."""
        self._shutdown.clear()
        tts_ok = self.tts.start()

        if self._allow_wake_word:
            self.wake.load()
        self.barge_in.arm()

        audio_ok = self.hub.start()
        if not audio_ok:
            if self._allow_wake_word:
                logger.warning(
                    "Sin micrófono: JARVIS funcionará en modo texto "
                    "(escribe tus órdenes en esta ventana)."
                )
                self.set_state(STATE_IDLE, "modo texto")

        self._running.set()
        self.stt.preload()
        if self.killswitch is not None:
            self.killswitch.register("voice_engine", self._on_killswitch)
        logger.info(
            "Motor de voz listo (micrófono=%s, voz=%s, palabra clave=%s).",
            audio_ok, tts_ok, self.wake.available,
        )
        return audio_ok or tts_ok

    def stop(self) -> None:
        """Detiene la escucha y la voz (pero mantiene el objeto utilizable)."""
        self._running.clear()
        self.hub.stop()
        self.tts.stop()

    def shutdown(self) -> None:
        """Apaga todo y libera recursos."""
        self.stop()
        self._shutdown.set()
        self.barge_in.disarm()
        self.tts.shutdown()
        if self.killswitch is not None:
            self.killswitch.unregister("voice_engine")

    # --- Interrupciones ----------------------------------------------------

    def _on_killswitch(self, event: Any) -> None:
        """El kill-switch manda callar a JARVIS inmediatamente."""
        self.tts.stop()
        self.set_state(STATE_KILLSWITCH, "Control liberado")
        logger.warning("Motor de voz silenciado por el kill-switch.")

    def _on_barge_in(self, event: BargeInEvent) -> None:
        """Pablo ha hablado: se corta la voz y se pasa a escuchar."""
        if self.tts.speaking:
            self.tts.stop()
        self.set_state(STATE_LISTENING, "Le escucho")
        self._start_capture(reason="barge_in")

    # --- Audio entrante ----------------------------------------------------

    def _on_frame(self, frame: np.ndarray) -> None:
        """Reparte cada trama de micrófono a todos los consumidores interesados."""
        if self._shutdown.is_set():
            return
        self._last_activity = time.monotonic()

        # 1) Doble palmada (si está conectada).
        if self.clap_trigger is not None:
            with suppress(Exception):
                self.clap_trigger.on_frame(frame)

        # 2) Interrupción de la propia voz / automatización.
        if self._allow_barge_in and self.tts.speaking:
            with suppress(Exception):
                self.barge_in.on_frame(frame)

        # 3) Palabra clave (solo si estamos dormidos).
        if self._allow_wake_word and not self._capturing and not self.tts.speaking:
            if self.wake.available and self.state in (STATE_SLEEPING, STATE_IDLE):
                if self.wake.feed(frame):
                    self._on_wake_word()

        # 4) Captura de la frase de Pablo.
        if self._capturing:
            self._capture_frame(frame)

    def _on_wake_word(self) -> None:
        """Se ha oído "Jarvis": a escuchar."""
        logger.info("¡Despierto! Escuchando a Pablo.")
        self.set_state(STATE_LISTENING, "Le escucho")
        if cfg.NATIVE_SOUND_ACK:
            with suppress(Exception):
                if sys.platform == "win32":
                    import winsound

                    winsound.Beep(cfg.ACK_FREQUENCY_HZ, cfg.ACK_DURATION_MS)
        if self.on_wake is not None:
            with suppress(Exception):
                self.on_wake()
        self._start_capture(reason="wake_word")

    def _start_capture(self, reason: str = "") -> None:
        """Empieza a acumular audio hasta que Pablo termine de hablar."""
        self._capture = []
        self._capturing = True
        self._silence_ms = 0.0
        self._voiced_ms = 0.0
        self._capture_started = time.monotonic()
        self._capture_reason = reason

    def _stop_capture(self) -> None:
        self._capturing = False

    def _capture_frame(self, frame: np.ndarray) -> None:
        """Añade audio a la frase en curso y decide cuándo ha terminado."""
        now = time.monotonic()
        self._capture.append(frame)
        speaking_now = self.vad.is_speech(frame, boost=1.0 if not self.tts.speaking else 1.8)

        if speaking_now:
            self._voiced_ms += cfg.AUDIO_FRAME_MS
            self._silence_ms = 0.0
        else:
            self._silence_ms += cfg.AUDIO_FRAME_MS

        elapsed_ms = (now - self._capture_started) * 1000.0
        voiced_enough = self._voiced_ms >= cfg.VAD_SPEECH_TRIGGER_MS
        finished = voiced_enough and self._silence_ms >= cfg.VAD_SILENCE_HANGOVER_MS
        too_long = elapsed_ms >= cfg.WAKE_WORD_ACTIVE_WINDOW_S * 1000.0
        no_speech = not voiced_enough and elapsed_ms >= 2500.0

        if finished or too_long or no_speech:
            audio = np.concatenate(self._capture) if self._capture else np.zeros(0, dtype=np.int16)
            self._stop_capture()
            self._capture = []
            if not voiced_enough or len(audio) < cfg.AUDIO_SAMPLE_RATE * cfg.STT_MIN_AUDIO_S:
                logger.debug("Frase descartada (%.0f ms con voz).", self._voiced_ms)
                if self._allow_wake_word:
                    self.set_state(STATE_SLEEPING, "")
                return
            run_in_thread(self._transcribe_and_dispatch, audio, self._capture_reason, name="jarvis-voz-stt")

    def _transcribe_and_dispatch(self, audio: np.ndarray, reason: str = "") -> None:
        """Transcribe la frase y se la pasa al cerebro de JARVIS."""
        self.set_state(STATE_TRANSCRIBING, "")
        text = self.stt.transcribe(audio)
        if not text:
            self.set_state(STATE_LISTENING if self._allow_wake_word else STATE_IDLE, "")
            self._maybe_sleep()
            return
        self.dispatch(text, source=reason or "voz")

    def dispatch(self, text: str, source: str = "texto") -> str | None:
        """Envía una orden al cerebro y dice la respuesta que devuelva."""
        clean = " ".join(str(text or "").split())
        if not clean:
            return None
        if self.killswitch is not None and self.killswitch.is_tripped():
            self.set_state(STATE_KILLSWITCH, "Control liberado")
            return None

        with self._process_lock:
            self.history.append({"momento": time.time(), "origen": source, "texto": clean})
            self.history = self.history[-40:]
            self.set_state(STATE_THINKING, clean[:60])
            reply: str | None = None
            if self.on_command is not None:
                try:
                    reply = self.on_command(clean)
                except Exception as exc:
                    logger.error("El cerebro de JARVIS falló procesando la orden: %s", exc)
                    reply = cfg.PHRASES["error"]
            if reply:
                self.say(reply, blocking=False)
            else:
                self._maybe_sleep()
        return reply

    # --- Salida de voz -----------------------------------------------------

    def say(self, text: str, blocking: bool = False, interrupt: bool = False) -> None:
        """Dice algo en voz alta. Si `interrupt`, corta lo que estuviera diciendo."""
        if interrupt and self.tts.speaking:
            self.tts.stop()
        if self.clap_trigger is not None:
            with suppress(Exception):
                # JARVIS no debe oírse a sí mismo como si fueran palmadas.
                self.clap_trigger.detector.suspend(
                    max(0.6, len(str(text)) / 14.0) + cfg.CLAP_COOLDOWN_AFTER_WAKE_MS / 1000.0
                )
        self.tts.say(text, blocking=blocking)

    def stop_speaking(self) -> None:
        """Corta la voz (lo usa el kill-switch y el barge-in)."""
        self.tts.stop()

    # --- Bucles ------------------------------------------------------------

    def listen_loop(self) -> None:
        """Mantiene vivo el motor (el audio lo mueve el callback del hub)."""
        self._listen_thread = threading.current_thread()
        while self._running.is_set() and not self._shutdown.is_set():
            self._maybe_sleep()
            time.sleep(cfg.MAIN_LOOP_SLEEP_S)

    def _maybe_sleep(self) -> None:
        """Vuelve a dormir tras un tiempo sin actividad (ahorra CPU)."""
        if not self._allow_wake_word or not self.wake.available:
            self.set_state(STATE_IDLE, "")
            return
        if self._capturing or self.tts.speaking:
            return
        if time.monotonic() - self._last_activity >= cfg.WAKE_WORD_SLEEP_AFTER_S:
            self.set_state(STATE_SLEEPING, "")

    def listen_once(self, timeout: float = 8.0, say_beep: bool = True) -> str:
        """Escucha UNA frase y la devuelve transcrita (bloqueante).

        Es la herramienta que usan el modo texto, el diagnóstico y las pruebas
        manuales de micrófono.
        """
        if say_beep and cfg.NATIVE_SOUND_ACK and sys.platform == "win32":
            with suppress(Exception):
                import winsound

                winsound.Beep(cfg.ACK_FREQUENCY_HZ, cfg.ACK_DURATION_MS)
        self.set_state(STATE_LISTENING, "Escuchando...")
        self._start_capture(reason="listen_once")
        deadline = time.monotonic() + timeout
        while self._capturing and time.monotonic() < deadline:
            time.sleep(0.05)
        self._stop_capture()
        audio = np.concatenate(self._capture) if self._capture else np.zeros(0, dtype=np.int16)
        self._capture = []
        if len(audio) < cfg.AUDIO_SAMPLE_RATE * cfg.STT_MIN_AUDIO_S:
            return ""
        self.set_state(STATE_TRANSCRIBING, "")
        return self.stt.transcribe(audio)

    def text_loop(self, prompt: str = "Tú > ") -> None:
        """Modo consola: escribe órdenes por teclado (sin micrófono).

        Es el modo de pruebas y también la red de seguridad si el audio falla:
        `python main.py --texto`.
        """
        print("\n" + "=" * 66)
        print("  MODO TEXTO — escribe tus órdenes y pulsa Enter.")
        print("  Escribe 'salir' para terminar.")
        print("=" * 66 + "\n")
        while not self._shutdown.is_set():
            try:
                text = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if normalize_text(text) in {"salir", "adios", "exit", "quit", "cerrar jarvis"}:
                self.say("Hasta luego, Pablo.")
                break
            self.dispatch(text, source="texto")

    # --- Diagnóstico y pruebas --------------------------------------------

    def speak_test(self) -> bool:
        """Prueba rápida de la voz (para install.py y el diagnóstico)."""
        if not self.tts.start():
            logger.error("No hay ningún motor de voz disponible.")
            return False
        self.tts.say("Hola Pablo. Soy JARVIS. La voz funciona correctamente.", blocking=True)
        return True

    def status(self) -> dict:
        """Informe completo del subsistema de voz."""
        return {
            "estado": self.state,
            "detalle": self.state_detail,
            "microfono": self.hub.status(),
            "palabra_clave": self.wake.status(),
            "transcripcion": self.stt.status(),
            "sintesis": self.tts.status(),
            "interrupcion": self.barge_in.status(),
            "frases_recientes": [h["texto"] for h in self.history[-5:]],
        }

    def simulate_command(self, text: str, reply: str | None = None) -> None:
        """Inyecta una orden como si la hubiera dicho Pablo (pruebas automáticas)."""
        if reply is not None:
            original = self.on_command

            def _fixed(_text: str) -> str:
                return reply

            self.on_command = _fixed
            try:
                self.dispatch(text, source="simulacion")
            finally:
                self.on_command = original
        else:
            self.dispatch(text, source="simulacion")
