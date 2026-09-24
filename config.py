"""
config.py — Configuración central de JARVIS.
================================================================================
Este archivo es el "panel de control" de todo el sistema. Ninguna otra parte del
código guarda valores mágicos: todo se lee desde aquí, de modo que Pablo pueda
ajustar el comportamiento de JARVIS editando un único archivo.

Se puede sobreescribir cualquier valor sin tocar este archivo creando un
`local_config.py` en la raíz del proyecto (ver `_apply_local_overrides`).

IMPORTANTE: este módulo no importa ninguna dependencia externa pesada. Debe
poder importarse siempre, incluso antes de instalar `requirements.txt`, para
que el diagnóstico de arranque pueda explicar en español qué falta.
================================================================================
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import sys
from pathlib import Path

# =============================================================================
# 0. IDENTIDAD
# =============================================================================

APP_NAME = "JARVIS"
APP_TITLE = "J.A.R.V.I.S. — Asistente de escritorio local"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Pablo"
USER_NAME = "Pablo"  # Nombre con el que JARVIS se dirige a su dueño.
APP_LANGUAGE = "es"

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"

# =============================================================================
# 1. RUTAS DEL PROYECTO
# =============================================================================

ROOT_DIR = Path(__file__).resolve().parent
CORE_DIR = ROOT_DIR / "core"
GUI_DIR = ROOT_DIR / "gui"
SAFETY_DIR = ROOT_DIR / "safety"
TESTS_DIR = ROOT_DIR / "tests"
DOCS_DIR = ROOT_DIR / "docs"

WORKSPACE_DIR = ROOT_DIR / "workspace"
PROJECTS_DIR = WORKSPACE_DIR / "projects"
SANDBOX_VENVS_DIR = WORKSPACE_DIR / ".venvs"

LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"          # Modelos ONNX/Whisper descargados.
TTS_MODELS_DIR = MODELS_DIR / "tts"
WAKE_MODELS_DIR = MODELS_DIR / "wakeword"
STT_MODELS_DIR = MODELS_DIR / "whisper"
CACHE_DIR = ROOT_DIR / ".cache"
STATE_DIR = ROOT_DIR / "state"

LOG_FILE = LOGS_DIR / "jarvis.log"
HISTORY_FILE = STATE_DIR / "conversation_history.jsonl"
PID_FILE = STATE_DIR / "jarvis.pid"
STARTUP_LOG = LOGS_DIR / "arranque.log"

ALL_DIRECTORIES = (
    WORKSPACE_DIR,
    PROJECTS_DIR,
    SANDBOX_VENVS_DIR,
    LOGS_DIR,
    MODELS_DIR,
    TTS_MODELS_DIR,
    WAKE_MODELS_DIR,
    STT_MODELS_DIR,
    CACHE_DIR,
    STATE_DIR,
)

# =============================================================================
# 2. OLLAMA — EL MOTOR DE LOS MODELOS DE LENGUAJE
# =============================================================================

OLLAMA_HOST = os.environ.get("JARVIS_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_BIN = os.environ.get("JARVIS_OLLAMA_BIN", "ollama")  # Se busca en el PATH.

OLLAMA_API_CHAT = f"{OLLAMA_HOST}/api/chat"
OLLAMA_API_GENERATE = f"{OLLAMA_HOST}/api/generate"
OLLAMA_API_TAGS = f"{OLLAMA_HOST}/api/tags"
OLLAMA_API_PS = f"{OLLAMA_HOST}/api/ps"
OLLAMA_API_PULL = f"{OLLAMA_HOST}/api/pull"
OLLAMA_API_SHOW = f"{OLLAMA_HOST}/api/show"

OLLAMA_HEALTH_TIMEOUT = 2.5        # Segundos para saber si Ollama está vivo.
OLLAMA_SHORT_TIMEOUT = 30.0        # Operaciones administrativas (ps, unload).
OLLAMA_CHAT_TIMEOUT = 300.0        # Chat normal.
OLLAMA_CODE_TIMEOUT = 900.0        # Generación de código larga.
OLLAMA_VISION_TIMEOUT = 240.0      # Inferencia de visión sobre captura.
OLLAMA_PULL_TIMEOUT = 7200.0       # Descarga de modelos (2 h máx).

OLLAMA_RETRY_ATTEMPTS = 3
OLLAMA_RETRY_BACKOFF_S = 1.5

# =============================================================================
# 3. MATRIZ DE DISTRIBUCIÓN DE MODELOS (DINÁMICA)
# =============================================================================

MODEL_CHAT = "llama3.1:8b"            # Chat, planificación, despacho, routing.
MODEL_VISION = "llama3.2-vision:latest"  # Percepción de pantalla / UI.
MODEL_CODER = "qwen2.5-coder:7b"      # Auto-programación y generación de apps.

# Modelos que `install.bat` descarga automáticamente en el primer arranque.
# (El de chat se asume presente o se descarga también si falta.)
MODELS_TO_AUTO_PULL = (MODEL_CHAT, MODEL_VISION, MODEL_CODER)

# --- Catálogo de tareas -> modelo ------------------------------------------
# El enrutador dinámico traduce una "tarea lógica" a un modelo concreto.
TASK_CHAT = "chat"
TASK_PLANNING = "planning"
TASK_DISPATCH = "dispatch"
TASK_ROUTING = "routing"
TASK_SUMMARY = "summary"
TASK_VISION = "vision"
TASK_SCREEN_PERCEPTION = "screen_perception"
TASK_UI_DETECTION = "ui_detection"
TASK_CODE = "code"
TASK_SELF_PROGRAMMING = "self_programming"
TASK_APP_BUILDER = "app_builder"
TASK_WEB_GENERATION = "web_generation"
TASK_TESTS = "tests"

TASK_ROUTES: dict[str, str] = {
    TASK_CHAT: MODEL_CHAT,
    TASK_PLANNING: MODEL_CHAT,
    TASK_DISPATCH: MODEL_CHAT,
    TASK_ROUTING: MODEL_CHAT,
    TASK_SUMMARY: MODEL_CHAT,
    TASK_VISION: MODEL_VISION,
    TASK_SCREEN_PERCEPTION: MODEL_VISION,
    TASK_UI_DETECTION: MODEL_VISION,
    TASK_CODE: MODEL_CODER,
    TASK_SELF_PROGRAMMING: MODEL_CODER,
    TASK_APP_BUILDER: MODEL_CODER,
    TASK_WEB_GENERATION: MODEL_CODER,
    TASK_TESTS: MODEL_CODER,
}

# Descripción humana de cada tarea (usada por el README y por el log de swaps).
TASK_DESCRIPTIONS: dict[str, str] = {
    TASK_CHAT: "Conversación natural, memoria y personalidad",
    TASK_PLANNING: "Descomposición de órdenes complejas en pasos",
    TASK_DISPATCH: "Decisión de qué habilidad ejecutar",
    TASK_ROUTING: "Clasificación de intenciones",
    TASK_SUMMARY: "Resúmenes y síntesis de información",
    TASK_VISION: "Comprensión visual de la pantalla",
    TASK_SCREEN_PERCEPTION: "Descripción de lo que se ve en pantalla",
    TASK_UI_DETECTION: "Localización de botones y campos de UI",
    TASK_CODE: "Escritura y corrección de código",
    TASK_SELF_PROGRAMMING: "Añadir funcionalidades nuevas a JARVIS",
    TASK_APP_BUILDER: "Construcción de apps y webs completas",
    TASK_WEB_GENERATION: "HTML, CSS y JavaScript",
    TASK_TESTS: "Generación de pruebas automáticas",
}

# --- Parámetros de inferencia por tarea ------------------------------------
# `num_ctx` se limita deliberadamente para no desbordar los 8 GB de VRAM.
TASK_OPTIONS: dict[str, dict] = {
    TASK_CHAT: {"temperature": 0.7, "num_ctx": 8192, "num_predict": 512, "top_p": 0.9},
    TASK_PLANNING: {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1024, "top_p": 0.9},
    TASK_DISPATCH: {"temperature": 0.0, "num_ctx": 4096, "num_predict": 256, "top_p": 1.0},
    TASK_ROUTING: {"temperature": 0.0, "num_ctx": 4096, "num_predict": 128, "top_p": 1.0},
    TASK_SUMMARY: {"temperature": 0.3, "num_ctx": 8192, "num_predict": 512, "top_p": 0.9},
    TASK_VISION: {"temperature": 0.1, "num_ctx": 4096, "num_predict": 1024, "top_p": 0.9},
    TASK_SCREEN_PERCEPTION: {"temperature": 0.1, "num_ctx": 4096, "num_predict": 768, "top_p": 0.9},
    TASK_UI_DETECTION: {"temperature": 0.0, "num_ctx": 4096, "num_predict": 1024, "top_p": 1.0},
    TASK_CODE: {"temperature": 0.15, "num_ctx": 8192, "num_predict": 4096, "top_p": 0.95},
    TASK_SELF_PROGRAMMING: {"temperature": 0.15, "num_ctx": 8192, "num_predict": 4096, "top_p": 0.95},
    TASK_APP_BUILDER: {"temperature": 0.2, "num_ctx": 8192, "num_predict": 4096, "top_p": 0.95},
    TASK_WEB_GENERATION: {"temperature": 0.2, "num_ctx": 8192, "num_predict": 4096, "top_p": 0.95},
    TASK_TESTS: {"temperature": 0.1, "num_ctx": 8192, "num_predict": 2048, "top_p": 0.95},
}

# Tiempo que Ollama mantiene el modelo en memoria tras cada tarea.
# `keep_alive: 0` = descarga inmediata (ley de monogamia de VRAM).
KEEP_ALIVE_CHAT = "10m"     # El modelo conversacional se queda: es el más usado.
KEEP_ALIVE_VISION = 0       # La visión se descarga EN CUANTO termina.
KEEP_ALIVE_CODER = "3m"     # El programador sobrevive a ráfagas de edición.
KEEP_ALIVE_UNLOAD = 0       # Valor explícito de descarga.

# =============================================================================
# 4. LEY DE MONOGAMIA DE VRAM (ZERO-OVERLAP DYNAMIC SWAPPER)
# =============================================================================

VRAM_TOTAL_GB = 8.0            # RTX 4060 Laptop.
VRAM_SAFE_BUDGET_GB = 6.5      # Nunca reservamos los 8 GB completos.
VRAM_BASELINE_TARGET_GB = 3.9  # Objetivo en reposo (SO + HUD + audio).
VRAM_UNLOAD_WAIT_S = 12.0      # Espera máxima a que Ollama libere la VRAM.
VRAM_UNLOAD_POLL_S = 0.25      # Frecuencia de comprobación durante la espera.
ENFORCE_VRAM_MONOGAMY = True   # Si es True: un solo LLM en GPU, siempre.

# Política de "carga justa": los modelos voluminosos se cargan bajo demanda y
# se expulsan inmediatamente después de inferir.
EPHEMERAL_MODELS = (MODEL_VISION,)

# Comprobación pasiva con nvidia-smi (si no existe, se ignora silenciosamente).
NVIDIA_SMI_BIN = "nvidia-smi"
GPU_METRICS_ENABLED = True
GPU_QUERY_TIMEOUT_S = 3.0

# =============================================================================
# 5. AUDIO — CAPTURA COMÚN (un solo micrófono para todo el sistema)
# =============================================================================

AUDIO_SAMPLE_RATE = 16000      # 16 kHz: el estándar de Whisper y openWakeWord.
AUDIO_CHANNELS = 1
AUDIO_DTYPE = "int16"
AUDIO_FRAME_MS = 20            # Trama de análisis de 20 ms (320 muestras).
AUDIO_FRAME_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_FRAME_MS // 1000
AUDIO_INPUT_DEVICE = None      # None = dispositivo predeterminado del sistema.
AUDIO_OUTPUT_DEVICE = None
AUDIO_QUEUE_MAXSIZE = 256      # Trames en cola antes de descartar (anti-bloqueo).
AUDIO_STREAM_BLOCKSIZE = AUDIO_FRAME_SAMPLES * 2

# =============================================================================
# 6. VOICE ACTIVITY DETECTION (VAD) — el oído que nunca duerme
# =============================================================================

VAD_FRAME_MS = 20
VAD_ENERGY_THRESHOLD = 0.012         # RMS mínimo para considerar "voz".
VAD_ADAPTIVE_NOISE = True            # Aprende el ruido ambiente de la habitación.
VAD_NOISE_FLOOR_INIT = 0.0035
VAD_NOISE_FLOOR_ALPHA = 0.995        # Suavizado exponencial del suelo de ruido.
VAD_NOISE_MARGIN = 3.2               # voz si RMS > ruido_ambiente * margen.
VAD_SPEECH_TRIGGER_MS = 120          # Voz confirmada tras 120 ms continuos.
VAD_SILENCE_HANGOVER_MS = 700        # Silencio final que cierra la frase.
VAD_MIN_VOICE_RATIO = 0.35           # Proporción de tramas con voz en la frase.
VAD_USE_WEBRTCVAD = True             # Se usa si la librería está instalada.
VAD_WEBRTC_AGGRESSIVENESS = 2        # 0-3: 2 = equilibrado para micrófono de portátil.

# =============================================================================
# 7. SPEECH-TO-TEXT (faster-whisper)
# =============================================================================

STT_ENGINE = "faster-whisper"
STT_MODEL = "base"                 # 'base' = mejor relación precisión/velocidad.
STT_DEVICE = "cpu"                 # CPU a propósito: la VRAM es del LLM.
STT_COMPUTE_TYPE = "int8"          # Cuantización: 2x más rápido en CPU, -50% RAM.
STT_LANGUAGE = "es"                # Español.
STT_BEAM_SIZE = 1
STT_BEST_OF = 1
STT_TEMPERATURE = 0.0
STT_CONDITION_ON_PREVIOUS = False  # Evita que repita frases anteriores.
STT_VAD_FILTER = True              # Filtro interno de Silero para descartar ruido.
STT_DOWNLOAD_ROOT = str(STT_MODELS_DIR)
STT_CPU_THREADS = 4                # 8 núcleos disponibles; 4 es el punto dulce.
STT_NUM_WORKERS = 1
STT_MIN_AUDIO_S = 0.30             # Frases más cortas se descartan.
STT_MAX_AUDIO_S = 30.0             # Corte duro de seguridad.
STT_HALLUCINATION_BLACKLIST = (
    "subtítulos realizados por la comunidad de amara.org",
    "subtítulos por la comunidad de amara.org",
    "gracias por ver el vídeo",
    "gracias por ver el video",
    "¡suscríbete!",
    "amara.org",
    "[música]",
    "(música)",
    "¡gracias!",
    "thank you.",
)
STT_INITIAL_PROMPT = (
    "Conversación en español con JARVIS, el asistente de escritorio de Pablo. "
    "Se dicen órdenes cortas: abrir aplicaciones, buscar en internet, "
    "hacer clic en botones, programar, música, hora y fecha."
)

# =============================================================================
# 8. TEXT-TO-SPEECH (Kokoro-82M -> Piper -> SAPI de Windows)
# =============================================================================

TTS_ENGINE_ORDER = ("kokoro", "piper", "sapi", "null")
TTS_KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    "kokoro-v1.0.int8.onnx"
)
TTS_KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    "voices-v1.0.bin"
)
TTS_KOKORO_MODEL_FILE = "kokoro-v1.0.int8.onnx"
TTS_KOKORO_VOICES_FILE = "voices-v1.0.bin"
TTS_KOKORO_VOICE = "ef_dora"        # Voz femenina nativa en español (Kokoro v1.0).
TTS_KOKORO_LANG = "es"
TTS_KOKORO_SPEED = 1.0
TTS_KOKORO_FALLBACK_VOICE = "af_heart"   # Si `ef_dora` no existiera en el binario.

TTS_PIPER_VOICE = "es_ES-davefx-medium"
TTS_PIPER_FALLBACK_VOICE = "es_ES-sharvard-medium"
TTS_PIPER_VOICES_DIR = str(TTS_MODELS_DIR / "piper")

TTS_SAPI_VOICE_HINTS = ("español", "spanish", "helena", "sabina", "laura", "raul", "pablo")
TTS_SAPI_RATE = 0                      # -10 (lento) .. 10 (rápido)
TTS_SAPI_VOLUME = 1.0

TTS_SAMPLE_RATE = 24000                # Kokoro genera a 24 kHz.
TTS_SPEED = 1.0
TTS_CHUNK_MAX_CHARS = 180              # Frase corta = primera palabra antes.
TTS_SENTENCE_GAP_MS = 40               # Micropausa natural entre frases.
TTS_STOP_FADE_MS = 12                  # Corte del stream sin chasquido.
TTS_MAX_UTTERANCE_CHARS = 1200         # Corte de seguridad para respuestas.
TTS_PREBUFFER_CHUNKS = 1               # Sintetiza la siguiente frase mientras habla.
TTS_VOLUME = 0.9

# =============================================================================
# 9. WAKE WORD — openWakeWord con el modelo "Jarvis"
# =============================================================================

WAKE_WORD_ENABLED = True
WAKE_WORD_MODEL = "hey_jarvis"          # Modelo preentrenado oficial.
WAKE_WORD_MODEL_FILE = "hey_jarvis_v0.1.onnx"
WAKE_WORD_FALLBACK_MODEL_FILES = ("hey_jarvis_v0.1.onnx", "hey_jarvis_v0.1.tflite")
WAKE_WORD_THRESHOLD = 0.55              # Sensibilidad (0.5-0.6 = equilibrado).
WAKE_WORD_CHUNK_SAMPLES = 1280          # 80 ms exactos: requisito de openWakeWord.
WAKE_WORD_ACTIVE_WINDOW_S = 6.0         # Tras despertar, escucha directa este tiempo.
WAKE_WORD_SLEEP_AFTER_S = 25.0          # Vuelve a dormir tras esta inactividad.
WAKE_WORD_ACK_SOUND = True              # Pitido suave de confirmación.
WAKE_WORD_REQUIRE_NAME = True           # Debe decir "Jarvis" + la orden.
WAKE_WORD_ALIASES = (
    "jarvis", "yarvis", "jarbis", "harvis", "jarvi", "jarvis,",
    "oye jarvis", "hey jarvis", "hola jarvis",
)

# =============================================================================
# 10. DOBLE PALPADA (ACOUSTIC TRIGGER)
# =============================================================================

CLAP_ENABLED = True
CLAP_MIN_GAP_MS = 200               # Separación mínima válida entre palmas.
CLAP_MAX_GAP_MS = 750               # Separación máxima válida entre palmas.
CLAP_REFRACTORY_MS = 2500           # Tiempo muerto tras un disparo.
CLAP_MIN_PEAK_RMS = 0.055           # Volumen mínimo de una palmada real.
CLAP_NOISE_MULTIPLIER = 4.5         # Debe superar 4.5x el ruido ambiente.
CLAP_ATTACK_RATIO = 3.0             # Transitorio: subida brusca de energía.
CLAP_FLATNESS_MIN = 0.22            # Una palmada es ruido de banda ancha.
CLAP_FLATNESS_MAX = 0.95            # ...pero no un silbido/tono puro.
CLAP_SPECTRAL_CENTROID_MIN_HZ = 900  # Descarta golpes sordos (mesa, taza).
CLAP_MAX_CLAP_DURATION_MS = 60      # Una palmada dura menos de 60 ms.
CLAP_CONFIRM_MS = 250               # Espera tras la 2ª palmada: si llega una 3ª, no es orden.
CLAP_HISTORY_FRAMES = 64            # Ventana deslizante de detección de picos.
CLAP_MAX_TRIPLES = 3                # 3+ golpes seguidos = no es una orden.
CLAP_COOLDOWN_AFTER_WAKE_MS = 1200  # No confundir el "gracias" del TTS con palmas.

# --- Acción disparada por la doble palmada ---------------------------------
CLAP_TRACK_TITLE = "Loser"
CLAP_TRACK_ARTIST = "Tame Impala"
CLAP_TRACK_QUERY = f"{CLAP_TRACK_TITLE} {CLAP_TRACK_ARTIST}"
# URI opcional: pégalo aquí (spotify:track:XXXX) para reproducción exacta.
# Si está vacío, JARVIS usa la búsqueda profunda de Spotify o YouTube.
CLAP_SPOTIFY_URI = ""
CLAP_SPOTIFY_SEARCH_URI = f"spotify:search:{CLAP_TRACK_TITLE}%20{CLAP_TRACK_ARTIST}"
CLAP_SPOTIFY_PLAY_DELAY_S = 4.5     # Espera a que Spotify cargue antes del Play.
CLAP_YOUTUBE_URL = (
    "https://www.youtube.com/results?search_query="
    "loser+tame+impala"
)
CLAP_YOUTUBE_MUSIC_FIRST = True     # Intenta music.youtube.com si está disponible.

# =============================================================================
# 11. NAVEGADOR Y APLICACIONES
# =============================================================================

def _local_appdata(*parts: str) -> str:
    """Construye una ruta dentro de %LOCALAPPDATA% (Windows)."""
    base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local"))
    return os.path.join(base, *parts)


def _program_files(*parts: str) -> str:
    base = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    return os.path.join(base, *parts)


# Rutas candidatas de Comet (navegador de Perplexity). Se prueban en orden.
BROWSER_COMET_CANDIDATES = (
    _local_appdata("Perplexity", "Comet", "Application", "comet.exe"),
    _local_appdata("Perplexity", "Comet", "Application", "Comet.exe"),
    r"C:\Perplexity\Comet\Application\comet.exe",
    _program_files("Perplexity", "Comet", "Application", "comet.exe"),
)

# Navegadores de reserva si Comet no existe (en orden de preferencia).
BROWSER_FALLBACK_CANDIDATES = (
    _local_appdata("Google", "Chrome", "Application", "chrome.exe"),
    _program_files("Google", "Chrome", "Application", "chrome.exe"),
    _program_files("Microsoft", "Edge", "Application", "msedge.exe"),
    _program_files("Mozilla Firefox", "firefox.exe"),
    _local_appdata("Programs", "Opera", "launcher.exe"),
    _local_appdata("Programs", "BraveSoftware", "Brave-Browser", "Application", "brave.exe"),
    "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
)

BROWSER_PREFER_COMET = True
BROWSER_NEW_WINDOW = False
BROWSER_START_TIMEOUT_S = 6.0

SPOTIFY_CANDIDATES = (
    _local_appdata("Microsoft", "WindowsApps", "Spotify.exe"),
    _local_appdata("Spotify", "Spotify.exe"),
    r"C:\Program Files\WindowsApps\SpotifyAB.SpotifyMusic_zpdnekdrzrea0\Spotify.exe",
    _program_files("Spotify", "Spotify.exe"),
    "spotify",  # Confía en el PATH / protocolo de Windows.
)

# Apps conocidas para la habilidad "abrir <app>" (alias -> ejecutable/comando).
KNOWN_APPS: dict[str, str] = {
    "notepad": "notepad.exe",
    "bloc de notas": "notepad.exe",
    "calculadora": "calc.exe",
    "calculadora científica": "calc.exe",
    "explorador": "explorer.exe",
    "explorador de archivos": "explorer.exe",
    "terminal": "wt.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "administrador de tareas": "taskmgr.exe",
    "panel de control": "control.exe",
    "configuración": "ms-settings:",
    "ajustes": "ms-settings:",
    "spotify": r"spotify",
    "vscode": "code",
    "código": "code",
    "visual studio code": "code",
    "navegador": "comet",
    "comet": "comet",
    "chrome": "chrome",
    "word": "winword.exe",
    "excel": "excel.exe",
    "paint": "mspaint.exe",
    "captura de pantalla": "snippingtool.exe",
    "reproductor": "wmplayer.exe",
    "ollama": "ollama",
}

# =============================================================================
# 12. PERCEPCIÓN DE PANTALLA Y ACTUADORES
# =============================================================================

VISION_ENABLED = True
VISION_MONITOR_INDEX = 0            # 0 = todos los monitores combinados.
VISION_SCALE_MAX_WIDTH = 1600       # Redimensiona antes de enviar al modelo.
VISION_JPEG_QUALITY = 78            # Compresión de la captura (velocidad).
VISION_GRID = 1000                  # Coordenadas normalizadas 0-1000 (estilo Qwen-VL).
VISION_MAX_ELEMENTS = 40            # Techo de elementos devueltos por el modelo.
VISION_SCREENSHOTS_DIR = WORKSPACE_DIR / "capturas"
VISION_KEEP_LAST_SCREENSHOTS = 20   # Rotación de capturas temporales.

ACTUATOR_ENABLED = True
ACTUATOR_MOVE_DURATION = 0.25       # Movimiento de ratón humano (segundos).
ACTUATOR_MOVE_TWEEN = "easeInOutQuad"
ACTUATOR_CLICK_DELAY = 0.12         # Pausa antes de pulsar.
ACTUATOR_POST_CLICK_DELAY = 0.35    # Pausa tras pulsar (deja reaccionar la UI).
ACTUATOR_TYPE_INTERVAL = 0.02       # Velocidad de tecleo (s/carácter).
ACTUATOR_SAFE_FORESHOT = True       # Re-verifica la pantalla antes de cada clic.
ACTUATOR_MAX_STEPS_PER_TASK = 40    # Corte de seguridad anti-bucles.
ACTUATOR_STEP_TIMEOUT_S = 120.0     # Tiempo máximo por paso de automatización.
ACTUATOR_SCROLL_CLICKS = 5
PYAUTOGUI_FAILSAFE = True           # Llevar el ratón a la esquina 0,0 aborta.

# =============================================================================
# 13. GHOST HUD — INTERFAZ HOLOGRÁFICA
# =============================================================================

HUD_ENABLED = True
HUD_FPS = 60
HUD_CAPSULE_WIDTH = 236
HUD_CAPSULE_HEIGHT = 62
HUD_CAPSULE_MARGIN = 26             # Separación con el borde de la pantalla.
HUD_BORDER_THICKNESS = 3            # Grosor del marco de neón.
HUD_BORDER_RADIUS = 16
HUD_GLOW_BLUR = 26                  # Radio del halo luminoso.
HUD_FADE_MS = 320                   # Transición de opacidad.
HUD_RING_SPIN_MS = 1100             # Vuelta completa del anillo de progreso.
HUD_OPACITY_SLEEPING = 0.45
HUD_OPACITY_ACTIVE = 0.95
HUD_WINDOW_FLAGS_CLICK_THROUGH = True  # WS_EX_TRANSPARENT | WS_EX_LAYERED.

# Paleta (hex ARGB/RGB). Electric cyan para escuchar, gris para dormido.
HUD_COLOR_IDLE = "#6E7B8B"          # Gris azulado apagado.
HUD_COLOR_SLEEPING = "#4A5560"      # Gris dormido.
HUD_COLOR_LISTENING = "#00E5FF"     # Cian eléctrico vibrante.
HUD_COLOR_THINKING = "#00B4D8"      # Cian profundo.
HUD_COLOR_SPEAKING = "#22D3EE"      # Cian claro.
HUD_COLOR_ACTING = "#0EA5E9"        # Azul de acción.
HUD_COLOR_ERROR = "#FF3B5C"         # Rojo de error.
HUD_COLOR_KILL = "#FF8A00"          # Ámbar del kill-switch.
HUD_CAPSULE_BG = "#0B1220"          # Cápsula oscura semitransparente.
HUD_CAPSULE_ALPHA = 205
HUD_TEXT_COLOR = "#DCE9F5"
HUD_MIC_LISTENING_COLOR = "#00E5FF"
HUD_MIC_MUTED_COLOR = "#5A6673"
HUD_BORDER_ACTIVE_COLOR = "#00E5FF"
HUD_SHOW_STATUS_TEXT = True
HUD_STATUS_TEXT = {
    "sleeping": "En espera",
    "idle": "Listo",
    "listening": "Escuchando",
    "transcribing": "Transcribiendo",
    "thinking": "Pensando",
    "swapping": "Cambiando modelo",
    "speaking": "Hablando",
    "acting": "Actuando",
    "vision": "Mirando la pantalla",
    "error": "Error",
    "killswitch": "KILL-SWITCH",
}

# =============================================================================
# 14. SEGURIDAD — KILL-SWITCH E INTEGRIDAD
# =============================================================================

KILLSWITCH_FILE = SAFETY_DIR / "killswitch.py"
KILLSWITCH_HASH_FILE = SAFETY_DIR / "killswitch.sha256"
COMMAND_GUARD_FILE = SAFETY_DIR / "command_guard.py"

KILLSWITCH_HOTKEY = "ctrl+shift+space"
KILLSWITCH_HOTKEY_ALT = "ctrl+alt+k"
KILLSWITCH_MOUSE_TRIGGER_ENABLED = True
KILLSWITCH_MOUSE_TRAVEL_PX = 1500     # Sacudida violenta del ratón.
KILLSWITCH_MOUSE_WINDOW_S = 0.3       # ...en menos de 300 ms.
KILLSWITCH_MOUSE_MIN_SAMPLES = 4      # Evita falsos positivos por un salto.
KILLSWITCH_MOUSE_POLL_S = 0.02        # 50 Hz de muestreo de posición.
KILLSWITCH_GRACE_PERIOD_S = 1.5       # Ignora sacudidas del propio JARVIS.
KILLSWITCH_HARDEN_FILES = True        # Marca los archivos como solo-lectura.
KILLSWITCH_INTEGRITY_STRICT = True    # Si el hash falla: JARVIS NO arranca.
KILLSWITCH_ALWAYS_ALLOW_READONLY_OFF = True  # `--desproteger` desbloquea a Pablo.
KILLSWITCH_TRIP_BEEP = True
KILLSWITCH_MAX_TRIPS_BEFORE_LOCK = 5  # Tras 5 abortos, pide reinicio manual.

PROTECTED_PATHS = (
    "safety/killswitch.py",
    "safety/killswitch.sha256",
    "safety/__init__.py",
    "safety/command_guard.py",
)
PROTECTED_NAME_FRAGMENTS = ("killswitch", "command_guard", ".sha256")

# =============================================================================
# 15. AUTO-PROGRAMACIÓN Y CONSTRUCTOR DE APPS
# =============================================================================

SELF_PROGRAMMING_ENABLED = True
SELF_PROGRAMMING_BRANCH_PREFIX = "feature/staging-"
SELF_PROGRAMMING_BRANCH_TS_FORMAT = "%Y%m%d-%H%M%S"
SELF_PROGRAMMING_MAX_FILES = 12        # Archivos por iteración generada.
SELF_PROGRAMMING_MAX_ATTEMPTS = 3      # Reintentos si pytest falla.
SELF_PROGRAMMING_TEST_TIMEOUT_S = 420.0
SELF_PROGRAMMING_DRY_RUN = False       # True = prepara todo pero no fusiona.
SELF_PROGRAMMING_HOT_RELOAD = True     # os.execv tras fusionar en main.
SELF_PROGRAMMING_REQUIRE_CONFIRMATION = False  # Pablo dio la orden de palabra.
SELF_PROGRAMMING_ALLOWED_EXTENSIONS = (
    ".py", ".md", ".txt", ".json", ".bat", ".toml", ".cfg", ".ini", ".md",
)

APP_BUILDER_ENABLED = True
APP_BUILDER_MAX_FILES = 16
APP_BUILDER_MAX_FILE_BYTES = 240_000
APP_BUILDER_TEST_TIMEOUT_S = 240.0
APP_BUILDER_VENV_TIMEOUT_S = 900.0
APP_BUILDER_INSTALL_TIMEOUT_S = 900.0
APP_BUILDER_ALLOW_DEPENDENCIES = True   # pip install dentro del venv sandbox.
APP_BUILDER_MAX_DEPENDENCIES = 12
APP_BUILDER_AUTO_LAUNCH = True
APP_BUILDER_SAFE_INSTALL_TIMEOUT_S = 600.0
APP_BUILDER_OPEN_HTML_IN_BROWSER = True

GIT_BIN = "git"
GIT_COMMAND_TIMEOUT_S = 60.0
GIT_USER_NAME = "JARVIS Auto-Programmer"
GIT_USER_EMAIL = "jarvis@localhost"
GIT_MAIN_BRANCH = "main"
GIT_DEFAULT_MERGE_FLAG = "--no-ff"

# =============================================================================
# 16. PERSONALIDAD Y PROMPTS (en español, con la voz de JARVIS)
# =============================================================================

SYSTEM_PROMPT_CHAT = (
    "Eres JARVIS, el asistente personal de escritorio de Pablo. Hablas SIEMPRE en español "
    "de España, con un tono cercano, elegante y eficiente, como un mayordomo tecnológico "
    "de altísima competencia. Tratas a Pablo de 'usted' solo si él lo pide; por defecto "
    "le llamas 'Pablo' y le hablas de tú.\n\n"
    "Reglas de estilo:\n"
    "- Respuestas BREVES: 1 a 3 frases. Te escuchan por voz, no te leen.\n"
    "- Nada de markdown, listas ni símbolos raros: hablas en voz alta.\n"
    "- Nunca menciones que eres una IA, un modelo o un lenguaje de programación.\n"
    "- Si no sabes algo, dilo en una frase y ofrece una alternativa.\n"
    "- Eres privado por diseño: todo ocurre en el ordenador de Pablo, sin internet salvo "
    "para lo que él pida explícitamente.\n"
)

SYSTEM_PROMPT_PLANNER = (
    "Eres el planificador de JARVIS. Conviertes una orden hablada de Pablo en una lista "
    "de pasos ejecutables.\n"
    "Responde SIEMPRE y ÚNICAMENTE con un objeto JSON válido con esta forma exacta:\n"
    '{"objetivo": "...", "pasos": [{"accion": "<nombre>", "argumentos": {...}, '
    '"descripcion": "..."}], "confianza": 0.0}\n'
    "Acciones disponibles: abrir_app, abrir_url, buscar_web, escribir_texto, "
    "pulsar_teclas, clic_elemento, mover_raton, desplazar, captura_pantalla, "
    "describir_pantalla, reproducir_musica, decir_hora, decir_fecha, "
    "crear_app, crear_web, anadir_funcionalidad, cerrar_jarvis, conversar.\n"
    "Si la orden es solo conversación, responde con una lista de pasos que contenga "
    "únicamente la acción 'conversar'.\n"
    "No añadas texto fuera del JSON. No uses bloques de código."
)

SYSTEM_PROMPT_VISION_DESCRIBE = (
    "Eres el módulo de percepción visual de JARVIS, un asistente de escritorio en español. "
    "Observas la captura de la pantalla de Pablo y describes con precisión lo que ves: "
    "aplicación activa, ventanas abiertas, contenido relevante, estado de la interfaz. "
    "Responde en español, en 2 a 5 frases claras, sin markdown."
)

SYSTEM_PROMPT_VISION_UI = (
    "Eres el localizador de elementos de interfaz de JARVIS. Recibes una captura de "
    "pantalla y una descripción de un elemento que hay que pulsar.\n"
    "Responde ÚNICAMENTE con JSON válido, sin texto adicional ni bloques de código:\n"
    '{"elementos": [{"nombre": "...", "descripcion": "...", "confianza": 0.0, '
    '"x": 0, "y": 0, "ancho": 0, "alto": 0}], "encontrado": true}\n'
    "Las coordenadas x/y son el CENTRO del elemento en una rejilla normalizada de 0 a 1000 "
    "sobre la imagen completa (x=0 izquierda, x=1000 derecha, y=0 arriba, y=1000 abajo). "
    "Ordena la lista del elemento más probable al menos probable. "
    "Si no encuentras el elemento, devuelve la lista vacía y \"encontrado\": false."
)

SYSTEM_PROMPT_CODER = (
    "Eres el motor de programación de JARVIS. Escribes código en español para los "
    "comentarios y nombres de cara al usuario, con calidad de producción: sin "
    "placeholders, sin TODOs, sin pseudocódigo, manejo de errores, docstrings y "
    "código completo y ejecutable.\n"
    "Nunca modificas, importas ni mencionas nada relacionado con la carpeta `safety/` "
    "ni con el kill-switch: son archivos protegidos e inmutables."
)

SYSTEM_PROMPT_APP_BUILDER = (
    "Eres el constructor de aplicaciones de JARVIS. Pablo describe una app o una web "
    "hablando y tú entregas el proyecto completo y funcional.\n"
    "Respondes ÚNICAMENTE con un objeto JSON válido, sin markdown:\n"
    '{"nombre": "slug-del-proyecto", "descripcion": "...", "tipo": "web|escritorio|script", '
    '"dependencias": ["paquete"], "comando": "comando de ejecución", '
    '"archivos": [{"ruta": "index.html", "contenido": "<!DOCTYPE html>..."}]}\n'
    "Reglas: rutas relativas simples dentro del proyecto (nunca '..', nunca rutas "
    "absolutas, nunca .exe ni .bat), dependencias solo de PyPI con nombre válido, "
    "y código final completo listo para ejecutar."
)

# Frases de JARVIS (personalidad consistente en toda la aplicación).
PHRASES = {
    "wake": ("Dígame, Pablo.", "Le escucho.", "¿Qué necesita?", "Aquí estoy."),
    "thinking": ("Un momento...", "Déjeme ver...", "Procesando."),
    "not_understood": (
        "Perdone, Pablo, no le he entendido bien. ¿Puede repetirlo?",
        "No estoy seguro de haberle entendido. ¿Me lo dice de otra forma?",
    ),
    "no_audio": "No encuentro el micrófono, Pablo. Puede escribirme por texto.",
    "killswitch": "Control liberado. Mando en sus manos, Pablo.",
    "vision_loading": "Voy a mirar la pantalla, deme un segundo.",
    "app_built": "Listo, Pablo. Su aplicación está construida y la he abierto.",
    "error": "Ha ocurrido un error, Pablo. Lo he anotado en el registro.",
    "goodbye": "Hasta luego, Pablo. Estaré aquí si me necesita.",
}

# =============================================================================
# 17. RENDIMIENTO Y COMPORTAMIENTO GLOBAL
# =============================================================================

MAIN_LOOP_SLEEP_S = 0.02
CONVERSATION_HISTORY_LIMIT = 24      # Turnos recordados en memoria RAM.
CONVERSATION_PERSIST = True
LOG_LEVEL = os.environ.get("JARVIS_LOG_LEVEL", "INFO").upper()
LOG_MAX_BYTES = 2 * 1024 * 1024
LOG_BACKUP_COUNT = 3
CONSOLE_UTF8 = True
SINGLE_INSTANCE = True               # Evita dos JARVIS peleando por el micrófono.
STARTUP_GREETING = False             # Saludo solo con doble palmada o por voz.
TEXT_MODE_DEFAULT = False            # `python main.py --texto` para modo consola.
NATIVE_SOUND_ACK = True              # Pitidos nativos de Windows (winsound).
ACK_FREQUENCY_HZ = 880
ACK_DURATION_MS = 90

# =============================================================================
# 18. GUIAS DE HARDWARE (usadas por el diagnóstico del README/install.bat)
# =============================================================================

HARDWARE_TIERS = {
    "bajo": {
        "ram_gb": (8, 12),
        "vram_gb": (0, 4),
        "chat": "llama3.2:3b",
        "vision": "llama3.2-vision:11b (solo si hay 8 GB+ de VRAM)",
        "coder": "qwen2.5-coder:1.5b",
        "stt": "tiny",
        "nota": "Funciona, pero con respuestas más simples y visión desactivada.",
    },
    "medio": {
        "ram_gb": (12, 24),
        "vram_gb": (4, 8),
        "chat": "llama3.1:8b",
        "vision": "llama3.2-vision:latest",
        "coder": "qwen2.5-coder:7b",
        "stt": "base",
        "nota": "Configuración objetivo de Pablo (Ryzen 7 7840HS / RTX 4060 8 GB).",
    },
    "alto": {
        "ram_gb": (24, 128),
        "vram_gb": (8, 48),
        "chat": "llama3.1:8b",
        "vision": "llama3.2-vision:11b",
        "coder": "qwen2.5-coder:14b",
        "stt": "small",
        "nota": "Permite modelos mayores manteniendo la monogamia de VRAM.",
    },
}


# =============================================================================
# 19. UTILIDADES
# =============================================================================

def ensure_directories() -> None:
    """Crea (si no existen) todas las carpetas de trabajo de JARVIS.

    Se llama en el arranque y es intencionadamente tolerante a fallos: si una
    carpeta no se puede crear, se registra el aviso y se continúa.
    """
    logger = logging.getLogger("jarvis.config")
    for directory in ALL_DIRECTORIES:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - depende del sistema de archivos
            logger.warning("No se pudo crear la carpeta %s: %s", directory, exc)
    for keep in (WORKSPACE_DIR / ".gitkeep", MODELS_DIR / ".gitkeep"):
        try:
            if not keep.exists():
                keep.touch()
        except OSError:  # pragma: no cover
            pass


def setup_logging(level: str | None = None, console: bool = True) -> logging.Logger:
    """Configura el registro de JARVIS (consola + archivo rotativo en `logs/`).

    Devuelve el logger raíz "jarvis". Es idempotente: llamarlo dos veces no
    duplica los manejadores.
    """
    logger = logging.getLogger("jarvis")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO))
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if console:
        try:
            if CONSOLE_UTF8 and hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover
            pass
        stream = logging.StreamHandler(stream=sys.stdout)
        stream.setFormatter(formatter)
        logger.addHandler(stream)

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:  # pragma: no cover
        pass

    return logger


def greeting_for_hour(hour: int | None = None, name: str = USER_NAME) -> str:
    """Devuelve el saludo cálido en español según la hora del reloj.

    * 06:00 – 11:59 -> "Buenos días Pablo"
    * 12:00 – 19:59 -> "Buenas tardes Pablo"
    * 20:00 – 05:59 -> "Buenas noches Pablo"

    >>> greeting_for_hour(6), greeting_for_hour(12), greeting_for_hour(20)
    ('Buenos días Pablo', 'Buenas tardes Pablo', 'Buenas noches Pablo')
    """
    if hour is None:
        import datetime as _dt

        hour = _dt.datetime.now().hour
    hour = int(hour) % 24
    if 6 <= hour < 12:
        return f"Buenos días {name}"
    if 12 <= hour < 20:
        return f"Buenas tardes {name}"
    return f"Buenas noches {name}"


def detect_hardware_tier(ram_gb: float | None = None, vram_gb: float | None = None) -> str:
    """Clasifica el equipo en 'bajo', 'medio' o 'alto' según RAM y VRAM.

    Se usa en el diagnóstico de arranque para recomendar la matriz de modelos
    adecuada sin que el usuario tenga que entender nada.
    """
    if ram_gb is None:
        try:
            import psutil  # type: ignore

            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            ram_gb = 16.0
    if vram_gb is None:
        vram_gb = 0.0
    if ram_gb >= 24 or vram_gb >= 8:
        return "alto" if ram_gb >= 32 else "medio"
    if ram_gb >= 12:
        return "medio"
    return "bajo"


def summarize_config() -> dict:
    """Resumen legible de la configuración activa (para `main.py --diagnostico`)."""
    return {
        "version": APP_VERSION,
        "modelos": {
            "chat": MODEL_CHAT,
            "vision": MODEL_VISION,
            "coder": MODEL_CODER,
        },
        "monogamia_vram": ENFORCE_VRAM_MONOGAMY,
        "audio_hz": AUDIO_SAMPLE_RATE,
        "stt": f"faster-whisper {STT_MODEL} ({STT_DEVICE}/{STT_COMPUTE_TYPE})",
        "tts": f"{TTS_ENGINE_ORDER[0]} voz={TTS_KOKORO_VOICE}",
        "wake_word": f"{WAKE_WORD_MODEL} (umbral {WAKE_WORD_THRESHOLD})",
        "doble_palpada": CLAP_ENABLED,
        "killswitch": f"{KILLSWITCH_HOTKEY} + sacudida de ratón",
        "hud": HUD_ENABLED,
        "carpetas": {d.name: str(d) for d in ALL_DIRECTORIES if d.parent == ROOT_DIR},
    }


def _apply_local_overrides() -> None:
    """Carga `local_config.py` (si existe) y sobreescribe los valores definidos.

    Permite personalizar sin ensuciar el repositorio (el archivo está en
    `.gitignore`) y sin romper el historial de actualizaciones.
    """
    path = ROOT_DIR / "local_config.py"
    if not path.exists():
        return
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("jarvis_local_config", path)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace = globals()
        for key, value in vars(module).items():
            if key.isupper() and not key.startswith("_"):
                namespace[key] = value
    except Exception as exc:  # pragma: no cover
        logging.getLogger("jarvis.config").warning(
            "local_config.py no se pudo aplicar: %s", exc
        )


_apply_local_overrides()
