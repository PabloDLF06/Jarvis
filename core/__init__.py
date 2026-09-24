"""
core — Núcleo funcional de JARVIS.
================================================================================
Mapa de subsistemas (el orden refleja el camino de una orden hablada):

1. `voice_engine`      — Oídos y boca: micrófono, VAD, Whisper, Kokoro,
                         openWakeWord y el doble palmada. Publica cambios de
                         estado que el HUD dibuja.
2. `barge_in`          — Interrupción en vuelo: Pablo habla y JARVIS calla.
3. `acoustic_detector` — Doble palmada: música + saludo por hora del día.
4. `model_router`      — Monogamia de VRAM: elige y cambia de modelo LLM.
5. `vision_actuator`   — Mira la pantalla (llama3.2-vision) y mueve el ratón.
6. `self_programmer`   — Ramas git aisladas, tests, fusión y recarga en caliente.
7. `app_builder`       — De la voz a una app: scaffold, código, venv y arranque.
8. `ollama_client`     — Cliente HTTP de la API local de Ollama.
9. `media_dispatcher`  — Spotify, Comet/YouTube y apertura de URLs.
10. `utils`            — Utilidades compartidas.

`main.py` es el director de orquesta: construye estos módulos, los conecta al
kill-switch y los arranca en su propio hilo.

Este `__init__` es deliberadamente ligero: no importa los subsistemas para que
importar `core` nunca arrastre dependencias pesadas (PyQt6, Whisper, etc.).
================================================================================
"""

from __future__ import annotations

__all__ = [
    "SUBSYSTEMS",
    "describe_architecture",
    "get_version",
]

#: Mapa de subsistemas -> responsabilidad (usado por `--diagnostico` y el README).
SUBSYSTEMS: dict[str, str] = {
    "ollama_client": "Cliente HTTP de Ollama con flujo por fragmentos y descarga forzada.",
    "model_router": "Enrutador dinámico y ley de monogamia de VRAM.",
    "voice_engine": "Wake word, VAD, STT (faster-whisper), TTS (Kokoro/Piper) y bucle principal.",
    "barge_in": "Detección de interrupción mientras JARVIS habla o actúa.",
    "acoustic_detector": "Doble palmada: música de Tame Impala y saludo por hora.",
    "vision_actuator": "Percepción de pantalla con llama3.2-vision y control de ratón/teclado.",
    "self_programmer": "Auto-programación en rama git aislada con tests y rollback.",
    "app_builder": "Constructor de apps y webs en espacio de trabajo aislado.",
    "media_dispatcher": "Reproductor de música y apertura de navegador/URLs.",
    "utils": "Utilidades: reintentos, JSON atómico, texto y concurrencia.",
}


def describe_architecture() -> str:
    """Devuelve un texto legible con la arquitectura de JARVIS (para la consola)."""
    lines = ["Arquitectura de JARVIS:", ""]
    for index, (module, description) in enumerate(SUBSYSTEMS.items(), start=1):
        lines.append(f"  {index:>2}. core/{module}.py — {description}")
    return "\n".join(lines)


def get_version() -> str:
    """Versión del paquete (leída de config sin importar dependencias pesadas)."""
    try:
        import config  # noqa: PLC0415

        return config.APP_VERSION
    except Exception:  # pragma: no cover
        return "0.0.0"
