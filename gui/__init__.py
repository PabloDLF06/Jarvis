"""
gui — Interfaz holográfica de JARVIS (GHOST HUD).
================================================================================
`hud_overlay.py` dibuja una capa completamente transparente sobre el escritorio
con dos elementos:

1. **Cápsula flotante** (esquina superior derecha): icono de micrófono que se
   enciende en cian eléctrico cuando JARVIS escucha, anillo de progreso que gira
   durante la inferencia y el cambio de modelo, y un punto de estado.
2. **Marco de neón** alrededor de todo el monitor: se ilumina solo cuando
   JARVIS mira la pantalla o mueve el ratón, y se apaga con una transición suave.

La ventana es **click-through**: los clics atraviesan el HUD y llegan a las
aplicaciones de debajo, para que nunca estorbe. En Windows se consigue con los
estilos extendidos `WS_EX_TRANSPARENT | WS_EX_LAYERED`; en otros sistemas se
usan las banderas equivalentes de Qt.

Este módulo importa PyQt6 de forma perezosa: si no está instalado, JARVIS sigue
funcionando sin interfaz (modo consola) en lugar de fallar.
================================================================================
"""

from __future__ import annotations

__all__ = ["describe_hud", "HudState"]


def describe_hud() -> str:
    """Descripción breve del HUD (para el diagnóstico de `main.py`)."""
    return (
        "GHOST HUD: cápsula flotante click-through (micrófono cian al escuchar, "
        "anillo giratorio al pensar) + marco de neón que se ilumina cuando JARVIS "
        "mira la pantalla o actúa."
    )


class HudState:
    """Estados que puede mostrar la cápsula (espejo de `config.HUD_STATUS_TEXT`)."""

    SLEEPING = "sleeping"
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SWAPPING = "swapping"
    SPEAKING = "speaking"
    ACTING = "acting"
    VISION = "vision"
    ERROR = "error"
    KILLSWITCH = "killswitch"
