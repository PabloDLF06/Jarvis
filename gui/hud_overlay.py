"""
gui/hud_overlay.py — GHOST HUD: la interfaz holográfica de JARVIS.
================================================================================
Dos capas flotan sobre el escritorio, sin bordes, sin barra de tareas y sin
estorbar nunca, porque **todos los clics las atraviesan**:

1. **Cápsula** (esquina superior derecha)
   * Fondo oscuro translúcido con un halo cian sutil.
   * **Micrófono vectorial**: gris apagado cuando duerme, **cian eléctrico
     vibrante** cuando está escuchando (con un latido suave).
   * **Anillo de progreso** que gira mientras el modelo piensa o mientras
     Ollama cambia de modelo (la ley de monogamia de VRAM).
   * Punto de estado y texto corto ("Le escucho", "Pensando", "Hablando"...).

2. **Marco de neón** (los cuatro bordes de la pantalla)
   * Se ilumina en cian SOLO cuando JARVIS captura la pantalla, procesa visión
     o mueve el ratón y escribe.
   * Se apaga con una transición suave (~320 ms) cuando vuelve al reposo.

DETALLES TÉCNICOS
-----------------
* `Qt.WindowTransparentForInput` (Qt 6) y, en Windows, los estilos extendidos
  `WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW` garantizan el
  click-through y que la ventana no aparezca en la barra de tareas.
* Las animaciones corren a 60 FPS con un único `QTimer`; cuando JARVIS duerme,
  el temporizador baja a 12 FPS para no gastar batería.
* Los cambios de estado llegan desde otros hilos: se comunican por señales Qt
  (conexión en cola), que es la forma segura de tocar la interfaz.
================================================================================
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import config as cfg  # noqa: E402

logger = logging.getLogger("jarvis.hud")

# Importación perezosa: JARVIS debe poder arrancar en modo texto sin PyQt6.
try:  # pragma: no cover - depende del entorno
    from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal, QObject
    from PyQt6.QtGui import (
        QBrush,
        QColor,
        QFont,
        QFontMetrics,
        QGuiApplication,
        QLinearGradient,
        QPainter,
        QPainterPath,
        QPen,
        QRadialGradient,
    )
    from PyQt6.QtWidgets import QApplication, QWidget

    PYQT_AVAILABLE = True
except Exception:  # pragma: no cover - sin PyQt6 se degrada con elegancia
    PYQT_AVAILABLE = False
    QApplication = object  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment]


HUD_STATES_COLOR = {
    "sleeping": cfg.HUD_COLOR_SLEEPING,
    "idle": cfg.HUD_COLOR_IDLE,
    "listening": cfg.HUD_COLOR_LISTENING,
    "transcribing": cfg.HUD_COLOR_THINKING,
    "thinking": cfg.HUD_COLOR_THINKING,
    "swapping": cfg.HUD_COLOR_THINKING,
    "speaking": cfg.HUD_COLOR_SPEAKING,
    "acting": cfg.HUD_COLOR_ACTING,
    "vision": cfg.HUD_COLOR_ACTING,
    "error": cfg.HUD_COLOR_ERROR,
    "killswitch": cfg.HUD_COLOR_KILL,
}

# Estados durante los que el anillo gira.
SPINNER_STATES = {"thinking", "transcribing", "swapping"}


# =============================================================================
# AYUDANTES DE PINTURA (independientes de Qt para poder probarse)
# =============================================================================

def color_for_state(state: str) -> str:
    """Color (hex) asociado a un estado de JARVIS.

    >>> color_for_state("listening") == cfg.HUD_COLOR_LISTENING
    True
    >>> color_for_state("estado-inexistente") == cfg.HUD_COLOR_IDLE
    True
    """
    return HUD_STATES_COLOR.get(str(state).strip().lower(), cfg.HUD_COLOR_IDLE)


def should_spin(state: str) -> bool:
    """¿Debe girar el anillo de progreso en este estado?"""
    return str(state).strip().lower() in SPINNER_STATES


def opacity_for_state(state: str) -> float:
    """Opacidad de la cápsula según el estado (dormido = más discreta)."""
    if str(state).strip().lower() in ("sleeping", "idle"):
        return cfg.HUD_OPACITY_SLEEPING
    return cfg.HUD_OPACITY_ACTIVE


def lerp(current: float, target: float, factor: float) -> float:
    """Interpolación lineal exponencial para transiciones suaves."""
    return current + (target - current) * factor


# =============================================================================
# SEÑALES ENTRE HILOS
# =============================================================================

if PYQT_AVAILABLE:

    class HudSignals(QObject):
        """Puente seguro entre los hilos de JARVIS y el hilo de la interfaz."""

        state_changed = pyqtSignal(str, str)     # (estado, detalle)
        glow_requested = pyqtSignal(float)       # segundos de iluminación
        border_level = pyqtSignal(float)         # 0..1 nivel directo del marco
        close_requested = pyqtSignal()

else:  # pragma: no cover

    class HudSignals:  # type: ignore[no-redef]
        """Sustituto inerte cuando PyQt6 no está disponible."""


# =============================================================================
# CÁPSULA
# =============================================================================

if PYQT_AVAILABLE:

    class CapsuleWidget(QWidget):
        """Cápsula flotante con micrófono, anillo y texto de estado."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
                | Qt.WindowType.WindowTransparentForInput
                | Qt.WindowType.WindowDoesNotAcceptFocus
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            self.setFixedSize(cfg.HUD_CAPSULE_WIDTH, cfg.HUD_CAPSULE_HEIGHT)

            self.state = "sleeping"
            self.detail = ""
            self._pulse = 0.0        # Latido del micrófono al escuchar.
            self._spin = 0.0         # Ángulo del anillo.
            self._opacity_value = cfg.HUD_OPACITY_SLEEPING
            self._target_opacity = cfg.HUD_OPACITY_SLEEPING
            self._mic_color = QColor(cfg.HUD_MIC_MUTED_COLOR)
            self._state_color = QColor(color_for_state(self.state))
            self._font = QFont("Segoe UI", 8)
            self._font.setBold(True)
            self.setWindowOpacity(self._opacity_value)
            self._move_to_corner()

        # --- Colocación ---------------------------------------------------

        def _move_to_corner(self) -> None:
            """Coloca la cápsula en la esquina superior derecha de la pantalla."""
            screen = QGuiApplication.primaryScreen()
            if screen is None:
                return
            area = screen.availableGeometry()
            x = area.right() - self.width() - cfg.HUD_CAPSULE_MARGIN
            y = area.top() + cfg.HUD_CAPSULE_MARGIN
            self.move(max(area.left(), int(x)), max(area.top(), int(y)))

        def reposition(self) -> None:
            """Recoloca la cápsula (al cambiar de pantalla o resolución)."""
            self._move_to_corner()

        # --- Estado -------------------------------------------------------

        def set_state(self, state: str, detail: str = "") -> None:
            """Actualiza el estado visual de la cápsula."""
            self.state = str(state or "idle").strip().lower()
            self.detail = detail or ""
            self._state_color = QColor(color_for_state(self.state))
            self._target_opacity = opacity_for_state(self.state)
            self._mic_color = QColor(
                cfg.HUD_MIC_LISTENING_COLOR
                if self.state in ("listening", "transcribing")
                else cfg.HUD_MIC_MUTED_COLOR
            )
            if self.state == "killswitch":
                self._mic_color = QColor(cfg.HUD_COLOR_KILL)
            self.update()

        def tick(self, dt: float) -> None:
            """Avanza las animaciones (lo llama el temporizador del HUD)."""
            self._opacity_value = lerp(self._opacity_value, self._target_opacity, 0.15)
            try:
                self.setWindowOpacity(max(0.05, min(1.0, self._opacity_value)))
            except Exception:
                pass

            if self.state in ("listening", "speaking"):
                self._pulse = (self._pulse + dt * 2.4) % (2 * 3.14159265)
            else:
                self._pulse = lerp(self._pulse, 0.0, 0.1)

            if should_spin(self.state):
                self._spin = (self._spin + dt * 360.0 / max(0.2, cfg.HUD_RING_SPIN_MS / 1000.0)) % 360.0
            self.update()

        # --- Pintura ------------------------------------------------------

        def paintEvent(self, event) -> None:  # noqa: N802 - API de Qt
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
            radius = cfg.HUD_CAPSULE_HEIGHT / 2.0

            # Fondo de la cápsula con degradado oscuro.
            background = QLinearGradient(rect.topLeft(), rect.bottomRight())
            base = QColor(cfg.HUD_CAPSULE_BG)
            base.setAlpha(cfg.HUD_CAPSULE_ALPHA)
            background.setColorAt(0.0, base)
            darker = QColor(cfg.HUD_CAPSULE_BG)
            darker.setAlpha(max(0, cfg.HUD_CAPSULE_ALPHA - 40))
            background.setColorAt(1.0, darker)
            painter.setBrush(QBrush(background))

            # Halo suave del color del estado.
            glow = QColor(self._state_color)
            glow.setAlpha(70 if self.state not in ("sleeping", "idle") else 30)
            painter.setPen(QPen(glow, 1.6))
            painter.drawRoundedRect(rect.adjusted(0.8, 0.8, -0.8, -0.8), radius, radius)

            # --- Micrófono (icono vectorial) ------------------------------
            mic_center_x = 26.0
            mic_center_y = self.height() / 2.0
            self._draw_microphone(painter, mic_center_x, mic_center_y)

            # --- Anillo de progreso ---------------------------------------
            ring_center = QPointF(self.width() - 30.0, self.height() / 2.0)
            self._draw_ring(painter, ring_center)

            # --- Punto de estado y texto ----------------------------------
            if cfg.HUD_SHOW_STATUS_TEXT:
                self._draw_text(painter)
            painter.end()

        def _draw_microphone(self, painter: QPainter, cx: float, cy: float) -> None:
            """Dibuja el micrófono, con latido cuando JARVIS escucha."""
            color = QColor(self._mic_color)
            if self.state in ("listening", "speaking"):
                # Latido suave: el micrófono "respira" mientras escucha.
                beat = 0.88 + 0.12 * math.sin(float(self._pulse))
                color.setAlpha(max(0, min(255, int(255 * beat))))
                halo = QRadialGradient(QPointF(cx, cy), 17.0)
                halo.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 90))
                halo.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
                painter.setBrush(QBrush(halo))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(cx, cy), 17.0, 17.0)

            pen = QPen(color, 2.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            body = QRectF(cx - 4.0, cy - 9.5, 8.0, 13.0)
            painter.drawRoundedRect(body, 4.0, 4.0)

            cradle = QPainterPath()
            cradle.moveTo(cx - 7.5, cy - 1.0)
            cradle.arcTo(QRectF(cx - 7.5, cy - 7.0, 15.0, 13.0), 180.0, 180.0)
            painter.drawPath(cradle)

            painter.drawLine(QPointF(cx, cy + 6.0), QPointF(cx, cy + 9.5))
            painter.drawLine(QPointF(cx - 4.0, cy + 9.5), QPointF(cx + 4.0, cy + 9.5))

        def _draw_ring(self, painter: QPainter, center: QPointF) -> None:
            """Anillo de progreso: gira al pensar y se completa al terminar."""
            radius = 9.0
            box = QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2)

            track = QColor(self._state_color)
            track.setAlpha(45)
            painter.setPen(QPen(track, 2.6))
            painter.drawEllipse(box)

            arc_pen = QPen(QColor(self._state_color), 2.6)
            arc_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(arc_pen)
            if should_spin(self.state):
                span = 110 * 16
                start = int(-self._spin * 16)
            elif self.state in ("listening", "speaking"):
                span = 250 * 16
                start = 90 * 16
            elif self.state in ("error", "killswitch"):
                span = 360 * 16
                start = 0
            else:
                span = 360 * 16
                start = 90 * 16
            painter.drawArc(box, start, span)

            if self.state in ("error", "killswitch", "idle", "sleeping"):
                dot = QColor(self._state_color)
                dot.setAlpha(200)
                painter.setBrush(QBrush(dot))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(center, 2.6, 2.6)

        def _draw_text(self, painter: QPainter) -> None:
            """Texto de estado a la izquierda del anillo."""
            painter.setFont(self._font)
            label = cfg.HUD_STATUS_TEXT.get(self.state, "")
            if self.detail and self.state in ("thinking", "listening"):
                label = f"{label} · {self.detail[:22]}"
            text_color = QColor(cfg.HUD_TEXT_COLOR)
            text_color.setAlpha(235 if self.state not in ("sleeping", "idle") else 150)
            painter.setPen(QPen(text_color))
            metrics = QFontMetrics(self._font)
            available = self.width() - 56 - 22
            text = metrics.elidedText(label, Qt.TextElideMode.ElideRight, available)
            painter.drawText(
                QRectF(44.0, 0.0, available, float(self.height())),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                text,
            )


    # =========================================================================
    # MARCO DE NEÓN
    # =========================================================================

    class BorderWidget(QWidget):
        """Marco luminoso alrededor de toda la pantalla (no intercepta clics)."""

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
                | Qt.WindowType.WindowTransparentForInput
                | Qt.WindowType.WindowDoesNotAcceptFocus
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self.level = 0.0          # 0 = apagado, 1 = completamente iluminado.
            self._target = 0.0
            self._until = 0.0         # Iluminación temporal (por actividad).
            self._color = QColor(cfg.HUD_BORDER_ACTIVE_COLOR)
            self._freeze_max = 0.0
            self._fit_screen()

        def _fit_screen(self) -> None:
            screen = QGuiApplication.primaryScreen()
            if screen is None:
                return
            geometry = screen.geometry()
            self.setGeometry(geometry)

        def reposition(self) -> None:
            self._fit_screen()

        def illuminate(self, seconds: float = 1.2) -> None:
            """Ilumina el marco durante `seconds` y vuelve a apagarse solo."""
            self._until = max(self._until, time.monotonic() + max(0.1, seconds))
            self._target = 1.0
            self.update()

        def set_level(self, level: float) -> None:
            """Fija un nivel constante (0..1) sin temporizador."""
            self._target = max(0.0, min(1.0, float(level)))
            self._until = 0.0
            self.update()

        def freeze(self, active: bool) -> None:
            """Mantiene el marco encendido mientras JARVIS trabaja (o lo libera)."""
            self._freeze_max = 1.0 if active else 0.0
            if active:
                self._target = 1.0
            self.update()

        def set_color(self, color: str) -> None:
            self._color = QColor(color)
            self.update()

        def tick(self, dt: float) -> None:
            """Suaviza la transición de encendido/apagado."""
            now = time.monotonic()
            if self._until and now > self._until:
                self._until = 0.0
                self._target = self._freeze_max
            speed = dt / max(0.05, cfg.HUD_FADE_MS / 1000.0)
            self.level = lerp(self.level, self._target, min(1.0, speed))
            if self.level < 0.004 and self._target == 0.0:
                self.level = 0.0
            self.update()

        def paintEvent(self, event) -> None:  # noqa: N802 - API de Qt
            if self.level <= 0.0:
                return
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            rect = QRectF(1.5, 1.5, self.width() - 3.0, self.height() - 3.0)
            thickness = float(cfg.HUD_BORDER_THICKNESS)

            # Capa exterior difusa (el "halo").
            halo = QColor(self._color)
            halo.setAlpha(int(52 * self.level))
            painter.setPen(QPen(halo, thickness + cfg.HUD_GLOW_BLUR * 0.9))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, cfg.HUD_BORDER_RADIUS, cfg.HUD_BORDER_RADIUS)

            # Capa intermedia.
            mid = QColor(self._color)
            mid.setAlpha(int(120 * self.level))
            painter.setPen(QPen(mid, thickness + 3.0))
            painter.drawRoundedRect(rect, cfg.HUD_BORDER_RADIUS, cfg.HUD_BORDER_RADIUS)

            # Línea principal brillante.
            core = QColor(self._color)
            core.setAlpha(int(235 * self.level))
            painter.setPen(QPen(core, thickness))
            painter.drawRoundedRect(rect, cfg.HUD_BORDER_RADIUS, cfg.HUD_BORDER_RADIUS)
            painter.end()


# =============================================================================
# CONTROLADOR DEL HUD
# =============================================================================

class GhostHud:
    """Gestiona la ventana, los widgets y el bucle de Qt.

    Es el único objeto de la interfaz que ve el resto de JARVIS, y su API es
    segura entre hilos: se puede llamar desde el motor de voz, desde el
    enrutador o desde el kill-switch sin preocuparse por Qt.
    """

    def __init__(self, killswitch: Any | None = None, app: Any | None = None) -> None:
        if not PYQT_AVAILABLE:
            raise RuntimeError(
                "PyQt6 no está instalado: el HUD no puede mostrarse. "
                "Ejecuta install.bat o usa 'python main.py --texto'."
            )
        self.killswitch = killswitch
        self._own_app = app is None
        self.app = app or QApplication.instance() or QApplication(sys.argv[:1])
        self.app.setQuitOnLastWindowClosed(False)

        self.signals = HudSignals()
        self.capsule = CapsuleWidget()
        self.border = BorderWidget()
        self.state = "sleeping"
        self.detail = ""
        self._running = False
        self._last_tick = time.monotonic()
        self._idle_frames = 0

        self._timer = QTimer()
        self._timer.setInterval(int(1000 / max(10, cfg.HUD_FPS)))
        self._timer.timeout.connect(self._tick)

        self.signals.state_changed.connect(self._apply_state)
        self.signals.glow_requested.connect(lambda seconds: self.border.illuminate(seconds))
        self.signals.border_level.connect(self.border.set_level)
        self.signals.close_requested.connect(self.stop)

        if killswitch is not None:
            killswitch.register("hud", self._on_killswitch)

    # --- API pública (segura entre hilos) ---------------------------------

    def set_state(self, state: str, detail: str = "") -> None:
        """Cambia el estado visible (se puede llamar desde cualquier hilo)."""
        self.signals.state_changed.emit(str(state or "idle"), str(detail or ""))

    def glow(self, seconds: float = 1.2) -> None:
        """Ilumina el marco de neón (capturas, visión, acciones sobre el equipo)."""
        self.signals.glow_requested.emit(float(seconds))

    def set_border(self, level: float) -> None:
        """Fija el nivel del marco (por ejemplo, 1.0 mientras se automatiza)."""
        self.signals.border_level.emit(float(level))

    def spinner(self, active: bool) -> None:
        """Activa o desactiva el anillo de progreso (inferencia / cambio de modelo)."""
        self.set_state("thinking" if active else "idle")

    def quit(self) -> None:
        """Pide el cierre ordenado desde cualquier hilo."""
        self.signals.close_requested.emit()

    # --- Interno -----------------------------------------------------------

    def _apply_state(self, state: str, detail: str) -> None:
        self.state = state
        self.detail = detail
        self.capsule.set_state(state, detail)
        if state in ("vision", "acting"):
            self.border.set_color(cfg.HUD_BORDER_ACTIVE_COLOR)
            self.border.freeze(True)
        elif state == "killswitch":
            self.border.set_color(cfg.HUD_COLOR_KILL)
            self.border.set_level(0.0)
            self.border.freeze(False)
        elif state in ("error",):
            self.border.set_color(cfg.HUD_COLOR_ERROR)
            self.border.illuminate(0.8)
        else:
            self.border.set_color(cfg.HUD_BORDER_ACTIVE_COLOR)
            self.border.freeze(False)

    def _on_killswitch(self, event: Any) -> None:
        """El kill-switch apaga el marco y marca la cápsula en ámbar."""
        try:
            self.set_border(0.0)
            self.set_state("killswitch", "Control liberado")
        except Exception:  # pragma: no cover
            pass

    def _tick(self) -> None:
        """Latido de las animaciones (60 FPS activo, 12 FPS en reposo)."""
        now = time.monotonic()
        dt = max(0.001, now - self._last_tick)
        self._last_tick = now
        self.capsule.tick(dt)
        self.border.tick(dt)

        idle = self.state in ("sleeping", "idle") and self.border.level <= 0.01
        self._idle_frames = self._idle_frames + 1 if idle else 0
        target_interval = int(1000 / max(10, cfg.HUD_FPS)) if self._idle_frames < 60 else 83
        if self._timer.interval() != target_interval:
            self._timer.setInterval(target_interval)

        if self.killswitch is not None and self.killswitch.is_tripped():
            if self.state != "killswitch":
                self._on_killswitch(None)

    # --- Ciclo de vida -----------------------------------------------------

    def start(self) -> None:
        """Muestra los widgets y arranca el temporizador (no bloquea)."""
        self.capsule.reposition()
        self.border.reposition()
        self.capsule.show()
        # El marco se muestra transparente; solo se ve cuando se ilumina.
        self.border.show()
        self.border.set_level(0.0)
        self._last_tick = time.monotonic()
        self._timer.start()
        self._running = True
        logger.info("GHOST HUD visible (click-through activo).")

    def run(self) -> int:
        """Muestra el HUD y ejecuta el bucle de Qt (BLOQUEA el hilo actual)."""
        self.start()
        return int(self.app.exec())

    def stop(self) -> None:
        """Oculta los widgets y detiene el bucle de Qt."""
        if not self._running:
            return
        self._running = False
        self._timer.stop()
        try:
            self.capsule.hide()
            self.border.hide()
        except Exception:  # pragma: no cover
            pass
        if self._own_app:
            self.app.quit()
        logger.info("GHOST HUD oculto.")

    def status(self) -> dict:
        return {
            "activo": self._running,
            "estado": self.state,
            "detalle": self.detail,
            "marco": round(self.border.level, 3),
            "click_through": self.click_through_enabled(),
        }

    def click_through_enabled(self) -> bool:
        """¿Están aplicados los estilos de traspaso de clics?"""
        if not PYQT_AVAILABLE:
            return False
        flags = self.capsule.windowFlags()
        return bool(flags & Qt.WindowType.WindowTransparentForInput)


# =============================================================================
# INTEGRACIÓN CON WINDOWS Y WATCHDOG DE ESTADO
# =============================================================================

def apply_windows_click_through(hud: "GhostHud") -> bool:
    """Refuerza el click-through en Windows con los estilos extendidos.

    Qt ya aplica `WindowTransparentForInput`, pero en Windows conviene añadir
    `WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE`
    para garantizar que el HUD nunca roba un clic ni aparece en Alt+Tab.
    """
    if sys.platform != "win32" or not PYQT_AVAILABLE:  # pragma: no cover
        return False
    try:
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020
        WS_EX_TOOLWINDOW = 0x00000080
        WS_EX_NOACTIVATE = 0x08000000

        user32 = ctypes.windll.user32
        ok = False
        for widget in (hud.capsule, hud.border):
            handle = int(widget.winId())
            style = user32.GetWindowLongW(handle, GWL_EXSTYLE)
            style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
            user32.SetWindowLongW(handle, GWL_EXSTYLE, style)
            ok = True
        logger.debug("Click-through de Windows aplicado.")
        return ok
    except Exception as exc:  # pragma: no cover
        logger.debug("No se pudo reforzar el click-through: %s", exc)
        return False


def create_hud(killswitch: Any | None = None, enabled: bool | None = None) -> "GhostHud | None":
    """Crea el HUD si es posible; devuelve None (sin excepción) si no lo es.

    Así `main.py` puede arrancar en consola aunque PyQt6 falle, que es
    exactamente lo que se quiere en un equipo recién configurado.
    """
    if enabled is None:
        enabled = cfg.HUD_ENABLED
    if not enabled:
        logger.info("El HUD está desactivado en config.py.")
        return None
    if not PYQT_AVAILABLE:
        logger.warning(
            "PyQt6 no está instalado: JARVIS funcionará sin HUD. "
            "Instálalo con: pip install -r requirements.txt"
        )
        return None
    try:
        hud = GhostHud(killswitch=killswitch)
        apply_windows_click_through(hud)
        return hud
    except Exception as exc:  # pragma: no cover
        logger.error("No he podido crear el HUD: %s", exc)
        return None


class HudBridge:
    """Objeto ligero que `main.py` entrega a los demás módulos como `notifier`.

    Traduce las notificaciones internas ("vision", "acting", "swapping"...) a
    estados del HUD y a iluminaciones del marco, sin que los módulos de lógica
    sepan nada de Qt. Si no hay HUD, guarda el último estado y no hace nada más.
    """

    def __init__(self, hud: "GhostHud | None" = None) -> None:
        self.hud = hud
        self.last_state = "sleeping"
        self.last_detail = ""
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}

    def __call__(self, state: str, detail: str = "") -> None:
        """Se usa como función `notifier(estado, detalle)`."""
        self.notify(state, detail)

    def notify(self, state: str, detail: str = "") -> None:
        """Actualiza el HUD y, si procede, ilumina el marco."""
        with self._lock:
            self.last_state = state
            self.last_detail = detail
            self.counts[state] = self.counts.get(state, 0) + 1
        if self.hud is None:
            return
        try:
            self.hud.set_state(state, detail)
            if state == "vision":
                self.hud.glow(1.4)
            elif state == "acting":
                self.hud.glow(0.9)
        except Exception as exc:  # pragma: no cover
            logger.debug("El HUD no pudo actualizarse: %s", exc)

    def glow(self, seconds: float = 1.2) -> None:
        if self.hud is not None:
            try:
                self.hud.glow(seconds)
            except Exception:  # pragma: no cover
                pass

    def spinner(self, active: bool) -> None:
        if self.hud is not None:
            try:
                self.hud.spinner(active)
            except Exception:  # pragma: no cover
                pass

    def status(self) -> dict:
        return {
            "estado": self.last_state,
            "detalle": self.last_detail,
            "contadores": dict(self.counts),
            "hud": self.hud.status() if self.hud is not None else None,
        }
