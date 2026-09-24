"""
core/media_dispatcher.py — Música, navegador y apertura de aplicaciones.
================================================================================
Aquí vive la rutina de la **doble palmada**: poner "Loser" de Tame Impala y
saludar a Pablo según la hora del reloj. También es el módulo que sabe dónde
está cada cosa en el ordenador de Pablo.

CADENA DE REPRODUCCIÓN (con reserva automática)
-----------------------------------------------
1. **Spotify local**  — se lanza el cliente, se envía la URI de búsqueda y se
   pulsa reproducir.
2. **Comet**          — si Spotify no está instalado, está cerrado o falla, se
   abre el navegador de Perplexity en YouTube con la canción.
3. **Navegador del sistema** — si Comet no existe, se usa Chrome, Edge,
   Firefox, Opera o Brave, el que haya.
4. **Aviso por voz**  — si nada funciona, JARVIS lo dice claramente.

Rutas de Comet que se comprueban (en orden)
-------------------------------------------
    %LOCALAPPDATA%\\Perplexity\\Comet\\Application\\comet.exe
    C:\\Perplexity\\Comet\\Application\\comet.exe
    %PROGRAMFILES%\\Perplexity\\Comet\\Application\\comet.exe
================================================================================
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import config as cfg  # noqa: E402
from core.utils import find_executable, normalize_text, run_in_thread, truncate  # noqa: E402

logger = logging.getLogger("jarvis.media")


# =============================================================================
# RESULTADO DE UNA ACCIÓN DE MEDIOS
# =============================================================================

@dataclass
class DispatchResult:
    """Qué se ha conseguido hacer, con el detalle suficiente para explicarlo."""

    accion: str = ""
    exito: bool = False
    metodo: str = ""
    saludo: str = ""
    detalles: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.detalles.append(message)

    def describe(self) -> str:
        estado = "OK" if self.exito else "FALLO"
        extra = f" ({self.metodo})" if self.metodo else ""
        return f"{self.accion}: {estado}{extra}"


# =============================================================================
# LANZADOR DE PROCESOS
# =============================================================================

def _popen_detached(argv: Sequence[str] | str, shell: bool = False) -> bool:
    """Lanza un programa sin bloquear ni heredar la consola de JARVIS.

    Es el único lugar del proyecto (con el guardián de comandos) autorizado a
    lanzar procesos "de sistema": aquí no entra código generado por un modelo,
    solo rutas de navegadores y aplicaciones que Pablo usa a diario.
    """
    try:
        kwargs: dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
            "close_fds": True,
        }
        if sys.platform == "win32":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP: el programa sobrevive a JARVIS.
            kwargs["creationflags"] = 0x00000008 | 0x00000200
        subprocess.Popen(argv, shell=shell, **kwargs)  # noqa: S603 - rutas verificadas
        return True
    except (OSError, ValueError) as exc:
        logger.warning("No se pudo lanzar %s: %s", argv, exc)
        return False


def _open_with_os_startfile(target: str) -> bool:
    """Abre una ruta, una URL o un protocolo con la aplicación asociada."""
    if sys.platform == "win32":
        try:
            os.startfile(target)  # type: ignore[attr-defined]  # noqa: S606
            return True
        except OSError as exc:
            logger.debug("os.startfile falló para %s: %s", target, exc)
            return False
    if sys.platform == "darwin":
        return _popen_detached(["open", target])
    return _popen_detached(["xdg-open", target])


def open_uri(uri: str) -> bool:
    """Abre un protocolo del sistema (``spotify:``, ``ms-settings:``, ``http:``)."""
    if _open_with_os_startfile(uri):
        return True
    if sys.platform == "win32":
        # Alternativa clásica: el shell de Windows resuelve los protocolos.
        return _popen_detached(["cmd", "/c", "start", "", uri], shell=False)
    return False


# =============================================================================
# DESPACHADOR
# =============================================================================

class MediaDispatcher:
    """Ejecuta las acciones de música, navegador y aplicaciones.

    Parameters
    ----------
    notifier:
        Función `(estado, detalle)` para el HUD.
    killswitch:
        Interruptor de emergencia (se consulta antes de cada paso).
    speaker:
        Función que dice una frase en voz alta (la inyecta `main.py`). Si no se
        pasa, el saludo se escribe en el registro pero no se pronuncia.
    """

    def __init__(
        self,
        notifier: Callable[[str, str], None] | None = None,
        killswitch: Any | None = None,
        speaker: Callable[[str], None] | None = None,
    ) -> None:
        self.notifier = notifier
        self.killswitch = killswitch
        self.speaker = speaker
        self._spotify_lock = threading.Lock()

    # --- Notificación y control -------------------------------------------

    def _notify(self, state: str, detail: str = "") -> None:
        if self.notifier is None:
            return
        try:
            self.notifier(state, detail)
        except Exception as exc:  # pragma: no cover
            logger.debug("Notificador del HUD falló: %s", exc)

    def _cancelled(self) -> bool:
        return bool(self.killswitch is not None and self.killswitch.is_tripped())

    def say(self, text: str) -> None:
        """Dice algo en voz alta (si hay motor de voz conectado)."""
        logger.info("JARVIS dice: %s", text)
        if self.speaker is None:
            return
        try:
            self.speaker(text)
        except Exception as exc:  # pragma: no cover
            logger.error("No se pudo pronunciar el saludo: %s", exc)

    # --- Localización de programas ----------------------------------------

    @staticmethod
    def find_comet() -> str | None:
        """Ruta de Comet, el navegador de Perplexity (o None si no está)."""
        return find_executable(cfg.BROWSER_COMET_CANDIDATES)

    @staticmethod
    def find_system_browser() -> str | None:
        """Primer navegador de reserva disponible en el equipo."""
        return find_executable(cfg.BROWSER_FALLBACK_CANDIDATES)

    @classmethod
    def find_browser(cls, prefer_comet: bool | None = None) -> tuple[str | None, str]:
        """Devuelve `(ruta, nombre)` del navegador a usar."""
        prefer_comet = cfg.BROWSER_PREFER_COMET if prefer_comet is None else prefer_comet
        comet = cls.find_comet()
        if comet and prefer_comet:
            return comet, "Comet"
        fallback = cls.find_system_browser()
        if fallback:
            name = Path(fallback).stem.capitalize()
            return fallback, name
        if comet:
            return comet, "Comet"
        return None, ""

    @staticmethod
    def find_spotify() -> str | None:
        """Ruta del cliente de Spotify, si está instalado."""
        for candidate in cfg.SPOTIFY_CANDIDATES:
            found = find_executable([candidate])
            if found:
                return found
        return None

    @staticmethod
    def is_spotify_running() -> bool:
        """¿Hay un proceso de Spotify activo? (Sin dependencias externas)."""
        try:
            import psutil  # type: ignore

            for process in psutil.process_iter(["name"]):
                name = (process.info.get("name") or "").lower()
                if "spotify" in name:
                    return True
        except Exception:
            return False
        return False

    # --- Navegador ---------------------------------------------------------

    def open_url(
        self,
        url: str,
        prefer_comet: bool | None = None,
        new_window: bool | None = None,
    ) -> tuple[bool, str]:
        """Abre una URL en Comet (o en el navegador disponible).

        Returns
        -------
        (exito, metodo)
        """
        if self._cancelled():
            return False, "kill-switch"
        browser, browser_name = self.find_browser(prefer_comet)
        new_window = cfg.BROWSER_NEW_WINDOW if new_window is None else new_window
        args = [browser]
        if new_window:
            args.append("--new-window")
        args.append(url)

        if browser and _popen_detached(args):
            logger.info("Abriendo %s en %s", url, browser_name)
            return True, browser_name

        if _open_with_os_startfile(url):
            logger.info("Abriendo %s en el navegador predeterminado del sistema", url)
            return True, "navegador del sistema"

        logger.error("No he podido abrir la URL %s", url)
        return False, ""

    # --- Spotify -----------------------------------------------------------

    def launch_spotify(self) -> bool:
        """Lanza el cliente de Spotify y espera unos segundos a que levante."""
        exe = self.find_spotify()
        if not exe:
            logger.info("Spotify no está instalado en este equipo.")
            return False
        if self.is_spotify_running():
            logger.info("Spotify ya estaba abierto.")
            return True
        if not _popen_detached([exe]):
            return False
        logger.info("Cliente de Spotify lanzado. Esperando a que cargue...")
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            if self._cancelled():
                return False
            if self.is_spotify_running():
                time.sleep(1.5)  # Un respiro para que la interfaz esté lista.
                return True
            time.sleep(0.4)
        logger.warning("Spotify tardó demasiado en arrancar; se continúa de todas formas.")
        return True

    def play_on_spotify(self, uri: str | None = None, press_play: bool = True) -> bool:
        """Reproduce una pista en Spotify a través de su URI.

        Primero abre la aplicación (si hace falta) y luego envía la URI con el
        protocolo ``spotify:``. Si Pablo ha pegado una URI de pista exacta en
        `config.CLAP_SPOTIFY_URI`, esa tiene prioridad.
        """
        with self._spotify_lock:
            if self._cancelled():
                return False
            target_uri = uri or cfg.CLAP_SPOTIFY_URI or cfg.CLAP_SPOTIFY_SEARCH_URI
            if not self.launch_spotify():
                return False

            logger.info("Enviando a Spotify: %s", target_uri)
            if not open_uri(target_uri):
                return False

            if press_play and self._cancelled():
                return False
            time.sleep(max(0.5, cfg.CLAP_SPOTIFY_PLAY_DELAY_S))

            if press_play and cfg.CLAP_SPOTIFY_URI == "":
                # Al buscar por texto, Spotify deja la lista de resultados abierta:
                # se activa la ventana y se pulsa "reproducir" para empezar la 1ª pista.
                self._focus_window("spotify")
                self._press_key("space")
            elif press_play:
                self._focus_window("spotify")
                self._press_key("space")
            return True

    # --- YouTube -----------------------------------------------------------

    def play_on_youtube(self, url: str | None = None, auto_play: bool = True) -> bool:
        """Abre la canción en YouTube dentro de Comet y arranca la reproducción."""
        url = url or cfg.CLAP_YOUTUBE_URL
        ok, method = self.open_url(url)
        if not ok:
            return False
        logger.info("YouTube abierto en %s: %s", method, url)
        if auto_play and not self._cancelled():
            time.sleep(max(1.0, cfg.BROWSER_START_TIMEOUT_S))
            # La tecla 'k' reproduce/pausa en el reproductor de YouTube.
            self._focus_window("youtube")
            self._press_key("k")
        return True

    # --- La rutina completa de la doble palmada ---------------------------

    def double_clap_routine(self, deliver_music_first: bool = True) -> DispatchResult:
        """Lo que hace JARVIS cuando Pablo da dos palmadas.

        1. Calcula el saludo según la hora del reloj (buenos días / tardes / noches).
        2. Lanza la música por la mejor vía disponible.
        3. Saluda en voz alta a la vez que suena la canción.
        """
        result = DispatchResult(accion="doble_palmada")
        result.saludo = cfg.greeting_for_hour()
        self._notify("acting", "Doble palmada")

        if self._cancelled():
            result.add("Kill-switch activado: no se lanza nada.")
            return result

        if deliver_music_first:
            # El saludo entra primero en la cola de voz (suena al instante) y la
            # música arranca en paralelo: JARVIS saluda mientras suena la canción.
            self.say(result.saludo)
            music_thread = run_in_thread(
                self._dispatch_music_chain,
                result,
                name="jarvis-musica-palmada",
            )
            music_thread.join(timeout=45.0)
        else:
            self.say(result.saludo)
            self._dispatch_music_chain(result)

        self._notify("idle", "")
        logger.info("Rutina de doble palmada: %s", " | ".join(result.detalles) or result.describe())
        return result

    def _dispatch_music_chain(self, result: DispatchResult) -> None:
        """Cadena Spotify -> Comet/YouTube -> navegador del sistema."""
        # 1) Spotify local -----------------------------------------------------
        try:
            if self.play_on_spotify():
                result.exito = True
                result.metodo = "Spotify"
                result.add(f"Reproduciendo '{cfg.CLAP_TRACK_QUERY}' en Spotify.")
                return
            result.add("Spotify no disponible.")
        except Exception as exc:  # pragma: no cover - depende del sistema
            logger.error("Spotify falló: %s", exc)
            result.add(f"Spotify falló: {truncate(str(exc), 120)}")

        if self._cancelled():
            result.add("Cancelado por el kill-switch antes del plan B.")
            return

        # 2) Comet / YouTube ---------------------------------------------------
        try:
            if self.play_on_youtube():
                browser, name = self.find_browser()
                result.exito = True
                result.metodo = f"{name} + YouTube"
                result.add(f"YouTube abierto en {name} con '{cfg.CLAP_TRACK_QUERY}'.")
                return
            result.add("Comet/YouTube no respondió.")
        except Exception as exc:  # pragma: no cover
            logger.error("YouTube falló: %s", exc)
            result.add(f"YouTube falló: {truncate(str(exc), 120)}")

        # 3) Navegador predeterminado del sistema ------------------------------
        if _open_with_os_startfile(cfg.CLAP_YOUTUBE_URL):
            result.exito = True
            result.metodo = "navegador del sistema"
            result.add("Abierto en el navegador predeterminado.")
        else:
            result.add("No he podido abrir ningún navegador.")
            self.say(
                "Lo siento, Pablo: no encuentro ni Spotify ni un navegador para poner la música."
            )

    # --- Aplicaciones ------------------------------------------------------

    def launch_app(self, name: str) -> tuple[bool, str]:
        """Abre una aplicación por su nombre hablado.

        Busca en el catálogo `config.KNOWN_APPS` (con alias en español), en el
        PATH del sistema y, como último recurso, dejar que Windows resuelva el
        comando con `start`.
        """
        if not name:
            return False, ""
        key = normalize_text(name)
        if key in ("comet", "navegador", "el navegador", "perplexity"):
            browser, browser_name = self.find_browser()
            if browser and _popen_detached([browser]):
                return True, browser_name

        command = cfg.KNOWN_APPS.get(key)
        if command is None:
            for alias, candidate in cfg.KNOWN_APPS.items():
                if key and (key in alias or alias in key):
                    command = candidate
                    break
        if command is None:
            command = name.strip()

        if command.lower().startswith("spotify") or command == "spotify":
            if self.launch_spotify():
                return True, "Spotify"

        resolved = find_executable([command])
        if resolved and _popen_detached([resolved]):
            return True, Path(resolved).name

        if sys.platform == "win32":
            if _popen_detached(["cmd", "/c", "start", "", command], shell=False):
                return True, command
        elif _open_with_os_startfile(command):
            return True, command

        logger.warning("No he podido abrir la aplicación '%s'.", name)
        return False, ""

    # --- Ayudantes de automatización sobre ventanas ------------------------

    @staticmethod
    def _focus_window(title_fragment: str) -> bool:
        """Trae al frente la primera ventana cuyo título contenga el texto.

        Se usa solo como apoyo para pulsar "play" tras abrir un reproductor.
        """
        try:
            import pygetwindow  # type: ignore

            fragment = title_fragment.lower()
            for window in pygetwindow.getAllWindows():
                if fragment and fragment in str(window.title).lower():
                    try:
                        if getattr(window, "isMinimized", False):
                            window.restore()
                        window.activate()
                    except Exception:
                        try:
                            window.minimize()
                            window.restore()
                        except Exception:
                            pass
                    time.sleep(0.6)
                    return True
        except Exception as exc:
            logger.debug("No se pudo enfocar la ventana '%s': %s", title_fragment, exc)
        return False

    @staticmethod
    def _press_key(key: str) -> bool:
        """Pulsa una tecla con pyautogui (con tolerancia a fallos)."""
        try:
            import pyautogui  # type: ignore

            pyautogui.press(key)
            return True
        except Exception as exc:
            logger.debug("No se pudo pulsar '%s': %s", key, exc)
            return False
