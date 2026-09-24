"""
main.py — JARVIS: asistente de escritorio local, privado y gratuito.
================================================================================
Este archivo es el director de orquesta. Arranca los diez subsistemas, los
conecta entre sí y entrega el control a Pablo.

    main.py
      |
      +-- 1. Puerta de seguridad ...... hash del kill-switch + archivos en solo lectura
      +-- 2. Configuración ............ carpetas, registro, un solo JARVIS a la vez
      +-- 3. Cerebros ................. Ollama + enrutador de modelos (monogamia de VRAM)
      +-- 4. Sentidos ................. voz (Whisper/Kokoro), doble palmada, visión
      +-- 5. Manos .................... ratón, teclado, navegador, Spotify
      +-- 6. Creadores ................ auto-programación y constructor de apps
      +-- 7. Interfaz ................. GHOST HUD (cápsula + marco de neón)
      |
      +-- Bucle principal: escucha, entiende, responde y actúa.

USO
---
    python main.py                 # arranque normal (HUD + voz)
    python main.py --texto         # modo consola, sin micrófono (para probar)
    python main.py --sin-hud       # sin interfaz flotante
    python main.py --diagnostico   # informe completo del estado del sistema
    python main.py --prueba-voz    # comprueba la voz y los oídos
    python main.py --matar         # detiene la instancia en marcha
    python main.py --sellar-seguridad   # recalcula el hash del kill-switch
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable

# --- Bootstrap: la raíz del proyecto siempre en el camino de importación ------
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config as cfg  # noqa: E402
from core.acoustic_detector import AcousticTrigger, ClapDetector  # noqa: E402
from core.app_builder import AppBuilder  # noqa: E402
from core.media_dispatcher import MediaDispatcher  # noqa: E402
from core.model_router import ModelRouter  # noqa: E402
from core.ollama_client import OllamaClient, OllamaNotRunningError  # noqa: E402
from core.self_programmer import SelfProgrammer  # noqa: E402
from core.utils import normalize_text, normalize_text_map  # noqa: E402
from core.vision_actuator import (  # noqa: E402
    AutomationInterrupted,
    ScreenActuator,
    ScreenCapture,
    ScreenPerception,
    VisionExecutor,
)
from core.voice_engine import VoiceEngine  # noqa: E402
from gui.hud_overlay import HudBridge, create_hud  # noqa: E402
from safety.killswitch import (  # noqa: E402
    KillSwitchIntegrityError,
    TripEvent,
    get_killswitch,
    harden_files,
    integrity_gate,
    verify_integrity,
    write_hash_file,
)

logger = logging.getLogger("jarvis.main")

SEARCH_URL = "https://www.google.com/search?q={query}"
SPOTIFY_SEARCH_URI = "spotify:search:{query}"


# =============================================================================
# LÍNEA DE COMANDOS
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Define las opciones de arranque de JARVIS."""
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="JARVIS — asistente de escritorio local, privado y gratuito.",
        epilog="Sin opciones, JARVIS arranca con voz, oídos y HUD.",
    )
    parser.add_argument("--texto", action="store_true", help="Modo consola (sin micrófono ni voz).")
    parser.add_argument("--sin-hud", action="store_true", help="Arranca sin la interfaz flotante.")
    parser.add_argument("--sin-voz", action="store_true", help="Desactiva la síntesis de voz.")
    parser.add_argument("--sin-palmada", action="store_true", help="Desactiva la doble palmada.")
    parser.add_argument("--sin-vision", action="store_true", help="Desactiva la visión por pantalla.")
    parser.add_argument("--diagnostico", action="store_true", help="Muestra un informe y sale.")
    parser.add_argument("--prueba-voz", action="store_true", help="Prueba micrófono y voz y sale.")
    parser.add_argument("--prueba-palmada", action="store_true", help="Escucha palmadas 60 s y sale.")
    parser.add_argument("--verificar-seguridad", action="store_true", help="Comprueba el hash del kill-switch.")
    parser.add_argument("--sellar-seguridad", action="store_true", help="Recalcula el hash del kill-switch.")
    parser.add_argument("--matar", action="store_true", help="Detiene una instancia de JARVIS en marcha.")
    parser.add_argument("--forzar", action="store_true", help="Ignora el aviso de otra instancia activa.")
    parser.add_argument("--version", action="store_true", help="Muestra la versión y sale.")
    return parser


# =============================================================================
# NÚCLEO DE JARVIS
# =============================================================================

class Jarvis:
    """Une todos los subsistemas y da sentido a lo que dice Pablo.

    Es el único lugar donde se conectan voz, modelos, visión, actuadores,
    interfaz y seguridad. Cada subsistema puede fallar por separado: JARVIS
    degrada sus capacidades (por ejemplo, modo texto si no hay micrófono) en
    lugar de no arrancar.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.started_at = time.time()
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self.debug_mode = bool(getattr(args, "texto", False))

        # Subsistemas (se rellenan en `boot`).
        self.client: OllamaClient | None = None
        self.router: ModelRouter | None = None
        self.hud = None
        self.hud_bridge = HudBridge(None)
        self.killswitch = None
        self.media: MediaDispatcher | None = None
        self.perception: ScreenPerception | None = None
        self.actuator: ScreenActuator | None = None
        self.executor: VisionExecutor | None = None
        self.clap: AcousticTrigger | None = None
        self.voice: VoiceEngine | None = None
        self.app_builder: AppBuilder | None = None
        self.self_programmer: SelfProgrammer | None = None
        self.history: list[dict] = []
        self._listen_thread: threading.Thread | None = None
        self._keeper_thread: threading.Thread | None = None
        self._shutting_down = False

    # ------------------------------------------------------------------ #
    # ARRANQUE
    # ------------------------------------------------------------------ #

    def boot(self) -> bool:
        """Prepara seguridad, configuración y subsistemas. Devuelve si arrancó."""
        cfg.ensure_directories()
        cfg.setup_logging()
        logger.info("=" * 74)
        logger.info("  %s v%s — %s", cfg.APP_NAME, cfg.APP_VERSION, cfg.APP_TITLE)
        logger.info("=" * 74)

        if not self._security_gate():
            return False
        if not self._single_instance():
            return False
        self._bootstrap_models()
        self._bootstrap_senses()
        self._bootstrap_creators()
        self._bootstrap_interface()
        return True

    def _security_gate(self) -> bool:
        """Verifica la integridad del kill-switch y bloquea los archivos de seguridad."""
        ok, message = verify_integrity()
        if not ok and not self.args.forzar:
            logger.critical(message)
            logger.critical(
                "JARVIS no arranca: la capa de seguridad ha cambiado. Si fuiste tú, Pablo, "
                "ejecuta 'python main.py --sellar-seguridad'."
            )
            print("\n" + "!" * 70)
            print("  ALERTA DE SEGURIDAD")
            print(message)
            print("  Si el cambio es legítimo: python main.py --sellar-seguridad")
            print("!" * 70 + "\n")
            return False
        if not ok:
            logger.warning("Se continúa pese al fallo de integridad (--forzar): %s", message)

        try:
            integrity_gate(strict=False)
        except KillSwitchIntegrityError as exc:  # pragma: no cover
            logger.error("%s", exc)

        if cfg.KILLSWITCH_HARDEN_FILES:
            harden_files()
        self.killswitch = get_killswitch(on_trip=self._on_killswitch)
        logger.info("Kill-switch listo: %s", self.killswitch.status())
        return True

    def _single_instance(self) -> bool:
        """Evita que dos JARVIS se peleen por el micrófono."""
        if not cfg.SINGLE_INSTANCE or self.args.forzar:
            return True
        pid_file = Path(cfg.PID_FILE)
        if pid_file.exists():
            try:
                other = int(pid_file.read_text(encoding="utf-8").strip() or 0)
            except (OSError, ValueError):
                other = 0
            if other and other != os.getpid() and self._process_alive(other):
                print(
                    f"\nYa hay un JARVIS en marcha (PID {other}).\n"
                    "  · Para hablar con él, usa el micrófono o el HUD.\n"
                    "  · Para detenerlo: python main.py --matar\n"
                    "  · Para arrancar otro de todos modos: python main.py --forzar\n"
                )
                return False
        try:
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(str(os.getpid()), encoding="utf-8")
        except OSError as exc:
            logger.debug("No pude escribir el archivo de PID: %s", exc)
        return True

    @staticmethod
    def _process_alive(pid: int) -> bool:
        """¿Sigue vivo ese proceso? (sin depender de psutil)."""
        try:
            if sys.platform == "win32":  # pragma: no cover
                import ctypes

                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    return True
                return False
            os.kill(pid, 0)
            return True
        except Exception:
            return False

    def _bootstrap_models(self) -> None:
        """Crea el cliente de Ollama y el enrutador; verifica la VRAM."""
        self.client = OllamaClient(cancel_check=lambda: bool(self.killswitch and self.killswitch.is_tripped()))
        self.router = ModelRouter(
            client=self.client,
            killswitch=self.killswitch,
            notifier=self.hud_bridge,
            auto_pull=True,
        )
        if not self.client.health():
            logger.warning(
                "Ollama no responde: las respuestas del modelo no estarán disponibles.\n"
                "  · Abre la aplicación Ollama o ejecuta 'ollama serve'."
            )
            self.hud_bridge.notify("error", "Ollama apagado")
        else:
            version = self.client.version()
            logger.info("Ollama %s activo en %s", version or "?", cfg.OLLAMA_HOST)
            freed = self.router.enforce_monogamy_now()
            if freed:
                logger.info("Monogamia de VRAM aplicada al arrancar: %s", ", ".join(freed))
            self.hud_bridge.notify("idle", "")

    def _bootstrap_senses(self) -> None:
        """Monta la visión, la voz, la doble palmada y el cortafuegos de interrupción."""
        assert self.router is not None

        # --- Visión y actuadores -----------------------------------------
        self.perception = ScreenPerception(self.router, ScreenCapture(), notifier=self.hud_bridge)
        self.actuator = ScreenActuator(
            gate=None, killswitch=self.killswitch, notifier=self.hud_bridge
        )
        self.executor = VisionExecutor(
            perception=self.perception,
            actuator=self.actuator,
            router=self.router,
            notifier=self.hud_bridge,
        )

        # --- Medios (Spotify / Comet / YouTube / aplicaciones) -----------
        self.media = MediaDispatcher(notifier=self.hud_bridge, killswitch=self.killswitch)

        # --- Doble palmada -----------------------------------------------
        if cfg.CLAP_ENABLED and not self.args.sin_palmada:
            self.clap = AcousticTrigger(
                action=self._on_double_clap,
                detector=ClapDetector(),
                notifier=self.hud_bridge,
            )

        # --- Voz ----------------------------------------------------------
        self.voice = VoiceEngine(
            on_command=self.handle_command,
            on_state=self.hud_bridge.notify,
            killswitch=self.killswitch,
            clap_trigger=self.clap,
            on_wake=self._on_wake,
            allow_wake_word=bool(cfg.WAKE_WORD_ENABLED and not self.args.texto),
        )
        # Barge-in conectado al ejecutor: congelar la automatización y corregir el plan.
        self.voice.barge_in.gate = self.actuator.gate
        if self.executor is not None:
            self.executor.voice = self.voice
        if self.media is not None:
            self.media.speaker = self.voice.say

    def _bootstrap_creators(self) -> None:
        """Auto-programación y constructor de aplicaciones."""
        assert self.router is not None
        self.app_builder = AppBuilder(
            router=self.router,
            media=self.media,
            killswitch=self.killswitch,
            notifier=self.hud_bridge,
        )
        self.self_programmer = SelfProgrammer(
            router=self.router,
            killswitch=self.killswitch,
            notifier=self.hud_bridge,
        )

    def _bootstrap_interface(self) -> None:
        """Crea el GHOST HUD (si PyQt6 está disponible y no se ha pedido lo contrario)."""
        if self.args.texto or self.args.sin_hud or self.args.diagnostico:
            self.hud_bridge = HudBridge(None)
            return
        self.hud = create_hud(killswitch=self.killswitch, enabled=cfg.HUD_ENABLED)
        self.hud_bridge = HudBridge(self.hud)
        if self.hud is not None:
            self.hud.set_state("sleeping", "")
        # Reengancha el notificador del HUD en los módulos ya creados.
        for module in (self.router, self.perception, self.actuator, self.media, self.voice,
                       self.app_builder, self.self_programmer):
            if module is not None and hasattr(module, "notifier"):
                try:
                    module.notifier = self.hud_bridge
                except Exception:  # pragma: no cover
                    pass

    # ------------------------------------------------------------------ #
    # EVENTOS
    # ------------------------------------------------------------------ #

    def _on_wake(self) -> None:
        """Se ha oído "Jarvis": el HUD se enciende en cian."""
        self.hud_bridge.notify("listening", "")

    def _on_double_clap(self, event: Any) -> None:
        """Rutina de la doble palmada: música + saludo según la hora."""
        if self.killswitch is not None and self.killswitch.is_tripped():
            return
        logger.info("Doble palmada confirmada: %s", getattr(event, "describe", lambda: "")())
        self.hud_bridge.notify("acting", "Doble palmada")
        if self.media is not None:
            outcome = self.media.double_clap_routine()
            logger.info("Resultado de la rutina: %s", outcome.describe())

    def _on_killswitch(self, event: TripEvent) -> None:
        """Todo lo que hay que parar cuando Pablo activa el interruptor."""
        logger.critical("KILL-SWITCH [%s]: %s", event.reason, event.detail or "sin detalle")
        with self._lock:
            try:
                if self.voice is not None:
                    self.voice.stop_speaking()
                if self.actuator is not None:
                    self.actuator.gate.cancel("kill-switch")
                if self.hud is not None:
                    self.hud.set_border(0.0)
                    self.hud.set_state("killswitch", "Control liberado")
            except Exception as exc:  # pragma: no cover
                logger.error("Error liberando el control: %s", exc)
        print("\n" + "=" * 66)
        print("  KILL-SWITCH ACTIVADO — Control devuelto a Pablo.")
        print(f"  Motivo: {event.reason} {event.detail}")
        print("  Di 'Jarvis, continúa' para que vuelva a obedecerte.")
        print("=" * 66 + "\n")

    # ------------------------------------------------------------------ #
    # EL CEREBRO: DE UNA FRASE A UNA ACCIÓN
    # ------------------------------------------------------------------ #

    def handle_command(self, text: str) -> str | None:
        """Interpreta una orden de Pablo y devuelve la respuesta hablada.

        Orden de comprobaciones (de lo instantáneo a lo costoso):
        comandos de seguridad -> habilidades locales -> planificación con el
        modelo -> conversación. Así lo simple responde en milisegundos y lo
        complejo se reserva para cuando hace falta de verdad.
        """
        original = " ".join(str(text or "").split())
        if not original:
            return None
        logger.info("Orden de Pablo: %s", original)

        # Se quita el saludo inicial («Jarvis, …») para interpretar la orden:
        # las habilidades deben recibir el mandato, no el nombre del asistente.
        clean = self._strip_wake_word(original)
        if not clean:
            return cfg.PHRASES["wake"][0]
        normalized = normalize_text(clean)

        # 0) Seguridad y control.
        if self.killswitch is not None and self.killswitch.is_tripped():
            if any(word in normalized for word in ("continua", "reanuda", "reactivate", "vuelve")):
                self.killswitch.reset()
                if self.actuator is not None:
                    self.actuator.gate.clear()
                return "Control recuperado, Pablo. Le escucho."
            return "El control está en sus manos, Pablo. Diga 'continúa' cuando quiera que vuelva."

        conversation_handlers: list[tuple[Callable[[str], str | None], str]] = [
            ("time", self._handle_time),
            ("date", self._handle_date),
            ("stop", self._handle_stop),
            ("music", self._handle_music),
            ("screen", self._handle_screen),
            ("web", self._handle_web),
            ("build", self._handle_build),
            ("selfprogram", self._handle_self_programming),
            ("app", self._handle_open_app),
            ("automation", self._handle_automation),
            ("goodbye", self._handle_goodbye),
        ]
        for name, handler in conversation_handlers:
            try:
                reply = handler(clean)
            except AutomationInterrupted as exc:
                return f"De acuerdo, Pablo. {exc}"
            except Exception as exc:  # pragma: no cover - ninguna habilidad tumba el cerebro
                logger.exception("La habilidad '%s' falló: %s", name, exc)
                continue
            if reply is not None:
                self.history.append({"momento": time.time(), "orden": clean, "respuesta": reply})
                self.history = self.history[-cfg.CONVERSATION_HISTORY_LIMIT * 2 :]
                return reply

        # 1) Conversación con el modelo principal.
        return self._handle_chat(clean)

    # --- Habilidades locales ----------------------------------------------

    @staticmethod
    def _strip_wake_word(text: str) -> str:
        """Quita el saludo inicial («Jarvis, …») conservando tildes y mayúsculas.

        >>> Jarvis._strip_wake_word("Jarvis, abre el bloc de notas")
        'abre el bloc de notas'
        >>> Jarvis._strip_wake_word("Oye Jarvis: ¿qué hora es?")
        '¿qué hora es?'
        >>> Jarvis._strip_wake_word("Jarvis")
        ''
        """
        stripped = str(text or "").strip()
        normalized, mapping = normalize_text_map(stripped)
        for alias in sorted(cfg.WAKE_WORD_ALIASES, key=len, reverse=True):
            alias_norm = normalize_text(alias)
            if not alias_norm:
                continue
            if normalized == alias_norm:
                return ""
            if normalized.startswith(alias_norm + " "):
                start = len(alias_norm) + 1
                origin = mapping[start] if start < len(mapping) else len(stripped)
                # Recupera los signos de apertura que la normalización borró
                # («¿», «¡»): la orden de Pablo se devuelve tal cual la dijo.
                while origin > 0 and stripped[origin - 1] in " ¿¡":
                    origin -= 1
                return stripped[origin:].lstrip(" ,.:;")
        return stripped

    def _handle_time(self, text: str) -> str | None:
        """¿Qué hora es?"""
        if not any(w in normalize_text(text) for w in ("que hora", "hora es", "dime la hora", "hora tienes")):
            return None
        ahora = time.localtime()
        momento = "de la mañana" if ahora.tm_hour < 12 else ("de la tarde" if ahora.tm_hour < 20 else "de la noche")
        return f"Son las {ahora.tm_hour}:{ahora.tm_min:02d} {momento}, Pablo."

    def _handle_date(self, text: str) -> str | None:
        """¿Qué día es hoy?"""
        if not any(w in normalize_text(text) for w in ("que dia", "fecha", "dia es hoy", "dia estamos")):
            return None
        dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        meses = [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
        ]
        hoy = time.localtime()
        return (
            f"Hoy es {dias[hoy.tm_wday]}, {hoy.tm_mday} de {meses[hoy.tm_mon - 1]} "
            f"de {hoy.tm_year}."
        )

    def _handle_stop(self, text: str) -> str | None:
        """Detener la voz y la automatización en curso."""
        normalized = normalize_text(text)
        if not any(w in normalized for w in ("detente", "parate", "para de", "callate", "silencio", "basta", "cancelalo")):
            return None
        if self.voice is not None:
            self.voice.stop_speaking()
        if self.actuator is not None:
            self.actuator.gate.cancel("orden de voz de Pablo")
        self.hud_bridge.notify("idle", "Detenido")
        return "De acuerdo, Pablo. Me detengo."

    def _handle_music(self, text: str) -> str | None:
        """Poner música (Spotify y, si falla, YouTube)."""
        normalized = normalize_text(text)
        if not any(w in normalized for w in ("pon musica", "poner musica", "pon la cancion", "reproduce",
                                             "pon el tema", "musica de", "pon a ")):
            return None
        if self.media is None:
            return "No tengo acceso al reproductor, Pablo."
        if "musica" in normalized and len(normalized.split()) <= 4:
            outcome = self.media.double_clap_routine(deliver_music_first=True)
            return None if outcome.exito else "No he podido poner la música, Pablo."

        query = normalized
        for prefix in ("pon la cancion ", "pon el tema ", "pon musica de ", "reproduce ", "pon a ", "pon "):
            if query.startswith(prefix):
                query = query[len(prefix):]
                break
        if not query:
            query = cfg.CLAP_TRACK_QUERY
        # 1) Spotify (búsqueda); 2) YouTube en Comet.
        if self.media.find_spotify() and self.media.play_on_spotify(
            SPOTIFY_SEARCH_URI.format(query=urllib.parse.quote(query))
        ):
            return f"Reproduciendo {query} en Spotify."
        ok, method = self.media.open_url(
            f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
        )
        if ok:
            return f"Pongo {query} en YouTube, Pablo."
        return "No he encontrado dónde reproducir eso, Pablo."

    def _handle_screen(self, text: str) -> str | None:
        """Mirar la pantalla, describirla o hacer clic en algo."""
        normalized = normalize_text(text)
        if self.args.sin_vision or not cfg.VISION_ENABLED:
            return None
        if self.perception is None:
            return None

        # Clic en un elemento descrito por Pablo.
        click_markers = ("haz clic", "haz click", "pulsa el boton", "pulsa en", "pincha en")
        if any(marker in normalized for marker in click_markers):
            element = self._extract_element_description(clean=text)
            if not element:
                return "¿En qué quieres que haga clic, Pablo?"
            if self.router is not None and not self.router.client.has_model(cfg.MODEL_VISION):
                return "Necesito el modelo de visión descargado para hacer eso, Pablo."
            plan = {
                "objetivo": f"Hacer clic en {element}",
                "pasos": [{"accion": "clic_elemento", "argumentos": {"descripcion": element}}],
            }
            assert self.executor is not None
            outcome = self.executor.execute(plan, speak=self._speak)
            return None if outcome.exito else f"No he podido hacer clic ahí: {outcome.motivo}"

        # Escribir texto en la ventana activa.
        if normalized.startswith("escribe ") or normalized.startswith("escribir "):
            content = text.split(" ", 1)[1] if " " in text else ""
            if not content:
                return "¿Qué quieres que escriba, Pablo?"
            assert self.actuator is not None
            self.actuator.write(content)
            return "Escrito, Pablo."

        # Describir la pantalla.
        if any(w in normalized for w in ("que ves", "mira mi pantalla", "describe la pantalla",
                                         "que hay en pantalla", "mira la pantalla", "en que estoy trabajando")):
            description = self.perception.describe(
                "Describe en español y en pocas frases qué aplicación está activa y qué se ve."
            )
            return description or "No he podido ver la pantalla, Pablo."
        return None

    @staticmethod
    def _extract_element_description(clean: str) -> str:
        """Rescata la descripción del elemento de una frase con "haz clic en...".

        >>> Jarvis._extract_element_description("Jarvis, haz clic en el botón azul")
        'el botón azul'
        >>> Jarvis._extract_element_description("haz click en Guardar")
        'Guardar'
        """
        normalized, mapping = normalize_text_map(clean)
        for marker in ("haz clic en", "haz click en", "pulsa en", "pulsa el boton", "pincha en"):
            index = normalized.find(marker)
            if index >= 0:
                start = index + len(marker)
                origin = mapping[start] if start < len(mapping) else len(clean)
                fragment = clean[origin:].strip(" ,.:;¡!¿?")
                return fragment[:120]
        return ""

    def _handle_web(self, text: str) -> str | None:
        """Buscar en internet y abrir el resultado en Comet."""
        normalized = normalize_text(text)
        markers = ("busca en internet", "busca en google", "busca en la web", "buscar en internet",
                   "buscame", "busca ", "abre la web", "abre la pagina", "navega a")
        if not any(marker in normalized for marker in markers):
            return None
        query = text
        _, mapping = normalize_text_map(text)
        for marker in ("busca en internet", "busca en google", "busca en la web", "buscar en internet",
                       "buscame", "busca", "abre la web", "abre la pagina", "navega a"):
            index = normalized.find(marker)
            if index >= 0:
                start = index + len(marker)
                origin = mapping[start] if start < len(mapping) else len(text)
                query = text[origin:].strip(" ,.:;¡!¿?") or text
                break
        if self.media is None or not query:
            return "No he podido buscar eso, Pablo."
        ok, method = self.media.open_url(SEARCH_URL.format(query=urllib.parse.quote(query)))
        if ok:
            return f"Buscando {query} en {method}, Pablo."
        return "No he podido abrir el navegador, Pablo."

    def _handle_open_app(self, text: str) -> str | None:
        """Abrir una aplicación instalada."""
        normalized = normalize_text(text)
        markers = ("abre ", "abrir ", "lanza ", "ejecuta ", "pon el ")
        if not any(normalized.startswith(m) for m in markers):
            return None
        if self.media is None:
            return None
        target = normalized
        for marker in ("abre ", "abrir ", "lanza ", "ejecuta ", "pon el "):
            if normalized.startswith(marker):
                target = text[len(marker):].strip(" ,.:;¡!¿?")
                break
        if not target or len(target.split()) > 4:
            return None
        # «abre el bloc de notas» -> «bloc de notas» (los artículos no son parte
        # del nombre de la aplicación y estorban al buscarla).
        for article in ("el ", "la ", "los ", "las ", "un ", "una "):
            if normalize_text(target).startswith(article):
                target = target[len(article):].strip()
                break
        lower = normalize_text(target)
        known = any(lower == alias or lower in alias or alias in lower for alias in cfg.KNOWN_APPS)
        if not known and lower not in ("comet", "spotify", "navegador"):
            return None
        ok, method = self.media.launch_app(target)
        return f"Abriendo {target}, Pablo." if ok else f"No he encontrado la aplicación {target}, Pablo."

    def _handle_automation(self, text: str) -> str | None:
        """Automatización guiada por visión: "en la pantalla, haz esto paso a paso"."""
        normalized = normalize_text(text)
        triggers = ("en la pantalla", "paso a paso", "automatiza", "haz una tarea", "navega por",
                    "rellena el formulario", "abre y escribe", "instala la aplicacion")
        if not any(trigger in normalized for trigger in triggers):
            return None
        if self.args.sin_vision or self.router is None or self.executor is None:
            return "La visión está desactivada, Pablo; no puedo hacer eso."
        if not self.router.client.has_model(cfg.MODEL_VISION):
            return "Me falta el modelo de visión para eso, Pablo. Ejecuta install.bat."

        context = self.perception.describe(
            "Describe la aplicación activa y los elementos de interfaz relevantes."
        ) if self.perception is not None else ""
        plan = self.router.plan(text, screen_context=context)
        outcome = self.executor.execute(plan, speak=self._speak)
        if outcome.exito:
            return f"Hecho, Pablo. {outcome.motivo}"
        return f"No he podido completarlo: {outcome.motivo}"

    def _handle_build(self, text: str) -> str | None:
        """Construir una aplicación o una web completa."""
        if self.app_builder is None or not AppBuilder.looks_like_app_request(text):
            return None
        outcome = self.app_builder.build(text, speak=None)
        if outcome.exito:
            return f"Su aplicación '{outcome.nombre}' está lista y la he abierto, Pablo."
        return f"No he podido construirla: {outcome.motivo}"

    def _handle_self_programming(self, text: str) -> str | None:
        """Añadir una funcionalidad al propio JARVIS (rama aislada + pruebas)."""
        if self.self_programmer is None or not SelfProgrammer.looks_like_self_programming(text):
            return None
        if not self.self_programmer.enabled:
            return "La auto-programación está desactivada, Pablo."
        self._speak("Voy a programarme, Pablo. Dame un momento y ejecuto las pruebas.")
        outcome = self.self_programmer.implement(text, speak=None)
        if outcome.exito:
            return None if outcome.recargado else "Listo, Pablo. Funcionalidad añadida y probada."
        return f"No he aplicado cambios, Pablo. {outcome.motivo}"

    def _handle_goodbye(self, text: str) -> str | None:
        """Cerrar JARVIS por voz."""
        normalized = normalize_text(text)
        if not any(w in normalized for w in ("apagate", "apaga jarvis", "cierra jarvis",
                                             "cerrar jarvis", "adios", "hasta luego",
                                             "hasta pronto", "me voy")):
            return None
        # La despedida la pronuncia el motor de voz al recibir esta respuesta;
        # el apagado se programa con margen para que termine de sonar.
        threading.Timer(2.5, self.request_stop).start()
        return "Hasta luego, Pablo."

    def _handle_chat(self, text: str) -> str | None:
        """Conversación normal con llama3.1:8b (con memoria de la sesión)."""
        if self.router is None:
            return cfg.PHRASES["not_understood"][0]
        if self.client is not None and not self.client.health():
            return "Ahora mismo no tengo acceso a mi modelo de lenguaje, Pablo. ¿Está Ollama abierto?"
        try:
            answer = self.router.chat(text, history=self._chat_history())
        except Exception as exc:
            logger.error("La conversación falló: %s", exc)
            return cfg.PHRASES["error"]
        if not answer:
            return cfg.PHRASES["not_understood"][0]
        self.history.append({"momento": time.time(), "orden": text, "respuesta": answer})
        return answer.strip()

    def _chat_history(self, turns: int = 6) -> list[dict]:
        """Convierte el historial reciente en mensajes para el modelo."""
        messages: list[dict] = []
        for entry in self.history[-turns:]:
            messages.append({"role": "user", "content": entry.get("orden", "")})
            messages.append({"role": "assistant", "content": entry.get("respuesta", "")})
        return messages

    def _speak(self, text: str) -> None:
        """Dice algo en voz alta (si hay motor de voz)."""
        if self.voice is not None:
            self.voice.say(text, blocking=False)
        else:
            logger.info("(sin voz) %s", text)

    # ------------------------------------------------------------------ #
    # BUCLE PRINCIPAL
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Arranca los hilos de fondo (voz y escucha)."""
        assert self.voice is not None
        if self.voice.start():
            logger.info("Oídos activos. Di 'Jarvis' o da dos palmadas.")
        else:
            logger.warning("Sin micrófono: usa el modo texto ('python main.py --texto').")

        if self.args.texto:
            return
        self._listen_thread = threading.Thread(
            target=self.voice.listen_loop, name="jarvis-escucha", daemon=True
        )
        self._listen_thread.start()
        self._keeper_thread = threading.Thread(
            target=self._keep_alive_loop, name="jarvis-vigilante", daemon=True
        )
        self._keeper_thread.start()

    def _keep_alive_loop(self) -> None:
        """Vigila el estado general: kill-switch, HUD, memoria y limpieza."""
        while not self._stop.is_set():
            try:
                if self.killswitch is not None and self.killswitch.is_tripped():
                    self.hud_bridge.notify("killswitch", "Control liberado")
                elif self.voice is not None and self.voice.state == "killswitch":
                    self.hud_bridge.notify("idle", "")
            except Exception as exc:  # pragma: no cover
                logger.debug("Vigilante: %s", exc)
            self._stop.wait(1.0)

    def run(self) -> int:
        """Ejecuta JARVIS hasta que Pablo lo detenga."""
        if not self.boot():
            return 1
        self.start()

        if self.args.texto:
            assert self.voice is not None
            try:
                self.voice.text_loop()
            except KeyboardInterrupt:
                print()
            finally:
                self.shutdown()
            return 0

        self._install_signal_handlers()
        if self.hud is not None:
            self.hud.set_state("sleeping", "")
            try:
                return int(self.hud.run())  # El bucle de Qt ocupa el hilo principal.
            finally:
                self.shutdown()
        try:
            while not self._stop.is_set():
                time.sleep(0.25)
        except KeyboardInterrupt:
            self.shutdown()
        return 0

    def _install_signal_handlers(self) -> None:
        """Ctrl+C en la consola cierra JARVIS con orden."""
        def handler(_signum, _frame):
            logger.info("Interrupción de teclado: cerrando JARVIS...")
            self.request_stop()

        try:
            signal.signal(signal.SIGINT, handler)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError):  # pragma: no cover
            pass

    def request_stop(self) -> None:
        """Pide el cierre desde cualquier hilo (incluido el de Qt)."""
        self._stop.set()
        if self.hud is not None:
            try:
                self.hud.quit()
            except Exception:  # pragma: no cover
                pass

    def shutdown(self) -> None:
        """Cierra todo con orden: voz, HUD, GPU y archivo de PID."""
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        logger.info("Cerrando JARVIS...")
        self._stop.set()

        for name, action in (
            ("voz", lambda: self.voice and self.voice.shutdown()),
            ("palmada", lambda: self.clap and self.clap.stop()),
            ("hud", lambda: self.hud and self.hud.stop()),
            ("modelos", lambda: self.router and self.router.close()),
            ("cliente", lambda: self.client and self.client.close()),
            ("killswitch", lambda: self.killswitch and self.killswitch.shutdown()),
        ):
            try:
                action()
            except Exception as exc:  # pragma: no cover
                logger.debug("Error cerrando %s: %s", name, exc)

        try:
            Path(cfg.PID_FILE).unlink(missing_ok=True)
        except OSError:
            pass
        logger.info("JARVIS detenido. Hasta luego, Pablo.")

    # ------------------------------------------------------------------ #
    # DIAGNÓSTICO
    # ------------------------------------------------------------------ #

    def status(self) -> dict:
        """Informe completo del estado de todos los subsistemas."""
        return {
            "version": cfg.APP_VERSION,
            "en_marcha_desde_s": round(time.time() - self.started_at, 1),
            "seguridad": {
                "integro": verify_integrity()[0],
                "detalle": verify_integrity()[1],
                "killswitch": self.killswitch.status() if self.killswitch else None,
            },
            "modelos": self.router.status() if self.router else {},
            "ollama": self.client.summary() if self.client else {},
            "voz": self.voice.status() if self.voice else {},
            "vision": self.executor.status() if self.executor else {},
            "palmada": self.clap.detector.status() if self.clap else {},
            "constructor": self.app_builder.status() if self.app_builder else {},
            "autoprogramacion": self.self_programmer.status() if self.self_programmer else {},
            "hud": self.hud_bridge.status(),
            "ordenes": len(self.history),
        }


# =============================================================================
# COMANDOS DE MANTENIMIENTO
# =============================================================================

def command_kill_running() -> int:
    """Detiene la instancia de JARVIS que esté en marcha (opción `--matar`)."""
    pid_file = Path(cfg.PID_FILE)
    if not pid_file.exists():
        print("No hay ningún JARVIS en marcha (no existe el archivo de PID).")
        return 0
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        print("El archivo de PID está corrupto; se elimina.")
        pid_file.unlink(missing_ok=True)
        return 0

    if pid <= 0 or pid == os.getpid():
        pid_file.unlink(missing_ok=True)
        return 0

    killed = False
    try:
        import psutil  # type: ignore

        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=5)
        killed = True
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
            killed = True
        except Exception as exc:
            print(f"No he podido detener el proceso {pid}: {exc}")
    if killed:
        print(f"JARVIS (PID {pid}) detenido. Control devuelto a Pablo.")
    pid_file.unlink(missing_ok=True)
    return 0 if killed else 1


def command_diagnostic(jarvis: Jarvis) -> int:
    """Informe detallado de todo el sistema (opción `--diagnostico`)."""
    cfg.setup_logging()
    print("\n" + "=" * 74)
    print(f"  DIAGNÓSTICO DE {cfg.APP_NAME} v{cfg.APP_VERSION}")
    print("=" * 74)

    ok, message = verify_integrity()
    print(f"\n[SEGURIDAD]\n  Integridad del kill-switch: {'OK' if ok else 'FALLO'}\n  {message}")
    print(f"  Atajo de emergencia: {cfg.KILLSWITCH_HOTKEY.upper()} (alternativo {cfg.KILLSWITCH_HOTKEY_ALT.upper()})")
    print(f"  Sacudida del ratón: {cfg.KILLSWITCH_MOUSE_TRAVEL_PX} px en {cfg.KILLSWITCH_MOUSE_WINDOW_S} s")

    print("\n[HARDWARE Y MODELOS]")
    tier = cfg.detect_hardware_tier()
    print(f"  Perfil detectado: {tier.upper()} — {cfg.HARDWARE_TIERS[tier]['nota']}")
    print(f"  Modelo de conversación: {cfg.MODEL_CHAT}")
    print(f"  Modelo de visión:       {cfg.MODEL_VISION} (efímero, se descarga al terminar)")
    print(f"  Modelo de código:       {cfg.MODEL_CODER}")
    if jarvis.client is not None:
        resumen = jarvis.client.summary()
        print(f"  Ollama: {resumen['servicio']} ({resumen.get('version') or 'versión desconocida'})")
        print(f"  Modelos descargados: {resumen['modelos_descargados']}")
        for entry in resumen["modelos_cargados"]:
            print(f"    · cargado: {entry['nombre']} ({entry['vram_gb']:.2f} GB de VRAM)")
        faltan = jarvis.client.missing_models()
        if faltan:
            print(f"  Faltan por descargar: {', '.join(faltan)}")
            print("    Se descargarán solos en el primer uso (o ejecuta install.bat).")

    print("\n[AUDIO Y VOZ]")
    if jarvis.voice is not None:
        voz = jarvis.voice.status()
        print(f"  Micrófono: {'abierto' if voz['microfono']['activo'] else 'no disponible'}")
        print(f"  Motor de voz: {voz['sintesis']['motor'] or 'ninguno'} (voz {voz['sintesis']['voz']})")
        print(f"  Transcripción: {voz['transcripcion']['modelo']} "
              f"({voz['transcripcion']['dispositivo']}, "
              f"{'cargado' if voz['transcripcion']['disponible'] else 'sin cargar'})")
        print(f"  Palabra clave: {voz['palabra_clave']['modelo']} "
              f"({'lista' if voz['palabra_clave']['disponible'] else voz['palabra_clave']['motivo'] or 'sin cargar'})")
        print(f"  Doble palmada: {'activa' if jarvis.clap else 'desactivada'} "
              f"(separación {cfg.CLAP_MIN_GAP_MS}-{cfg.CLAP_MAX_GAP_MS} ms)")

    print("\n[INTERFAZ]")
    print(f"  GHOST HUD: {'activo' if jarvis.hud is not None else 'no disponible (PyQt6 o --sin-hud)'}")
    if jarvis.hud is not None:
        print(f"  Click-through: {'sí' if jarvis.hud.click_through_enabled() else 'no'}")

    print("\n[SUBSISTEMAS]")
    for modulo, descripcion in __import__("core").SUBSYSTEMS.items():
        print(f"  · {modulo:<20} {descripcion}")

    print("\n[CARPETAS]")
    for directory in cfg.ALL_DIRECTORIES:
        print(f"  · {directory}")

    print("\n" + "=" * 74)
    print("  Para arrancar JARVIS: python main.py")
    print("  Para probar sin micrófono: python main.py --texto")
    print("=" * 74 + "\n")
    return 0


def command_voice_test(jarvis: Jarvis) -> int:
    """Prueba rápida de oídos y boca (opción `--prueba-voz`)."""
    assert jarvis.voice is not None
    print("\n=== PRUEBA DE VOZ ===")
    print("1) Comprobando la síntesis (JARVIS hablará)...")
    ok = jarvis.voice.speak_test()
    print(f"   Síntesis: {'correcta' if ok else 'no disponible'}")

    print("2) Abriendo el micrófono...")
    jarvis.voice.start()
    print("   Diga una frase cualquiera en los próximos 8 segundos...")
    text = jarvis.voice.listen_once(timeout=8.0)
    print(f"   He entendido: {text or '(nada)'}")
    jarvis.voice.shutdown()
    print("=== FIN DE LA PRUEBA ===\n")
    return 0 if ok or text else 1


def command_clap_test(jarvis: Jarvis, seconds: float = 60.0) -> int:
    """Escucha dobles palmadas sin arrancar todo JARVIS (opción `--prueba-palmada`)."""
    print("\n=== PRUEBA DE DOBLE PALPADA ===")
    print(f"Escuchando {int(seconds)} segundos. Da DOS palmadas seguidas...")

    def announce(event: Any) -> None:
        print(f"  ¡Detectada! {getattr(event, 'describe', lambda: '')()}")
        print(f"  Saludo que usaría: {cfg.greeting_for_hour()}")
        print(f"  Y pondría: {cfg.CLAP_TRACK_QUERY}")

    trigger = AcousticTrigger(action=announce, detector=ClapDetector(), notifier=None)
    if not trigger.start():
        print("No he podido abrir el micrófono.")
        return 1
    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        pass
    finally:
        trigger.stop()
    print("Estadísticas:", json.dumps(trigger.detector.status(), ensure_ascii=False))
    print("=== FIN DE LA PRUEBA ===\n")
    return 0


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de JARVIS."""
    args = build_parser().parse_args(argv)

    if args.version:
        print(f"{cfg.APP_NAME} v{cfg.APP_VERSION} — {cfg.APP_TITLE}")
        return 0
    if args.verificar_seguridad:
        ok, message = verify_integrity()
        print(("✅ " if ok else "❌ ") + message)
        return 0 if ok else 1
    if args.sellar_seguridad:
        digest = write_hash_file()
        print(f"✅ Sello de integridad actualizado: {digest}")
        return 0
    if args.matar:
        return command_kill_running()
    if args.prueba_palmada:
        cfg.ensure_directories()
        cfg.setup_logging()
        return command_clap_test(Jarvis(args))
    if args.diagnostico:
        jarvis = Jarvis(args)
        jarvis._security_gate()
        jarvis._bootstrap_models()
        jarvis._bootstrap_senses()
        return command_diagnostic(jarvis)
    if args.prueba_voz:
        cfg.ensure_directories()
        cfg.setup_logging()
        jarvis = Jarvis(args)
        jarvis._security_gate()
        jarvis._bootstrap_models()
        jarvis._bootstrap_senses()
        return command_voice_test(jarvis)

    jarvis = Jarvis(args)
    try:
        return jarvis.run()
    except KillSwitchIntegrityError as exc:
        print(f"\n❌ {exc}\n")
        return 2
    except KeyboardInterrupt:
        jarvis.shutdown()
        return 0


if __name__ == "__main__":
    sys.exit(main())
