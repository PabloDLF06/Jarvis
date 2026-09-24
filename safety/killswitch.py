"""
safety/killswitch.py — INTERRUPTOR DE EMERGENCIA INMUTABLE DE JARVIS.
================================================================================
                    ██  ARCHIVO PROTEGIDO — NO EDITAR  ██

Este archivo es el botón rojo de JARVIS. Su única misión es devolver el control
del ordenador a Pablo de forma INSTANTÁNEA, pase lo que pase.

DOS DISPARADORES
----------------
1. **Atajo global de teclado:** `Ctrl + Shift + Space` (y `Ctrl + Alt + K`).
   Funciona aunque la ventana activa no sea JARVIS.
2. **Sacudida violenta del ratón:** más de 1500 píxeles de recorrido en menos
   de 0,3 segundos (unas 4 muestras). Es el gesto de pánico: agarrar el ratón
   y zarandearlo.

QUÉ HACE AL ACTIVARSE
---------------------
* Detiene la automatización de ratón y teclado (el actuador consulta
  `is_tripped()` antes de cada paso).
* Cancela las llamadas a modelos en curso (el enrutador verifica el estado
  entre fragmentos).
* Vacía la cola de ejecución y detiene las subrutinas activas (los módulos se
  suscriben con `register()`).
* Apaga el marco de neón del HUD (estado KILLSWITCH).
* Libera el control al usuario: JARVIS queda como observador hasta que Pablo
  diga "Jarvis, continúa" o reinicie.

INTEGRIDAD
----------
El hash SHA-256 de este archivo vive en `safety/killswitch.sha256`. `main.py`
lo verifica ANTES de arrancar. El motor de auto-programación tiene prohibido
(y además no puede, por el guardián de comandos) modificar `safety/`.
Si necesitas cambiar este archivo a propósito, edítalo, ejecuta
`python main.py --sellar-seguridad` y el hash se recalcula.
================================================================================
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import stat
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

# --- Bootstrap de rutas: permite usar este archivo de forma aislada ---------
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import config as cfg  # noqa: E402  (después del bootstrap, a propósito)

logger = logging.getLogger("jarvis.safety.killswitch")

TRIP_HOTKEY = "hotkey"
TRIP_MOUSE = "sacudida_raton"
TRIP_MANUAL = "manual"
TRIP_API = "api"
TRIP_BEHAVIORAL = "anomalia"


# =============================================================================
# INTEGRIDAD SHA-256
# =============================================================================

class KillSwitchIntegrityError(RuntimeError):
    """El archivo del kill-switch no coincide con su hash sellado."""


def compute_file_hash(path: str | os.PathLike[str]) -> str:
    """Devuelve el SHA-256 (hexadecimal) del archivo indicado.

    Lee en bloques de 1 MiB para no cargar archivos grandes en memoria.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_hash_file(hash_path: str | os.PathLike[str] = cfg.KILLSWITCH_HASH_FILE) -> str | None:
    """Lee el hash sellado. Acepta comentarios con `#` y el formato `sha256sum`."""
    try:
        with open(hash_path, "r", encoding="utf-8") as handle:
            for line in handle:
                clean = line.strip()
                if not clean or clean.startswith("#"):
                    continue
                token = clean.split()[0] if clean.split() else ""
                if len(token) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in token):
                    return token.lower()
    except (OSError, IndexError):
        return None
    return None


def write_hash_file(
    file_path: str | os.PathLike[str] = cfg.KILLSWITCH_FILE,
    hash_path: str | os.PathLike[str] = cfg.KILLSWITCH_HASH_FILE,
) -> str:
    """Sella el hash actual del kill-switch. Devuelve el hash escrito.

    Solo debe llamarse de forma deliberada (opción `--sellar-seguridad`).
    """
    digest = compute_file_hash(file_path)
    content = (
        "# ============================================================\n"
        "#  JARVIS — Sello de integridad del kill-switch\n"
        "#  Si este hash no coincide con safety/killswitch.py, JARVIS\n"
        "#  se negará a arrancar. Regenerar: python main.py --sellar-seguridad\n"
        "# ============================================================\n"
        f"{digest}  killswitch.py\n"
    )
    hash_path = Path(hash_path)
    previous_mode = None
    if hash_path.exists():
        previous_mode = hash_path.stat().st_mode
        try:
            os.chmod(hash_path, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            previous_mode = None
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hash_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    if previous_mode is not None:
        try:
            os.chmod(hash_path, previous_mode)
        except OSError:
            pass
    logger.info("Sello de integridad actualizado: %s", digest[:16] + "...")
    return digest


def verify_integrity(
    file_path: str | os.PathLike[str] = cfg.KILLSWITCH_FILE,
    hash_path: str | os.PathLike[str] = cfg.KILLSWITCH_HASH_FILE,
) -> tuple[bool, str]:
    """Comprueba que el kill-switch no ha sido alterado.

    Returns
    -------
    (ok, mensaje)
        ``ok`` es True solo si el hash calculado coincide con el sellado.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return False, f"No existe el archivo del kill-switch: {file_path}"
    expected = read_hash_file(hash_path)
    if not expected:
        return False, (
            f"No hay sello de integridad válido en {hash_path}. "
            "Ejecuta: python main.py --sellar-seguridad"
        )
    actual = compute_file_hash(file_path)
    if actual != expected:
        return False, (
            "¡ALERTA DE INTEGRIDAD! El kill-switch ha cambiado.\n"
            f"  Esperado: {expected}\n"
            f"  Actual:   {actual}\n"
            "  Si fuiste tú, Pablo: python main.py --sellar-seguridad"
        )
    return True, f"Kill-switch íntegro ({actual[:16]}...)"


def integrity_gate(strict: bool | None = None) -> bool:
    """Puerta de arranque: verifica la integridad y aborta si el modo es estricto.

    Lanza `KillSwitchIntegrityError` cuando ``strict`` es True y la verificación
    falla. Si ``strict`` es False, se limita a registrar una advertencia.
    """
    strict = cfg.KILLSWITCH_INTEGRITY_STRICT if strict is None else strict
    ok, message = verify_integrity()
    if ok:
        logger.info(message)
        return True
    logger.critical(message)
    if strict:
        raise KillSwitchIntegrityError(message)
    logger.warning("Continuando sin verificación estricta (modo no estricto).")
    return False


# =============================================================================
# PROTECCIÓN DE ARCHIVOS (SOLO LECTURA)
# =============================================================================

def _protected_paths() -> list[Path]:
    """Lista absoluta de los archivos protegidos que existan en disco."""
    paths: list[Path] = []
    for relative in cfg.PROTECTED_PATHS:
        candidate = (_ROOT_DIR / relative).resolve()
        if candidate.exists():
            paths.append(candidate)
    return paths


def set_readonly(path: str | os.PathLike[str], readonly: bool = True) -> bool:
    """Marca (o desmarca) un archivo como solo lectura. Devuelve si lo consiguió."""
    path = Path(path)
    try:
        current = path.stat().st_mode
        if readonly:
            os.chmod(path, current & ~stat.S_IWRITE & ~stat.S_IWGRP & ~stat.S_IWOTH)
        else:
            os.chmod(path, current | stat.S_IWRITE | stat.S_IREAD)
    except OSError as exc:
        logger.debug("No se pudo cambiar el atributo de %s: %s", path, exc)
        return False

    if sys.platform == "win32" and readonly:
        try:
            ctypes.windll.kernel32.SetFileAttributesW(str(path), 0x01)  # FILE_ATTRIBUTE_READONLY
        except Exception:  # pragma: no cover - API de Windows
            pass
    return True


def harden_files(paths: Iterable[str | os.PathLike[str]] | None = None) -> list[str]:
    """Bloquea la escritura sobre los archivos de seguridad de JARVIS.

    Es una defensa en profundidad: aunque el código ya está vetado por el
    guardián de comandos, el sistema operativo también se niega a modificarlos.
    """
    targets = [Path(p) for p in paths] if paths is not None else _protected_paths()
    hardened: list[str] = []
    for target in targets:
        if set_readonly(target, True):
            hardened.append(str(target))
    if hardened:
        logger.info("Archivos de seguridad protegidos (solo lectura): %d", len(hardened))
    return hardened


def relax_files(paths: Iterable[str | os.PathLike[str]] | None = None) -> list[str]:
    """Quita el atributo de solo lectura (para poder editar a propósito)."""
    targets = [Path(p) for p in paths] if paths is not None else _protected_paths()
    relaxed: list[str] = []
    for target in targets:
        if set_readonly(target, False):
            relaxed.append(str(target))
    if relaxed:
        logger.warning("Archivos de seguridad liberados para edición: %d", len(relaxed))
    return relaxed


# =============================================================================
# SONIDO NATIVO Y POSICIÓN DEL CURSOR
# =============================================================================

def _beep(frequency: int = 1200, duration_ms: int = 260) -> None:
    """Pitido nativo (winsound en Windows). Silencioso y sin dependencias."""
    if not cfg.NATIVE_SOUND_ACK:
        return
    try:  # pragma: no cover - depende del sistema
        if sys.platform == "win32":
            import winsound

            winsound.Beep(frequency, duration_ms)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass


def _cursor_position() -> tuple[int, int]:
    """Posición del cursor sin cargar dependencias pesadas (ctypes en Windows)."""
    if sys.platform == "win32":
        try:

            class _POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            pt = _POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                return int(pt.x), int(pt.y)
        except Exception:  # pragma: no cover - depende de la API de Windows
            pass
    try:
        import pyautogui  # type: ignore

        x, y = pyautogui.position()
        return int(x), int(y)
    except Exception:
        return (-1, -1)


# =============================================================================
# EL KILL-SWITCH
# =============================================================================

@dataclass(order=True)
class TripEvent:
    """Registro de una activación del interruptor de emergencia."""

    timestamp: float = field(compare=False)
    reason: str = field(compare=False)
    detail: str = field(compare=False, default="")

    def as_dict(self) -> dict:
        return {"momento": self.timestamp, "motivo": self.reason, "detalle": self.detail}

    def describe(self) -> str:
        return f"{self.reason}: {self.detail}" if self.detail else self.reason


class KillSwitch:
    """Interruptor de emergencia con doble disparador (teclado y ratón).

    Es intencionadamente **sin dependencias obligatorias**: si la librería de
    atajos globales no está disponible, los disparadores que sí puedan
    registrarse se activan y el resto se documenta en el registro.
    """

    def __init__(
        self,
        on_trip: Callable[[TripEvent], None] | None = None,
        watch_mouse: bool | None = None,
        register_hotkeys: bool = True,
        armed: bool = True,
    ) -> None:
        self._trip_event = threading.Event()
        self._lock = threading.RLock()
        self._trips: list[TripEvent] = []
        self._callbacks: list[tuple[str, Callable[[TripEvent], None]]] = []
        self._threads: list[threading.Thread] = []
        self._stop_flag = threading.Event()
        self._hotkey_handles: list[tuple[object, str]] = []
        self._ignore_mouse_until = 0.0
        self._armed = armed
        self._last_position: tuple[int, int] | None = None
        self._samples: deque[tuple[float, int, int]] = deque(maxlen=256)
        self._watch_mouse = cfg.KILLSWITCH_MOUSE_TRIGGER_ENABLED if watch_mouse is None else watch_mouse

        if on_trip is not None:
            self.register("callback_inicial", on_trip)

        if register_hotkeys:
            self._register_hotkeys()
        if self._watch_mouse:
            self._start_mouse_watchdog()
        logger.info(
            "Kill-switch armado (atajo=%s, sacudida=%s px en %.1f s).",
            cfg.KILLSWITCH_HOTKEY,
            cfg.KILLSWITCH_MOUSE_TRAVEL_PX,
            cfg.KILLSWITCH_MOUSE_WINDOW_S,
        )

    # --- Estado ------------------------------------------------------------

    @property
    def tripped(self) -> bool:
        """¿Está el interruptor activado (control devuelto a Pablo)?"""
        return self._trip_event.is_set()

    @property
    def armed(self) -> bool:
        """¿Está JARVIS autorizado a controlar el equipo?"""
        return self._armed and not self.tripped

    @property
    def trip_count(self) -> int:
        return len(self._trips)

    @property
    def trips(self) -> tuple[TripEvent, ...]:
        return tuple(self._trips)

    def is_tripped(self) -> bool:
        """Alias de `tripped` (lo consultan el actuador y el enrutador)."""
        return self._trip_event.is_set()

    def status(self) -> dict:
        """Resumen del estado para el HUD y el diagnóstico."""
        return {
            "armado": self._armed,
            "activado": self.tripped,
            "activaciones": len(self._trips),
            "ultimo_motivo": self._trips[-1].describe() if self._trips else "",
            "vigilancia_raton": self._watch_mouse,
            "atajo": cfg.KILLSWITCH_HOTKEY,
        }

    # --- Suscripción de módulos -------------------------------------------

    def register(self, name: str, callback: Callable[[TripEvent], None]) -> None:
        """Suscribe una función que se ejecutará al activarse el interruptor.

        Se usa para que el motor de voz corte el TTS, el actuador suelte el
        ratón, el constructor de apps cancele su cola, etc. Las funciones se
        llaman SIEMPRE, en orden de suscripción y con protección frente a
        excepciones: un módulo roto no puede impedir que los demás reaccionen.
        """
        with self._lock:
            self._callbacks.append((name, callback))
        logger.debug("Kill-switch: suscrito '%s'.", name)

    def unregister(self, name: str) -> None:
        """Elimina las suscripciones con ese nombre."""
        with self._lock:
            self._callbacks = [(n, cb) for n, cb in self._callbacks if n != name]

    def clear_subscriptions(self) -> None:
        with self._lock:
            self._callbacks.clear()

    # --- Disparadores ------------------------------------------------------

    def trip(self, reason: str = TRIP_MANUAL, detail: str = "") -> TripEvent:
        """ACTIVA EL INTERRUPTOR. Idempotente y seguro desde cualquier hilo.

        Devuelve el evento registrado (aunque ya estuviera activado).
        """
        event = TripEvent(timestamp=time.time(), reason=reason, detail=detail)
        first_time = not self._trip_event.is_set()
        with self._lock:
            self._armed = False
            self._trip_event.set()
            self._trips.append(event)
            if len(self._trips) > 200:
                del self._trips[:-200]
            callbacks = list(self._callbacks)

        if first_time:
            logger.critical("KILL-SWITCH ACTIVADO [%s] %s", reason, detail)
        else:
            logger.warning("Kill-switch ya estaba activado (nueva señal: %s).", reason)

        if cfg.KILLSWITCH_TRIP_BEEP:
            threading.Thread(target=_beep, args=(1400, 320), daemon=True).start()

        for name, callback in callbacks:
            try:
                callback(event)
            except Exception as exc:  # Nunca dejamos que un módulo roto bloquee el rescate
                logger.error("Fallo al notificar al módulo '%s': %s", name, exc)

        if len(self._trips) >= cfg.KILLSWITCH_MAX_TRIPS_BEFORE_LOCK:
            logger.critical(
                "Se han alcanzado %d activaciones: se recomienda reiniciar JARVIS por completo.",
                len(self._trips),
            )
        return event

    def reset(self, rearm: bool = True) -> None:
        """Rearma el interruptor (lo llama Pablo explícitamente: "Jarvis, continúa")."""
        with self._lock:
            self._trip_event.clear()
            self._armed = rearm
            self._samples.clear()
            self._last_position = None
            self._ignore_mouse_until = time.monotonic() + cfg.KILLSWITCH_GRACE_PERIOD_S
        logger.info("Kill-switch rearmado. JARVIS recupera el control por orden de Pablo.")

    def arm(self) -> None:
        self._armed = True
        logger.info("Kill-switch armado.")

    def disarm(self) -> None:
        """Desarma los disparadores sin activar el interruptor (mantenimiento)."""
        self._armed = False
        logger.warning("Kill-switch desarmado temporalmente.")

    def raise_if_tripped(self) -> None:
        """Lanza `KillSwitchTrippedError` si el interruptor está activado.

        La usan el actuador y el enrutador de modelos para abortar en seco
        antes de cada paso de automatización y entre fragmentos de inferencia.
        """
        if self.is_tripped():
            raise KillSwitchTrippedError("Operación cancelada por el kill-switch.")

    def wait_if_tripped(self, timeout: float | None = None) -> bool:
        """Bloquea hasta que se rearme. Devuelve True si sigue activado.

        El actuador la consulta entre pasos: la ejecución se congela en el
        instante, sin perder el plan, y se reanuda cuando Pablo lo pide.
        """
        if not self.is_tripped():
            return False
        return self._trip_event.wait(timeout=timeout)

    def notify_activity(self, duration_s: float | None = None) -> None:
        """Avisa de que JARVIS va a mover el ratón: ignora esas sacudidas.

        Sin esto, el propio JARVIS podría disparar su kill-switch al arrastrar
        el cursor rápidamente por culpa de una animación.
        """
        window = cfg.KILLSWITCH_GRACE_PERIOD_S if duration_s is None else duration_s
        self._ignore_mouse_until = time.monotonic() + max(0.2, float(window))

    # --- Atajos globales ---------------------------------------------------

    def _register_hotkeys(self) -> None:
        """Registra `Ctrl+Shift+Space` (y el atajo alternativo) a nivel global."""
        registered = False
        try:
            import keyboard  # type: ignore

            for hotkey in {cfg.KILLSWITCH_HOTKEY, cfg.KILLSWITCH_HOTKEY_ALT}:
                handle = keyboard.add_hotkey(hotkey, self._on_hotkey, suppress=False, trigger_on_release=False)
                self._hotkey_handles.append((keyboard, str(handle)))
                logger.info("Atajo de emergencia registrado: %s", hotkey.upper())
            registered = True
        except ImportError:
            logger.debug("La librería 'keyboard' no está instalada.")
        except Exception as exc:  # pragma: no cover - permisos del sistema
            logger.warning("No se pudo registrar el atajo con 'keyboard': %s", exc)

        if registered:
            return

        try:
            from pynput import keyboard as pynput_keyboard  # type: ignore

            # Alternativa nativa de pynput para el atajo global.
            hotkeys = pynput_keyboard.GlobalHotKeys(
                {"<ctrl>+<shift>+<space>": self._on_hotkey}
            )
            hotkeys.daemon = True
            hotkeys.start()
            self._hotkey_handles.append((hotkeys, "pynput"))
            logger.info(
                "Atajo de emergencia registrado con pynput (%s).", cfg.KILLSWITCH_HOTKEY.upper()
            )
        except Exception as exc:
            logger.warning(
                "No hay atajos globales disponibles (%s). El kill-switch sigue activo "
                "mediante la sacudida del ratón y `main.py --kill`.",
                exc,
            )

    def _on_hotkey(self, *_args, **_kwargs) -> None:
        """Manejador del atajo: dispara el interruptor inmediatamente."""
        self.trip(TRIP_HOTKEY, cfg.KILLSWITCH_HOTKEY.upper())

    # --- Vigilante del ratón ----------------------------------------------

    def _start_mouse_watchdog(self) -> None:
        thread = threading.Thread(
            target=self._mouse_watchdog_loop,
            name="jarvis-killswitch-raton",
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)

    def _mouse_watchdog_loop(self) -> None:
        """Muestrea el cursor y detecta sacudidas violentas de pánico."""
        interval = max(0.005, float(cfg.KILLSWITCH_MOUSE_POLL_S))
        window = max(0.05, float(cfg.KILLSWITCH_MOUSE_WINDOW_S))
        travel_limit = float(cfg.KILLSWITCH_MOUSE_TRAVEL_PX)
        min_samples = int(cfg.KILLSWITCH_MOUSE_MIN_SAMPLES)

        while not self._stop_flag.is_set():
            try:
                now = time.monotonic()
                x, y = _cursor_position()
                if x < 0 and y < 0:
                    time.sleep(0.5)
                    continue

                if self._ignore_mouse_until > now or not self._armed or self.tripped:
                    self._samples.clear()
                    self._last_position = (x, y)
                    time.sleep(interval)
                    continue

                previous = self._last_position
                self._last_position = (x, y)
                if previous is None:
                    time.sleep(interval)
                    continue

                step = abs(x - previous[0]) + abs(y - previous[1])
                self._samples.append((now, x, y))

                # Limpia muestras antiguas (ventana deslizante).
                while self._samples and now - self._samples[0][0] > window:
                    self._samples.popleft()

                if step <= 0 or len(self._samples) < min_samples:
                    time.sleep(interval)
                    continue

                travel = 0.0
                points = list(self._samples)
                for (_, px, py), (_, cx, cy) in zip(points, points[1:]):
                    travel += abs(cx - px) + abs(cy - py)

                if travel >= travel_limit:
                    self.trip(
                        TRIP_MOUSE,
                        f"{travel:.0f} px en {window:.2f} s (umbral {travel_limit:.0f} px)",
                    )
                    self._samples.clear()

                time.sleep(interval)
            except Exception as exc:  # pragma: no cover - el vigilante nunca debe morir
                logger.debug("Vigilante del ratón: %s", exc)
                time.sleep(0.5)

    # --- Cierre ------------------------------------------------------------

    def shutdown(self) -> None:
        """Desmonta atajos y vigilantes. Se llama al cerrar JARVIS."""
        self._stop_flag.set()
        for handle, kind in self._hotkey_handles:
            try:
                if kind == "pynput":
                    handle.stop()
                else:
                    handle.remove_hotkey(handle)  # type: ignore[attr-defined]
            except Exception:
                pass
        self._hotkey_handles.clear()
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()
        logger.info("Kill-switch desmontado.")


class KillSwitchTrippedError(RuntimeError):
    """Se lanza cuando una operación intenta continuar con el interruptor activo."""


# =============================================================================
# INSTANCIA GLOBAL PEREZOSA
# =============================================================================

_INSTANCE: KillSwitch | None = None
_INSTANCE_LOCK = threading.Lock()


def get_killswitch(
    on_trip: Callable[[TripEvent], None] | None = None,
    register_hotkeys: bool = True,
    watch_mouse: bool | None = None,
) -> KillSwitch:
    """Devuelve (creando si hace falta) la instancia global del kill-switch.

    Cualquier módulo de JARVIS puede llamar a esta función y recibir siempre el
    mismo interruptor: eso garantiza que exista un único botón rojo.
    """
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = KillSwitch(
                on_trip=on_trip,
                watch_mouse=watch_mouse,
                register_hotkeys=register_hotkeys,
            )
        elif on_trip is not None:
            _INSTANCE.register("callback_externo", on_trip)
    return _INSTANCE


def reset_killswitch_singleton() -> None:
    """Destruye la instancia global (solo para pruebas automatizadas)."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None:
            _INSTANCE.shutdown()
        _INSTANCE = None


def is_tripped() -> bool:
    """Atajo de módulo: ¿está activado el interruptor de emergencia?"""
    return _INSTANCE is not None and _INSTANCE.is_tripped()


# =============================================================================
# INTERFAZ DE LÍNEA DE COMANDOS (diagnóstico y mantenimiento)
# =============================================================================

def _cli(argv: list[str]) -> int:
    """Pequeña CLI para verificar, sellar, proteger o disparar el kill-switch."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="killswitch",
        description="Herramienta de mantenimiento del kill-switch de JARVIS.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verificar", action="store_true", help="Comprueba el hash de integridad.")
    group.add_argument("--sellar", action="store_true", help="Recalcula y guarda el hash.")
    group.add_argument("--proteger", action="store_true", help="Marca los archivos como solo lectura.")
    group.add_argument("--desproteger", action="store_true", help="Quita el solo lectura (para editar).")
    group.add_argument("--estado", action="store_true", help="Muestra el estado actual.")
    group.add_argument("--disparar", action="store_true", help="Prueba el interruptor (sin automatización).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.verificar:
        ok, message = verify_integrity()
        print(("✅ " if ok else "❌ ") + message)
        return 0 if ok else 1
    if args.sellar:
        digest = write_hash_file()
        print(f"✅ Sello actualizado: {digest}")
        return 0
    if args.proteger:
        files = harden_files()
        print(f"✅ Protegidos {len(files)} archivos (solo lectura).")
        return 0
    if args.desproteger:
        files = relax_files()
        print(f"⚠️  Liberados {len(files)} archivos para edición. Vuelve a sellar al terminar.")
        return 0
    if args.estado:
        ok, message = verify_integrity()
        print(f"Integridad: {'OK' if ok else 'FALLIDA'} — {message}")
        print(f"Atajo: {cfg.KILLSWITCH_HOTKEY.upper()}")
        print(f"Sacudida: {cfg.KILLSWITCH_MOUSE_TRAVEL_PX} px en {cfg.KILLSWITCH_MOUSE_WINDOW_S} s")
        print(f"Modo estricto: {cfg.KILLSWITCH_INTEGRITY_STRICT}")
        return 0
    if args.disparar:
        switch = KillSwitch(register_hotkeys=False, watch_mouse=False)
        switch.register("consola", lambda event: print(f"   → {event.describe()}"))
        switch.trip(TRIP_MANUAL, "prueba desde la línea de comandos")
        print("✅ El kill-switch responde correctamente.")
        switch.shutdown()
        return 0
    return 2


if __name__ == "__main__":  # pragma: no cover - ejecución manual
    sys.exit(_cli(sys.argv[1:]))
