"""
core/self_programmer.py — JARVIS se programa a sí mismo (con cinturones y tirantes).
================================================================================
Pablo dice: *"Jarvis, añade una funcionalidad para que me avises de las
reuniones"* y JARVIS se escribe su propio código. El proceso es deliberadamente
prudente, porque aquí un error no se queda en una frase rara: se queda en el
programa.

PROTOCOLO (SIEMPRE EL MISMO)
---------------------------
    1. ¿El repositorio está limpio?  Si hay cambios sin guardar, NO se toca nada.
    2. Rama aislada         -> feature/staging-<marca de tiempo>
    3. Código con           -> qwen2.5-coder:7b (JSON con archivos completos)
    4. Auditoría de código  -> guardián de comandos (prohibido tocar safety/)
    5. Escritura con        -> rutas confinadas al proyecto, nunca fuera
    6. Pruebas              -> pytest tests/ en un subproceso con tiempo límite
    7. Veredicto:
         * TODO PASA  -> fusión a main (--no-ff), borrado de la rama y recarga
                         en caliente con os.execv: JARVIS renace con la mejora.
         * ALGO FALLA -> git checkout main + git reset --hard + borrado de la
                         rama y de los archivos generados. JARVIS sigue
                         funcionando exactamente como antes (rollback total).

LO QUE ESTE MÓDULO JAMÁS HARÁ
-----------------------------
* Tocar `safety/` (kill-switch, guardián de comandos o sus hashes).
* Escribir fuera de la carpeta del proyecto.
* Fusionar si las pruebas no pasan al 100%.
* Ejecutar un comando que el guardián no haya autorizado.
================================================================================
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

import config as cfg  # noqa: E402
from core.utils import extract_json, normalize_text, timestamp_slug, truncate  # noqa: E402
from safety.command_guard import (  # noqa: E402
    CommandGuardError,
    UnsafePathError,
    audit_generated_code,
    is_protected_path,
    run_guarded,
    sanitize_relative_file_path,
)

logger = logging.getLogger("jarvis.autoprogramador")


# =============================================================================
# RESULTADOS
# =============================================================================

@dataclass
class TestReport:
    """Resultado de ejecutar la batería de pruebas."""

    ok: bool = False
    passed: int = 0
    failed: int = 0
    errors: int = 0
    duration_s: float = 0.0
    output: str = ""

    def describe(self) -> str:
        return f"{self.passed} correctas, {self.failed} fallidas, {self.errors} errores en {self.duration_s:.1f} s"


@dataclass
class CodeProposal:
    """Lo que el modelo propone escribir."""

    resumen: str = ""
    archivos: dict[str, str] = field(default_factory=dict)
    pruebas: dict[str, str] = field(default_factory=dict)
    notas: str = ""

    @property
    def total_files(self) -> int:
        return len(self.archivos) + len(self.pruebas)


@dataclass
class CodingOutcome:
    """Resultado completo de un ciclo de auto-programación."""

    exito: bool = False
    motivo: str = ""
    rama: str = ""
    archivos_escritos: list[str] = field(default_factory=list)
    pruebas: TestReport = field(default_factory=TestReport)
    fusionado: bool = False
    recargado: bool = False
    revertido: bool = False
    resumen: str = ""

    def describe(self) -> str:
        estado = "ÉXITO" if self.exito else "SIN CAMBIOS"
        lineas = [
            f"Auto-programación: {estado} — {self.motivo}",
            f"  Rama: {self.rama or '(no creada)'}",
            f"  Archivos: {len(self.archivos_escritos)}",
            f"  Pruebas: {self.pruebas.describe()}",
        ]
        if self.fusionado:
            lineas.append("  Fusión en main: SÍ" + (" (recarga en caliente)" if self.recargado else ""))
        if self.revertido:
            lineas.append("  Reversión completa: SÍ (todo sigue como estaba)")
        return "\n".join(lineas)


# =============================================================================
# MOTOR
# =============================================================================

class SelfProgrammer:
    """Ciclo completo de auto-programación con rama aislada y rollback.

    Parameters
    ----------
    router:
        Enrutador de modelos (usa `qwen2.5-coder:7b` para programar).
    killswitch:
        Interruptor de emergencia: aborta el ciclo si Pablo lo activa.
    notifier:
        Función `(estado, detalle)` para el HUD.
    repo_root:
        Carpeta del repositorio sobre la que trabajar (por defecto, la raíz).
    """

    def __init__(
        self,
        router: Any,
        killswitch: Any | None = None,
        notifier: Callable[[str, str], None] | None = None,
        repo_root: Path | None = None,
        hot_reload: bool | None = None,
        dry_run: bool | None = None,
    ) -> None:
        self.router = router
        self.killswitch = killswitch
        self.notifier = notifier
        self.repo_root = Path(repo_root or cfg.ROOT_DIR).resolve()
        self.hot_reload = cfg.SELF_PROGRAMMING_HOT_RELOAD if hot_reload is None else hot_reload
        self.dry_run = cfg.SELF_PROGRAMMING_DRY_RUN if dry_run is None else dry_run
        self.enabled = cfg.SELF_PROGRAMMING_ENABLED
        self.history: list[CodingOutcome] = []

    # --- Ayudantes ---------------------------------------------------------

    def _notify(self, state: str, detail: str = "") -> None:
        if self.notifier is None:
            return
        try:
            self.notifier(state, detail)
        except Exception:  # pragma: no cover
            pass

    def _check_killswitch(self) -> None:
        if self.killswitch is not None and self.killswitch.is_tripped():
            raise CommandGuardError("Kill-switch activado: el ciclo de auto-programación se aborta.")

    @staticmethod
    def looks_like_self_programming(text: str) -> bool:
        """¿La orden de Pablo pide claramente modificar el propio JARVIS?

        Se usa en el despachador para NO activar el auto-programador por error
        (solo con una petición explícita de añadir o cambiar funcionalidades).
        """
        normalized = normalize_text(text)
        triggers = (
            "anade una funcionalidad", "añade una funcionalidad", "nueva funcionalidad",
            "anade una funcion", "anade una caracteristica", "anade una capacidad",
            "programate", "programa te", "modifica tu codigo", "cambia tu codigo",
            "mejora tu", "anade al codigo", "programa una funcionalidad",
            "anade un comando", "añade un comando", "amplia tus funciones",
        )
        return any(normalize_text(t) in normalized for t in triggers)

    # --- Git ---------------------------------------------------------------

    def _git(self, args: Sequence[str], timeout: float | None = None) -> tuple[int, str]:
        """Ejecuta un comando git auditado. Devuelve `(código, salida)`."""
        argv = [cfg.GIT_BIN, *[str(a) for a in args]]
        try:
            result = run_guarded(
                argv,
                cwd=self.repo_root,
                timeout=timeout or cfg.GIT_COMMAND_TIMEOUT_S,
                trust="internal",  # Rutas y comandos vetados por nosotros.
            )
        except (CommandGuardError, OSError) as exc:
            logger.error("Git bloqueado o no disponible: %s", exc)
            return 1, str(exc)
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            logger.debug("git %s -> %d: %s", " ".join(argv[1:]), result.returncode, truncate(output, 300))
        return result.returncode, output.strip()

    def _git_ready(self) -> tuple[bool, str]:
        """Comprueba que estamos en un repositorio git y con el árbol limpio."""
        code, output = self._git(["rev-parse", "--is-inside-work-tree"])
        if code != 0 or "true" not in output:
            return False, "La carpeta no es un repositorio git (o git no está instalado)."

        code, status = self._git(["status", "--porcelain"])
        if code != 0:
            return False, f"No he podido consultar el estado de git: {truncate(status, 200)}"
        dirty = [line for line in status.splitlines() if line.strip()]
        if dirty:
            return False, (
                "Hay cambios sin guardar en el repositorio; no tocaré nada para no "
                f"mezclarlos: {truncate('; '.join(dirty[:5]), 200)}"
            )
        return True, ""

    def current_branch(self) -> str:
        code, output = self._git(["rev-parse", "--abbrev-ref", "HEAD"])
        return output.strip() if code == 0 else ""

    def _create_branch(self, branch: str) -> bool:
        code, output = self._git(["checkout", "-b", branch])
        if code != 0:
            logger.error("No he podido crear la rama '%s': %s", branch, truncate(output, 200))
            return False
        logger.info("Rama de trabajo creada: %s", branch)
        return True

    def _commit(self, message: str, files: Sequence[Path]) -> bool:
        """Añade y confirma SOLO los archivos generados por nosotros."""
        relative = [str(Path(f).relative_to(self.repo_root)) for f in files]
        if not relative:
            return False
        code, output = self._git(["add", "--", *relative])
        if code != 0:
            logger.error("No he podido preparar los archivos: %s", truncate(output, 200))
            return False
        code, output = self._git(
            [
                "-c", f"user.name={cfg.GIT_USER_NAME}",
                "-c", f"user.email={cfg.GIT_USER_EMAIL}",
                "commit", "-m", message,
            ]
        )
        if code != 0:
            logger.error("No he podido confirmar los cambios: %s", truncate(output, 300))
            return False
        return True

    def _rollback(self, branch: str, created_files: Sequence[Path]) -> None:
        """Vuelve a main y borra todo rastro del intento fallido."""
        main_branch = cfg.GIT_MAIN_BRANCH
        self._git(["checkout", "--force", main_branch])
        self._git(["reset", "--hard", main_branch])
        self._git(["branch", "-D", branch])
        for path in created_files:
            try:
                if path.exists() and not is_protected_path(path):
                    path.unlink()
            except OSError as exc:
                logger.debug("No se pudo borrar %s: %s", path, exc)
        logger.warning("Reversión completada: JARVIS sigue como estaba.")

    # --- Pruebas -----------------------------------------------------------

    def run_tests(self, tests_path: str = "tests/") -> TestReport:
        """Ejecuta pytest en un subproceso aislado y resume el resultado."""
        report = TestReport()
        started = time.perf_counter()
        argv = [
            sys.executable, "-m", "pytest", tests_path,
            "-q", "--no-header", "-p", "no:cacheprovider",
        ]
        try:
            result = run_guarded(
                argv,
                cwd=self.repo_root,
                timeout=cfg.SELF_PROGRAMMING_TEST_TIMEOUT_S,
                trust="internal",
                extra_env={"PYTHONPATH": str(self.repo_root), "JARVIS_TESTING": "1"},
            )
        except CommandGuardError as exc:
            report.output = f"Pruebas bloqueadas: {exc}"
            return report
        except Exception as exc:  # TimeoutExpired u otros
            report.output = f"Las pruebas no pudieron ejecutarse: {exc}"
            return report

        report.duration_s = time.perf_counter() - started
        report.output = ((result.stdout or "") + (result.stderr or "")).strip()
        report.passed = sum(int(n) for n in re.findall(r"(\d+) passed", report.output))
        report.failed = sum(int(n) for n in re.findall(r"(\d+) failed", report.output))
        report.errors = sum(int(n) for n in re.findall(r"(\d+) error", report.output))
        report.ok = result.returncode == 0 and report.failed == 0 and report.errors == 0
        logger.info("Pruebas: %s", report.describe())
        return report

    # --- Generación de código ---------------------------------------------

    def _build_prompt(self, instruction: str, context: str = "") -> str:
        """Prompt de auto-programación: JSON estricto con archivos completos."""
        return (
            "Eres JARVIS modificando su propio código fuente. Devuelve SOLO un JSON válido.\n"
            "Formato exacto:\n"
            '{"resumen": "qué cambias en una frase", '
            '"archivos": [{"ruta": "core/ejemplo.py", "contenido": "<código completo>"}], '
            '"pruebas": [{"ruta": "tests/test_ejemplo.py", "contenido": "<tests pytest>"}], '
            '"notas": "cualquier advertencia"}\n\n'
            "Reglas obligatorias:\n"
            "1. Devuelve el contenido COMPLETO de cada archivo (no parches ni fragmentos).\n"
            "2. Rutas relativas dentro del proyecto. Está TERMINANTEMENTE PROHIBIDO tocar "
            "cualquier archivo de la carpeta `safety/` (kill-switch incluido).\n"
            "3. Comentarios y textos de usuario en español; sin placeholders ni TODO.\n"
            f"4. Máximo {cfg.SELF_PROGRAMMING_MAX_FILES} archivos.\n"
            "5. Incluye al menos un archivo de pruebas pytest para la funcionalidad nueva.\n"
            "6. No uses `shell=True`, ni `os.system`, ni descargas de red.\n"
            f"\nPetición de Pablo:\n{instruction}"
            + (f"\n\nContexto del proyecto:\n{context}" if context else "")
        )

    def propose(self, instruction: str, context: str = "") -> CodeProposal:
        """Pide al modelo el código y lo audita antes de escribir nada."""
        prompt = self._build_prompt(instruction, context)
        text = self.router.write_code(prompt, json_mode=True)
        payload = extract_json(text)
        if not isinstance(payload, dict):
            logger.error("La propuesta del modelo no era JSON válido: %s", truncate(text, 200))
            return CodeProposal(resumen="", notas="El modelo no devolvió JSON.")

        proposal = CodeProposal(
            resumen=str(payload.get("resumen", "")),
            notas=str(payload.get("notas", "")),
        )
        for key, target in (("archivos", proposal.archivos), ("pruebas", proposal.pruebas)):
            for item in payload.get(key) or []:
                if not isinstance(item, dict):
                    continue
                try:
                    route = sanitize_relative_file_path(str(item.get("ruta", "")))
                except UnsafePathError as exc:
                    logger.warning("Ruta generada rechazada: %s", exc)
                    continue
                if not any(route.endswith(ext) for ext in cfg.SELF_PROGRAMMING_ALLOWED_EXTENSIONS):
                    logger.warning("Extensión no permitida en '%s': se descarta.", route)
                    continue
                content = str(item.get("contenido", ""))
                audit = audit_generated_code(content, route)
                if not audit.safe:
                    logger.error("Código rechazado por el guardián (%s): %s", route, audit.summary())
                    continue
                target[route] = content
        return proposal

    # --- Ciclo completo ----------------------------------------------------

    def implement(
        self,
        instruction: str,
        context: str = "",
        speak: Callable[[str], None] | None = None,
    ) -> CodingOutcome:
        """Ciclo completo: rama, código, pruebas, fusión o reversión."""
        outcome = CodingOutcome()
        if not self.enabled:
            outcome.motivo = "La auto-programación está desactivada en config.py."
            return outcome

        self._notify("thinking", "Auto-programación")
        try:
            self._check_killswitch()
        except CommandGuardError as exc:
            outcome.motivo = str(exc)
            return outcome

        ready, reason = self._git_ready()
        if not ready:
            outcome.motivo = reason
            logger.warning("Auto-programación no iniciada: %s", reason)
            return outcome

        branch = f"{cfg.SELF_PROGRAMMING_BRANCH_PREFIX}{timestamp_slug()}"
        outcome.rama = branch
        if not self._create_branch(branch):
            outcome.motivo = "No he podido crear la rama aislada."
            return outcome

        created: list[Path] = []
        try:
            proposal = self.propose(instruction, context)
            outcome.resumen = proposal.resumen
            if proposal.total_files == 0:
                outcome.motivo = "El modelo no propuso código válido (o lo rechazó el guardián)."
                self._rollback(branch, created)
                outcome.revertido = True
                return outcome

            created = self._write_proposal(proposal)
            outcome.archivos_escritos = [str(p.relative_to(self.repo_root)) for p in created]
            if not created:
                outcome.motivo = "No se pudo escribir ningún archivo."
                self._rollback(branch, created)
                outcome.revertido = True
                return outcome

            self._notify("thinking", "Ejecutando pruebas")
            if not self._commit(f"JARVIS: {proposal.resumen or truncate(instruction, 60)}", created):
                outcome.motivo = "No he podido confirmar los cambios en la rama."
                self._rollback(branch, created)
                outcome.revertido = True
                return outcome

            outcome.pruebas = self.run_tests()

            if outcome.pruebas.ok:
                if self.dry_run:
                    outcome.exito = True
                    outcome.motivo = "Modo de prueba: todo pasa, pero no se fusiona nada."
                    self._rollback(branch, created)
                    outcome.revertido = True
                    return outcome

                if self._merge(branch):
                    outcome.exito = True
                    outcome.fusionado = True
                    outcome.motivo = f"Fusión completada: {proposal.resumen or 'mejora aplicada'}"
                    if self.hot_reload:
                        outcome.recargado = self._hot_reload()
                else:
                    outcome.motivo = "La fusión falló; se revierte para no dejar main a medias."
                    self._rollback(branch, created)
                    outcome.revertido = True
            else:
                outcome.motivo = (
                    f"Las pruebas fallaron ({outcome.pruebas.describe()}). "
                    "No se toca main: todo queda como estaba."
                )
                self._rollback(branch, created)
                outcome.revertido = True
        except Exception as exc:  # pragma: no cover - red de seguridad final
            logger.exception("Fallo inesperado en el ciclo de auto-programación")
            outcome.motivo = f"Error inesperado: {truncate(str(exc), 200)}"
            with_guard = suppress_errors(self._rollback)
            with_guard(branch, created)
            outcome.revertido = True
        finally:
            self._notify("idle", "")
            self.history.append(outcome)
            logger.info("\n%s", outcome.describe())
            if speak is not None:
                with_guard = suppress_errors(speak)
                mensaje = (
                    "Listo, Pablo. He añadido esa funcionalidad y todas las pruebas pasan."
                    if outcome.exito else
                    f"No he aplicado cambios, Pablo. {outcome.motivo}"
                )
                with_guard(mensaje)
        return outcome

    def _write_proposal(self, proposal: CodeProposal) -> list[Path]:
        """Escribe los archivos aprobados, siempre dentro del proyecto."""
        written: list[Path] = []
        for route, content in {**proposal.archivos, **proposal.pruebas}.items():
            if is_protected_path(route):
                logger.error("El guardián impide escribir en '%s'.", route)
                continue
            target = (self.repo_root / route).resolve()
            try:
                target.relative_to(self.repo_root)
            except ValueError:
                logger.error("Ruta fuera del proyecto rechazada: %s", route)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.write_text(content, encoding="utf-8")
                written.append(target)
                logger.info("Escrito: %s (%d caracteres)", route, len(content))
            except OSError as exc:
                logger.error("No pude escribir '%s': %s", route, exc)
        return written

    def _merge(self, branch: str) -> bool:
        """Fusiona la rama de trabajo en main (sin fast-forward) y la borra."""
        code, output = self._git(["checkout", cfg.GIT_MAIN_BRANCH])
        if code != 0:
            logger.error("No he podido volver a main: %s", truncate(output, 200))
            return False
        code, output = self._git(
            [
                "-c", f"user.name={cfg.GIT_USER_NAME}",
                "-c", f"user.email={cfg.GIT_USER_EMAIL}",
                "merge", cfg.GIT_DEFAULT_MERGE_FLAG, branch,
                "-m", f"JARVIS: fusión de {branch}",
            ]
        )
        if code != 0:
            logger.error("La fusión falló: %s", truncate(output, 300))
            return False
        self._git(["branch", "-d", branch])
        logger.info("Rama '%s' fusionada en %s y eliminada.", branch, cfg.GIT_MAIN_BRANCH)
        return True

    def _hot_reload(self) -> bool:
        """Renace con el código nuevo (`os.execv`), sin intervención de Pablo."""
        if not cfg.SELF_PROGRAMMING_HOT_RELOAD:
            return False
        logger.critical("Recarga en caliente: JARVIS se reinicia con la mejora aplicada.")
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            for handler in logging.getLogger("jarvis").handlers:
                with_guard = suppress_errors(handler.flush)
                with_guard()
        except Exception:
            pass
        try:
            os.execv(sys.executable, [sys.executable, *sys.argv])
        except Exception as exc:  # pragma: no cover - depende del SO
            logger.error("No he podido recargarme en caliente: %s", exc)
            return False
        return True  # pragma: no cover - execv no vuelve

    # --- Diagnóstico -------------------------------------------------------

    def status(self) -> dict:
        last = self.history[-1] if self.history else None
        return {
            "activo": self.enabled,
            "modo_prueba": self.dry_run,
            "recarga_caliente": self.hot_reload,
            "rama_actual": self.current_branch(),
            "ciclos": len(self.history),
            "ultimo": last.describe() if last else "",
        }


def suppress_errors(func: Callable) -> Callable:
    """Envuelve una función para que sus errores nunca corten el ciclo."""

    def wrapper(*args: Any, **kwargs: Any):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover
            logger.debug("Error ignorado en %s: %s", getattr(func, "__name__", func), exc)
            return None

    return wrapper
