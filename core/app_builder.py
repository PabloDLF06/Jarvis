"""
core/app_builder.py — De la voz a una aplicación funcionando.
================================================================================
Pablo dice: *"Jarvis, hazme una web para llevar mis gastos"* o *"créame una
aplicación de escritorio para cronometrar tareas"* y JARVIS entrega un proyecto
completo, probado y abierto en pantalla.

FLUJO
-----
    1. Se crea una carpeta aislada: workspace/projects/<nombre>/
    2. qwen2.5-coder:7b escribe TODOS los archivos (HTML/CSS/JS o Python/PyQt6)
       en un JSON con el proyecto completo.
    3. El guardián de comandos audita el código generado y las rutas: nada de
       `shell=True`, nada fuera de la carpeta del proyecto, nada en `safety/`.
    4. Si hay dependencias, se crean en un entorno virtual propio
       (workspace/.venvs/<nombre>) y se validan los nombres de paquete antes de
       instalar nada.
    5. Se ejecuta una comprobación automática (compilación de los .py y un
       "smoke test" del servidor o del script).
    6. Se lanza la aplicación: las webs se abren en Comet y las apps de
       escritorio arrancan con el Python del entorno sandbox.

Nada de este módulo puede salir de `workspace/`, tocar `safety/` ni instalar
paquetes fuera de la lista blanca de PyPI.
================================================================================
"""

from __future__ import annotations

import logging
import os
import subprocess
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
    resolve_within,
    run_guarded,
    sanitize_project_name,
    sanitize_relative_file_path,
    validate_dependency_list,
)

logger = logging.getLogger("jarvis.constructor")

WEB_MARKERS = ("index.html", "web/index.html", "public/index.html")


# =============================================================================
# RESULTADOS
# =============================================================================

@dataclass
class AppProposal:
    """Proyecto propuesto por el modelo."""

    nombre: str = ""
    descripcion: str = ""
    tipo: str = "web"
    dependencias: list[str] = field(default_factory=list)
    comando: str = ""
    archivos: dict[str, str] = field(default_factory=dict)
    notas: str = ""


@dataclass
class BuildOutcome:
    """Resultado completo de la construcción de una aplicación."""

    nombre: str = ""
    carpeta: str = ""
    exito: bool = False
    motivo: str = ""
    archivos_escritos: list[str] = field(default_factory=list)
    dependencias_instaladas: list[str] = field(default_factory=list)
    entorno: str = ""
    lanzada: bool = False
    como_lanzar: str = ""

    def describe(self) -> str:
        estado = "LISTA" if self.exito else "FALLIDA"
        lineas = [
            f"Construcción de '{self.nombre}': {estado} — {self.motivo}",
            f"  Carpeta: {self.carpeta}",
            f"  Archivos: {len(self.archivos_escritos)}",
        ]
        if self.dependencias_instaladas:
            lineas.append(f"  Dependencias: {', '.join(self.dependencias_instaladas)}")
        if self.lanzada:
            lineas.append(f"  Lanzada en pantalla: SÍ ({self.como_lanzar})")
        else:
            lineas.append(f"  Para abrirla: {self.como_lanzar}")
        return "\n".join(lineas)


# =============================================================================
# CONSTRUCTOR
# =============================================================================

class AppBuilder:
    """Construye aplicaciones y webs completas a partir de una descripción hablada.

    Parameters
    ----------
    router:
        Enrutador de modelos (genera el código con `qwen2.5-coder:7b`).
    media:
        Despachador de medios (para abrir la web en Comet). Puede ser None.
    killswitch:
        Interruptor de emergencia.
    notifier:
        Función `(estado, detalle)` para el HUD (marco de neón durante el trabajo).
    projects_dir / venvs_dir:
        Carpetas de trabajo (por defecto `workspace/projects` y `workspace/.venvs`).
    """

    def __init__(
        self,
        router: Any,
        media: Any | None = None,
        killswitch: Any | None = None,
        notifier: Callable[[str, str], None] | None = None,
        projects_dir: Path | None = None,
        venvs_dir: Path | None = None,
    ) -> None:
        self.router = router
        self.media = media
        self.killswitch = killswitch
        self.notifier = notifier
        self.projects_dir = Path(projects_dir or cfg.PROJECTS_DIR).resolve()
        self.venvs_dir = Path(venvs_dir or cfg.SANDBOX_VENVS_DIR).resolve()
        self.enabled = cfg.APP_BUILDER_ENABLED
        self.history: list[BuildOutcome] = []
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.venvs_dir.mkdir(parents=True, exist_ok=True)

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
            raise CommandGuardError("Kill-switch activado: se cancela la construcción.")

    @staticmethod
    def looks_like_app_request(text: str) -> bool:
        """¿Pablo está pidiendo construir algo? (web, app, script, juego...)."""
        normalized = normalize_text(text)
        verbs = ("hazme", "creame", "crea", "construye", "genera", "programa", "desarrolla", "monta")
        nouns = ("web", "pagina web", "aplicacion", "app", "programa", "script", "juego",
                 "calculadora", "gestor", "lista de tareas", "cronometro", "formulario")
        return any(v in normalized for v in verbs) and any(n in normalized for n in nouns)

    # --- Generación --------------------------------------------------------

    def _build_prompt(self, description: str, name_hint: str) -> str:
        """Prompt estricto: JSON con el proyecto completo y la voz de JARVIS."""
        return (
            "Eres JARVIS construyendo una aplicación completa para Pablo. "
            "Devuelve SOLO un objeto JSON válido, sin markdown ni texto extra.\n"
            "Formato exacto:\n"
            '{"nombre": "slug-del-proyecto", "descripcion": "qué hace", '
            '"tipo": "web|escritorio|script", "dependencias": ["paquete"], '
            '"comando": "como se ejecuta", '
            '"archivos": [{"ruta": "index.html", "contenido": "<!DOCTYPE html>..."}]}\n\n'
            "Reglas obligatorias:\n"
            "1. El contenido de cada archivo debe estar COMPLETO y ser ejecutable tal cual "
            "(sin fragmentos, sin placeholders, sin TODO).\n"
            "2. Rutas relativas simples dentro del proyecto. Nunca '..', nunca rutas "
            "absolutas, nunca .exe ni .bat.\n"
            "3. Si es una web: un `index.html` autocontenido con CSS y JavaScript dentro, "
            "funcionando sin servidor y con diseño moderno y elegante.\n"
            "4. Si es de escritorio: Python con PyQt6 (import PyQt6) y un `main.py`.\n"
            "5. Dependencias: solo paquetes de PyPI con nombre válido (nada de URLs). "
            "Si no hacen falta, deja la lista vacía.\n"
            "6. Prohibido: `shell=True`, `os.system`, sockets, descargas de red, "
            "referencias a la carpeta `safety/`.\n"
            "7. Textos de interfaz y comentarios en español.\n"
            f"8. Nombre sugerido para la carpeta: {name_hint}\n"
            f"\nLo que pide Pablo:\n{description}"
        )

    def propose(self, description: str, name_hint: str | None = None) -> AppProposal:
        """Pide al modelo el proyecto y lo audita (código, rutas y dependencias)."""
        hint = name_hint or sanitize_project_name(description)[:32] or "proyecto"
        text = self.router.write_code(
            self._build_prompt(description, hint),
            json_mode=True,
        )
        payload = extract_json(text)
        if not isinstance(payload, dict):
            logger.error("La propuesta de la app no era JSON válido: %s", truncate(text, 240))
            return AppProposal(notas="El modelo no devolvió JSON.")

        proposal = AppProposal(
            nombre=sanitize_project_name(str(payload.get("nombre") or hint)),
            descripcion=str(payload.get("descripcion", "")),
            tipo=str(payload.get("tipo", "web")).strip().lower(),
            comando=str(payload.get("comando", "")),
            notas=str(payload.get("notas", "")),
        )
        if proposal.tipo not in ("web", "escritorio", "script"):
            proposal.tipo = "web"

        try:
            proposal.dependencias = validate_dependency_list(
                payload.get("dependencias") or [], limit=cfg.APP_BUILDER_MAX_DEPENDENCIES
            )
        except CommandGuardError as exc:
            logger.warning("Dependencias rechazadas (%s): se construye sin ellas.", exc)
            proposal.dependencias = []

        for item in payload.get("archivos") or []:
            if not isinstance(item, dict):
                continue
            try:
                route = sanitize_relative_file_path(str(item.get("ruta", "")))
            except UnsafePathError as exc:
                logger.warning("Archivo rechazado por su ruta: %s", exc)
                continue
            content = str(item.get("contenido", ""))
            if len(content) > cfg.APP_BUILDER_MAX_FILE_BYTES:
                logger.warning("Archivo '%s' demasiado grande: se descarta.", route)
                continue
            audit = audit_generated_code(content, route)
            if not audit.safe:
                logger.error("Código rechazado en '%s': %s", route, audit.summary())
                continue
            proposal.archivos[route] = content
        return proposal

    # --- Construcción ------------------------------------------------------

    def build(self, description: str, name_hint: str | None = None, speak: Callable[[str], None] | None = None) -> BuildOutcome:
        """Construye, prueba y lanza una aplicación completa."""
        outcome = BuildOutcome()
        if not self.enabled:
            outcome.motivo = "El constructor de aplicaciones está desactivado en config.py."
            return outcome

        self._notify("acting", "Construyendo aplicación")
        try:
            self._check_killswitch()
            proposal = self.propose(description, name_hint)
            if not proposal.archivos:
                outcome.motivo = "El modelo no propuso archivos válidos para el proyecto."
                outcome.nombre = proposal.nombre or (name_hint or "proyecto")
                return self._finish(outcome, speak)

            outcome.nombre = proposal.nombre
            project_dir = self._scaffold(proposal)
            outcome.carpeta = str(project_dir)

            written = self._write_files(project_dir, proposal.archivos)
            outcome.archivos_escritos = [str(p.relative_to(project_dir)) for p in written]
            if not written:
                outcome.motivo = "No se pudo escribir ningún archivo del proyecto."
                return self._finish(outcome, speak)

            # Entorno virtual + dependencias (solo si hacen falta).
            if proposal.dependencias:
                outcome.entorno, installed = self._prepare_environment(
                    project_dir, proposal.nombre, proposal.dependencias
                )
                outcome.dependencias_instaladas = installed
                if not installed:
                    logger.warning("No se pudieron instalar las dependencias; se continúa igualmente.")

            # Comprobación automática antes de dar nada por bueno.
            self._notify("thinking", "Comprobando la aplicación")
            checks_ok, check_output = self._smoke_test(project_dir, proposal)
            if not checks_ok:
                outcome.motivo = f"La comprobación automática encontró un problema: {truncate(check_output, 200)}"
                logger.warning(outcome.motivo)
                self._write_readme(project_dir, proposal, outcome, check_output)
                return self._finish(outcome, speak)

            self._write_readme(project_dir, proposal, outcome, check_output)

            # Lanzar en pantalla.
            if cfg.APP_BUILDER_AUTO_LAUNCH:
                launched, how = self._launch(project_dir, proposal, outcome.entorno)
                outcome.lanzada = launched
                outcome.como_lanzar = how
            else:
                outcome.como_lanzar = self._manual_instructions(project_dir, proposal)

            outcome.exito = True
            outcome.motivo = proposal.descripcion or "Aplicación construida y verificada."
        except CommandGuardError as exc:
            outcome.motivo = str(exc)
        except Exception as exc:  # pragma: no cover - red de seguridad
            logger.exception("Fallo construyendo la aplicación")
            outcome.motivo = f"Error inesperado: {truncate(str(exc), 200)}"
        return self._finish(outcome, speak)

    def _finish(self, outcome: BuildOutcome, speak: Callable[[str], None] | None) -> BuildOutcome:
        """Cierra el ciclo: avisa al HUD, guarda historial y habla si procede."""
        self._notify("idle", "")
        self.history.append(outcome)
        logger.info("\n%s", outcome.describe())
        if speak is not None:
            try:
                speak(
                    f"Listo, Pablo. Su aplicación '{outcome.nombre}' está preparada y la he abierto."
                    if outcome.exito
                    else f"No he podido terminar la aplicación, Pablo. {outcome.motivo}"
                )
            except Exception:  # pragma: no cover
                pass
        return outcome

    def _scaffold(self, proposal: AppProposal) -> Path:
        """Crea (con nombre único si hace falta) la carpeta del proyecto."""
        base = self.projects_dir / (proposal.nombre or f"proyecto-{timestamp_slug()}")
        target = base
        suffix = 2
        while target.exists():
            target = Path(f"{base}-{suffix}")
            suffix += 1
        target.mkdir(parents=True, exist_ok=False)
        logger.info("Proyecto creado en %s", target)
        return target

    def _write_files(self, project_dir: Path, files: dict[str, str]) -> list[Path]:
        """Escribe los archivos del proyecto, confinados a su carpeta."""
        written: list[Path] = []
        for route, content in list(files.items())[: cfg.APP_BUILDER_MAX_FILES]:
            try:
                target = resolve_within(project_dir, route)
            except UnsafePathError as exc:
                logger.error("Ruta fuera del proyecto: %s", exc)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.write_text(content, encoding="utf-8")
                written.append(target)
                logger.info("  + %s (%d bytes)", route, len(content.encode("utf-8")))
            except OSError as exc:
                logger.error("No pude escribir %s: %s", route, exc)
        return written

    # --- Entorno virtual ---------------------------------------------------

    def _venv_python(self, venv_dir: Path) -> Path:
        """Ruta del intérprete dentro de un entorno virtual."""
        if os.name == "nt":
            return venv_dir / "Scripts" / "python.exe"
        return venv_dir / "bin" / "python"

    def _prepare_environment(
        self, project_dir: Path, name: str, dependencies: Sequence[str] | None = None
    ) -> tuple[str, list[str]]:
        """Crea el entorno virtual e instala las dependencias validadas."""
        if not cfg.APP_BUILDER_ALLOW_DEPENDENCIES:
            logger.info("Las dependencias están desactivadas en config.py.")
            return "", []

        venv_dir = self.venvs_dir / sanitize_project_name(name)
        python = self._venv_python(venv_dir)
        if not python.exists():
            self._notify("thinking", "Creando entorno aislado")
            logger.info("Creando entorno virtual en %s", venv_dir)
            try:
                result = run_guarded(
                    [sys.executable, "-m", "venv", str(venv_dir)],
                    cwd=project_dir,
                    timeout=cfg.APP_BUILDER_VENV_TIMEOUT_S,
                    trust="internal",
                )
            except (CommandGuardError, OSError) as exc:
                logger.error("No pude crear el entorno virtual: %s", exc)
                return "", []
            if result.returncode != 0 or not python.exists():
                logger.error("El entorno virtual no se creó correctamente: %s", truncate(result.stderr, 200))
                return "", []

        # Nota: aquí llega una lista ya validada por el guardián de comandos
        # (`validate_dependency_list`), así que ningún nombre puede ser una bandera
        # de pip ni un paquete vetado.
        proposal_deps = list(dependencies or [])
        if not proposal_deps:
            return str(venv_dir), []

        self._notify("thinking", "Instalando dependencias")
        installed: list[str] = []
        for dependency in proposal_deps:
            try:
                self._check_killswitch()
                validate_dependency_list([dependency])
                result = run_guarded(
                    [str(python), "-m", "pip", "install", "--disable-pip-version-check", dependency],
                    cwd=project_dir,
                    timeout=cfg.APP_BUILDER_INSTALL_TIMEOUT_S,
                    trust="internal",
                )
            except (CommandGuardError, OSError) as exc:
                logger.error("Dependencia '%s' rechazada: %s", dependency, exc)
                continue
            if result.returncode == 0:
                installed.append(dependency)
                logger.info("Dependencia instalada: %s", dependency)
            else:
                logger.warning("No se pudo instalar '%s': %s", dependency, truncate(result.stderr, 200))
        return str(venv_dir), installed

    # --- Comprobaciones ----------------------------------------------------

    def _smoke_test(self, project_dir: Path, proposal: AppProposal) -> tuple[bool, str]:
        """Comprobación automática del proyecto recién generado.

        * Los `.py` se compilan con `compileall` (detecta errores de sintaxis).
        * Si es una web, se verifica que exista el HTML principal y que no esté vacío.
        """
        python = self._python_for(project_dir)
        python_files = list(project_dir.rglob("*.py"))
        if python_files:
            try:
                result = run_guarded(
                    [str(python), "-m", "compileall", "-q", str(project_dir)],
                    cwd=project_dir,
                    timeout=cfg.APP_BUILDER_TEST_TIMEOUT_S,
                    trust="internal",
                )
            except (CommandGuardError, OSError) as exc:
                return False, f"No se pudo compilar el código generado: {exc}"
            if result.returncode != 0:
                return False, (result.stdout or "") + (result.stderr or "")

        if proposal.tipo == "web" or not python_files:
            html = self._find_entry_html(project_dir)
            if html is None:
                return False, "No encuentro el archivo HTML principal del proyecto."
            if html.stat().st_size < 120:
                return False, f"El archivo {html.name} está prácticamente vacío."
        return True, "Compilación correcta y archivo principal presente."

    @staticmethod
    def _find_entry_html(project_dir: Path) -> Path | None:
        """Localiza el HTML de entrada del proyecto."""
        for marker in WEB_MARKERS:
            candidate = project_dir / marker
            if candidate.exists():
                return candidate
        found = sorted(project_dir.rglob("*.html"))
        return found[0] if found else None

    def _python_for(self, project_dir: Path) -> Path:
        """Intérprete a usar en el proyecto (el del entorno si existe)."""
        venv_python = self._venv_python(self.venvs_dir / sanitize_project_name(project_dir.name))
        return venv_python if venv_python.exists() else Path(sys.executable)

    def _write_readme(self, project_dir: Path, proposal: AppProposal, outcome: BuildOutcome, checks: str) -> None:
        """Documenta el proyecto generado (para que Pablo sepa qué tiene)."""
        content = (
            f"# {proposal.nombre}\n\n"
            f"{proposal.descripcion}\n\n"
            "## Cómo abrirlo\n\n"
            f"{self._manual_instructions(project_dir, proposal)}\n\n"
            "## Detalles\n\n"
            f"- Tipo: {proposal.tipo}\n"
            f"- Creado por JARVIS el {time.strftime('%d/%m/%Y a las %H:%M')}\n"
            f"- Dependencias: {', '.join(proposal.dependencias) or 'ninguna'}\n"
            f"- Comprobación automática: {checks}\n"
            f"- Archivos: {len(outcome.archivos_escritos)}\n"
        )
        try:
            (project_dir / "README.md").write_text(content, encoding="utf-8")
            outcome.archivos_escritos.append("README.md")
        except OSError as exc:
            logger.debug("No pude escribir el README del proyecto: %s", exc)

    # --- Lanzamiento -------------------------------------------------------

    def _manual_instructions(self, project_dir: Path, proposal: AppProposal) -> str:
        """Instrucciones humanas para abrir el proyecto."""
        html = self._find_entry_html(project_dir)
        if proposal.tipo == "web" and html is not None:
            return f"Abre el archivo {html.name} (doble clic) o pídele a JARVIS que lo abra."
        main = project_dir / "main.py"
        python = self._python_for(project_dir)
        return f"Ejecuta: \"{python}\" \"{main}\"" if main.exists() else "Revisa la carpeta del proyecto."

    def _launch(self, project_dir: Path, proposal: AppProposal, venv: str) -> tuple[bool, str]:
        """Abre la aplicación: web en Comet; escritorio con el Python del entorno."""
        html = self._find_entry_html(project_dir)
        if proposal.tipo == "web" and html is not None:
            if self.media is not None:
                try:
                    ok, method = self.media.open_url(html.as_uri())
                    if ok:
                        return True, f"Comet/{method}"
                except Exception as exc:
                    logger.debug("No pude abrir la web en Comet: %s", exc)
            try:
                if sys.platform == "win32":
                    os.startfile(str(html))  # type: ignore[attr-defined]  # noqa: S606
                    return True, "navegador predeterminado"
            except OSError as exc:
                logger.warning("No pude abrir la web: %s", exc)
            return False, self._manual_instructions(project_dir, proposal)

        entry = project_dir / "main.py"
        if not entry.exists():
            candidates = [p for p in project_dir.glob("*.py") if p.name != "conftest.py"]
            entry = candidates[0] if candidates else None
        if entry is None:
            return False, self._manual_instructions(project_dir, proposal)

        python = self._venv_python(Path(venv)) if venv else self._python_for(project_dir)
        if not python.exists():
            python = self._python_for(project_dir)
        try:
            kwargs: dict[str, Any] = {"cwd": str(project_dir), "close_fds": True}
            if sys.platform == "win32":
                kwargs["creationflags"] = 0x00000008 | 0x00000200
            subprocess.Popen(  # noqa: S603 - intérprete verificado y ruta confinada
                [str(python), str(entry)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **kwargs,
            )
            return True, f"ejecutándose con {python.name}"
        except OSError as exc:
            logger.warning("No pude lanzar la aplicación: %s", exc)
            return False, self._manual_instructions(project_dir, proposal)

    # --- Utilidades públicas ----------------------------------------------

    def list_projects(self) -> list[dict]:
        """Lista los proyectos construidos (para `--proyectos`)."""
        projects: list[dict] = []
        for folder in sorted(self.projects_dir.iterdir()):
            if not folder.is_dir():
                continue
            files = [p for p in folder.rglob("*") if p.is_file()]
            projects.append(
                {
                    "nombre": folder.name,
                    "archivos": len(files),
                    "modificado": time.strftime("%d/%m/%Y %H:%M", time.localtime(folder.stat().st_mtime)),
                    "ruta": str(folder),
                }
            )
        return projects

    def status(self) -> dict:
        return {
            "activo": self.enabled,
            "proyectos": len(list(self.projects_dir.iterdir())) if self.projects_dir.exists() else 0,
            "carpeta": str(self.projects_dir),
            "construcciones": len(self.history),
        }
