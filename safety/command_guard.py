"""
safety/command_guard.py — Escudo antivandálico de JARVIS.
================================================================================
Todo comando de shell que JARVIS se disponga a ejecutar (incluidos los que
genera su propio motor de auto-programación o el constructor de apps) pasa por
aquí. El guardián tiene tres misiones:

1. **Bloquear comandos destructivos** (borrado de disco, formateo, apagado,
   escalada de privilegios, borrado de registros, descargas y ejecución
   remota de scripts) mediante patrones de severidad CRÍTICA.
2. **Bloquear comandos dudosos** (resetear repositorios, matar procesos en
   masa, instalar paquetes desde URLs) cuando provienen de código generado.
3. **Encerrar rutas**: ninguna ruta generada puede salir de su sandbox ni
   tocar la carpeta `safety/`.

Este archivo también es un punto único de ejecución auditable: `run_guarded()`
es la ÚNICA función del proyecto autorizada a lanzar subprocesos con shell
desactivado, entorno saneado y directorio de trabajo confinado.
================================================================================
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger("jarvis.safety.guard")

# =============================================================================
# TIPOS DE DATOS
# =============================================================================

SEVERITY_CRITICAL = "CRITICA"
SEVERITY_HIGH = "ALTA"
SEVERITY_MEDIUM = "MEDIA"

TRUST_GENERATED = "generated"   # Código salido de un modelo: cero confianza.
TRUST_INTERNAL = "internal"     # Comandos vetados escritos por JARVIS.


class CommandGuardError(Exception):
    """Se lanza cuando un comando o una ruta violan las reglas de seguridad."""


class UnsafePathError(CommandGuardError):
    """Se lanza cuando una ruta intenta salir de su espacio permitido."""


@dataclass(frozen=True)
class CommandVerdict:
    """Resultado de auditar un comando."""

    allowed: bool
    severity: str = ""
    reason: str = ""
    pattern: str = ""
    command: str = ""

    def __bool__(self) -> bool:  # Permite `if verdict: ...`
        return self.allowed

    def describe(self) -> str:
        if self.allowed:
            return f"PERMITIDO: {self.command}"
        return f"BLOQUEADO [{self.severity}] {self.reason} (patrón: {self.pattern})"


@dataclass
class CodeAudit:
    """Resultado de auditar código fuente generado por un modelo."""

    safe: bool
    findings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.safe:
            return "Código auditado: sin patrones peligrosos."
        return "Código rechazado: " + "; ".join(self.findings)


# =============================================================================
# 1. PATRONES PROHIBIDOS SIEMPRE (nunca, jamás, bajo ninguna excusa)
# =============================================================================

# Formato: (expresión regular, severidad, motivo para Pablo)
FORBIDDEN_PATTERNS: tuple[tuple[str, str, str], ...] = (
    # --- Borrado total / formateo ------------------------------------------
    (r"\brm\s+(-[a-zA-Z]*\s+)*-?[rRf]{1,2}\w*\s+(/|/\*|~/|~|\$home|\$userprofile|\*)\/?\s*$",
     SEVERITY_CRITICAL, "Borrado recursivo de la raíz o del directorio personal"),
    (r"\brm\s+-[rRf]{1,2}\w*\s+(/etc|/usr|/bin|/boot|/var|/home|/opt|/lib)\b", SEVERITY_CRITICAL,
     "Borrado de directorios del sistema"),
    (r"\brm\s+-[rRf]{1,2}\w*\s+\.\.?(/|\s|$)", SEVERITY_CRITICAL,
     "Borrado recursivo del directorio actual o del padre"),
    (r"\b(del|erase)\b[^\n]*?(/s\b[^\n]*?/q\b|/q\b[^\n]*?/s\b)[^\n]*?\b(c:\\|%systemdrive%|\\\\\?\\|\*)", SEVERITY_CRITICAL,
     "Borrado silencioso y recursivo de una unidad completa"),
    (r"\b(rd|rmdir)\s+/s\s+/q\s+[a-zA-Z]:\\?\s*$", SEVERITY_CRITICAL,
     "Eliminación recursiva de una unidad completa"),
    (r"\bformat\s+[a-zA-Z]:", SEVERITY_CRITICAL, "Formateo de disco"),
    (r"\bmkfs(\.\w+)?\b", SEVERITY_CRITICAL, "Creación de sistema de archivos (destruye datos)"),
    (r"\bdiskpart\b", SEVERITY_CRITICAL, "Particionado de discos"),
    (r"\bdd\s+.*\bof=/dev/(sd|nvme|hd|disk)", SEVERITY_CRITICAL,
     "Escritura cruda sobre un dispositivo de bloques"),
    (r">\s*/dev/(sd|nvme|hd|disk)", SEVERITY_CRITICAL, "Sobrescritura de un dispositivo"),
    (r"\b(shred|wipefs)\b", SEVERITY_CRITICAL, "Destrucción irreversible de datos"),
    (r"\bcipher\s+/w", SEVERITY_CRITICAL, "Borrado seguro de espacio libre"),

    # --- Apagado / reinicio / cierre de sesión -----------------------------
    (r"\b(shutdown|poweroff|halt|reboot)\b", SEVERITY_CRITICAL, "Apagado o reinicio del equipo"),
    (r"\b(Stop-Computer|Restart-Computer)\b", SEVERITY_CRITICAL, "Apagado o reinicio vía PowerShell"),
    (r"\blogoff\b", SEVERITY_CRITICAL, "Cierre de sesión del usuario"),

    # --- Registro y arranque de Windows ------------------------------------
    (r"\breg\s+(delete|add|import)\b", SEVERITY_CRITICAL, "Modificación del registro de Windows"),
    (r"\b(bcdedit|vssadmin|wbadmin|diskpart)\b", SEVERITY_CRITICAL,
     "Modificación de arranque o copias de seguridad"),
    (r"\bbootrec\b", SEVERITY_CRITICAL, "Manipulación del sector de arranque"),

    # --- Cuentas, permisos y servicios -------------------------------------
    (r"\bnet\s+(user|localgroup)\b", SEVERITY_CRITICAL, "Creación o modificación de cuentas"),
    (r"\bsc\s+(delete|stop|config)\b", SEVERITY_CRITICAL, "Manipulación de servicios de Windows"),
    (r"\b(taskkill|tskill)\s+/f\s+/im\s+(\*|(winlogon|csrss|lsass|services|explorer|wininit)\.exe)", SEVERITY_CRITICAL,
     "Intento de matar procesos vitales del sistema"),
    (r"\btaskkill\b[^\n]*?/im\s+\*", SEVERITY_CRITICAL, "Intento de matar todos los procesos"),
    (r"\b(sudo|runas)\b", SEVERITY_CRITICAL, "Escalada de privilegios"),
    (r"\b(takeown|icacls|cacls|attrib)\b[^\n]*?/(grant|deny|setowner|f|r)", SEVERITY_CRITICAL,
     "Manipulación de permisos de archivos"),

    # --- Ejecución remota / inyección --------------------------------------
    (r"\b(curl|wget|iwr|Invoke-WebRequest|certutil|bitsadmin)\b[^\n]*?(\|\s*(bash|sh|zsh|python|python3|powershell|cmd))",
     SEVERITY_CRITICAL, "Descarga de un script y ejecución directa (riesgo de código remoto)"),
    (r"\b(iex|Invoke-Expression)\b", SEVERITY_CRITICAL, "Ejecución de expresión dinámica en PowerShell"),
    (r"-(enc|EncodedCommand|e)\s+[A-Za-z0-9+/=]{20,}", SEVERITY_CRITICAL,
     "Comando PowerShell codificado en Base64 (técnica típica de malware)"),
    (r"\bIWR\b[^\n]*?-\s*UseBasicParsing", SEVERITY_CRITICAL, "Descarga remota silenciosa"),
    (r"\bDownloadString\b|\bDownloadFile\b", SEVERITY_CRITICAL, "Descarga de código remoto"),
    (r"\bmshta\b|\brundll32\b|\bregsvr32\b|\bcscript\b|\bwscript\b", SEVERITY_CRITICAL,
     "Ejecución de scripts a través de binarios de sistema (técnica de evasión)"),
    (r":\s*\(\s*\)\s*\{.*\|\s*:.*\}", SEVERITY_CRITICAL, "Bomba fork"),
    (r"\bnc\s+-[a-z]*e\b|\bncat\b[^\n]*?(-e|--exec)", SEVERITY_CRITICAL, "Puerta trasera con shell remota"),

    # --- JARVIS: activos protegidos ----------------------------------------
    (r"(>|>>|Set-Content|Out-File|del|del\s+/f|Remove-Item|truncate)\s*[^\n]*?(killswitch|command_guard|\.sha256)",
     SEVERITY_CRITICAL, "Intento de escribir o borrar un archivo protegido de seguridad"),
    (r"\b(Set-ItemProperty|New-Item)\b[^\n]*?(killswitch|command_guard)", SEVERITY_CRITICAL,
     "Manipulación de los archivos de seguridad"),
    (r"\bgit\s+push\b[^\n]*?--force", SEVERITY_CRITICAL, "Sobrescritura forzada del historial remoto"),
    (r"\bgit\s+filter-(branch|repo)\b", SEVERITY_CRITICAL, "Reescritura destructiva del historial"),
    (r"\b(pip|pip3)\s+(install|download)\b[^\n]*?(-i|--index-url|--extra-index-url)", SEVERITY_CRITICAL,
     "Instalación desde un índice de paquetes no oficial"),
    (r"\b(pip|pip3)\s+install\b[^\n]*?(git\+|https?://|\.\./|/|\\\\)", SEVERITY_CRITICAL,
     "Instalación de un paquete desde URL o ruta arbitraria"),
    (r"\b(npm|yarn|pnpm)\s+(install|i|add)\b[^\n]*?(https?://|git\+)", SEVERITY_CRITICAL,
     "Instalación de dependencia remota arbitraria"),
    (r"\bchmod\s+(-R\s+)?777\s+/", SEVERITY_CRITICAL, "Permisos inseguros en la raíz del sistema"),
    (r"\bchown\s+-R\b[^\n]*?\s/\s*$", SEVERITY_CRITICAL, "Cambio de propietario sobre el sistema"),
    (r"\bwmic\b[^\n]*?\bdelete\b", SEVERITY_CRITICAL, "Borrado de recursos vía WMIC"),
    (r"\bGet-ChildItem\b[^\n]*?\bRemove-Item\b", SEVERITY_CRITICAL, "Borrado masivo recursivo"),
)

# =============================================================================
# 2. PATRONES DUDOSOS (bloqueados solo para código generado por IA)
# =============================================================================

RISKY_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (r"\bgit\s+reset\s+--hard\b", SEVERITY_HIGH, "Descarte destructivo de cambios locales"),
    (r"\bgit\s+clean\s+-[a-z]*f", SEVERITY_HIGH, "Borrado de archivos no rastreados"),
    (r"\bgit\s+branch\s+-D\b", SEVERITY_HIGH, "Borrado forzado de ramas"),
    (r"\bgit\s+checkout\s+--\s+\.", SEVERITY_HIGH, "Descarte de todos los cambios"),
    (r"\bgit\s+submodule\s+deinit\b", SEVERITY_HIGH, "Desactivación de submódulos"),
    (r"\b(taskkill|Stop-Process)\b", SEVERITY_HIGH, "Cierre de procesos del usuario"),
    (r"\b(pip|pip3)\s+uninstall\b", SEVERITY_HIGH, "Desinstalación de paquetes"),
    (r"\bconda\s+(remove|env\s+remove)\b", SEVERITY_HIGH, "Eliminación de entornos"),
    (r"\brm\b|\bdel\b|\bRemove-Item\b|\bshutil\.rmtree\b", SEVERITY_MEDIUM, "Borrado de archivos"),
    (r"\b(move|mv|ren|rename)\b", SEVERITY_MEDIUM, "Movimiento o renombrado de archivos"),
    (r"\b(net\s+use|robocopy|rclone)\b", SEVERITY_HIGH, "Acceso a recursos de red"),
    (r"\b(start|explorer)\b[^\n]*?(\\\\)", SEVERITY_HIGH, "Ejecución desde una ruta de red (UNC)"),
    (r"\bimport\s+safety\b|\bfrom\s+safety\b", SEVERITY_CRITICAL, "Importación del módulo de seguridad protegido"),
    (r"\b(AutoRun|Startup|Run)\b[^\n]*?\b(reg|shell:startup)\b", SEVERITY_HIGH, "Persistencia en el arranque"),
)

# Comandos cuyo uso SÍ está autorizado dentro del proyecto (allowlist interna).
ALLOWED_EXECUTABLES: frozenset[str] = frozenset(
    {
        "python", "python3", "py", "pip", "pytest", "git", "ollama",
        "cmd", "powershell", "pyinstaller",
    }
)

# Nombres de paquete PyPI válidos (PEP 503 simplificado).
PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")
PACKAGE_SPEC_RE = re.compile(
    r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?\s*([<>=!~]=?\s*[A-Za-z0-9._*+!-]+)?(\s*,\s*[<>=!~]=?\s*[A-Za-z0-9._*+!-]+)*$"
)

# Paquetes vetados: nada que abra puertas traseras o telemetría encubierta.
FORBIDDEN_PACKAGES: frozenset[str] = frozenset(
    {
        "subprocess32",  # obsoleto y sustituible
        "pycrypto",      # abandonado, con CVEs graves
        "request",       # confusión de nombres con `requests`
        "python3-pip",
        "pip3",
        "colourama",     # typosquatting de colorama
        "djanga",        # typosquatting de django
        "openzeppelin",
    }
)

# Fragmentos que delatan código generado que ataca la capa de seguridad.
CODE_RED_FLAGS: tuple[tuple[str, str], ...] = (
    (r"safety\s*/\s*killswitch|killswitch\.py", "Referencia al kill-switch protegido"),
    (r"safety\s*/\s*command_guard|command_guard", "Referencia al guardián de comandos"),
    (r"\.sha256\b", "Manipulación del archivo de integridad"),
    (r"os\.remove\s*\(\s*__file__", "Autodestrucción del propio código"),
    (r"shutil\.rmtree\s*\(\s*[\"']?\s*(/|C:\\\\|\"\"|\.\.)", "Borrado recursivo peligroso"),
    (r"subprocess\.[a-z]+\([^)]*shell\s*=\s*True", "Uso de shell=True (inyección de comandos)"),
    (r"os\.system\s*\(", "os.system (sin control de argumentos)"),
    (r"eval\s*\(\s*(requests|urllib|urlopen|open\()", "Evaluación de contenido externo"),
    (r"exec\s*\(\s*(requests|urllib|urlopen)", "Ejecución de código descargado"),
    (r"ctypes\.windll|VirtualAllocEx|WriteProcessMemory", "Inyección de memoria en otros procesos"),
    (r"socket\.socket\s*\(", "Apertura de sockets a bajo nivel"),
    (r"base64\.b64decode\s*\([^)]{40,}", "Carga útil codificada oculta"),
    (r"while\s+True\s*:\s*pass", "Bucle infinito que consume CPU"),
    (r"keyboard\.(write|press|hook)\s*\(", "Captura o inyección global de teclado"),
)


# =============================================================================
# 3. AUDITORÍA DE COMANDOS
# =============================================================================

def _normalize(command: str) -> str:
    """Normaliza un comando para que los patrones no se burlen del guardián.

    * Pasa todo a minúsculas.
    * Convierte comillas y barras escapadas tipo ``k"i"llswitch`` en su forma
      simple, ya que los patrones no deben poder ser evadidos con trucos.
    * Colapsa espacios repetidos.
    """
    text = command.strip().lower()
    text = text.replace("^", "")                      # escudos de cmd.exe
    text = re.sub(r"[\"'`]", "", text)                # comillas de todo tipo
    text = re.sub(r"(\\\\)+", r"\\", text)
    text = re.sub(r"\s+", " ", text)
    return text


def inspect_command(command: str, trust: str = TRUST_GENERATED) -> CommandVerdict:
    """Audita un comando y devuelve un veredicto motivado.

    Parameters
    ----------
    command:
        Comando completo (cadena) tal y como se va a ejecutar.
    trust:
        ``TRUST_GENERATED`` (por defecto) aplica TODAS las reglas.
        ``TRUST_INTERNAL`` solo aplica las reglas críticas, porque el comando
        lo ha escrito el propio JARVIS (por ejemplo ``git reset --hard`` para
        revertir una fusión fallida).
    """
    if not isinstance(command, str) or not command.strip():
        return CommandVerdict(False, SEVERITY_HIGH, "Comando vacío", "", command or "")

    normalized = _normalize(command)

    for pattern, severity, reason in FORBIDDEN_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            verdict = CommandVerdict(False, severity, reason, pattern, command)
            logger.error("Guardián: %s", verdict.describe())
            return verdict

    # Rutas absolutas hacia directorios del sistema operativo.
    for token in shlex.split(command.replace("\\", "\\\\")) if " " in command else [command]:
        cleaned = token.strip("\"'")
        if cleaned.startswith(("/etc/", "/usr/", "/bin/", "/boot/", "/var/", "/dev/", "/sys/")):
            verdict = CommandVerdict(False, SEVERITY_CRITICAL, f"Ruta de sistema: {cleaned}", "system-path", command)
            logger.error("Guardián: %s", verdict.describe())
            return verdict

    if trust == TRUST_GENERATED:
        for pattern, severity, reason in RISKY_PATTERNS:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                verdict = CommandVerdict(False, severity, reason, pattern, command)
                logger.warning("Guardián (código generado): %s", verdict.describe())
                return verdict

    return CommandVerdict(True, "", "Sin patrones peligrosos", "", command)


def is_command_safe(command: str, trust: str = TRUST_GENERATED) -> bool:
    """Atajo booleano de `inspect_command`."""
    return inspect_command(command, trust=trust).allowed


def assert_command_safe(command: str, trust: str = TRUST_GENERATED) -> CommandVerdict:
    """Como `inspect_command`, pero lanza `CommandGuardError` si se bloquea."""
    verdict = inspect_command(command, trust=trust)
    if not verdict.allowed:
        raise CommandGuardError(verdict.describe())
    return verdict


def validate_command_list(argv: Sequence[str], trust: str = TRUST_GENERATED) -> CommandVerdict:
    """Valida una lista de argumentos (modo seguro, ``shell=False``).

    Además de los patrones, comprueba que el ejecutable esté en la allowlist
    cuando el origen es código generado.
    """
    if not argv:
        return CommandVerdict(False, SEVERITY_HIGH, "Lista de argumentos vacía", "", "")
    executable = Path(str(argv[0])).name.lower()
    stem = executable[:-4] if executable.endswith(".exe") else executable
    if trust == TRUST_GENERATED and stem not in ALLOWED_EXECUTABLES:
        return CommandVerdict(
            False, SEVERITY_HIGH, f"Ejecutable no autorizado: {executable}", "allowlist",
            " ".join(str(a) for a in argv),
        )
    return inspect_command(" ".join(str(a) for a in argv), trust=trust)


def validate_package_name(name: str) -> str:
    """Valida y normaliza el nombre de un paquete PyPI.

    Rechaza typosquatting evidente, separadores de ruta, banderas de pip y
    cualquier carácter fuera de PEP 503. Devuelve el nombre limpio.
    """
    if not isinstance(name, str):
        raise CommandGuardError("El nombre del paquete debe ser texto.")
    clean = name.strip()
    if not clean:
        raise CommandGuardError("Nombre de paquete vacío.")
    if clean.startswith("-"):
        raise CommandGuardError(f"Banderas de pip disfrazadas de paquete: {clean}")
    if not PACKAGE_SPEC_RE.match(clean):
        raise CommandGuardError(f"Nombre de paquete inválido: {clean!r}")
    base = re.split(r"[<>=!~]", clean, maxsplit=1)[0].strip().lower()
    if base in FORBIDDEN_PACKAGES:
        raise CommandGuardError(f"Paquete vetado por seguridad: {base}")
    return clean


def validate_dependency_list(dependencies: Iterable[str], limit: int = 12) -> list[str]:
    """Valida una lista completa de dependencias y devuelve la lista limpia."""
    clean: list[str] = []
    for dep in dependencies or ():
        if len(clean) >= limit:
            raise CommandGuardError(f"Demasiadas dependencias (máximo {limit}).")
        clean.append(validate_package_name(str(dep)))
    return clean


# =============================================================================
# 4. CONFINAMIENTO DE RUTAS
# =============================================================================

def is_protected_path(path: str | os.PathLike[str]) -> bool:
    """¿La ruta apunta a un activo protegido de la capa de seguridad?"""
    import config  # Importación tardía para evitar ciclos en tests aislados.

    text = str(path).replace("\\", "/").lower()
    for protected in config.PROTECTED_PATHS:
        if text.endswith(protected.lower()):
            return True
    return any(fragment in text for fragment in config.PROTECTED_NAME_FRAGMENTS)


def resolve_within(root: str | os.PathLike[str], relative: str | os.PathLike[str]) -> Path:
    """Resuelve ``relative`` dentro de ``root`` y garantiza que NO se escapa.

    Bloquea rutas absolutas, unidades de Windows, ``..``, rutas UNC y enlaces
    simbólicos que apunten fuera del sandbox. Devuelve la ruta absoluta.
    """
    root_path = Path(root).resolve()
    raw = str(relative).strip().replace("\\", "/")

    if not raw:
        raise UnsafePathError("Ruta vacía.")
    if raw.startswith(("~", "$", "%")):
        raise UnsafePathError(f"Ruta con variable de entorno no permitida: {relative!r}")
    if raw.startswith("//") or raw.startswith("\\\\"):
        raise UnsafePathError(f"Ruta de red (UNC) no permitida: {relative!r}")
    if re.match(r"^[a-zA-Z]:", raw):
        raise UnsafePathError(f"Ruta absoluta de Windows no permitida: {relative!r}")
    if raw.startswith("/") or raw.startswith("\\"):
        raise UnsafePathError(f"Ruta absoluta no permitida: {relative!r}")
    if any(part == ".." for part in raw.split("/")):
        raise UnsafePathError(f"Ruta con '..' no permitida: {relative!r}")
    if is_protected_path(raw):
        raise UnsafePathError(f"Ruta protegida por seguridad: {relative!r}")

    candidate = (root_path / raw).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise UnsafePathError(
            f"La ruta {relative!r} queda fuera del espacio permitido {root_path}."
        ) from exc
    return candidate


def sanitize_project_name(name: str, max_length: int = 48) -> str:
    """Convierte un nombre hablado en un nombre de carpeta seguro y legible.

    >>> sanitize_project_name("¡Mi Súper Web de Recetas!")
    'mi-super-web-de-recetas'
    >>> sanitize_project_name("../../etc/passwd")
    'etc-passwd'
    """
    import unicodedata

    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    text = text[:max_length].strip("-")
    if not text or text in {"con", "prn", "aux", "nul", "com1", "lpt1"}:
        text = "proyecto"
    return text


def sanitize_relative_file_path(relative: str, max_depth: int = 4) -> str:
    """Limpia una ruta relativa de archivo generada por un modelo.

    Devuelve una ruta POSIX relativa y segura, con extensión obligatoria.
    """
    raw = str(relative).strip().replace("\\", "/")
    raw = re.sub(r"^\./+", "", raw)
    parts = [sanitize_project_name(p, max_length=64) if "." not in p else p for p in raw.split("/")]
    parts = []
    for piece in raw.split("/"):
        if piece == "..":
            # Un modelo que intenta salir de la carpeta no merece confianza:
            # se rechaza en lugar de recolocar el archivo en silencio.
            raise UnsafePathError(f"Ruta con salto de carpeta: {relative!r}")
        if not piece or piece == ".":
            continue
        safe_piece = re.sub(r"[^A-Za-z0-9._-]", "_", piece)
        safe_piece = re.sub(r"_{2,}", "_", safe_piece).strip("._-")
        if safe_piece:
            parts.append(safe_piece)
    if not parts:
        raise UnsafePathError(f"Ruta de archivo inválida: {relative!r}")
    if len(parts) > max_depth:
        parts = parts[-max_depth:]
    final = "/".join(parts)
    if final.startswith(".") or is_protected_path(final):
        raise UnsafePathError(f"Ruta reservada: {final}")
    if "." not in parts[-1]:
        final += ".txt"
    return final


# =============================================================================
# 5. AUDITORÍA DE CÓDIGO GENERADO
# =============================================================================

def audit_generated_code(code: str, filename: str = "<generado>") -> CodeAudit:
    """Inspecciona código propuesto por un modelo antes de escribirlo a disco.

    Detecta referencias a la capa de seguridad, borrados recursivos, uso de
    ``shell=True``, ejecución de código descargado, inyección de memoria y
    cargas útiles codificadas. Devuelve un informe que el motor de
    auto-programación usa como veto.
    """
    findings: list[str] = []
    if not isinstance(code, str):
        return CodeAudit(False, ["Contenido no textual."])

    for pattern, reason in CODE_RED_FLAGS:
        if re.search(pattern, code, flags=re.IGNORECASE):
            findings.append(f"{reason} (línea {_first_line(code, pattern)})")

    # Rutas de escritura hacia fuera del proyecto.
    for match in re.finditer(r"open\s*\(\s*[\"']([^\"']+)[\"']\s*,\s*[\"'][wa]", code):
        target = match.group(1)
        if is_protected_path(target) or target.startswith(("/", "C:\\", "..", "~")):
            findings.append(f"Escritura en ruta no permitida: {target}")

    if filename and is_protected_path(filename):
        findings.append(f"El nombre de archivo '{filename}' está protegido.")

    safe = not findings
    if not safe:
        logger.warning("Auditoría de código FALLIDA para %s: %s", filename, "; ".join(findings))
    return CodeAudit(safe, findings)


def _first_line(code: str, pattern: str) -> int:
    """Número de línea (1-based) de la primera coincidencia del patrón."""
    regex = re.compile(pattern, flags=re.IGNORECASE)
    for index, line in enumerate(code.splitlines(), start=1):
        if regex.search(line):
            return index
    return 0


# =============================================================================
# 6. EJECUCIÓN AUDITADA DE SUBPROCESOS
# =============================================================================

def build_safe_env(base: dict | None = None, extra: dict | None = None) -> dict:
    """Construye un entorno sin secretos ni trucos de inyección de rutas.

    Se eliminan ``PYTHONPATH``, ``PYTHONSTARTUP`` y ``PYTHONHOME`` para evitar
    que código generado secuestre el intérprete del proyecto principal.
    """
    env = dict(base if base is not None else os.environ)
    for key in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME", "PYTHONWARNINGS"):
        env.pop(key, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["JARVIS_SANDBOX"] = "1"
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def run_guarded(
    argv: Sequence[str],
    cwd: str | os.PathLike[str] | None = None,
    timeout: float = 120.0,
    trust: str = TRUST_INTERNAL,
    extra_env: dict | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Ejecuta un subproceso con todas las defensas activas.

    * ``shell=False`` siempre (nada de interpolación de cadenas en el shell).
    * Validación previa del ejecutable y de los argumentos.
    * Entorno saneado y directorio de trabajo confinado al proyecto.
    * Tiempo máximo de ejecución garantizado.

    Lanza ``CommandGuardError`` si la auditoría falla, y
    ``subprocess.TimeoutExpired`` si se agota el tiempo.
    """
    if isinstance(argv, str):
        raise CommandGuardError(
            "run_guarded() exige una lista de argumentos, nunca una cadena de shell."
        )
    args = [str(a) for a in argv]
    verdict = validate_command_list(args, trust=trust)
    if not verdict.allowed:
        raise CommandGuardError(verdict.describe())

    workdir: Path | None = None
    if cwd is not None:
        workdir = Path(cwd).resolve()
        if not workdir.is_dir():
            raise CommandGuardError(f"El directorio de trabajo no existe: {workdir}")

    logger.debug("Ejecutando (auditado): %s", " ".join(args))
    try:
        return subprocess.run(  # noqa: S603 - auditoría previa obligatoria
            args,
            cwd=str(workdir) if workdir else None,
            env=build_safe_env(extra=extra_env),
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=check,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0,
        )
    except subprocess.TimeoutExpired:
        logger.error("El comando superó el tiempo máximo de %.0f s: %s", timeout, " ".join(args))
        raise
    except FileNotFoundError as exc:
        logger.error("Ejecutable no encontrado: %s", args[0])
        raise CommandGuardError(f"No se encontró el ejecutable '{args[0]}'.") from exc


def shell_quote(command: str) -> str:
    """Comillas seguras para mostrar comandos en logs o interfaces."""
    return shlex.quote(command)


def is_protected_write(target: str | os.PathLike[str]) -> bool:
    """¿Se está intentando escribir sobre un archivo inmutable?"""
    return is_protected_path(target)
