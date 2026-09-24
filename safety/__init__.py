"""
safety/__init__.py — Capa de seguridad de JARVIS.

Reúne el kill-switch inmutable y el guardián de comandos. Ambos módulos son
deliberadamente independientes de PyQt6, Ollama o cualquier servicio: deben
funcionar aunque el resto del sistema esté degradado.
"""

from __future__ import annotations

from . import command_guard
from .command_guard import (
    CommandGuardError,
    CommandVerdict,
    CodeAudit,
    UnsafePathError,
    assert_command_safe,
    audit_generated_code,
    inspect_command,
    is_command_safe,
    is_protected_path,
    resolve_within,
    run_guarded,
    sanitize_project_name,
    sanitize_relative_file_path,
    validate_dependency_list,
    validate_package_name,
)

__all__ = [
    "command_guard",
    "CommandGuardError",
    "CommandVerdict",
    "CodeAudit",
    "UnsafePathError",
    "assert_command_safe",
    "audit_generated_code",
    "inspect_command",
    "is_command_safe",
    "is_protected_path",
    "resolve_within",
    "run_guarded",
    "sanitize_project_name",
    "sanitize_relative_file_path",
    "validate_dependency_list",
    "validate_package_name",
]


def load_killswitch():
    """Carga `safety.killswitch` de forma perezosa.

    Se hace así porque el kill-switch instala ganchos globales de teclado y
    ratón: solo debe importarse cuando JARVIS arranca de verdad, nunca al
    inspeccionar el paquete (por ejemplo, durante los tests).
    """
    from . import killswitch

    return killswitch
