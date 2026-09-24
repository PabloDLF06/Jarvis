"""
tests/conftest.py — Utilidades compartidas por las pruebas.
================================================================================
Aquí viven los dobles de prueba que hacen posible probar JARVIS sin hardware:

* `FakeOllama`   — cliente de Ollama simulado (modelos, VRAM, descargas).
* `FakeRouter`   — enrutador simulado que devuelve código o texto preparado.
* `FakeCapture`  — captura de pantalla simulada (sin ratón ni monitores).
* `fake_clock`   — reloj controlado a mano para los detectores de audio.
* `tmp_git_repo` — repositorio git temporal para el auto-programador.
================================================================================
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("JARVIS_TESTING", "1")

import config as cfg  # noqa: E402  (tras el bootstrap de rutas)


# =============================================================================
# DOBLES DE PRUEBA
# =============================================================================

class FakeOllama:
    """Cliente de Ollama simulado: reproduce modelos, VRAM y descargas."""

    def __init__(
        self,
        models: tuple[str, ...] = (cfg.MODEL_CHAT, cfg.MODEL_VISION, cfg.MODEL_CODER),
        loaded: tuple[str, ...] = (),
        vram_per_model: float = 2.5,
        alive: bool = True,
    ) -> None:
        self.models = set(models)
        self.loaded = list(loaded)
        self.vram_per_model = float(vram_per_model)
        self.alive = alive
        self.host = "http://fake-ollama"
        self.unloaded: list[str] = []
        self.pulled: list[str] = []
        self.responses: dict[str, str] = {}

    # --- Estado -----------------------------------------------------------

    def health(self, timeout: float | None = None) -> bool:
        return self.alive

    def require_running(self) -> None:
        if not self.alive:
            from core.ollama_client import OllamaNotRunningError

            raise OllamaNotRunningError(self.host)

    def version(self) -> str:
        return "0.0.0-fake"

    def list_models(self, use_cache: bool = True, ttl_s: float = 20.0) -> list[dict]:
        return [{"name": name} for name in sorted(self.models)]

    def has_model(self, model: str) -> bool:
        return model in self.models

    def resolve_model_name(self, model: str) -> str:
        return model

    def missing_models(self, models=None) -> list[str]:
        return [m for m in (models or cfg.MODELS_TO_AUTO_PULL) if m not in self.models]

    def pull(self, model: str, progress_cb=None) -> bool:
        self.models.add(model)
        self.pulled.append(model)
        return True

    # --- Modelos cargados y VRAM -----------------------------------------

    def ps(self):
        from core.ollama_client import LoadedModel

        return [
            LoadedModel(
                name=name,
                size_bytes=int(self.vram_per_model * (1024 ** 3)),
                size_vram_bytes=int(self.vram_per_model * (1024 ** 3)),
            )
            for name in self.loaded
        ]

    def loaded_names(self) -> list[str]:
        return list(self.loaded)

    def vram_in_use_gb(self) -> float:
        return len(self.loaded) * self.vram_per_model

    def unload(self, model: str, wait: bool = True) -> bool:
        if model in self.loaded:
            self.loaded.remove(model)
            self.unloaded.append(model)
        return True

    def unload_all(self, keep: str | None = None) -> list[str]:
        freed = [name for name in self.loaded if name != keep]
        self.loaded = [keep] if keep else []
        self.unloaded.extend(freed)
        return freed

    def wait_until_unloaded(self, model: str, timeout: float = 5.0) -> bool:
        return model not in self.loaded

    def warmup(self, model: str) -> bool:
        if model not in self.loaded:
            self.loaded.append(model)
        return True

    # --- Inferencia -------------------------------------------------------

    def _result(self, model: str, prompt_key: str = ""):
        from core.ollama_client import ChatResult

        text = self.responses.get(prompt_key or model, "Respuesta simulada de JARVIS.")
        return ChatResult(text=text, model=model, output_tokens=len(text.split()), duration_s=0.01)

    def chat(self, model: str, messages, options=None, keep_alive=None, stream=False,
             on_token=None, timeout=None, format_json=False, tools=None):
        return self._result(model, str(messages[-1].get("content", "")) if messages else "")

    def generate(self, model: str, prompt: str, images=None, options=None, keep_alive=None,
                 stream=False, system=None, timeout=None, format_json=False):
        return self._result(model, prompt)

    def close(self) -> None:
        pass


class FakeRouter:
    """Enrutador simulado: entrega el texto que se le haya configurado."""

    def __init__(self, code_text: str = "", chat_text: str = "Respuesta simulada.",
                 plan_payload: dict | None = None) -> None:
        self.code_text = code_text
        self.chat_text = chat_text
        self.plan_payload = plan_payload or {"objetivo": "demo", "pasos": [], "confianza": 0.9}
        self.prompts: list[str] = []

    class _Client:
        def __init__(self) -> None:
            self.host = "http://fake"

        def has_model(self, _name: str) -> bool:
            return True

        def health(self) -> bool:
            return True

    @property
    def client(self) -> "FakeRouter._Client":
        return self._Client()

    def write_code(self, instruction: str, context: str = "", json_mode: bool = True) -> str:
        self.prompts.append(instruction)
        return self.code_text

    def chat(self, prompt: str, system=None, history=None, on_token=None, stream=False) -> str:
        self.prompts.append(prompt)
        return self.chat_text

    def plan(self, order: str, screen_context: str = "") -> dict:
        return dict(self.plan_payload, objetivo=order)

    def look_at_screen(self, image_path, question: str = "") -> str:
        return "Veo el escritorio con el navegador abierto."

    def find_ui_elements(self, image_path, description: str) -> dict:
        return {
            "elementos": [
                {"nombre": "boton azul", "descripcion": description, "x": 500, "y": 250,
                 "ancho": 100, "alto": 40, "confianza": 0.9}
            ],
            "encontrado": True,
        }


class FakeCapture:
    """Sustituto de `ScreenCapture`: crea un JPEG sintético sin tocar la pantalla."""

    def __init__(self, width: int = 1600, height: int = 900, scale: float = 0.5) -> None:
        self.width = width
        self.height = height
        self.scale = scale
        self.stats = {"capturas": 0, "segundos": 0.0}

    def capture(self, monitor: int = 0, force: bool = False):
        from core.vision_actuator import ScreenShot

        self.stats["capturas"] += 1
        return ScreenShot(
            path=Path("captura_falsa.jpg"),
            width=int(self.width * self.scale),
            height=int(self.height * self.scale),
            scale=self.scale,
            monitor=monitor,
        )


class FakeClock:
    """Reloj controlado a mano (segundos), para los detectores de audio."""

    def __init__(self, start: float = 1000.0) -> None:
        self.value = float(start)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> float:
        self.value += float(seconds)
        return self.value


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fake_ollama() -> FakeOllama:
    return FakeOllama()


@pytest.fixture
def fake_router() -> FakeRouter:
    return FakeRouter()


@pytest.fixture
def fake_capture() -> FakeCapture:
    return FakeCapture()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def speech_frame() -> np.ndarray:
    """Trama de voz sintética (ruido con envolvente suave, 320 muestras)."""
    rng = np.random.default_rng(7)
    return (rng.normal(0, 0.18, cfg.AUDIO_FRAME_SAMPLES).clip(-1, 1) * 32767).astype(np.int16)


@pytest.fixture
def silence_frame() -> np.ndarray:
    """Trama de silencio con un mínimo de ruido de fondo."""
    rng = np.random.default_rng(11)
    return (rng.normal(0, 0.0005, cfg.AUDIO_FRAME_SAMPLES).clip(-1, 1) * 32767).astype(np.int16)


@pytest.fixture
def clap_frame() -> np.ndarray:
    """Trama que imita una palmada: ruido blanco fuerte y corto (20 ms)."""
    rng = np.random.default_rng(3)
    return (rng.uniform(-0.6, 0.6, cfg.AUDIO_FRAME_SAMPLES) * 32767).astype(np.int16)


@pytest.fixture
def tone_frame() -> np.ndarray:
    """Trama tonal (silbido): espectralmente estrecha, no debe parecer una palmada."""
    t = np.arange(cfg.AUDIO_FRAME_SAMPLES) / cfg.AUDIO_SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)


@pytest.fixture
def tmp_git_repo(tmp_path: Path) -> Path:
    """Repositorio git temporal con la rama `main` y un archivo inicial."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = shutil.which("git")
    assert git, "Se necesita git para esta prueba"
    _git(git, repo, "-c", "init.defaultBranch=main", "init", "-q")
    (repo / "README.md").write_text("# Proyecto de prueba\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_base.py").write_text("def test_base():\n    assert True\n", encoding="utf-8")
    _git(git, repo, "add", "-A")
    _git(git, repo, "-c", "user.name=Pruebas", "-c", "user.email=pruebas@local", "commit", "-q", "-m", "inicial")
    _git(git, repo, "branch", "-M", "main")
    return repo


def _git(git: str, cwd: Path, *args: str) -> str:
    """Ejecuta git en las pruebas (aquí sí se permite subprocess directo)."""
    result = subprocess.run(  # noqa: S603 - solo en la suite de pruebas
        [git, *args], cwd=str(cwd), capture_output=True, text=True, timeout=60, check=False
    )
    return result.stdout.strip() + result.stderr.strip()
