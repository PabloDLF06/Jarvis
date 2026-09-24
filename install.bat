@echo off
rem ============================================================================
rem  JARVIS - Instalacion en un solo clic
rem ============================================================================
rem  Este archivo deja JARVIS listo para usar sin conocimientos tecnicos:
rem    1. Comprueba que Python este instalado.
rem    2. Crea un entorno aislado (.venv) e instala las librerias.
rem    3. Comprueba Ollama y descarga los modelos que falten.
rem    4. Descarga las voces y el modelo de transcripcion.
rem    5. Sella el interruptor de emergencia (kill-switch).
rem
rem  Uso: haz doble clic sobre install.bat y espera a que termine.
rem ============================================================================

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title JARVIS - Instalacion
cd /d "%~dp0"

set "PASO_OK=  [OK]"
set "PASO_AVISO=  [!]"
set "PASO_FALLO=  [X]"

echo.
echo ============================================================================
echo    J A R V I S   -   Instalacion automatica
echo    Asistente de escritorio local, privado y gratuito
echo ============================================================================
echo.
echo  Todo se instala en esta carpeta. Nada sale de tu ordenador.
echo.

rem ---------------------------------------------------------------------------
rem  PASO 0 - Comprobaciones previas del sistema
rem ---------------------------------------------------------------------------
echo --------------------------------------------------------------------------
echo  PASO 0 de 5 . Comprobando el sistema
echo --------------------------------------------------------------------------
echo.

ver | findstr /i "10\.0" >nul
if errorlevel 1 (
    echo %PASO_AVISO% Este asistente esta pensado para Windows 10 u 11.
)

set "PY_CMD="
where py >nul 2>nul && set "PY_CMD=py -3"
if not defined PY_CMD (
    where python >nul 2>nul && set "PY_CMD=python"
)
if not defined PY_CMD (
    echo %PASO_FALLO% No encuentro Python en este ordenador.
    echo.
    echo      Python es el motor que hace funcionar a JARVIS.
    echo      1^) Se abrira la pagina de descarga.
    echo      2^) Descarga "Windows installer ^(64-bit^)".
    echo      3^) MUY IMPORTANTE: marca la casilla "Add python.exe to PATH".
    echo      4^) Vuelve a ejecutar install.bat.
    echo.
    start "" "https://www.python.org/downloads/windows/"
    pause
    exit /b 1
)

%PY_CMD% -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    %PY_CMD% -c "import sys; print('      Version detectada: ' + sys.version.split()[0])"
    echo %PASO_FALLO% Necesito Python 3.10 o superior ^(recomendado 3.11^).
    echo      Descarga la version mas reciente en https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

%PY_CMD% -c "import sys; print('      Python ' + sys.version.split()[0] + ' encontrado: ' + sys.executable)"
echo %PASO_OK% Sistema compatible.
echo.

rem ---------------------------------------------------------------------------
rem  PASO 1 - Entorno aislado e instalacion de librerias
rem ---------------------------------------------------------------------------
echo --------------------------------------------------------------------------
echo  PASO 1 de 5 . Preparando el entorno y las librerias
echo --------------------------------------------------------------------------
echo.

if not exist ".venv\Scripts\python.exe" (
    echo   Creando entorno aislado ^(.venv^)... puede tardar un minuto.
    %PY_CMD% -m venv .venv
    if errorlevel 1 (
        echo %PASO_FALLO% No pude crear el entorno virtual.
        echo      Prueba a ejecutar este archivo como administrador.
        pause
        exit /b 1
    )
) else (
    echo   El entorno aislado ya existia: se reutiliza.
)

set "VPY=%CD%\.venv\Scripts\python.exe"

"%VPY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
if errorlevel 1 (
    echo %PASO_AVISO% No pude actualizar pip, continuo con la version instalada.
)

echo.
echo   Instalando librerias (audio, vision, interfaz, modelos)...
echo   Esto tarda unos minutos la primera vez. Es normal.
echo.
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo %PASO_FALLO% Algo fallo instalando las librerias.
    echo      Revisa tu conexion a internet y vuelve a ejecutar install.bat.
    pause
    exit /b 1
)
echo %PASO_OK% Librerias instaladas.
echo.

rem ---------------------------------------------------------------------------
rem  PASO 2 - Ollama y los modelos de lenguaje
rem ---------------------------------------------------------------------------
echo --------------------------------------------------------------------------
echo  PASO 2 de 5 . Ollama y los modelos de lenguaje
echo --------------------------------------------------------------------------
echo.

where ollama >nul 2>nul
if errorlevel 1 (
    echo %PASO_AVISO% Ollama no esta instalado.
    echo.
    echo      Ollama es el programa que ejecuta los modelos de lenguaje
    echo      en TU ordenador, sin enviar nada a internet.
    echo      1^) Se abrira la pagina de descarga.
    echo      2^) Instala Ollama con las opciones por defecto.
    echo      3^) Vuelve a ejecutar install.bat.
    echo.
    start "" "https://ollama.com/download/windows"
    pause
    exit /b 1
)
echo %PASO_OK% Ollama instalado.

ollama list >nul 2>nul
if errorlevel 1 (
    echo   Arrancando el servicio de Ollama...
    start "Ollama" /min ollama serve
    timeout /t 6 /nobreak >nul
    ollama list >nul 2>nul
    if errorlevel 1 (
        echo %PASO_AVISO% No consigo hablar con Ollama. Abre la aplicacion Ollama
        echo      desde el menu Inicio y vuelve a ejecutar install.bat.
        pause
        exit /b 1
    )
)

set "MODELOS=llama3.1:8b llama3.2-vision:latest qwen2.5-coder:7b"
for %%M in (%MODELOS%) do (
    ollama list | findstr /i /c:"%%M" >nul
    if errorlevel 1 (
        echo   Descargando %%M ... ^(puede tardar varios minutos^)
        ollama pull %%M
        if errorlevel 1 (
            echo %PASO_AVISO% No pude descargar %%M. JARVIS lo intentara de nuevo al usarlo.
        ) else (
            echo %PASO_OK% Modelo %%M listo.
        )
    ) else (
        echo %PASO_OK% Modelo %%M ya estaba descargado.
    )
)
echo.

rem ---------------------------------------------------------------------------
rem  PASO 3 - Oido y voz (transcripcion, palabra clave y voces espanolas)
rem ---------------------------------------------------------------------------
echo --------------------------------------------------------------------------
echo  PASO 3 de 5 . Oido y voz de JARVIS
echo --------------------------------------------------------------------------
echo.

echo   Descargando el modelo de transcripcion ^(faster-whisper base^)...
"%VPY%" -c "import config; from faster_whisper import WhisperModel; WhisperModel(config.STT_MODEL, download_root=config.STT_DOWNLOAD_ROOT); print('      Transcripcion lista.')"
if errorlevel 1 echo %PASO_AVISO% No pude preparar la transcripcion; JARVIS lo hara al primer uso.

echo   Descargando las voces de Kokoro ^(espanol^)...
"%VPY%" -c "import config, pathlib, urllib.request; d = pathlib.Path(config.TTS_MODELS_DIR); d.mkdir(parents=True, exist_ok=True); [urllib.request.urlretrieve(u, str(d / n)) for u, n in ((config.TTS_KOKORO_MODEL_URL, config.TTS_KOKORO_MODEL_FILE), (config.TTS_KOKORO_VOICES_URL, config.TTS_KOKORO_VOICES_FILE)) if not (d / n).exists()]; print('      Voces listas.')"
if errorlevel 1 echo %PASO_AVISO% No pude descargar las voces; JARVIS usara el sintetizador de Windows.

echo   Descargando la palabra clave "Jarvis"...
"%VPY%" -c "import config; from openwakeword.utils import download_models; download_models(model_names=[config.WAKE_WORD_MODEL], target_directory=str(config.WAKE_MODELS_DIR)); print('      Palabra clave lista.')"
if errorlevel 1 echo %PASO_AVISO% La palabra clave se descargara en el primer arranque.
echo.

rem ---------------------------------------------------------------------------
rem  PASO 4 - Interruptor de emergencia (kill-switch)
rem ---------------------------------------------------------------------------
echo --------------------------------------------------------------------------
echo  PASO 4 de 5 . Sellando el interruptor de emergencia
echo --------------------------------------------------------------------------
echo.

"%VPY%" safety\killswitch.py --sellar
if errorlevel 1 (
    echo %PASO_AVISO% No pude sellar el kill-switch. JARVIS lo hara en el primer arranque.
) else (
    echo %PASO_OK% Kill-switch verificado y protegido.
)
echo.

rem ---------------------------------------------------------------------------
rem  PASO 5 - Comprobacion final
rem ---------------------------------------------------------------------------
echo --------------------------------------------------------------------------
echo  PASO 5 de 5 . Comprobacion final
echo --------------------------------------------------------------------------
echo.

"%VPY%" main.py --diagnostico --sin-hud
if errorlevel 1 (
    echo %PASO_AVISO% El diagnostico avisa de algo pendiente. Revisa los mensajes de arriba.
)
echo.

echo ============================================================================
echo    INSTALACION TERMINADA
echo ============================================================================
echo.
echo    Para usar JARVIS, haz doble clic en:  start.bat
echo.
echo    Trucos rapidos:
echo      - Di "Jarvis" y despues tu orden.
echo      - Da DOS palmadas para poner musica y recibir el saludo.
echo      - CTRL + SHIFT + ESPACIO detiene todo al instante.
echo.
pause
endlocal
