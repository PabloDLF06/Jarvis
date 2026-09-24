@echo off
rem ============================================================================
rem  JARVIS - Arranque en un solo clic
rem ============================================================================
rem  Arranca JARVIS en segundo plano con la interfaz flotante (HUD) activada.
rem
rem  Uso normal:          doble clic en start.bat
rem  Modo texto:          start.bat --texto          (escribes las ordenes)
rem  Sin interfaz:        start.bat --sin-hud
rem  Diagnostico:         start.bat --diagnostico
rem  Detener JARVIS:      start.bat --matar
rem ============================================================================

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title JARVIS

set "VPY=%CD%\.venv\Scripts\python.exe"
set "VPYW=%CD%\.venv\Scripts\pythonw.exe"

if not exist "%VPY%" (
    echo.
    echo   Todavia no esta instalado. Ejecuta primero:  install.bat
    echo.
    pause
    exit /b 1
)

if not exist "%VPYW%" set "VPYW=%VPY%"

rem --- Parada limpia de una instancia en marcha -------------------------------
if /i "%~1"=="--matar" (
    echo   Deteniendo JARVIS...
    "%VPY%" main.py --matar
    pause
    exit /b 0
)

rem --- Modo consola: se queda en esta ventana para poder escribir -------------
if /i "%~1"=="--texto" (
    echo.
    echo   JARVIS en modo texto. Escribe tus ordenes y pulsa Enter.
    echo.
    "%VPY%" main.py %*
    pause
    exit /b 0
)

rem --- Diagnostico y utilidades de un solo uso --------------------------------
if /i "%~1"=="--diagnostico"  goto :directo
if /i "%~1"=="--verificar-seguridad" goto :directo
if /i "%~1"=="--sellar-seguridad"    goto :directo
if /i "%~1"=="--prueba-voz"          goto :directo
if /i "%~1"=="--prueba-palmada"      goto :directo
if /i "%~1"=="--version"             goto :directo

rem --- Arranque normal: en segundo plano, con HUD, sin ventana negra ---------
echo.
echo   Arrancando JARVIS...
echo.
echo   - La capsula de estado aparece arriba a la derecha.
echo   - Di "Jarvis" y despues tu orden.
echo   - Dos palmadas: musica y saludo.
echo   - CTRL + SHIFT + ESPACIO detiene todo al instante.
echo.

start "JARVIS" "%VPYW%" main.py %*

timeout /t 3 /nobreak >nul 2>nul
tasklist /fi "imagename eq pythonw.exe" 2>nul | findstr /i "pythonw.exe" >nul
if errorlevel 1 (
    echo   Aviso: JARVIS no parece haber arrancado.
    echo   Revisa el archivo de registro: logs\jarvis.log
    echo   O ejecuta "start.bat --diagnostico" para ver que ocurre.
    echo.
    pause
    exit /b 1
)

echo   JARVIS esta en marcha. Puedes cerrar esta ventana.
echo.
timeout /t 2 /nobreak >nul 2>nul
exit /b 0

:directo
"%VPY%" main.py %*
pause
exit /b 0
