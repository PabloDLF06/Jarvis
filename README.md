<div align="center">

# J A R V I S

**Asistente de escritorio local, privado y gratuito**

Escucha, habla, ve tu pantalla, programa por ti y no envía ni un byte a internet.

*Hecho a medida para Pablo · Ryzen 7 7840HS · RTX 4060 8 GB · 16 GB DDR5*

</div>

---

## Índice

1. [Qué es JARVIS y qué lo hace diferente](#1-qué-es-jarvis-y-qué-lo-hace-diferente)
2. [Instalación en un solo clic](#2-instalación-en-un-solo-clic)
3. [Arrancar JARVIS](#3-arrancar-jarvis)
4. [La arquitectura del Enrutador Dinámico de Modelos](#4-la-arquitectura-del-enrutador-dinámico-de-modelos)
5. [Guía de compatibilidad de hardware](#5-guía-de-compatibilidad-de-hardware)
6. [Chuleta de órdenes de voz](#6-chuleta-de-órdenes-de-voz)
7. [La doble palmada](#7-la-doble-palmada)
8. [Configurar el navegador Comet](#8-configurar-el-navegador-comet)
9. [Interruptor de emergencia (kill-switch)](#9-interruptor-de-emergencia-kill-switch)
10. [JARVIS que se programa a sí mismo](#10-jarvis-que-se-programa-a-sí-mismo)
11. [Constructor de aplicaciones y webs](#11-constructor-de-aplicaciones-y-webs)
12. [Estructura del proyecto](#12-estructura-del-proyecto)
13. [Preguntas frecuentes y solución de problemas](#13-preguntas-frecuentes-y-solución-de-problemas)

---

## 1. Qué es JARVIS y qué lo hace diferente

JARVIS es un asistente de escritorio que vive **entero dentro de tu ordenador**. No hay cuentas, no hay suscripciones, no hay claves de API y no hay facturas: todo el trabajo lo hacen modelos de lenguaje que se ejecutan en tu propia tarjeta gráfica mediante [Ollama](https://ollama.com).

Lo que sabe hacer:

| Capacidad | Cómo funciona |
| --- | --- |
| **Conversación natural** | `llama3.1:8b` con memoria de la sesión. |
| **Oído permanente** | Palabra clave *«Jarvis»* con `openWakeWord` (menos del 1 % de CPU). |
| **Voz en español** | `Kokoro-82M` (voz `ef_dora`); si falla, Piper; si falla, la voz de Windows. |
| **Interrupción real** | Puedes hablarle **mientras habla** o **mientras actúa**: se calla y te escucha. |
| **Visión de pantalla** | `llama3.2-vision` mira tu pantalla bajo demanda y la describe. |
| **Control de ratón y teclado** | Mueve el cursor, hace clic en botones descritos con palabras. |
| **Doble palmada** | Dos palmadas → música (*Loser* de Tame Impala) y saludo según la hora. |
| **Auto-programación** | Puede añadirse funcionalidades nuevas a sí mismo con ramas git y pruebas. |
| **Constructor de apps** | «Hazme una web para mis gastos» → proyecto real, probado y abierto. |
| **Privacidad total** | Nada sale de tu equipo. El modo avión no le molesta. |

### Las tres leyes de JARVIS

1. **Ley de monogamia de VRAM.** Nunca hay dos modelos de lenguaje cargados a la vez en la tarjeta gráfica. Antes de cargar uno, se descarga el anterior con `keep_alive: 0`. La visión y el programador son *efímeros*: entran, hacen su trabajo y se van.
2. **Ley de privacidad.** Todo se ejecuta en local. La única conexión a internet ocurre la primera vez, para descargar modelos; después, ninguna.
3. **Ley del interruptor.** `CTRL + SHIFT + ESPACIO` (o una sacudida fuerte del ratón) detiene cualquier cosa que esté haciendo, al instante y sin preguntas.

---

## 2. Instalación en un solo clic

### Lo que necesitas antes de empezar

| Requisito | Detalle |
| --- | --- |
| Windows | 10 u 11, 64 bits. |
| Python | 3.10 o superior (**marca «Add python.exe to PATH»** al instalarlo). |
| Ollama | Se instala desde [ollama.com/download](https://ollama.com/download/windows) — es el motor que ejecuta los modelos. |
| Espacio en disco | Unos 15 GB entre modelos, voces y librerías. |
| Tarjeta gráfica | Recomendada NVIDIA con 8 GB (RTX 4060 Laptop). También funciona sin GPU, más despacio. |

### Los tres pasos

1. **Descarga el proyecto.** Copia esta carpeta donde quieras (por ejemplo `C:\JARVIS`). Evita las rutas con acentos raros.
2. **Doble clic en `install.bat`.** El instalador hace todo solo, contándote en español lo que va haciendo:
   - Comprueba la versión de Windows y de Python.
   - Crea un entorno aislado en `.venv` (nada se instala «sucio» en tu sistema).
   - Instala las librerías de `requirements.txt` (audio, visión, interfaz y pruebas).
   - Comprueba que Ollama existe y lo arranca si hace falta.
   - Descarga los modelos que falten con `ollama pull`: `llama3.1:8b`, `llama3.2-vision:latest` y `qwen2.5-coder:7b`.
   - Descarga el oído (`faster-whisper base`), las voces de Kokoro y la palabra clave *«Jarvis»*.
   - **Sella el kill-switch** y muestra un diagnóstico final.
3. **Doble clic en `start.bat`.** Y ya está: JARVIS está escuchando.

> Si algo falla, el instalador **nunca te deja a medias**: te dice exactamente qué paso falló, en español, y qué hacer a continuación.

### Instalación manual (solo si te gusta el detalle)

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull llama3.2-vision:latest
ollama pull qwen2.5-coder:7b
.venv\Scripts\python main.py --diagnostico
```

---

## 3. Arrancar JARVIS

| Quiero… | Comando |
| --- | --- |
| Uso normal (con la cápsula flotante) | doble clic en `start.bat` |
| Escribir en vez de hablar | `start.bat --texto` |
| Sin interfaz flotante | `start.bat --sin-hud` |
| Ver el estado de todo | `start.bat --diagnostico` |
| Probar micrófono y voz | `start.bat --prueba-voz` |
| Calibrar la doble palmada | `start.bat --prueba-palmada` |
| Comprobar el kill-switch | `start.bat --verificar-seguridad` |
| Detener una instancia abierta | `start.bat --matar` |

Al arrancar, JARVIS comprueba **primero** su capa de seguridad: si el archivo del kill-switch cambió respecto al sello guardado, **se niega a arrancar** y te avisa. Eso es lo que impide que un programa (o una IA) manipule su interruptor de emergencia.

### La cápsula de estado (Ghost HUD)

Arriba a la derecha verás una cápsula discreta que **nunca te roba un clic**: siempre deja pasar el ratón.

| Estado | Qué verás |
| --- | --- |
| Dormido | Cápsula gris, micrófono apagado. |
| Escuchando | Micrófono **cian** encendido. |
| Pensando / cambiando de modelo | Anillo de progreso girando. |
| Hablando | Cápsula con onda de voz. |
| Actuando (clics, teclado, pantalla) | **Borde cian** en toda la pantalla: control activo. |
| Kill-switch | Cápsula apagada y borde fuera. |

El borde cian solo aparece cuando JARVIS de verdad está tocando tu equipo. Si no lo ves, no está tocando nada.

---

## 4. La arquitectura del Enrutador Dinámico de Modelos

Esta es la pieza que hace posible tener un asistente de verdad en un portátil de 8 GB de VRAM sin que el equipo se ahogue.

### El problema

`llama3.1:8b` ocupa unos 4,9 GB en la GPU. `llama3.2-vision` otros ~7 GB. Si ambos estuvieran cargados a la vez, la VRAM se desbordaría y el sistema empezaría a usar RAM como memoria gráfica: todo iría a tirones. Eso es exactamente lo que **no** pasa aquí.

### La solución: monogamia de VRAM

```
   Orden de Pablo
        │
        ▼
 ┌─────────────────────┐
 │  Clasificador de    │   «¿esto es hablar, ver, programar o construir?»
 │  intención (routing)│   → responde el modelo de chat en milisegundos
 └─────────┬───────────┘
           ▼
 ┌──────────────────────────────────────────────────────────────┐
 │   ENRUTADOR DINÁMICO  (core/model_router.py)                 │
 │                                                              │
 │   1. ¿Qué modelo necesita esta tarea?                        │
 │   2. ¿Está ya cargado ESE modelo?  → adelante, sin tocar nada│
 │   3. ¿Hay OTRO modelo cargado?     → keep_alive: 0 y esperar │
 │   4. ¿Falta descargarlo?           → ollama pull automático  │
 │   5. Comprobar presupuesto de VRAM (6,5 GB)                  │
 │   6. Cargar, inferir y, si es efímero, descargar YA           │
 └──────────────────────────────────────────────────────────────┘
```

Reglas concretas, tal y como están en el código:

- **Un solo candado de GPU.** Todo el trabajo de GPU pasa por un único `RLock`: dos peticiones nunca pueden solaparse.
- **Descarga explícita antes de cambiar.** Antes de cargar un modelo se llama a Ollama con `keep_alive: 0` sobre el modelo anterior y se espera a que la VRAM quede libre (`VRAM_UNLOAD_WAIT_S = 12 s`).
- **Modelos efímeros.** La visión (`llama3.2-vision:latest`) y el programador se cargan **solo cuando se usan** y se expulsan **en el `finally`**, pase lo que pase.
- **Presupuesto de seguridad.** Nunca se reservan los 8 GB completos: el techo son **6,5 GB**. Si un modelo lo superara, JARVIS descarga todo y te lo dice en vez de arrastrar el sistema.
- **Arranque limpio.** Al empezar, el enrutador revisa `/api/ps` y expulsa cualquier modelo que hubiera quedado suelto de una sesión anterior.

### Matriz de modelos

| Tarea | Modelo | `keep_alive` | VRAM aprox. |
| --- | --- | --- | --- |
| Conversación, planificación, despacho, resúmenes | `llama3.1:8b` | `10m` | ~4,9 GB |
| Visión de pantalla y localización de botones | `llama3.2-vision:latest` | **`0` (efímero)** | ~7 GB |
| Auto-programación, apps, webs y pruebas | `qwen2.5-coder:7b` | `3m` | ~4,4 GB |
| Transcripción de voz | `faster-whisper base` | en CPU | 0 GB |
| Palabra clave «Jarvis» | `openWakeWord` | en CPU (<1 %) | 0 GB |
| Voz española | `Kokoro-82M` | en CPU | 0 GB |

**Coste: 0 €.** No hay ninguna llamada a servicios de pago. Ni una.

### Cómo se ve en la práctica

```
Pablo: «Jarvis, ¿qué ves en mi pantalla?»
  → router.ensure_model(task="screen_perception")   # carga llama3.2-vision
  → inferencia de visión                             # ~7 GB durante segundos
  → router._release_ephemeral()                      # keep_alive: 0
  → «Veo Comet abierto con una pestaña de recetas…»  # VRAM de vuelta al reposo

Pablo: «Jarvis, añade un comando para lavarme las manos»
  → se descarga la visión, se carga qwen2.5-coder:7b
  → escribe el código, crea una rama git, ejecuta pytest
  → si todo pasa: fusiona y se reinicia en caliente
```

---

## 5. Guía de compatibilidad de hardware

JARVIS se adapta a tu equipo. `python main.py --diagnostico` te dice en qué perfil estás.

| Perfil | RAM | VRAM | Chat | Visión | Código | Voz (STT) |
| --- | --- | --- | --- | --- | --- | --- |
| **Bajo** | 8–12 GB | 0–4 GB | `llama3.2:3b` | desactivada | `qwen2.5-coder:1.5b` | `tiny` |
| **Medio** ← el tuyo | 12–24 GB | 4–8 GB | `llama3.1:8b` | `llama3.2-vision:latest` | `qwen2.5-coder:7b` | `base` |
| **Alto** | 24 GB+ | 8 GB+ | `llama3.1:8b` | `llama3.2-vision:11b` | `qwen2.5-coder:14b` | `small` |

### Tu equipo (perfil medio)

- **CPU:** Ryzen 7 7840HS — se encarga del oído, la voz y el VAD sin despeinarse.
- **GPU:** RTX 4060 Laptop, 8 GB — un modelo a la vez, siempre.
- **RAM:** 16 GB DDR5 — el HUD de PyQt6 y las capturas de pantalla caben de sobra.
- **Reposo realista:** ~3,5–3,9 GB de VRAM (sistema + HUD + audio) y el modelo de chat cargado cuando toca.

### Sin tarjeta gráfica

Funciona, pero despacio. Todo el trabajo cae sobre la CPU y una respuesta que tarda 1 segundo puede tardar 10. Para ese caso:

- Cambia los modelos por las variantes pequeñas (`llama3.2:3b`, `qwen2.5-coder:1.5b`).
- Baja el perfil en `local_config.py` o simplemente déjalo: el diagnóstico te lo recomienda.

### Ajustes finos

Crea un archivo `local_config.py` junto a `config.py` para sobrescribir cualquier valor sin tocar el original:

```python
# local_config.py — mis ajustes personales
STT_MODEL = "small"          # más precisión al transcribir (algo más lento)
TTS_KOKORO_VOICE = "ef_dora" # voz femenina española de Kokoro
HUD_OPACITY_IDLE = 0.55      # cápsula más discreta
CLAP_MIN_PEAK_RMS = 0.07     # exigir palmadas más fuertes
```

---

## 6. Chuleta de órdenes de voz

### Lo básico

| Le dices… | Hace… |
| --- | --- |
| «Jarvis» + «¿qué hora es?» | La hora, con el saludo que toca. |
| «Jarvis, ¿qué día es hoy?» | Día de la semana y fecha en español. |
| «Jarvis, pon música» | *Loser* de Tame Impala (Spotify → YouTube). |
| «Jarvis, pon a Daft Punk» | Busca ese artista en Spotify/YouTube. |
| «Jarvis, abre Comet» | Abre el navegador (o la app que digas). |
| «Jarvis, busca en internet recetas de tortilla» | Abre los resultados en Comet. |
| «Jarvis, ¿qué ves en mi pantalla?» | Mira la pantalla y te la describe. |
| «Jarvis, haz clic en el botón azul» | Localiza ese elemento y hace clic. |
| «Jarvis, escribe mi correo en el formulario» | Escribe en la ventana activa. |
| «Jarvis, detente» | Cállate y cancela lo que estuvieras haciendo. |
| «Jarvis, apágate» | Se despide y se cierra solo. |

### Crear cosas

| Le dices… | Hace… |
| --- | --- |
| «Jarvis, hazme una web para llevar mis gastos» | Crea un proyecto real en `workspace/projects/`, lo prueba y lo abre. |
| «Jarvis, créame una aplicación de tareas» | Proyecto con su propio entorno aislado. |
| «Jarvis, añade una funcionalidad para que me avises cada hora» | Se programa a sí mismo: rama git, código, pruebas y fusión. |

### Conversación

Todo lo que no sea una orden reconocida pasa a `llama3.1:8b`, que responde con memoria de la sesión. Puedes hablarle de lo que quieras; **nada de lo que digas sale de tu casa**.

### Trucos que casi nadie descubre

- **Puedes interrumpirle.** Si empieza a hablar y le hablas encima, se calla al instante y te escucha.
- **Puedes frenarle mientras actúa.** Si está moviendo el ratón y le dices «para», suelta el control en el acto.
- **«Jarvis, continúa»** reactiva el control después de un kill-switch.
- **Si no le dices «Jarvis» al principio**, ignorará la frase: así no reacciona a tus conversaciones.

---

## 7. La doble palmada

Dos palmadas y JARVIS entra en acción: música y saludo. Sin tocar el teclado, sin decir nada.

### Qué hace exactamente

1. Detecta el primer golpe y el segundo.
2. Comprueba que están separados entre **200 y 750 ms** (ni un golpe doble del ratón ni dos palmadas de canciones distintas).
3. Espera un margen de **250 ms**: si llega una **tercera** palmada, lo entiende como un aplauso y **no hace nada**.
4. Dice el saludo que corresponde a la hora:
   - **06:00 – 12:00** → «Buenos días Pablo»
   - **12:00 – 20:00** → «Buenas tardes Pablo»
   - **20:00 – 06:00** → «Buenas noches Pablo»
5. Pone **«Loser» de Tame Impala**:
   - Primero, **Spotify** (`spotify:search:Loser Tame Impala`, con `Play` automático).
   - Si no hay Spotify, **Comet** abriendo YouTube con la canción.
   - Si Comet no está, cualquier navegador del sistema.

### Cómo distinguir una palmada de otros ruidos

No es un simple «pico de volumen»: el detector mira **cómo suena** la palmada.

| Comprobación | Por qué |
| --- | --- |
| Pico de energía con ataque brusco (3× en 20 ms) | Una palmada explota de golpe. |
| Más de 4,5× el ruido ambiente | Sabe cuál es el ruido de tu habitación y lo aprende solo. |
| Duración menor de 60 ms | Una palmada es corta; una puerta, no. |
| Planitud espectral entre 0,22 y 0,95 | Es ruido de banda ancha, no un silbido ni un tono puro. |
| Centroide espectral > 900 Hz | Descarta golpes sordos (la mesa, una taza, tu propio carraspeo). |
| Tiempo muerto de 2,5 s | No se dispara dos veces por el mismo aplauso. |

### Calibrar en 60 segundos

```bat
start.bat --prueba-palmada
```

Da palmadas cuando te lo pida y verás, en directo, el nivel de ruido del aula, el umbral que está usando y si tus palmadas entran o no. Si el portátil está a mucho volumen o la habitación es muy ruidosa, ajusta en `local_config.py`:

```python
CLAP_MIN_PEAK_RMS = 0.075    # exigir palmadas más fuertes
CLAP_NOISE_MULTIPLIER = 5.5  # exigir más diferencia sobre el ruido ambiente
```

---

## 8. Configurar el navegador Comet

JARVIS usa **Comet** (el navegador de Perplexity) como navegador preferido, porque es el que usa Pablo a diario.

1. **Instala Comet** desde [perplexity.ai/comet](https://www.perplexity.ai/comet) con las opciones por defecto.
2. **Comprueba dónde quedó.** Lo normal en Windows es:

   ```
   %LOCALAPPDATA%\Perplexity\Comet\Application\comet.exe
   ```

   Ojo: en Windows se instala en la carpeta de **usuario** (`%LOCALAPPDATA%\Perplexity\Comet`), no en `Program Files`.

3. **Verifica que JARVIS lo encuentra:**

   ```bat
   start.bat --diagnostico
   ```

   En la sección de navegador debería aparecer Comet como navegador elegido.

4. **Si tienes la ruta en otro sitio**, declárala en `local_config.py`:

   ```python
   BROWSER_COMET_CANDIDATES = (
       r"D:\Navegadores\Comet\Application\comet.exe",
   )
   ```

5. **Si no usas Comet**, JARVIS cae en cadena sobre Chrome → Edge → Firefox → Opera → Brave. Y si no hay ninguno, abre con el navegador predeterminado de Windows.

### Spotify

Para que la doble palmada arranque la canción exacta:

- Instala **Spotify para Windows** (la versión de escritorio, no la web).
- Deja que JARVIS abra **Spotify** y, tras 4,5 segundos (lo que tarda en cargar), pulsa `Play` él mismo.
- Si prefieres un enlace exacto, pega su URI en `local_config.py`:

  ```python
  CLAP_SPOTIFY_URI = "spotify:track:5YcBkeHj4zBmUk1eFfBTLc"   # ej. de "Loser"
  ```

---

## 9. Interruptor de emergencia (kill-switch)

Es la pieza más importante del sistema. Si algo va mal, **tú siempre puedes pararlo**.

### Cómo se activa

| Forma | Detalle |
| --- | --- |
| **`CTRL + SHIFT + ESPACIO`** | Atajo global, funciona aunque JARVIS esté en segundo plano. |
| **`CTRL + ALT + K`** | Atajo alternativo, por si el primero choca con otro programa. |
| **Sacudida del ratón** | Más de **1500 píxeles en 0,3 segundos**: si el ratón se te va de las manos, se corta solo. |
| **`start.bat --matar`** | Desde fuera, si algo se quedara colgado. |
| **Por voz** | «Jarvis, detente». Cancela la acción en curso. |

### Qué hace, en orden

1. **Cancela** las llamadas pendientes al modelo.
2. **Congela** la automatización en el paso exacto donde estaba.
3. **Mata** las subrutinas y los hilos de acción.
4. **Esconde** el borde cian de la pantalla: deja de tener el control.
5. **Suelta** el ratón y el teclado, sin importar lo que estuviera haciendo.
6. **Se calla** — la voz se corta a media palabra.

Después, el ratón y el teclado son **tuyos otra vez**. Para devolverle el control: «Jarvis, continúa».

### Por qué es inmutable

El kill-switch está diseñado para que **ni siquiera JARVIS pueda tocarlo**:

- Al instalarse, se guarda su **huella SHA-256** en `safety/killswitch.sha256`.
- En cada arranque, JARVIS verifica esa huella **antes de nada**. Si no coincide, **no arranca** y te avisa en español.
- Los archivos de seguridad se marcan como **solo lectura**.
- El motor de auto-programación tiene **prohibido por diseño** tocar `safety/`: cualquier ruta que empiece por `safety/` se rechaza antes de escribir un solo byte.
- El guardián de comandos bloquea los intentos de modificar la capa de seguridad, incluso dentro de código generado por la propia IA.

Comandos útiles:

```bat
start.bat --verificar-seguridad     :: comprobar la huella
start.bat --sellar-seguridad        :: volver a sellar (solo si fuiste tú quien editó)
```

> Si alguna vez ves «INTEGRIDAD DEL KILL-SWITCH: FALLO», **no lo ignores**: alguien o algo tocó esos archivos. Séllalo tú conscientemente o restaura el original.

---

## 10. JARVIS que se programa a sí mismo

Pídele una funcionalidad nueva y la escribe, la prueba y la aplica **sin romper nada**.

```
Pablo: «Jarvis, añade una funcionalidad para que me avises cada hora en punto»
```

Lo que ocurre por dentro:

1. **Comprueba que el repositorio está limpio** (si tienes cambios sin guardar, no toca nada).
2. **Crea una rama aislada:** `feature/staging-<fecha>`. Tu `main` sigue intacto.
3. **Escribe el código** con `qwen2.5-coder:7b`, en formato estructurado (qué archivo, qué contenido).
4. **Audita el código generado** antes de escribirlo:
   - Nada de `os.system`, `shell=True` ni ejecución de código descargado.
   - Nada de `safety/` ni de rutas que se salgan de la carpeta.
   - Nada de tocar la capa del kill-switch.
5. **Ejecuta las pruebas** `pytest tests/` en un subproceso aislado.
6. **Si todo pasa:** fusiona en `main` y **se reinicia en caliente** con `os.execv`. Sigues donde estabas.
7. **Si algo falla:** `git reset --hard`, borra la rama y **sigue funcionando como si nada**. Ni un archivo a medias, ni un proceso roto.

Esa última línea es la clave: **es imposible que un cambio fallido deje a JARVIS sin arrancar.**

---

## 11. Constructor de aplicaciones y webs

Dile lo que necesitas en lenguaje natural y te construye un proyecto completo.

```
Pablo: «Jarvis, hazme una web para llevar el control de mis gastos»
```

Qué obtienes:

- Una carpeta de verdad: `workspace/projects/control-de-gastos/`.
- Un **entorno virtual propio** para el proyecto, aislado del de JARVIS.
- Las dependencias instaladas **previo paso por el guardián** (nombres validados, nada de paquetes raros ni de URLs externas).
- Un **README** del proyecto explicando cómo abrirlo.
- **Prueba automática** antes de dártelo por bueno (compilación + comprobación del HTML principal).
- Si es una web, **se abre sola en Comet**. Si es una app, se lanza en segundo plano.

Los proyectos viven en `workspace/`, que **no se sube a git**: son tus cosas, no parte del programa.

---

## 12. Estructura del proyecto

```
Jarvis\
│
├── install.bat               Instalación en un clic (modelos, voces, sellado)
├── start.bat                 Arranque en un clic (con HUD, en segundo plano)
├── requirements.txt          Librerías, agrupadas por subsistema
├── README.md                 Esto que estás leyendo
├── main.py                   Punto de entrada: arranque, cerebro y habilidades
├── config.py                 Todos los ajustes en un solo sitio
│
├── safety\                   CAPA INTOCABLE
│   ├── killswitch.py         Interruptor de emergencia + huella SHA-256
│   ├── killswitch.sha256     El sello de integridad
│   └── command_guard.py      Guardián de comandos, rutas y código generado
│
├── core\                     MOTORES
│   ├── ollama_client.py      Habla con Ollama (flujo, descarga, cancelación)
│   ├── model_router.py       Enrutador dinámico y monogamia de VRAM
│   ├── voice_engine.py       Micrófono, VAD, Whisper, Kokoro, palabra clave
│   ├── barge_in.py           Interrupción y congelación de automatizaciones
│   ├── acoustic_detector.py  Doble palmada (DSP puro, sin GPU)
│   ├── vision_actuator.py    Captura, percepción y control de ratón/teclado
│   ├── self_programmer.py    Auto-programación con ramas y pruebas
│   ├── app_builder.py        Constructor de apps y webs
│   ├── media_dispatcher.py   Spotify, Comet, YouTube y apertura de apps
│   └── utils.py              Reintentos, JSON atómico, texto, hilos
│
├── gui\
│   └── hud_overlay.py        Cápsula flotante y borde cian (PyQt6, clic pasante)
│
├── tests\
│   ├── test_core.py          Batería principal (más de 130 pruebas)
│   └── conftest.py           Dobles de prueba: sin hardware real
│
├── workspace\                Tus apps y webs (no se sube a git)
├── models\                   Voces y oído descargados
├── logs\                     Registro de lo que pasa
└── state\                    PID y memoria de la sesión
```

---

## 13. Preguntas frecuentes y solución de problemas

<details>
<summary><b>«No encuentro Python» durante la instalación</b></summary>

Descarga Python de [python.org](https://www.python.org/downloads/windows/) y **marca la casilla «Add python.exe to PATH»** en la primera pantalla del instalador. Es el error más común: si no la marcas, Windows no sabe dónde está Python.
</details>

<details>
<summary><b>«Ollama no responde»</b></summary>

Abre la aplicación **Ollama** desde el menú Inicio (se queda en la bandeja del sistema) o ejecuta `ollama serve` en una ventana. JARVIS funciona sin él, pero sin conversación: las órdenes locales (hora, música, palmada, apps) siguen funcionando.
</details>

<details>
<summary><b>No me oye</b></summary>

1. Comprueba el micrófono: `start.bat --prueba-voz`.
2. Windows → Configuración → Privacidad → Micrófono → «Permitir que las aplicaciones accedan al micrófono».
3. Si usas auriculares con micrófono, selecciónalos como dispositivo predeterminado.
4. En modo texto siempre puedes escribirle: `start.bat --texto`.
</details>

<details>
<summary><b>La doble palmada no salta, o salta sola</b></summary>

Calíbrala con `start.bat --prueba-palmada`. Verás el ruido ambiente y el umbral en vivo. Si salta sola con la música alta, sube `CLAP_MIN_PEAK_RMS`. Si no salta con tus palmadas, bájalo.
</details>

<details>
<summary><b>Va lento o la tarjeta gráfica se queda sin memoria</b></summary>

1. Ejecuta `start.bat --diagnostico` y mira el presupuesto de VRAM.
2. Cierra juegos, OBS o cualquier cosa que use la GPU: JARVIS necesita unos 5 GB libres.
3. La visión (`llama3.2-vision`) es la más pesada: si no la usas, arranca con `--sin-vision`.
4. Los modelos se descargan solos al terminar cada tarea. Si ves el anillo girando mucho rato, es que está cambiando de modelo: es normal la primera vez.
</details>

<details>
<summary><b>¿De verdad no envía nada a internet?</b></summary>

Sí. Los modelos se ejecutan en tu GPU con Ollama, la transcripción con faster-whisper en tu CPU y la voz con Kokoro en tu CPU. La única conexión ocurre durante la instalación, para descargar modelos. Puedes desconectar el cable de red y todo sigue funcionando igual.
</details>

<details>
<summary><b>¿Puedo cambiar los modelos?</b></summary>

Sí, en `local_config.py`:

```python
MODEL_CHAT = "llama3.1:8b"          # cambia por el que prefieras
STT_MODEL = "small"                 # o "tiny", "medium"
TTS_KOKORO_VOICE = "ef_dora"        # cualquier voz del catálogo de Kokoro
```
</details>

<details>
<summary><b>Se ha quedado «pillado», no responde a nada</b></summary>

`CTRL + SHIFT + ESPACIO` primero. Si aun así sigue, `start.bat --matar`. Y si el micrófono se quedó bloqueado, cierra y vuelve a arrancar: JARVIS detecta que ya había otra instancia y evita pelearse por el dispositivo.
</details>

<details>
<summary><b>¿Y si quiero cerrarlo del todo? Es un asistente que siempre está</b></summary>

Di «Jarvis, apágate», o pulsa `CTRL + SHIFT + ESPACIO` y luego cierra la cápsula. También puedes ponerlo en modo manual: `HUD_ENABLED = False` y `WAKE_WORD_ENABLED = False` en `local_config.py`, y así solo se activa cuando tú lo arranques.
</details>

---

<div align="center">

**JARVIS v1.0.0** — hecho para Pablo, funciona en tu casa, no cuesta nada.

*«A veces hay que correr antes de aprender a caminar.»* — Tony Stark

</div>
