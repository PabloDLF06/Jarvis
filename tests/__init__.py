"""
tests — Suite de pruebas de JARVIS.
================================================================================
Estas pruebas son también el **árbitro del motor de auto-programación**: cuando
JARVIS propone un cambio en su propio código, se ejecuta `pytest tests/` dentro
de la rama aislada. Si algo falla aquí, la rama se descarta y JARVIS sigue
funcionando exactamente como antes.

REGLA DE ORO: ninguna prueba puede necesitar micrófono, tarjeta gráfica,
Ollama en marcha ni conexión a internet. Todo el hardware se sustituye por
dobles de prueba (relojes falsos, clientes simulados, capturas sintéticas).
================================================================================
"""
