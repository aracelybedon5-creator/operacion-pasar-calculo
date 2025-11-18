@echo off
REM ============================================================================
REM SCRIPT DE INICIO RAPIDO PARA WINDOWS (CMD)
REM ============================================================================
REM Este script automatiza el proceso de instalacion y ejecucion
REM Doble clic en este archivo para ejecutar
REM ============================================================================

title Calculadora Vectorial 3D - Instalacion
color 0A

echo ========================================
echo   CALCULADORA VECTORIAL 3D
echo   Instalacion y Ejecucion Automatica
echo ========================================
echo.

REM Verificar que Python esta instalado
echo [1/5] Verificando instalacion de Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo       X ERROR: Python no esta instalado
    echo       Descargalo desde: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo       Exito Python encontrado
echo.

REM Crear entorno virtual si no existe
echo [2/5] Configurando entorno virtual...
if not exist "venv" (
    echo       Creando nuevo entorno virtual...
    python -m venv venv
    echo       Exito Entorno virtual creado
) else (
    echo       Exito Entorno virtual ya existe
)
echo.

REM Activar entorno virtual
echo [3/5] Activando entorno virtual...
call venv\Scripts\activate.bat
echo       Exito Entorno activado
echo.

REM Instalar dependencias
echo [4/5] Instalando dependencias...
echo       (Esto puede tomar unos minutos la primera vez)
pip install -q -r requirements.txt
echo       Exito Dependencias instaladas
echo.

REM Ejecutar la aplicacion
echo [5/5] Iniciando aplicacion...
echo.
echo ========================================
echo   Aplicacion lista!
echo   Se abrira en tu navegador automaticamente
echo   Presiona Ctrl+C para detener
echo ========================================
echo.

streamlit run app_vectorial.py

pause
