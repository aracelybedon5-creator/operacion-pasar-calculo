# ============================================================================
# SCRIPT DE INICIO RÁPIDO PARA WINDOWS (PowerShell)
# ============================================================================
# Este script automatiza el proceso de instalación y ejecución
# Guárdalo como: iniciar.ps1
# ============================================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CALCULADORA VECTORIAL 3D" -ForegroundColor Yellow
Write-Host "  Instalación y Ejecución Automática" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar que Python está instalado
Write-Host "[1/5] Verificando instalación de Python..." -ForegroundColor Green
try {
    $pythonVersion = python --version 2>&1
    Write-Host "      ✓ Python encontrado: $pythonVersion" -ForegroundColor Gray
}
catch {
    Write-Host "      ✗ ERROR: Python no está instalado" -ForegroundColor Red
    Write-Host "      Descárgalo desde: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Crear entorno virtual si no existe
Write-Host "[2/5] Configurando entorno virtual..." -ForegroundColor Green
if (!(Test-Path -Path "venv")) {
    Write-Host "      Creando nuevo entorno virtual..." -ForegroundColor Gray
    python -m venv venv
    Write-Host "      ✓ Entorno virtual creado" -ForegroundColor Gray
}
else {
    Write-Host "      ✓ Entorno virtual ya existe" -ForegroundColor Gray
}

# Activar entorno virtual
Write-Host "[3/5] Activando entorno virtual..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1
Write-Host "      ✓ Entorno activado" -ForegroundColor Gray

# Instalar dependencias
Write-Host "[4/5] Instalando dependencias..." -ForegroundColor Green
Write-Host "      (Esto puede tomar unos minutos la primera vez)" -ForegroundColor Gray
pip install -q -r requirements.txt
Write-Host "      ✓ Dependencias instaladas" -ForegroundColor Gray

# Ejecutar la aplicación
Write-Host "[5/5] Iniciando aplicación..." -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ¡Aplicación lista!" -ForegroundColor Yellow
Write-Host "  Se abrirá en tu navegador automáticamente" -ForegroundColor Yellow
Write-Host "  Presiona Ctrl+C para detener" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

streamlit run app_vectorial.py
