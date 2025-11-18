# 📋 INSTRUCCIONES PARA SUBIR A GITHUB

## 🎯 Pasos para Subir el Proyecto

### Opción 1: Usando Git desde la Terminal

#### 1. Inicializar Git en el proyecto (si no está inicializado)
```bash
cd "c:\Calculo vectorial\Version.1"
git init
```

#### 2. Configurar tu información (si es la primera vez)
```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tuemail@ejemplo.com"
```

#### 3. Agregar todos los archivos
```bash
git add .
```

#### 4. Hacer el primer commit
```bash
git commit -m "Initial commit: Aplicación completa de Cálculo Vectorial"
```

#### 5. Conectar con el repositorio remoto
```bash
# Reemplaza <URL_DEL_REPO> con la URL que te dé GitHub
git remote add origin <URL_DEL_REPO>
```

#### 6. Subir a GitHub
```bash
# Subir a la rama main
git push -u origin main

# O si prefieres crear una rama con tu nombre
git checkout -b nombre-usuario
git push -u origin nombre-usuario
```

---

### Opción 2: Crear el Repositorio desde Cero en GitHub

#### Paso 1: Crear repositorio en GitHub
1. Ve a https://github.com
2. Click en el botón **"New repository"** (verde)
3. Nombre del repositorio: `calculo-vectorial-3d`
4. Descripción: `Aplicación interactiva de Cálculo Vectorial con Streamlit`
5. Selecciona **Public** o **Private**
6. **NO** marques "Initialize this repository with a README"
7. Click **"Create repository"**

#### Paso 2: Subir archivos desde la terminal
GitHub te dará comandos similares a estos:

```bash
cd "c:\Calculo vectorial\Version.1"
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/calculo-vectorial-3d.git
git push -u origin main
```

---

### Opción 3: Usar GitHub Desktop (Más Fácil)

#### 1. Descargar GitHub Desktop
https://desktop.github.com/

#### 2. Abrir el proyecto
- File → Add Local Repository
- Selecciona la carpeta `c:\Calculo vectorial\Version.1`

#### 3. Crear repositorio
- Click en "Create a repository"
- Name: `calculo-vectorial-3d`
- Description: `Aplicación de Cálculo Vectorial`
- Click "Create Repository"

#### 4. Publicar a GitHub
- Click en "Publish repository"
- Marca o desmarca "Keep this code private"
- Click "Publish Repository"

---

## 📂 Archivos a Incluir

Asegúrate de que estos archivos estén en el repositorio:

```
Version.1/
├── app_vectorial.py          ✅ Aplicación principal
├── calculo_vectorial.py      ✅ Módulo de cálculo
├── viz_vectorial.py          ✅ Visualizaciones
├── viz_superficies.py        ✅ Visualizaciones
├── viz_curvas.py             ✅ Visualizaciones
├── requirements.txt          ✅ Dependencias
├── README.md                 ✅ Documentación
├── CASOS_DE_PRUEBA.md        ✅ Casos de prueba
└── .gitignore                ⚠️ Crear si no existe
```

---

## 🚫 Archivo .gitignore Recomendado

Crea un archivo `.gitignore` con este contenido:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Streamlit
.streamlit/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Sistema
.DS_Store
Thumbs.db

# Logs
*.log
```

---

## 🌿 Estructura de Ramas Recomendada

### Rama Principal: `main`
- Código estable y funcional
- Solo se hace merge después de probar

### Rama de Desarrollo: `develop`
- Nuevas características en desarrollo
- Se prueba antes de hacer merge a main

### Ramas de Características: `feature/nombre-feature`
```bash
git checkout -b feature/nuevas-visualizaciones
# Hacer cambios
git add .
git commit -m "Agregar visualizaciones mejoradas"
git push origin feature/nuevas-visualizaciones
```

---

## 🔄 Comandos Git Útiles

### Ver estado actual
```bash
git status
```

### Ver historial de commits
```bash
git log --oneline
```

### Crear nueva rama
```bash
git checkout -b nombre-rama
```

### Cambiar de rama
```bash
git checkout nombre-rama
```

### Ver diferencias
```bash
git diff
```

### Deshacer cambios no guardados
```bash
git restore archivo.py
```

### Ver ramas remotas
```bash
git branch -r
```

---

## 📤 Comandos para Colaboración

### Clonar repositorio
```bash
git clone https://github.com/usuario/calculo-vectorial-3d.git
```

### Actualizar desde GitHub
```bash
git pull origin main
```

### Hacer Push de una rama
```bash
git push origin nombre-rama
```

### Crear Pull Request
1. Haz push de tu rama
2. Ve a GitHub
3. Click en "Compare & pull request"
4. Escribe descripción
5. Click "Create pull request"

---

## ✅ Checklist Pre-Commit

Antes de hacer commit, verifica:

- [ ] El código funciona sin errores
- [ ] Se ejecuta `streamlit run app_vectorial.py` correctamente
- [ ] No hay archivos innecesarios (cache, logs)
- [ ] README.md está actualizado
- [ ] requirements.txt incluye todas las dependencias

---

## 🆘 Solución de Problemas Comunes

### Error: "fatal: not a git repository"
```bash
cd "c:\Calculo vectorial\Version.1"
git init
```

### Error: "remote origin already exists"
```bash
git remote remove origin
git remote add origin <URL_NUEVA>
```

### Error: "rejected - non-fast-forward"
```bash
git pull origin main --rebase
git push origin main
```

### Olvidé hacer commit antes de cambiar de rama
```bash
git stash
git checkout otra-rama
# Cuando vuelvas:
git stash pop
```

---

## 📞 Necesitas Ayuda?

1. **Pásame el link del repositorio** y puedo ayudarte con los comandos exactos
2. **Usa GitHub Desktop** si prefieres una interfaz gráfica
3. **Consulta la documentación**: https://docs.github.com/es

---

**Última actualización**: 17 de Noviembre, 2025
