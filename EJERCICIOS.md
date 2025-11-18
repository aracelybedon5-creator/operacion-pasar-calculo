# 📝 EJERCICIOS Y PROBLEMAS DE CÁLCULO VECTORIAL
## Para resolver usando la Calculadora Vectorial 3D

---

## 🎯 NIVEL BÁSICO - Introducción a Curvas Paramétricas

### Ejercicio 1: Círculo Unitario
**Objetivo**: Entender la parametrización básica

**Curva**:
- x(t) = cos(t)
- y(t) = sin(t)
- z(t) = 0
- t ∈ [0, 2π]

**Preguntas**:
1. ¿Cuál es el vector tangente en t = π/4?
2. ¿Cuál es la curvatura en cualquier punto?
3. ¿Cuál es la longitud total de la curva?
4. ¿Por qué la curvatura es constante?

**Respuestas esperadas**:
1. T(π/4) ≈ (-0.707, 0.707, 0)
2. κ = 1 (constante, radio = 1)
3. L = 2π ≈ 6.283
4. Porque es un círculo perfecto de radio constante

---

### Ejercicio 2: Hélice Simple
**Objetivo**: Explorar movimiento 3D

**Curva**:
- x(t) = cos(t)
- y(t) = sin(t)
- z(t) = t
- t ∈ [0, 4π]

**Preguntas**:
1. ¿Cómo cambia la altura por cada vuelta completa?
2. ¿La curvatura es constante?
3. ¿Cuál es la velocidad (||r'(t)||)?
4. Compara la longitud de arco con una hélice donde z(t) = 2t

**Pistas**:
- Cada 2π radianes hay una vuelta completa
- La velocidad combina componentes horizontal (circular) y vertical (lineal)

---

### Ejercicio 3: Elipse
**Objetivo**: Estudiar curvas con curvatura variable

**Curva**:
- x(t) = 3*cos(t)
- y(t) = 2*sin(t)
- z(t) = 0
- t ∈ [0, 2π]

**Preguntas**:
1. ¿En qué puntos la curvatura es máxima?
2. ¿En qué puntos la curvatura es mínima?
3. ¿Por qué varía la curvatura?
4. Estima la longitud total (aproximadamente)

**Exploración**:
- Mueve t de 0 a 2π observando cómo cambia κ(t)
- ¿Qué pasa si A = B? (se convierte en círculo)

---

## 🎯 NIVEL INTERMEDIO - Operaciones Vectoriales

### Ejercicio 4: Verificar Fórmulas de Curvatura
**Objetivo**: Comprobar cálculos a mano

**Curva**: Parábola 3D
- x(t) = t
- y(t) = t²
- z(t) = 0
- t ∈ [-2, 2]

**Tarea**:
1. Calcula r'(t) a mano: r'(t) = (1, 2t, 0)
2. Calcula r''(t) a mano: r''(t) = (0, 2, 0)
3. Calcula r'(t) × r''(t) a mano
4. Usa la fórmula κ = ||r' × r''|| / ||r'||³
5. Verifica en la app en t = 0, t = 1, t = 2

**Solución en t = 0**:
- r'(0) = (1, 0, 0), ||r'|| = 1
- r''(0) = (0, 2, 0)
- r' × r'' = (0, 0, 2)
- κ(0) = 2 / 1³ = 2

---

### Ejercicio 5: Triedro de Frenet
**Objetivo**: Entender el sistema T-N-B

**Curva**: Hélice
- x(t) = 2*cos(t)
- y(t) = 2*sin(t)
- z(t) = t
- t ∈ [0, 2π]

**Tarea**:
1. Activa la visualización de T, N, B en la app
2. Observa cómo se mueve el triedro al variar t
3. Verifica que T · N = 0 (ortogonales) en varios puntos
4. Verifica que B = T × N en t = π/2

**Preguntas**:
- ¿Hacia dónde apunta N? (hacia el eje Z)
- ¿B es siempre horizontal?
- ¿Qué representa el plano TN? (plano osculador)

---

### Ejercicio 6: Longitud de Arco vs Parámetro
**Objetivo**: Distinguir entre t y s (longitud de arco)

**Curva**: Espiral logarítmica
- x(t) = exp(0.1*t) * cos(t)
- y(t) = exp(0.1*t) * sin(t)
- z(t) = 0
- t ∈ [0, 4π]

**Tarea**:
1. Mide L cuando t va de 0 a π
2. Mide L cuando t va de π a 2π
3. ¿Los incrementos son iguales? ¿Por qué no?
4. ¿Qué representa ||r'(t)||? (rapidez con la que se recorre la curva)

---

## 🎯 NIVEL AVANZADO - Análisis Profundo

### Ejercicio 7: Curvas con Curvatura Constante
**Objetivo**: Caracterizar curvas especiales

**Hipótesis**: Las únicas curvas con curvatura constante son:
- Líneas rectas (κ = 0)
- Círculos (κ = 1/r)
- Hélices circulares (κ constante ≠ 0)

**Tarea**:
1. Verifica que el círculo x=cos(t), y=sin(t) tiene κ = 1
2. Encuentra el radio r de un círculo con κ = 0.5
3. Prueba diferentes hélices y verifica que κ es constante
4. Intenta encontrar otra curva con κ constante

**Desafío**:
- ¿Qué relación hay entre A, B de la hélice y su curvatura?

---

### Ejercicio 8: Torsión (Avanzado)
**Objetivo**: Introducir el concepto de torsión τ

**Curva**: Hélice con parámetros variables
- x(t) = A*cos(t)
- y(t) = A*sin(t)
- z(t) = B*t

**Concepto**: La torsión mide cuánto se "tuerce" la curva fuera de su plano osculador

**Fórmula**:
```
τ = (r' × r'') · r''' / ||r' × r''||²
```

**Tarea**:
1. Calcula r'''(t) para la hélice
2. Computa (r' × r'') · r''' a mano
3. ¿Qué pasa cuando B = 0? (curva plana, τ = 0)
4. Relaciona τ con A y B

---

### Ejercicio 9: Diseño de Trayectorias
**Objetivo**: Aplicación práctica - diseñar una montaña rusa

**Restricciones**:
1. Debe empezar en (0, 0, 10) y terminar en (10, 0, 0)
2. Curvatura máxima ≤ 0.5 (seguridad)
3. Longitud total ≈ 30 unidades
4. Debe tener al menos 2 "loops" o giros interesantes

**Tarea**:
- Diseña una curva paramétrica que cumpla las restricciones
- Usa la app para verificar κ(t) en todos los puntos
- Ajusta parámetros hasta lograr el objetivo

---

### Ejercicio 10: Análisis de Campos Vectoriales (Extensión)
**Objetivo**: Conectar con divergencia y rotacional

**Curva**: Cualquier hélice

**Conceptos**:
- Campo tangente: T(t) en cada punto define un campo vectorial
- Campo normal: N(t) define otro campo
- ¿Cómo calcularías div(T) o curl(T)?

**Desafío teórico**:
- ¿Qué representa físicamente un campo cuyo rotacional es la curvatura?
- Investiga las ecuaciones de Frenet-Serret

---

## 🎯 PROYECTOS CREATIVOS

### Proyecto 1: Galería de Curvas Famosas
**Objetivo**: Recrear curvas históricas

**Curvas a implementar**:
1. **Cicloide**: trayectoria de punto en rueda
2. **Cardioide**: curva en forma de corazón
3. **Rosa Polar**: pétalos matemáticos
4. **Lemniscata**: símbolo de infinito
5. **Espiral de Arquímedes**: crecimiento lineal

**Entregable**:
- Documento con ecuaciones, parámetros y capturas
- Análisis de T, κ, L para cada curva
- Contexto histórico/aplicaciones

---

### Proyecto 2: Curva que Deletree tu Nombre
**Objetivo**: Diseño paramétrico creativo

**Tarea**:
- Diseña una curva 3D que, vista desde arriba, deletree tu inicial
- Debe ser continua (una sola curva paramétrica)
- Bonus: añade altura (componente z) para efecto 3D

**Hint**: Usa funciones trigonométricas con diferentes frecuencias

---

### Proyecto 3: Optimización de Trayectorias
**Objetivo**: Encontrar la curva más corta con restricciones

**Problema**:
- Conectar (0, 0, 0) con (5, 5, 5)
- Curvatura máxima = 1
- Minimizar longitud total

**Tarea**:
1. Propón 3 curvas diferentes
2. Compara sus longitudes
3. Verifica que κ ≤ 1 en todo el trayecto
4. ¿Cuál es óptima?

---

## 📊 TABLA DE REFERENCIA RÁPIDA

| Curva | κ típica | Aplicación |
|-------|----------|------------|
| Línea recta | 0 | Movimiento uniforme |
| Círculo | 1/r | Órbitas, ruedas |
| Hélice | constante | ADN, resortes |
| Parábola | variable | Trayectorias balísticas |
| Elipse | variable | Órbitas planetarias |
| Lissajous | compleja | Oscilaciones acopladas |

---

## 🔬 LABORATORIO VIRTUAL

### Experimento 1: Efecto del Radio en la Curvatura
**Hipótesis**: κ = 1/r para círculos

**Procedimiento**:
1. Círculo con A = 1: mide κ
2. Círculo con A = 2: mide κ
3. Círculo con A = 3: mide κ
4. Grafica κ vs 1/A
5. Verifica que κ·A = 1

---

### Experimento 2: Composición de Movimientos
**Objetivo**: Entender superposición de ondas

**Curva Lissajous**:
- x(t) = sin(a·t)
- y(t) = sin(b·t)
- z(t) = 0

**Tarea**:
1. Prueba a=1, b=1 (círculo diagonal)
2. Prueba a=1, b=2 (figura 8)
3. Prueba a=2, b=3 (patrón complejo)
4. ¿Cuándo la curva se cierra?

---

## ✅ AUTOEVALUACIÓN

Después de cada ejercicio, responde:

1. ✓ ¿Entendí el concepto matemático subyacente?
2. ✓ ¿Puedo explicar los resultados sin mirar la app?
3. ✓ ¿Podría calcular T, κ, L a mano para curvas simples?
4. ✓ ¿Veo la conexión con aplicaciones reales?

---

## 📚 RECURSOS ADICIONALES

**Temas relacionados para profundizar**:
- Fórmulas de Frenet-Serret
- Torsión de curvas espaciales
- Coordenadas intrínsecas (s, κ, τ)
- Teorema fundamental de curvas
- Involuta y evoluta de curvas
- Curvaturas principales de superficies

**Libros recomendados**:
1. Stewart - Cálculo Multivariable (Cap. 13)
2. Marsden & Tromba - Cálculo Vectorial (Cap. 4)
3. Do Carmo - Differential Geometry of Curves and Surfaces

---

## 🎓 CRITERIOS DE EVALUACIÓN (Para proyectos)

| Aspecto | Excelente | Bueno | Mejorable |
|---------|-----------|-------|-----------|
| **Precisión matemática** | Todos los cálculos correctos | Errores menores | Errores conceptuales |
| **Interpretación** | Explica el significado físico/geométrico | Describe numéricamente | Solo reporta valores |
| **Uso de la app** | Explora creativamente | Usa funciones básicas | Uso limitado |
| **Documentación** | Reporte claro con gráficos | Respuestas completas | Respuestas parciales |

---

### ¡Buena suerte con los ejercicios! 🚀📐

**Recuerda**: La app es una herramienta para desarrollar intuición, pero siempre debes entender la teoría detrás de los cálculos.
