# 📋 CASOS DE PRUEBA - APLICACIÓN DE CÁLCULO VECTORIAL

## 🎯 Guía Completa de Pruebas por Sección

Este documento contiene casos de prueba organizados desde los **MÁS SIMPLES** hasta los **MÁS EXIGENTES** para cada funcionalidad de la aplicación.

---

## 1️⃣ CAMPO VECTORIAL (∇·F, ∇×F)

### 🟢 Nivel Básico - Campos Constantes

#### Prueba 1.1: Campo Constante en Z
```
P: 0
Q: 0
R: 1

Resultado Esperado:
- Divergencia: 0 (campo incompresible)
- Rotacional: (0, 0, 0) (campo irrotacional)
- Visualización: Vectores verticales uniformes
```

#### Prueba 1.2: Campo Radial Simple
```
P: x
Q: y
R: z

Resultado Esperado:
- Divergencia: 3 (expansión uniforme)
- Rotacional: (0, 0, 0) (campo irrotacional)
- Visualización: Vectores apuntando hacia afuera del origen
```

### 🟡 Nivel Intermedio - Campos Rotacionales

#### Prueba 1.3: Campo Rotacional Clásico
```
P: -y
Q: x
R: 0

Resultado Esperado:
- Divergencia: 0 (campo incompresible)
- Rotacional: (0, 0, 2) (rotación en sentido antihorario)
- Visualización: Vectores girando alrededor del eje Z
```

#### Prueba 1.4: Campo con Divergencia Variable
```
P: x^2
Q: y^2
R: z^2

Resultado Esperado:
- Divergencia: 2x + 2y + 2z
- Rotacional: (0, 0, 0)
- Visualización: Divergencia aumenta con la distancia
```

### 🔴 Nivel Avanzado - Campos Complejos

#### Prueba 1.5: Campo Electromagnético
```
P: y*z
Q: x*z
R: x*y

Resultado Esperado:
- Divergencia: 0
- Rotacional: (x - x, y - y, z - z) = (0, 0, 0)
- Nota: Campo conservativo
```

#### Prueba 1.6: Vórtice 3D
```
P: -y/(x^2 + y^2)
Q: x/(x^2 + y^2)
R: z

Resultado Esperado:
- Divergencia: 1
- Rotacional: componente Z singular en el origen
- Visualización: Vórtice con componente vertical
```

#### Prueba 1.7: Campo de Coulomb
```
P: x/(x^2 + y^2 + z^2)^(3/2)
Q: y/(x^2 + y^2 + z^2)^(3/2)
R: z/(x^2 + y^2 + z^2)^(3/2)

Resultado Esperado:
- Divergencia: 0 (excepto en el origen)
- Rotacional: (0, 0, 0)
- Campo conservativo esférico
```

---

## 2️⃣ GRADIENTE EN CAMPO ESCALAR (∇φ)

### 🟢 Nivel Básico - Funciones Polinomiales

#### Prueba 2.1: Plano Inclinado
```
φ: x + y + z

Resultado Esperado:
- Gradiente: (1, 1, 1)
- Dirección: Apunta hacia (1, 1, 1)
- Visualización: Plano con gradiente constante
```

#### Prueba 2.2: Paraboloide Simple
```
φ: x^2 + y^2

Resultado Esperado:
- Gradiente: (2x, 2y, 0)
- Dirección: Radial en el plano XY
- Visualización: Paraboloide con gradiente aumentando radialmente
```

### 🟡 Nivel Intermedio - Funciones No Lineales

#### Prueba 2.3: Esfera (Campo Cuadrático)
```
φ: x^2 + y^2 + z^2

Resultado Esperado:
- Gradiente: (2x, 2y, 2z)
- Superficies de nivel: Esferas concéntricas
- Visualización: Gradiente perpendicular a las esferas
```

#### Prueba 2.4: Silla de Montar
```
φ: x^2 - y^2

Resultado Esperado:
- Gradiente: (2x, -2y, 0)
- Punto crítico en (0, 0, 0)
- Visualización: Superficie hiperbólica
```

#### Prueba 2.5: Cono
```
φ: sqrt(x^2 + y^2)

Resultado Esperado:
- Gradiente: (x/√(x²+y²), y/√(x²+y²), 0)
- Singular en el origen
- Visualización: Cono con vértice en el origen
```

### 🔴 Nivel Avanzado - Funciones Trascendentales

#### Prueba 2.6: Gaussiana 3D
```
φ: exp(-(x^2 + y^2 + z^2))

Resultado Esperado:
- Gradiente: (-2x·e^(-r²), -2y·e^(-r²), -2z·e^(-r²))
- Máximo en el origen
- Visualización: Campana 3D
```

#### Prueba 2.7: Potencial Gravitacional
```
φ: -1/sqrt(x^2 + y^2 + z^2)

Resultado Esperado:
- Gradiente: Campo tipo 1/r²
- Singular en el origen
- Superficies de nivel: Esferas
```

#### Prueba 2.8: Función Trigonométrica
```
φ: sin(x)*cos(y)*z

Resultado Esperado:
- Gradiente: (cos(x)cos(y)z, -sin(x)sin(y)z, sin(x)cos(y))
- Patrón ondulatorio complejo
- Visualización: Ondas en 3D
```

---

## 3️⃣ INTEGRAL DE LÍNEA (∫ F·dr)

### 🟢 Nivel Básico - Curvas Simples

#### Prueba 3.1: Línea Recta con Campo Constante
```
Campo F:
P: 1
Q: 0
R: 0

Curva r(t):
x(t): t
y(t): 0
z(t): 0
t₀: 0
t₁: 1

Resultado Esperado:
- Integral: 1
- Integrando: F·dr/dt = 1 (constante)
- Gráfica: Línea horizontal en y=1
```

#### Prueba 3.2: Círculo en XY con Campo Tangencial
```
Campo F:
P: -y
Q: x
R: 0

Curva r(t):
x(t): cos(t)
y(t): sin(t)
z(t): 0
t₀: 0
t₁: pi

Resultado Esperado:
- Integral: -π ≈ -3.14159
- Integrando: Constante -1
- Gráfica: Línea horizontal en y=-1
```

### 🟡 Nivel Intermedio - Curvas en 3D

#### Prueba 3.3: Hélice con Campo Vertical
```
Campo F:
P: 0
Q: 0
R: z

Curva r(t):
x(t): cos(t)
y(t): sin(t)
z(t): t
t₀: 0
t₁: 2*pi

Resultado Esperado:
- Integral: ≈ 12.566 (2π²)
- Integrando: Creciente lineal
- Gráfica: Rampa ascendente
```

#### Prueba 3.4: Segmento de Parábola
```
Campo F:
P: x
Q: y
R: 0

Curva r(t):
x(t): t
y(t): t^2
z(t): 0
t₀: 0
t₁: 1

Resultado Esperado:
- Integral: ≈ 1.167 (7/6)
- Integrando: Variable
- Gráfica: Curva no lineal
```

### 🔴 Nivel Avanzado - Casos Especiales

#### Prueba 3.5: **CAMPO ROTACIONAL COMPLETO** (Caso Emblemático)
```
Campo F:
P: y
Q: -x
R: 0

Curva r(t):
x(t): cos(t)
y(t): sin(t)
z(t): 0
t₀: 0
t₁: 2*pi

Resultado Esperado:
- Integral: **-2π ≈ -6.283185307**
- Integrando: Constante -1
- Gráfica: Línea horizontal ROJA en y=-1
- Detección automática del caso clásico
```

#### Prueba 3.6: Campo Conservativo (Integral = 0)
```
Campo F (gradiente de x²+y²):
P: 2*x
Q: 2*y
R: 0

Curva r(t) (curva cerrada):
x(t): cos(t)
y(t): sin(t)
z(t): 0
t₀: 0
t₁: 2*pi

Resultado Esperado:
- Integral: 0 (campo conservativo en curva cerrada)
- Integrando: Oscilante simétrico
- Gráfica: Área positiva = Área negativa
```

#### Prueba 3.7: Espiral 3D Compleja
```
Campo F:
P: -y + z
Q: x - z
R: x + y

Curva r(t):
x(t): t*cos(t)
y(t): t*sin(t)
z(t): t
t₀: 0
t₁: 4*pi

Resultado Esperado:
- Integral: Valor numérico complejo
- Integrando: Oscilatorio con amplitud creciente
- Gráfica: Patrón ondulatorio
```

---

## 4️⃣ FLUJO DE SUPERFICIE (∬ F·n dS)

### 🟢 Nivel Básico - Superficies Planas

#### Prueba 4.1: Cuadrado en el Plano XY
```
Campo F:
P: 0
Q: 0
R: 1

Superficie r(u,v):
x(u,v): u
y(u,v): v
z(u,v): 0
u₀: 0, u₁: 1
v₀: 0, v₁: 1

Resultado Esperado:
- Flujo: 0 (campo perpendicular a la superficie)
- Normal: (0, 0, -1)
```

#### Prueba 4.2: Rectángulo Vertical
```
Campo F:
P: 1
Q: 0
R: 0

Superficie r(u,v):
x(u,v): 0
y(u,v): u
z(u,v): v
u₀: 0, u₁: 2
v₀: 0, v₁: 3

Resultado Esperado:
- Flujo: 6 (área × componente)
- Normal: (-1, 0, 0)
```

### 🟡 Nivel Intermedio - Superficies Curvas

#### Prueba 4.3: Cilindro
```
Campo F:
P: x
Q: y
R: 0

Superficie r(u,v):
x(u,v): cos(u)
y(u,v): sin(u)
z(u,v): v
u₀: 0, u₁: 2*pi
v₀: 0, v₁: 1

Resultado Esperado:
- Flujo: 2π (flujo radial saliente)
- Normal: Radial hacia afuera
```

#### Prueba 4.4: Paraboloide
```
Campo F:
P: 0
Q: 0
R: z

Superficie r(u,v):
x(u,v): u
y(u,v): v
z(u,v): u^2 + v^2
u₀: -1, u₁: 1
v₀: -1, v₁: 1

Resultado Esperado:
- Flujo: Positivo (campo apunta hacia arriba)
- Normal: Inclinada hacia arriba
```

### 🔴 Nivel Avanzado - Superficies Complejas

#### Prueba 4.5: Esfera Completa (Teorema de Divergencia)
```
Campo F:
P: x
Q: y
R: z

Superficie r(u,v):
x(u,v): sin(u)*cos(v)
y(u,v): sin(u)*sin(v)
z(u,v): cos(u)
u₀: 0, u₁: pi
v₀: 0, v₁: 2*pi

Resultado Esperado:
- Flujo: 4π (= volumen × divergencia)
- Verificar Teorema de Divergencia
```

#### Prueba 4.6: Toro
```
Campo F:
P: 0
Q: 0
R: 1

Superficie r(u,v):
x(u,v): (2 + cos(u))*cos(v)
y(u,v): (2 + cos(u))*sin(v)
z(u,v): sin(u)
u₀: 0, u₁: 2*pi
v₀: 0, v₁: 2*pi

Resultado Esperado:
- Flujo: 0 (campo constante vertical)
- Normal: Compleja, orientada hacia afuera
```

---

## 5️⃣ TEOREMA DE STOKES (∮ F·dr = ∬ (∇×F)·n dS)

### 🟢 Nivel Básico - Verificación Simple

#### Prueba 5.1: Disco Unitario con Campo Constante
```
Campo F:
P: 0
Q: 0
R: x + y

Frontera C:
x(t): cos(t)
y(t): sin(t)
z(t): 0
t₀: 0, t₁: 2*pi

Superficie S:
x(u,v): u*cos(v)
y(u,v): u*sin(v)
z(u,v): 0
u₀: 0, u₁: 1
v₀: 0, v₁: 2*pi

Resultado Esperado:
- Integral de línea: 0
- Integral de superficie: 0
- Verificación: ✅ Ambos iguales
```

### 🟡 Nivel Intermedio - Campo Rotacional

#### Prueba 5.2: Círculo con Campo Rotacional 2D
```
Campo F:
P: -y
Q: x
R: 0

Frontera C:
x(t): cos(t)
y(t): sin(t)
z(t): 0
t₀: 0, t₁: 2*pi

Superficie S:
x(u,v): u*cos(v)
y(u,v): u*sin(v)
z(u,v): 0
u₀: 0, u₁: 1
v₀: 0, v₁: 2*pi

Resultado Esperado:
- Integral de línea: 2π
- Rotacional: (0, 0, 2)
- Integral de superficie: 2π
- Verificación: ✅ Teorema cumplido
```

### 🔴 Nivel Avanzado - Superficies No Planas

#### Prueba 5.3: Hemisferio con Campo 3D
```
Campo F:
P: y
Q: -x
R: z^2

Frontera C:
x(t): cos(t)
y(t): sin(t)
z(t): 0
t₀: 0, t₁: 2*pi

Superficie S (hemisferio):
x(u,v): sin(u)*cos(v)
y(u,v): sin(u)*sin(v)
z(u,v): cos(u)
u₀: 0, u₁: pi/2
v₀: 0, v₁: 2*pi

Resultado Esperado:
- Integral de línea: 2π
- Rotacional: (-2z, 0, -2)
- Integral de superficie: ≈ 2π
- Verificación: ✅ Tolerancia < 0.01
```

#### Prueba 5.4: Paraboloide con Borde Circular
```
Campo F:
P: -y + z
Q: x - z
R: x*y

Frontera C:
x(t): 2*cos(t)
y(t): 2*sin(t)
z(t): 4
t₀: 0, t₁: 2*pi

Superficie S:
x(u,v): u*cos(v)
y(u,v): u*sin(v)
z(u,v): u^2
u₀: 0, u₁: 2
v₀: 0, v₁: 2*pi

Resultado Esperado:
- Ambas integrales deben coincidir
- Rotacional: (y, -x, 2 + 1)
- Verificación: Error relativo < 1%
```

---

## 🎯 CASOS DE PRUEBA EXTREMOS

### ⚠️ Casos Límite y Singularidades

#### Extremo 1: Campo con Singularidad
```
Campo F:
P: x/(x^2 + y^2 + z^2)
Q: y/(x^2 + y^2 + z^2)
R: z/(x^2 + y^2 + z^2)

Nota: Evitar evaluar en el origen
```

#### Extremo 2: Curva Muy Larga
```
Curva r(t):
x(t): t
y(t): t
z(t): t
t₀: 0
t₁: 100

Nota: Probar eficiencia del cálculo
```

#### Extremo 3: Superficie de Alta Resolución
```
Superficie con:
u₀: 0, u₁: 10
v₀: 0, v₁: 10
Nu: 100
Nv: 100

Nota: Probar rendimiento
```

---

## 📊 CHECKLIST DE VERIFICACIÓN

Para cada prueba, verificar:

- [ ] **Cálculo Correcto**: Resultado numérico coincide con el esperado
- [ ] **Pasos Mostrados**: Se muestran todos los pasos intermedios
- [ ] **Visualización**: Gráfica se genera sin errores
- [ ] **Interactividad**: Sliders y controles funcionan
- [ ] **Persistencia**: Gráfica no desaparece al mover controles
- [ ] **Descarga**: Botón de descarga funciona
- [ ] **Precisión**: Mínimo 6 decimales en resultados

---

## 🚀 SECUENCIA DE PRUEBAS RECOMENDADA

1. **Día 1**: Pruebas básicas (🟢) de todas las secciones
2. **Día 2**: Pruebas intermedias (🟡) de todas las secciones
3. **Día 3**: Pruebas avanzadas (🔴) de todas las secciones
4. **Día 4**: Casos extremos y verificación de rendimiento

---

## 📝 NOTAS IMPORTANTES

- **Tolerancia Numérica**: Aceptar diferencias < 1e-6 por errores de redondeo
- **Visualización**: Verificar que colores y leyendas sean claros
- **Documentación**: Cada resultado debe tener explicación matemática
- **Eficiencia**: Cálculos no deben tardar más de 5 segundos

---

**Última actualización**: 17 de Noviembre, 2025
**Versión del documento**: 1.0
