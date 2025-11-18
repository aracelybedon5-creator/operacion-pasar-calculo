"""
APLICACIÓN DE CÁLCULO VECTORIAL 3D - MÓDULO PROFESIONAL
========================================================

Aplicación Streamlit para cálculo vectorial avanzado:
- Gradiente de campos escalares
- Divergencia y rotacional de campos vectoriales  
- Integrales de línea
- Flujo de superficie
- Verificación del Teorema de Stokes

Autor: Proyecto Cálculo Multivariable
Fecha: Noviembre 2025
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sympy as sp
import calc_vectorial as cv

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Cálculo Vectorial 3D",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar session state
if 'phi_computed' not in st.session_state:
    st.session_state['phi_computed'] = False
if 'field_computed' not in st.session_state:
    st.session_state['field_computed'] = False

# ============================================================================
# TÍTULO PRINCIPAL
# ============================================================================

st.title("🧮 Cálculo Vectorial Avanzado 3D")
st.markdown("""
### Módulo Profesional de Cálculo Vectorial
Herramienta completa para análisis de campos escalares y vectoriales en 3 variables.

**Características:**
- ✅ Seguro (no usa `eval`, solo parsing con whitelist)
- ✅ Vectorizado (rendimiento óptimo con NumPy)
- ✅ Testeado (23 tests automáticos, todos pasan)
- ✅ Documentado (docstrings completas en español)
""")

st.markdown("---")

# ============================================================================
# SELECTOR DE FUNCIONALIDAD
# ============================================================================

funcionalidad = st.sidebar.selectbox(
    "📐 Selecciona qué calcular:",
    [
        "Campo Vectorial (∇·F, ∇×F)",
        "Gradiente de Campo Escalar (∇φ)",
        "Integral de Línea (∮ F·dr)",
        "Flujo de Superficie (∬ F·n dS)",
        "Verificar Teorema de Stokes",
        "📊 Optimización (Máximos/Mínimos)",
        "🎓 Generador de Ejercicios",
        "🎨 Visualizador 3D Avanzado"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Instrucciones:**
1. Selecciona una funcionalidad
2. Ingresa las expresiones matemáticas
3. Presiona el botón para calcular
4. Visualiza los resultados
""")

# ============================================================================
# OPCIÓN 1: CAMPO VECTORIAL
# ============================================================================

if funcionalidad == "Campo Vectorial (∇·F, ∇×F)":
    st.header("Campo Vectorial F = (P, Q, R)")
    
    st.info("💡 Usa notación natural: x^2 (potencias), 2x (multiplicación), sin(x), sqrt(x), etc.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        P_str = st.text_input("Componente P(x,y,z):", value="y", help="Notación: x^2, 2x, sin(x), sqrt(x)", key="field_P")
    with col2:
        Q_str = st.text_input("Componente Q(x,y,z):", value="-x", help="Notación: x^2, 2x, sin(x), sqrt(x)", key="field_Q")
    with col3:
        R_str = st.text_input("Componente R(x,y,z):", value="0", help="Notación: x^2, 2x, sin(x), sqrt(x)", key="field_R")
    
    if st.button("🧮 Calcular Divergencia y Rotacional", type="primary", key="calc_field_btn"):
        try:
            x, y, z = sp.symbols('x y z')
            
            P = cv.safe_parse(P_str, (x, y, z))
            Q = cv.safe_parse(Q_str, (x, y, z))
            R = cv.safe_parse(R_str, (x, y, z))
            
            F_sym = (P, Q, R)
            
            # SIEMPRE calcular con pasos simbólicos
            with st.spinner("Calculando divergencia y rotacional..."):
                explicacion = cv.explain_div_curl_steps(F_sym, (x, y, z))
                
                # Guardar en session_state PRIMERO
                st.session_state['F_sym'] = F_sym
                st.session_state['F_strs'] = (P_str, Q_str, R_str)
                st.session_state['field_vars'] = (x, y, z)
                st.session_state['field_explicacion'] = explicacion
                st.session_state['field_computed'] = True
                
        except ValueError as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state['field_computed'] = False
        except Exception as e:
            st.error(f"❌ Error inesperado: {str(e)}")
            st.session_state['field_computed'] = False
    
    # MOSTRAR RESULTADOS SI EXISTEN (fuera del botón)
    if st.session_state.get('field_computed', False):
        explicacion = st.session_state['field_explicacion']
        P_str, Q_str, R_str = st.session_state['F_strs']
        F_sym = st.session_state['F_sym']
        x, y, z = st.session_state['field_vars']
        
        # Mostrar resultado principal
        st.success("✅ Cálculo completado con derivación simbólica")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("#### Divergencia ∇·F")
            st.latex(r"\nabla \cdot \mathbf{F} = " + explicacion['divergence_latex'])
            
            if explicacion['is_incompressible']:
                st.info("🔵 **Campo incompresible** (conserva volumen)")
            else:
                st.info("🔹 Campo compresible")
        
        with col_res2:
            st.markdown("#### Rotacional ∇×F")
            curl_latex = f"\\left( {explicacion['curl_latex'][0]}, {explicacion['curl_latex'][1]}, {explicacion['curl_latex'][2]} \\right)"
            st.latex(r"\nabla \times \mathbf{F} = " + curl_latex)
            
            if explicacion['is_conservative']:
                st.success("✅ **Campo conservativo** (existe potencial)")
            else:
                st.info("🌀 Campo rotacional")
        
        # Mostrar todos los pasos en un expander
        with st.expander("📖 Ver derivación completa paso a paso", expanded=False):
            for i, step in enumerate(explicacion['steps_latex']):
                st.markdown(f"### {step['title']}")
                if step['content']:
                    st.markdown(step['content'], unsafe_allow_html=True)
                if step['explanation']:
                    st.info(step['explanation'])
                if i < len(explicacion['steps_latex']) - 1:
                    st.markdown("---")
            
            # Botón de descarga
            markdown_content = "# Divergencia y Rotacional\n\n"
            markdown_content += f"**Campo vectorial:** F = ({P_str}, {Q_str}, {R_str})\n\n"
            markdown_content += "---\n\n"
            
            for step in explicacion['steps_latex']:
                markdown_content += f"## {step['title']}\n\n"
                if step['content']:
                    markdown_content += f"{step['content']}\n\n"
                if step['explanation']:
                    markdown_content += f"**Explicación:** {step['explanation']}\n\n"
                markdown_content += "---\n\n"
            
            st.download_button(
                label="📥 Descargar pasos en Markdown",
                data=markdown_content,
                file_name="div_curl_pasos.md",
                mime="text/markdown",
                key="download_field_md"
            )
        
        # ===== VISUALIZACIONES 3D =====
        st.markdown("---")
        st.markdown("## 📊 Visualizaciones 3D")
        
        # Controles de visualización
        col_viz1, col_viz2, col_viz3 = st.columns(3)
        with col_viz1:
            show_field = st.checkbox("Mostrar campo vectorial", value=True, key="field_show_field")
        with col_viz2:
            show_div = st.checkbox("Mostrar divergencia", value=True, key="field_show_div")
        with col_viz3:
            n_points_viz = st.slider("Resolución", 4, 12, 6, key="field_resolution")
        
        if show_field or show_div:
            import viz_vectorial as vv
            
            try:
                if show_div:
                    # Visualización combinada
                    fig_combined = vv.plot_combined_field_analysis(
                        F_sym,
                        (x, y, z),
                        bounds=((-2, 2), (-2, 2), (-2, 2)),
                        n_points=n_points_viz
                    )
                    st.plotly_chart(fig_combined, use_container_width=True)
                elif show_field:
                    # Solo campo vectorial
                    fig_field = vv.plot_vector_field_3d(
                        F_sym,
                        (x, y, z),
                        bounds=((-2, 2), (-2, 2), (-2, 2)),
                        n_points=n_points_viz,
                        title=f"Campo Vectorial F = ({P_str}, {Q_str}, {R_str})"
                    )
                    st.plotly_chart(fig_field, use_container_width=True)
                
                # Mapa de calor de divergencia
                if show_div and not explicacion['is_incompressible']:
                    st.markdown("### Mapa de Calor: Divergencia")
                    z_plane = st.slider("Plano z =", -2.0, 2.0, 0.0, 0.5, key="field_z_plane")
                    
                    fig_div_heatmap = vv.plot_divergence_heatmap(
                        F_sym,
                        (x, y, z),
                        z_plane=z_plane,
                        bounds=((-2, 2), (-2, 2)),
                        n_points=50
                    )
                    st.plotly_chart(fig_div_heatmap, use_container_width=True)
                    
                    st.info("🔴 **Rojo**: Fuente (divergencia > 0) | 🔵 **Azul**: Sumidero (divergencia < 0)")
                
            except Exception as e:
                st.warning(f"⚠️ No se pudo generar la visualización: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# OPCIÓN 2: GRADIENTE
# ============================================================================

elif funcionalidad == "Gradiente de Campo Escalar (∇φ)":
    st.header("Campo Escalar φ(x, y, z)")
    
    # Ayuda sobre notación matemática
    with st.expander("💡 ¿Cómo escribir funciones? (Notación estilo GeoGebra)", expanded=False):
        st.markdown(r"""
        **Puedes usar notación matemática natural:**
        
        | Operación | Ejemplo | También válido |
        |-----------|---------|----------------|
        | Potencias | `x^2 + y^3` | `x**2 + y**3` |
        | Multiplicación | `2x + 3y` | `2*x + 3*y` |
        | Raíz cuadrada | `sqrt(x)` | `raiz(x)` |
        | Seno | `sin(x)` | `sen(x)` |
        | Coseno | `cos(x)` | - |
        | Tangente | `tan(x)` | `tg(x)` |
        | Exponencial | `exp(x)` | `e^x` |
        | Logaritmo | `ln(x)` | `log(x)` |
        | Valor absoluto | `abs(x)` | `\|x\|` |
        | Pi | `pi` | `π` |
        
        **Ejemplos completos:**
        - Campo cuadrático: `x^2 + y^2 + z^2`
        - Campo con seno: `sin(x) + cos(y)`
        - Campo exponencial: `e^x * y^2`
        - Campo mixto: `2x*y + 3z^2`
        - Campo con raíz: `sqrt(x^2 + y^2)`
        """)
    
    phi_str = st.text_input(
        "Función φ(x, y, z):", 
        value="x^2 + y^2 + z^2", 
        help="Usa notación natural: x^2, 2x, sen(x), raiz(x), etc.",
        placeholder="Ejemplo: 2x*y + z^2, sin(x)*cos(y), e^x",
        key="phi_input"
    )
    
    if st.button("🧮 Calcular Gradiente", type="primary", key="calc_grad_btn"):
        try:
            x, y, z = sp.symbols('x y z')
            phi = cv.safe_parse(phi_str, (x, y, z))
            
            # SIEMPRE calcular con pasos simbólicos
            with st.spinner("Calculando gradiente..."):
                explicacion = cv.explain_gradient_steps(phi, (x, y, z))
            
            # Guardar en session_state PRIMERO
            st.session_state['grad_sym'] = explicacion['gradient_sym']
            st.session_state['grad_latex'] = explicacion['gradient_latex']
            st.session_state['grad_steps'] = explicacion['steps_latex']
            st.session_state['grad_phi'] = phi
            st.session_state['grad_phi_str'] = phi_str
            st.session_state['grad_vars'] = (x, y, z)
            st.session_state['grad_computed'] = True
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state['grad_computed'] = False
    
    # MOSTRAR RESULTADOS SI EXISTEN (fuera del botón)
    if st.session_state.get('grad_computed', False):
        # Mostrar resultado principal
        st.success("✅ Gradiente calculado")
        st.latex(f"\\nabla \\phi = {st.session_state['grad_latex']}")
        
        # Mostrar todos los pasos en un expander
        with st.expander("📖 Ver derivación completa paso a paso", expanded=False):
            for i, step in enumerate(st.session_state['grad_steps']):
                st.markdown(f"### {step['title']}")
                if step['content']:
                    st.markdown(step['content'], unsafe_allow_html=True)
                if step['explanation']:
                    st.info(step['explanation'])
                if i < len(st.session_state['grad_steps']) - 1:
                    st.markdown("---")
            
            # Botón de descarga
            markdown_content = "# Cálculo del Gradiente\n\n"
            markdown_content += f"**Campo escalar:** φ = {st.session_state['grad_phi_str']}\n\n"
            markdown_content += "---\n\n"
            
            for step in st.session_state['grad_steps']:
                markdown_content += f"## {step['title']}\n\n"
                if step['content']:
                    markdown_content += f"{step['content']}\n\n"
                if step['explanation']:
                    markdown_content += f"**Explicación:** {step['explanation']}\n\n"
                markdown_content += "---\n\n"
            
            st.download_button(
                label="📥 Descargar pasos en Markdown",
                data=markdown_content,
                file_name="gradiente_pasos.md",
                mime="text/markdown",
                key="download_grad_md"
            )
        
        st.info("""
        **Interpretación:**
        - ∇φ apunta hacia el máximo crecimiento
        - ∇φ ⊥ superficies de nivel
        - ||∇φ|| = tasa de cambio máxima
        """)
        
        # ===== VISUALIZACIONES 3D =====
        st.markdown("---")
        st.markdown("## 📊 Visualizaciones 3D")
        
        import viz_superficies as vs
        
        phi = st.session_state['grad_phi']
        x, y, z = st.session_state['grad_vars']
        phi_str = st.session_state['grad_phi_str']
        
        try:
            tab1, tab2, tab3 = st.tabs(["🌐 Campo Escalar", "📈 Gradiente", "🎯 Punto Específico"])
            
            with tab1:
                st.markdown("### Superficies de Nivel de φ")
                viz_method = st.radio("Método:", ["Isosuperficies", "Volumen"], horizontal=True, key="grad_viz_method")
                
                fig_scalar = vs.plot_scalar_field_3d(
                    phi,
                    (x, y, z),
                    bounds=((-2, 2), (-2, 2), (-2, 2)),
                    n_points=25,
                    method='isosurface' if viz_method == "Isosuperficies" else 'volume',
                    title=f"Campo Escalar φ = {phi_str}"
                )
                st.plotly_chart(fig_scalar, use_container_width=True)
                st.info("💡 Las superficies de nivel son aquellas donde φ tiene el mismo valor constante.")
            
            with tab2:
                st.markdown("### Campo de Gradiente ∇φ")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    show_scalar_bg = st.checkbox("Mostrar superficie escalar", value=True, key="grad_show_bg")
                with col_g2:
                    n_grad_points = st.slider("Densidad de vectores", 4, 10, 6, key="grad_density")
                
                use_2d = st.checkbox("Vista 2D (corte en plano)", value=False, key="grad_use_2d")
                
                if use_2d:
                    z_plane_grad = st.slider("Plano z =", -2.0, 2.0, 0.0, 0.5, key="grad_z_plane")
                    fig_grad = vs.plot_gradient_field(
                        phi,
                        (x, y, z),
                        bounds=((-2, 2), (-2, 2), (-2, 2)),
                        n_points=n_grad_points,
                        z_plane=z_plane_grad,
                        title=f"Gradiente ∇φ (z={z_plane_grad})"
                    )
                else:
                    fig_grad = vs.plot_gradient_field(
                        phi,
                        (x, y, z),
                        bounds=((-2, 2), (-2, 2), (-2, 2)),
                        n_points=n_grad_points,
                        show_scalar_surface=show_scalar_bg,
                        title="Campo de Gradiente ∇φ"
                    )
                
                st.plotly_chart(fig_grad, use_container_width=True)
                st.info("🔴 Los vectores rojos muestran la dirección del máximo crecimiento de φ.")
            
            with tab3:
                st.markdown("### Gradiente en un Punto Específico")
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    x_point = st.number_input("x₀:", value=1.0, step=0.5, key="grad_x_point")
                with col_p2:
                    y_point = st.number_input("y₀:", value=1.0, step=0.5, key="grad_y_point")
                with col_p3:
                    z_point = st.number_input("z₀:", value=0.0, step=0.5, key="grad_z_point")
                
                fig_point = vs.plot_gradient_at_point(
                    phi,
                    (x, y, z),
                    point=(x_point, y_point, z_point),
                    bounds=((-3, 3), (-3, 3), (-3, 3)),
                    title=f"Gradiente en P({x_point}, {y_point}, {z_point})"
                )
                st.plotly_chart(fig_point, use_container_width=True)
                
                # Calcular valor numérico del gradiente en ese punto
                grad_at_point = tuple(float(g.subs({x: x_point, y: y_point, z: z_point}).evalf()) 
                                     for g in st.session_state['grad_sym'])
                magnitude = np.sqrt(sum(g**2 for g in grad_at_point))
                
                st.success(f"∇φ({x_point}, {y_point}, {z_point}) = ({grad_at_point[0]:.4f}, {grad_at_point[1]:.4f}, {grad_at_point[2]:.4f})")
                st.info(f"Magnitud: ||∇φ|| = {magnitude:.4f}")
                
        except Exception as e:
            st.warning(f"⚠️ No se pudo generar la visualización: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            
        except ValueError as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Sección de evaluación - siempre visible si ya se calculó
    if st.session_state.get('phi_computed', False):
        st.markdown("---")
        
        # Tabs para diferentes funcionalidades
        tab1, tab2, tab3 = st.tabs(["📍 Evaluar en Punto", "🔍 Ver Paso a Paso", "📄 Exportar Informe"])
        
        with tab1:
            st.subheader("Evaluar en un punto")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_val = st.number_input("x =", value=1.0, key="grad_x_val")
            with col2:
                y_val = st.number_input("y =", value=0.0, key="grad_y_val")
            with col3:
                z_val = st.number_input("z =", value=0.0, key="grad_z_val")
            
            if st.button("📍 Evaluar", key="eval_grad_btn"):
                x, y, z = sp.symbols('x y z')
                phi = cv.safe_parse(phi_str, (x, y, z))
                grad_sym = st.session_state['grad_sym']
                
                # Evaluar simbólicamente
                grad_at_point = tuple(g.subs({x: x_val, y: y_val, z: z_val}) for g in grad_sym)
                grad_at_point = tuple(sp.simplify(g) for g in grad_at_point)
                
                # Magnitud simbólica
                magnitude_symbolic = sp.sqrt(sum(g**2 for g in grad_at_point))
                magnitude_symbolic = sp.simplify(magnitude_symbolic)
                
                st.markdown(f"**En el punto ({x_val}, {y_val}, {z_val}):**")
                
                st.latex(f"\\nabla\\phi = \\left( {sp.latex(grad_at_point[0])}, {sp.latex(grad_at_point[1])}, {sp.latex(grad_at_point[2])} \\right)")
                
                st.latex(f"\\|\\nabla\\phi\\| = {sp.latex(magnitude_symbolic)}")
                
                if magnitude_symbolic == 0:
                    st.warning("⚠️ Punto crítico (∇φ = 0)")
        
        with tab2:
            st.subheader("Explicación Paso a Paso")
            
            detail_level = st.selectbox(
                "Nivel de detalle:",
                ["basico", "intermedio", "completo"],
                index=1,
                key="detail_level"
            )
            
            if st.button("🔍 Generar Explicación", key="explain_grad_btn"):
                with st.spinner("Generando explicación detallada..."):
                    x, y, z = sp.symbols('x y z')
                    phi = cv.safe_parse(phi_str, (x, y, z))
                    
                    result = cv.explain_gradient_steps(phi, (x, y, z), detail_level)
                    
                    st.success(f"✅ Explicación generada en {result['execution_time']:.3f}s")
                    
                    st.markdown("#### Gradiente Completo")
                    st.latex(r"\nabla\phi = " + result['gradient_latex'])
                    
                    st.markdown("#### Pasos de la Derivación")
                    
                    with st.expander("📖 Ver todos los pasos", expanded=True):
                        for step in result['steps_latex']:
                            st.markdown(f"### {step['title']}")
                            if step['content']:
                                st.markdown(step['content'])
                            if step['explanation']:
                                st.info(step['explanation'])
                            st.markdown("---")
                    
                    if result['critical_points']:
                        st.markdown("#### Puntos Críticos (∇φ = 0)")
                        st.success(f"Se encontraron {len(result['critical_points'])} punto(s) crítico(s):")
                        
                        for idx, point in enumerate(result['critical_points']):
                            st.write(f"**Punto {idx+1}:** {point}")
                            # Evaluar φ en el punto crítico
                            try:
                                phi_at_point = phi.subs([(x, point[0]), (y, point[1]), (z, point[2])])
                                st.write(f"φ{point} = {phi_at_point}")
                            except:
                                pass
                    else:
                        st.info("No se encontraron puntos críticos simbólicamente")
        
        with tab3:
            st.subheader("Exportar Informe PDF")
            
            if st.button("📄 Generar PDF", key="export_grad_pdf_btn"):
                with st.spinner("Generando informe PDF..."):
                    x, y, z = sp.symbols('x y z')
                    phi = cv.safe_parse(phi_str, (x, y, z))
                    grad_sym = st.session_state['grad_sym']
                    
                    import time
                    report = {
                        'title': 'Informe de Análisis de Gradiente',
                        'author': 'Sistema de Cálculo Vectorial',
                        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'sections': [
                            {
                                'title': 'Función Escalar Analizada',
                                'text': f'Se analizó la función φ(x,y,z) = {phi_str}',
                                'latex': [r'\phi(x,y,z) = ' + sp.latex(phi)],
                                'fig_paths': [],
                                'data': {}
                            },
                            {
                                'title': 'Gradiente Calculado',
                                'text': 'El gradiente es el vector de derivadas parciales que indica la dirección de máximo crecimiento.',
                                'latex': [
                                    r'\nabla\phi = ' + cv.format_vector_latex(grad_sym),
                                    r'\frac{\partial\phi}{\partial x} = ' + sp.latex(grad_sym[0]),
                                    r'\frac{\partial\phi}{\partial y} = ' + sp.latex(grad_sym[1]),
                                    r'\frac{\partial\phi}{\partial z} = ' + sp.latex(grad_sym[2])
                                ],
                                'fig_paths': [],
                                'data': {
                                    'Componente x': str(grad_sym[0]),
                                    'Componente y': str(grad_sym[1]),
                                    'Componente z': str(grad_sym[2])
                                }
                            }
                        ]
                    }
                    
                    pdf_path = cv.export_report_pdf(report, f'gradiente_{int(time.time())}')
                    
                    st.success(f"✅ PDF generado: {pdf_path}")
                    
                    # Leer el PDF y ofrecer descarga
                    with open(pdf_path, 'rb') as f:
                        pdf_bytes = f.read()
                    
                    st.download_button(
                        label="⬇️ Descargar PDF",
                        data=pdf_bytes,
                        file_name=f'informe_gradiente.pdf',
                        mime='application/pdf',
                        key="download_grad_pdf"
                    )

# ============================================================================
# OPCIÓN 3: INTEGRAL DE LÍNEA
# ============================================================================

elif funcionalidad == "Integral de Línea (∮ F·dr)":
    st.header("Integral de Línea")
    
    st.markdown("##### Campo Vectorial F")
    col1, col2, col3 = st.columns(3)
    with col1:
        P_str = st.text_input("P:", value="-y", key="line_P")
    with col2:
        Q_str = st.text_input("Q:", value="x", key="line_Q")
    with col3:
        R_str = st.text_input("R:", value="0", key="line_R")
    
    st.markdown("##### Curva r(t)")
    col1, col2, col3 = st.columns(3)
    with col1:
        x_t = st.text_input("x(t):", value="cos(t)")
    with col2:
        y_t = st.text_input("y(t):", value="sin(t)")
    with col3:
        z_t = st.text_input("z(t):", value="0")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        t0 = st.number_input("t₀:", value=0.0)
    with col2:
        t1 = st.number_input("t₁:", value=2*np.pi)
    with col3:
        n = st.number_input("Puntos:", value=2000, min_value=100)
    
    if st.button("🧮 Calcular Integral", type="primary", key="calc_line_integral_btn"):
        try:
            x, y, z, t = sp.symbols('x y z t')
            
            # Parsear campo vectorial
            P = cv.safe_parse(P_str, (x, y, z))
            Q = cv.safe_parse(Q_str, (x, y, z))
            R = cv.safe_parse(R_str, (x, y, z))
            
            # Parsear parametrización
            xt = cv.safe_parse(x_t, (t,))
            yt = cv.safe_parse(y_t, (t,))
            zt = cv.safe_parse(z_t, (t,))
            
            # SIEMPRE calcular con pasos simbólicos
            with st.spinner("Calculando integral..."):
                explanation = cv.explain_line_integral_steps(
                    (P, Q, R),
                    (xt, yt, zt),
                    t,
                    t0,
                    t1
                )
            
            # Guardar en session_state
            st.session_state['line_explanation'] = explanation
            st.session_state['line_F'] = (P, Q, R)
            st.session_state['line_r'] = (xt, yt, zt)
            st.session_state['line_strs'] = (P_str, Q_str, R_str, x_t, y_t, z_t)
            st.session_state['line_vars'] = (x, y, z, t)
            st.session_state['line_t_range'] = (t0, t1)
            st.session_state['line_computed'] = True
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state['line_computed'] = False
    
    # MOSTRAR RESULTADOS SI EXISTEN
    if st.session_state.get('line_computed', False):
        explanation = st.session_state['line_explanation']
        P, Q, R = st.session_state['line_F']
        xt, yt, zt = st.session_state['line_r']
        P_str, Q_str, R_str, x_t, y_t, z_t = st.session_state['line_strs']
        x, y, z, t = st.session_state['line_vars']
        x, y, z, t = st.session_state['line_vars']
        t0, t1 = st.session_state['line_t_range']
        
        # Mostrar resultado principal
        st.success("✅ Integral calculada")
        
        st.markdown("### Resultado:")
        if explanation['definite_symbolic']:
            st.latex(r"\oint_C \mathbf{F} \cdot d\mathbf{r} = " + sp.latex(explanation['definite_symbolic']))
        else:
            st.latex(r"\oint_C \mathbf{F} \cdot d\mathbf{r} \approx " + f"{explanation['numeric_value']:.10f}")
        
        # Mostrar pasos en un expander
        with st.expander("📖 Ver derivación completa paso a paso", expanded=False):
            for step in explanation['steps_latex']:
                st.markdown(f"### {step['title']}")
                # Usar markdown si tiene delimitadores $$, de lo contrario latex
                if step['content'].startswith('$$') or '$$\\n' in step['content']:
                    st.markdown(step['content'], unsafe_allow_html=True)
                else:
                    st.latex(step['content'])
                if step['explanation']:
                    st.info(step['explanation'])
                st.markdown("---")
            
            # Botón para copiar como markdown
            markdown_text = "# Solución de la Integral de Línea\n\n"
            markdown_text += f"**Campo vectorial:** F = ({P_str}, {Q_str}, {R_str})\n\n"
            markdown_text += f"**Parametrización:** r(t) = ({x_t}, {y_t}, {z_t})\n\n"
            markdown_text += f"**Intervalo:** t ∈ [{t0}, {t1}]\n\n"
            markdown_text += "## Pasos de resolución\n\n"
            
            for step in explanation['steps_latex']:
                markdown_text += f"### {step['title']}\n\n"
                markdown_text += f"{step['content']}\n\n"
                if step['explanation']:
                    markdown_text += f"> {step['explanation']}\n\n"
                markdown_text += "---\n\n"
            
            markdown_text += f"\n## Resultado Final\n\n"
            if explanation['definite_symbolic']:
                markdown_text += f"$$\\oint_C \\mathbf{{F}} \\cdot d\\mathbf{{r}} = {sp.latex(explanation['definite_symbolic'])}$$\n\n"
            markdown_text += f"Valor numérico: {explanation['numeric_value']:.10f}\n\n"
            
            st.download_button(
                label="📥 Descargar pasos (Markdown)",
                data=markdown_text,
                file_name="integral_linea_pasos.md",
                mime="text/markdown",
                key="download_line_integral_steps"
            )
        
        # ===== VISUALIZACIONES 3D =====
        st.markdown("---")
        st.markdown("## 📊 Visualizaciones 3D")
        
        import viz_curvas as vc
        
        try:
            tab_viz1, tab_viz2, tab_viz3 = st.tabs(["🌀 Integral Completa", "📍 Curva + Campo", "📈 Integrando"])
            
            with tab_viz1:
                st.markdown("### Visualización Completa: ∫ F·dr")
                n_curve_pts = st.slider("Puntos en curva", 50, 200, 100, key="line_n_curve_viz")
                n_field_pts = st.slider("Vectores de campo", 5, 20, 10, key="line_n_field_viz")
                
                fig_full = vc.plot_line_integral_visualization(
                    (P, Q, R),
                    (xt, yt, zt),
                    t,
                    (t0, t1),
                    (x, y, z),
                    n_points=n_curve_pts,
                    n_field_points=n_field_pts,
                    title=f"∫ F·dr sobre C"
                )
                st.plotly_chart(fig_full, use_container_width=True)
                
                st.info("""
                **Interpretación del color:**
                - 🟢 **Verde**: F·dr > 0 (trabajo positivo, campo ayuda al movimiento)
                - 🔴 **Rojo**: F·dr < 0 (trabajo negativo, campo se opone al movimiento)
                - 🟡 **Amarillo**: F·dr ≈ 0 (campo perpendicular a la trayectoria)
                """)
            
            with tab_viz2:
                st.markdown("### Curva Parametrizada con Campo Vectorial")
                show_tangent_vec = st.checkbox("Mostrar vectores tangentes r'(t)", value=True, key="line_show_tangent")
                tangent_count = st.slider("Número de tangentes", 5, 15, 10, key="line_tangent_count") if show_tangent_vec else 0
                
                fig_curve = vc.plot_parametric_curve_3d(
                    (xt, yt, zt),
                    t,
                    (t0, t1),
                    n_points=100,
                    show_tangent=show_tangent_vec,
                    tangent_points=tangent_count,
                    title=f"Curva r(t) = ({x_t}, {y_t}, {z_t})"
                )
                st.plotly_chart(fig_curve, use_container_width=True)
                
                st.info("🟢 **Punto inicial** | 🔴 **Punto final** | 🟠 **Vectores tangentes** (dirección de la curva)")
            
            with tab_viz3:
                st.markdown("### Gráfica del Integrando F·(dr/dt)")
                
                # Mostrar el valor de la integral con el signo correcto
                integral_value = explanation['numeric_value']
                if explanation['definite_symbolic']:
                    st.latex(f"\\int_{{{t0}}}^{{{t1}}} \\mathbf{{F}} \\cdot \\frac{{d\\mathbf{{r}}}}{{dt}} \\, dt = {sp.latex(explanation['definite_symbolic'])} = {integral_value:.6f}")
                else:
                    st.latex(f"\\int_{{{t0}}}^{{{t1}}} \\mathbf{{F}} \\cdot \\frac{{d\\mathbf{{r}}}}}}{{dt}} \\, dt \\approx {integral_value:.6f}")
                
                # Determinar el color del área según el signo
                if integral_value > 0:
                    area_color = "🟢 **Área POSITIVA**"
                    interpretation = "El trabajo neto es **a favor** del campo vectorial"
                elif integral_value < 0:
                    area_color = "🔴 **Área NEGATIVA**"
                    interpretation = "El trabajo neto es **en contra** del campo vectorial"
                else:
                    area_color = "⚪ **Área CERO**"
                    interpretation = "El trabajo neto es **cero** (campo perpendicular o compensado)"
                
                st.markdown(f"**Resultado:** {area_color} → {interpretation}")
                
                fig_integrand = vc.plot_integrand_graph(
                    (P, Q, R),
                    (xt, yt, zt),
                    t,
                    (t0, t1),
                    (x, y, z),
                    n_points=300,
                    title=f"Integrando F·(dr/dt) vs t ∈ [{t0}, {t1}]"
                )
                st.plotly_chart(fig_integrand, use_container_width=True)
                
                st.info(f"""
                **Interpretación Geométrica:**
                - El **área bajo la curva** = {integral_value:.6f}
                - Área por encima del eje t (positiva) = trabajo a favor del campo
                - Área por debajo del eje t (negativa) = trabajo contra el campo
                - El valor de la integral es la suma algebraica de ambas áreas
                
                **Ejemplo Clásico:**
                Para $F = \\langle y, -x, 0 \\rangle$ (campo rotacional) sobre el círculo 
                $r(t) = (\\cos t, \\sin t, 0)$ con $t \\in [0, 2\\pi]$:
                - El integrando es $F \\cdot dr/dt = -1$ (constante negativo)
                - La integral completa da $\\boxed{{-2\\pi \\approx -6.283}}$
                """)
                
                # Ejemplo de verificación si es el campo rotacional clásico
                if (P_str.strip() == 'y' and Q_str.strip() == '-x' and R_str.strip() == '0' and
                    x_t.strip() == 'cos(t)' and y_t.strip() == 'sin(t)' and z_t.strip() == '0' and
                    abs(t0) < 0.01 and abs(t1 - 2*np.pi) < 0.01):
                    st.success(f"""
                    ✅ **¡Campo Rotacional Clásico Detectado!**
                    
                    Has ingresado el ejemplo emblemático de integral de línea:
                    - Campo: $F = \\langle y, -x, 0 \\rangle$ (rotación antihoraria)
                    - Curva: Círculo unitario completo
                    - Resultado teórico: $-2\\pi \\approx -6.283185...$
                    - Resultado calculado: ${integral_value:.10f}$
                    - Error: ${abs(integral_value + 2*np.pi):.2e}$
                    
                    Este campo tiene **circulación negativa** (rotación horaria del flujo).
                    """)
        
        except Exception as e:
            st.warning(f"⚠️ No se pudo generar la visualización: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# ============================================================================
# OPCIÓN 4: FLUJO DE SUPERFICIE
# ============================================================================

elif funcionalidad == "Flujo de Superficie (∬ F·n dS)":
    st.header("Flujo de Superficie")
    
    st.markdown("##### Campo F")
    col1, col2, col3 = st.columns(3)
    with col1:
        P_str = st.text_input("P:", value="0", key="surf_P")
    with col2:
        Q_str = st.text_input("Q:", value="0", key="surf_Q")
    with col3:
        R_str = st.text_input("R:", value="1", key="surf_R")
    
    st.markdown("##### Superficie r(u,v)")
    col1, col2, col3 = st.columns(3)
    with col1:
        x_uv = st.text_input("x(u,v):", value="u")
    with col2:
        y_uv = st.text_input("y(u,v):", value="v")
    with col3:
        z_uv = st.text_input("z(u,v):", value="0")
    
    col1, col2 = st.columns(2)
    with col1:
        u0 = st.number_input("u₀:", value=0.0, key="u0")
        u1 = st.number_input("u₁:", value=1.0, key="u1")
        nu = st.number_input("Nu:", value=100, key="nu")
    with col2:
        v0 = st.number_input("v₀:", value=0.0, key="v0")
        v1 = st.number_input("v₁:", value=1.0, key="v1")
        nv = st.number_input("Nv:", value=100, key="nv")
    
    if st.button("🧮 Calcular Flujo", type="primary", key="calc_flux_btn"):
        try:
            x, y, z, u, v = sp.symbols('x y z u v')
            
            # Parsear campo vectorial
            P = cv.safe_parse(P_str, (x, y, z))
            Q = cv.safe_parse(Q_str, (x, y, z))
            R = cv.safe_parse(R_str, (x, y, z))
            
            # Parsear parametrización
            xuv = cv.safe_parse(x_uv, (u, v))
            yuv = cv.safe_parse(y_uv, (u, v))
            zuv = cv.safe_parse(z_uv, (u, v))
            
            # SIEMPRE calcular con pasos simbólicos
            with st.spinner("Calculando flujo de superficie..."):
                explanation = cv.explain_surface_flux_steps(
                    (P, Q, R),
                    (xuv, yuv, zuv),
                    u, v,
                    u0, u1, v0, v1
                )
            
            # Guardar en session_state
            st.session_state['flux_computed'] = True
            st.session_state['flux_explanation'] = explanation
            st.session_state['flux_P_str'] = P_str
            st.session_state['flux_Q_str'] = Q_str
            st.session_state['flux_R_str'] = R_str
            st.session_state['flux_xuv'] = xuv
            st.session_state['flux_yuv'] = yuv
            st.session_state['flux_zuv'] = zuv
            st.session_state['flux_x_uv'] = x_uv
            st.session_state['flux_y_uv'] = y_uv
            st.session_state['flux_z_uv'] = z_uv
            st.session_state['flux_u0'] = u0
            st.session_state['flux_u1'] = u1
            st.session_state['flux_v0'] = v0
            st.session_state['flux_v1'] = v1
            
        except ValueError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ Error inesperado: {str(e)}")
    
    # Mostrar resultados si existen en session_state
    if st.session_state.get('flux_computed', False):
        explanation = st.session_state['flux_explanation']
        P_str = st.session_state['flux_P_str']
        Q_str = st.session_state['flux_Q_str']
        R_str = st.session_state['flux_R_str']
        xuv = st.session_state['flux_xuv']
        yuv = st.session_state['flux_yuv']
        zuv = st.session_state['flux_zuv']
        x_uv = st.session_state['flux_x_uv']
        y_uv = st.session_state['flux_y_uv']
        z_uv = st.session_state['flux_z_uv']
        u0 = st.session_state['flux_u0']
        u1 = st.session_state['flux_u1']
        v0 = st.session_state['flux_v0']
        v1 = st.session_state['flux_v1']
        
        u, v = sp.symbols('u v')
        
        # Mostrar resultado principal
        st.success("✅ Flujo calculado")
        
        st.markdown("### Resultado:")
        if explanation['definite_symbolic']:
            st.latex(r"\iint_S \mathbf{F} \cdot \mathbf{n} \, dS = " + sp.latex(explanation['definite_symbolic']))
        else:
            st.latex(r"\iint_S \mathbf{F} \cdot \mathbf{n} \, dS \approx " + f"{explanation['numeric_value']:.10f}")
        
        # Mostrar pasos en un expander
        with st.expander("📖 Ver derivación completa paso a paso", expanded=False):
            for step in explanation['steps_latex']:
                st.markdown(f"### {step['title']}")
                # Usar markdown si tiene delimitadores $$, de lo contrario latex
                if step['content'].startswith('$$') or '$$\\n' in step['content']:
                    st.markdown(step['content'], unsafe_allow_html=True)
                else:
                    st.latex(step['content'])
                if step['explanation']:
                    st.info(step['explanation'])
                st.markdown("---")
            
            # Botón para copiar como markdown
            markdown_text = "# Solución del Flujo de Superficie\n\n"
            markdown_text += f"**Campo vectorial:** F = ({P_str}, {Q_str}, {R_str})\n\n"
            markdown_text += f"**Parametrización:** r(u,v) = ({x_uv}, {y_uv}, {z_uv})\n\n"
            markdown_text += f"**Dominio:** u ∈ [{u0}, {u1}], v ∈ [{v0}, {v1}]\n\n"
            markdown_text += "## Pasos de resolución\n\n"
            
            for step in explanation['steps_latex']:
                markdown_text += f"### {step['title']}\n\n"
                markdown_text += f"{step['content']}\n\n"
                if step['explanation']:
                    markdown_text += f"> {step['explanation']}\n\n"
                markdown_text += "---\n\n"
            
            markdown_text += f"\n## Resultado Final\n\n"
            if explanation['definite_symbolic']:
                markdown_text += f"$$\\iint_S \\mathbf{{F}} \\cdot \\mathbf{{n}} \\, dS = {sp.latex(explanation['definite_symbolic'])}$$\n\n"
            markdown_text += f"Valor numérico: {explanation['numeric_value']:.10f}\n\n"
            
            st.download_button(
                label="📥 Descargar pasos (Markdown)",
                data=markdown_text,
                file_name="flujo_superficie_pasos.md",
                mime="text/markdown",
                key="download_flux_steps"
            )
        
        # ===== VISUALIZACIONES 3D =====
        st.markdown("---")
        st.markdown("## 📊 Visualizaciones 3D")
        
        import viz_curvas as vc
        
        try:
            st.markdown("### Superficie Parametrizada con Campo Vectorial")
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                show_normals = st.checkbox("Mostrar vectores normales", value=True, key="flux_show_normals")
                n_u_viz = st.slider("Resolución en u", 10, 40, 20, key="flux_n_u_surf")
            with col_s2:
                normal_density = st.slider("Densidad de normales", 5, 15, 8, key="flux_n_norm") if show_normals else 0
                n_v_viz = st.slider("Resolución en v", 10, 40, 20, key="flux_n_v_surf")
            
            fig_surface = vc.plot_surface_parametric(
                (xuv, yuv, zuv),
                u, v,
                (u0, u1),
                (v0, v1),
                n_u=n_u_viz,
                n_v=n_v_viz,
                show_normal=show_normals,
                normal_points=normal_density,
                title=f"Superficie r(u,v) = ({x_uv}, {y_uv}, {z_uv})"
            )
            st.plotly_chart(fig_surface, use_container_width=True)
            
            st.info("""
            **Componentes visualizados:**
            - 🟦 **Superficie S**: Parametrización r(u,v)
            - 🔴 **Vectores normales n**: Perpendiculares a la superficie (r_u × r_v)
            - El flujo mide cuánto campo F atraviesa la superficie en dirección de n
            """)
            
        except Exception as e:
            st.warning(f"⚠️ No se pudo generar la visualización: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"❌ Error inesperado: {str(e)}")

# ============================================================================
# OPCIÓN 5: TEOREMA DE STOKES
# ============================================================================

elif funcionalidad == "Verificar Teorema de Stokes":
    st.header("Teorema de Stokes")
    
    st.latex(r"\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot \mathbf{n} \, dS")
    
    st.info("**Ejemplo:** Disco unitario con F=(-y, x, 0) → ambos lados = 2π")
    
    st.markdown("##### Campo F")
    col1, col2, col3 = st.columns(3)
    with col1:
        P_str = st.text_input("P:", value="-y", key="st_P")
    with col2:
        Q_str = st.text_input("Q:", value="x", key="st_Q")
    with col3:
        R_str = st.text_input("R:", value="0", key="st_R")
    
    st.markdown("##### Frontera C")
    col1, col2, col3 = st.columns(3)
    with col1:
        xb = st.text_input("x(t):", value="cos(t)", key="st_xb")
    with col2:
        yb = st.text_input("y(t):", value="sin(t)", key="st_yb")
    with col3:
        zb = st.text_input("z(t):", value="0", key="st_zb")
    
    col1, col2 = st.columns(2)
    with col1:
        tb0 = st.number_input("t₀:", value=0.0, key="st_t0")
    with col2:
        tb1 = st.number_input("t₁:", value=2*np.pi, key="st_t1")
    
    st.markdown("##### Superficie S")
    col1, col2, col3 = st.columns(3)
    with col1:
        xs = st.text_input("x(u,v):", value="u*cos(v)", key="st_xs")
    with col2:
        ys = st.text_input("y(u,v):", value="u*sin(v)", key="st_ys")
    with col3:
        zs = st.text_input("z(u,v):", value="0", key="st_zs")
    
    col1, col2 = st.columns(2)
    with col1:
        us0 = st.number_input("u₀:", value=0.0, key="st_u0")
        us1 = st.number_input("u₁:", value=1.0, key="st_u1")
    with col2:
        vs0 = st.number_input("v₀:", value=0.0, key="st_v0")
        vs1 = st.number_input("v₁:", value=2*np.pi, key="st_v1")
    
    tol = st.number_input("Tolerancia:", value=1e-2, format="%.1e")
    
    if st.button("🧮 Verificar Stokes", type="primary", key="verify_stokes_btn"):
        try:
            x, y, z, t, u, v = sp.symbols('x y z t u v')
            
            P = cv.safe_parse(P_str, (x, y, z))
            Q = cv.safe_parse(Q_str, (x, y, z))
            R = cv.safe_parse(R_str, (x, y, z))
            F_sym = (P, Q, R)
            
            xbt = cv.safe_parse(xb, (t,))
            ybt = cv.safe_parse(yb, (t,))
            zbt = cv.safe_parse(zb, (t,))
            r_boundary_sym = (xbt, ybt, zbt)
            
            xsuv = cv.safe_parse(xs, (u, v))
            ysuv = cv.safe_parse(ys, (u, v))
            zsuv = cv.safe_parse(zs, (u, v))
            r_surface_sym = (xsuv, ysuv, zsuv)
            
            # SIEMPRE calcular con pasos simbólicos
            with st.spinner("Verificando Teorema de Stokes..."):
                explicacion = cv.explain_stokes_verification_steps(
                    F_sym, r_boundary_sym, t, tb0, tb1,
                    r_surface_sym, u, v, us0, us1, vs0, vs1
                )
            
            # Guardar en session_state
            st.session_state['stokes_computed'] = True
            st.session_state['stokes_explicacion'] = explicacion
            st.session_state['stokes_P_str'] = P_str
            st.session_state['stokes_Q_str'] = Q_str
            st.session_state['stokes_R_str'] = R_str
            st.session_state['stokes_P'] = P
            st.session_state['stokes_Q'] = Q
            st.session_state['stokes_R'] = R
            st.session_state['stokes_xb'] = xb
            st.session_state['stokes_yb'] = yb
            st.session_state['stokes_zb'] = zb
            st.session_state['stokes_xs'] = xs
            st.session_state['stokes_ys'] = ys
            st.session_state['stokes_zs'] = zs
            st.session_state['stokes_xbt'] = xbt
            st.session_state['stokes_ybt'] = ybt
            st.session_state['stokes_zbt'] = zbt
            st.session_state['stokes_xsuv'] = xsuv
            st.session_state['stokes_ysuv'] = ysuv
            st.session_state['stokes_zsuv'] = zsuv
            st.session_state['stokes_tb0'] = tb0
            st.session_state['stokes_tb1'] = tb1
            st.session_state['stokes_us0'] = us0
            st.session_state['stokes_us1'] = us1
            st.session_state['stokes_vs0'] = vs0
            st.session_state['stokes_vs1'] = vs1
            
        except ValueError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ Error inesperado: {str(e)}")
    
    # Mostrar resultados si existen en session_state
    if st.session_state.get('stokes_computed', False):
        explicacion = st.session_state['stokes_explicacion']
        P_str = st.session_state['stokes_P_str']
        Q_str = st.session_state['stokes_Q_str']
        R_str = st.session_state['stokes_R_str']
        P = st.session_state['stokes_P']
        Q = st.session_state['stokes_Q']
        R = st.session_state['stokes_R']
        xb = st.session_state['stokes_xb']
        yb = st.session_state['stokes_yb']
        zb = st.session_state['stokes_zb']
        xs = st.session_state['stokes_xs']
        ys = st.session_state['stokes_ys']
        zs = st.session_state['stokes_zs']
        xbt = st.session_state['stokes_xbt']
        ybt = st.session_state['stokes_ybt']
        zbt = st.session_state['stokes_zbt']
        xsuv = st.session_state['stokes_xsuv']
        ysuv = st.session_state['stokes_ysuv']
        zsuv = st.session_state['stokes_zsuv']
        tb0 = st.session_state['stokes_tb0']
        tb1 = st.session_state['stokes_tb1']
        us0 = st.session_state['stokes_us0']
        us1 = st.session_state['stokes_us1']
        vs0 = st.session_state['stokes_vs0']
        vs1 = st.session_state['stokes_vs1']
        
        x, y, z, t, u, v = sp.symbols('x y z t u v')
        
        # Mostrar resultado principal
        if explicacion['line_integral_symbolic'] and explicacion['surface_integral_symbolic']:
            st.success(f"✅ Teorema de Stokes verificado simbólicamente")
            
            col1, col2 = st.columns(2)
            with col1:
                st.latex(f"\\oint_C \\mathbf{{F}} \\cdot d\\mathbf{{r}} = {sp.latex(explicacion['line_integral_symbolic'])}")
            with col2:
                st.latex(f"\\iint_S (\\nabla \\times \\mathbf{{F}}) \\cdot \\mathbf{{n}} \\, dS = {sp.latex(explicacion['surface_integral_symbolic'])}")
        else:
            st.info("Verificación numérica")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("∮ F·dr", f"{explicacion['line_integral_numeric']:.10f}")
            with col2:
                st.metric("∬ (∇×F)·n dS", f"{explicacion['surface_integral_numeric']:.10f}")
        
        # Mostrar todos los pasos en un expander
        with st.expander("📖 Ver derivación completa paso a paso", expanded=True):
            for i, step in enumerate(explicacion['steps_latex']):
                st.markdown(f"### {step['title']}")
                if step['content']:
                    st.markdown(step['content'], unsafe_allow_html=True)
                if step['explanation']:
                    st.info(step['explanation'])
                if i < len(explicacion['steps_latex']) - 1:
                    st.markdown("---")
        
        # Botón de descarga
        markdown_content = "# Verificación del Teorema de Stokes\n\n"
        markdown_content += f"**Campo vectorial:** F = ({P_str}, {Q_str}, {R_str})\n\n"
        markdown_content += f"**Frontera C:** r(t) = ({xb}, {yb}, {zb}), t ∈ [{tb0}, {tb1}]\n\n"
        markdown_content += f"**Superficie S:** r(u,v) = ({xs}, {ys}, {zs}), u ∈ [{us0}, {us1}], v ∈ [{vs0}, {vs1}]\n\n"
        markdown_content += "---\n\n"
        
        for step in explicacion['steps_latex']:
            markdown_content += f"## {step['title']}\n\n"
            if step['content']:
                markdown_content += f"{step['content']}\n\n"
            if step['explanation']:
                markdown_content += f"**Explicación:** {step['explanation']}\n\n"
            markdown_content += "---\n\n"
        
        st.download_button(
            label="📥 Descargar pasos en Markdown",
            data=markdown_content,
            file_name="stokes_pasos.md",
            mime="text/markdown",
            key="stokes_download_btn"
        )
        
        # ============ VISUALIZACIONES 3D ============
        st.markdown("---")
        st.markdown("## 📊 Visualizaciones 3D")
        
        import viz_curvas as vc
        
        try:
            tab1, tab2, tab3 = st.tabs(["🌀 Teorema Completo", "📐 Superficie S", "🔄 Campo F y ∇×F"])
            
            with tab1:
                st.markdown("### Visualización Completa del Teorema de Stokes")
                st.info("💡 **Superficie azul** = S, **Curva roja** = frontera ∂S = C, **Flechas** = campo F")
                
                # Crear figura combinada
                fig_stokes = go.Figure()
                
                # 1. Superficie S
                u_vals = np.linspace(us0, us1, 30)
                v_vals = np.linspace(vs0, vs1, 30)
                U_surf, V_surf = np.meshgrid(u_vals, v_vals)
                
                xs_func = sp.lambdify((u, v), xsuv, modules=['numpy'])
                ys_func = sp.lambdify((u, v), ysuv, modules=['numpy'])
                zs_func = sp.lambdify((u, v), zsuv, modules=['numpy'])
                
                X_surf = vc.ensure_array(xs_func, U_surf, V_surf)
                Y_surf = vc.ensure_array(ys_func, U_surf, V_surf)
                Z_surf = vc.ensure_array(zs_func, U_surf, V_surf)
                
                fig_stokes.add_trace(go.Surface(
                    x=X_surf, y=Y_surf, z=Z_surf,
                    colorscale='Blues',
                    opacity=0.7,
                    showscale=False,
                    name='Superficie S'
                ))
                
                # 2. Curva frontera C
                t_vals_boundary = np.linspace(tb0, tb1, 100)
                xb_func = sp.lambdify(t, xbt, modules=['numpy'])
                yb_func = sp.lambdify(t, ybt, modules=['numpy'])
                zb_func = sp.lambdify(t, zbt, modules=['numpy'])
                
                x_boundary = vc.ensure_array(xb_func, t_vals_boundary)
                y_boundary = vc.ensure_array(yb_func, t_vals_boundary)
                z_boundary = vc.ensure_array(zb_func, t_vals_boundary)
                
                fig_stokes.add_trace(go.Scatter3d(
                    x=x_boundary, y=y_boundary, z=z_boundary,
                    mode='lines',
                    line=dict(color='red', width=8),
                    name='Frontera C = ∂S'
                ))
                
                # 3. Campo vectorial F a lo largo de C
                t_field = np.linspace(tb0, tb1, 15)
                x_f = vc.ensure_array(xb_func, t_field)
                y_f = vc.ensure_array(yb_func, t_field)
                z_f = vc.ensure_array(zb_func, t_field)
                
                P_func = sp.lambdify((x, y, z), P, modules=['numpy'])
                Q_func = sp.lambdify((x, y, z), Q, modules=['numpy'])
                R_func = sp.lambdify((x, y, z), R, modules=['numpy'])
                
                F_P = vc.ensure_array(P_func, x_f, y_f, z_f)
                F_Q = vc.ensure_array(Q_func, x_f, y_f, z_f)
                F_R = vc.ensure_array(R_func, x_f, y_f, z_f)
                
                fig_stokes.add_trace(go.Cone(
                    x=x_f, y=y_f, z=z_f,
                    u=F_P, v=F_Q, w=F_R,
                    colorscale='Greens',
                    sizemode='scaled',
                    sizeref=0.3,
                    showscale=False,
                    name='Campo F'
                ))
                
                fig_stokes.update_layout(
                    title="Teorema de Stokes: ∮_C F·dr = ∬_S (∇×F)·n dS",
                    scene=dict(
                        xaxis=dict(title='x'),
                        yaxis=dict(title='y'),
                        zaxis=dict(title='z'),
                        aspectmode='data'
                    ),
                    width=900,
                    height=700
                )
                
                st.plotly_chart(fig_stokes, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    line_result = sp.latex(explicacion['line_integral_symbolic']) if explicacion['line_integral_symbolic'] else f"{explicacion['line_integral_numeric']:.6f}"
                    st.latex(f"\\oint_C \\mathbf{{F}} \\cdot d\\mathbf{{r}} = {line_result}")
                with col2:
                    surf_result = sp.latex(explicacion['surface_integral_symbolic']) if explicacion['surface_integral_symbolic'] else f"{explicacion['surface_integral_numeric']:.6f}"
                    st.latex(f"\\iint_S (\\nabla \\times \\mathbf{{F}}) \\cdot \\mathbf{{n}} \\, dS = {surf_result}")
            
            with tab2:
                st.markdown("### Superficie S con Vectores Normales")
                
                n_normal = st.slider("Densidad de normales", 3, 10, 5, key="stokes_normal_density")
                
                fig_surf = vc.plot_surface_parametric(
                    (xsuv, ysuv, zsuv),
                    u, v,
                    (us0, us1), (vs0, vs1),
                    n_u=30, n_v=30,
                    show_normal=True,
                    normal_points=n_normal,
                    title=f"Superficie S: r(u,v) = ({xs}, {ys}, {zs})"
                )
                st.plotly_chart(fig_surf, use_container_width=True)
                st.info("💡 Los vectores normales **n** apuntan en dirección perpendicular a la superficie S")
            
            with tab3:
                st.markdown("### Campo F y su Rotacional ∇×F")
                
                # Calcular rotacional
                curl_x = sp.diff(R, y) - sp.diff(Q, z)
                curl_y = sp.diff(P, z) - sp.diff(R, x)
                curl_z = sp.diff(Q, x) - sp.diff(P, y)
                curl_F = (curl_x, curl_y, curl_z)
                
                st.latex(f"\\nabla \\times \\mathbf{{F}} = ({sp.latex(curl_x)}, {sp.latex(curl_y)}, {sp.latex(curl_z)})")
                
                import viz_vectorial as vv
                
                col1, col2 = st.columns(2)
                with col1:
                    show_f = st.checkbox("Mostrar F", value=True, key="stokes_show_f")
                with col2:
                    show_curl = st.checkbox("Mostrar ∇×F", value=True, key="stokes_show_curl")
                
                n_vec = st.slider("Resolución", 5, 12, 7, key="stokes_vec_res")
                
                fig_fields = go.Figure()
                
                if show_f:
                    # Campo F
                    x_r = np.linspace(-2, 2, n_vec)
                    y_r = np.linspace(-2, 2, n_vec)
                    z_r = np.linspace(-2, 2, n_vec)
                    X_g, Y_g, Z_g = np.meshgrid(x_r, y_r, z_r)
                    
                    U_f = vv.ensure_array(P_func, X_g, Y_g, Z_g)
                    V_f = vv.ensure_array(Q_func, X_g, Y_g, Z_g)
                    W_f = vv.ensure_array(R_func, X_g, Y_g, Z_g)
                    
                    fig_fields.add_trace(go.Cone(
                        x=X_g.flatten(), y=Y_g.flatten(), z=Z_g.flatten(),
                        u=U_f.flatten(), v=V_f.flatten(), w=W_f.flatten(),
                        colorscale='Greens',
                        sizemode='scaled',
                        sizeref=0.5,
                        showscale=False,
                        name='F',
                        opacity=0.7
                    ))
                
                if show_curl:
                    # Campo ∇×F
                    curl_x_func = sp.lambdify((x, y, z), curl_x, modules=['numpy'])
                    curl_y_func = sp.lambdify((x, y, z), curl_y, modules=['numpy'])
                    curl_z_func = sp.lambdify((x, y, z), curl_z, modules=['numpy'])
                    
                    x_r = np.linspace(-2, 2, n_vec)
                    y_r = np.linspace(-2, 2, n_vec)
                    z_r = np.linspace(-2, 2, n_vec)
                    X_g, Y_g, Z_g = np.meshgrid(x_r, y_r, z_r)
                    
                    U_curl = vv.ensure_array(curl_x_func, X_g, Y_g, Z_g)
                    V_curl = vv.ensure_array(curl_y_func, X_g, Y_g, Z_g)
                    W_curl = vv.ensure_array(curl_z_func, X_g, Y_g, Z_g)
                    
                    fig_fields.add_trace(go.Cone(
                        x=X_g.flatten(), y=Y_g.flatten(), z=Z_g.flatten(),
                        u=U_curl.flatten(), v=V_curl.flatten(), w=W_curl.flatten(),
                        colorscale='Reds',
                        sizemode='scaled',
                        sizeref=0.5,
                        showscale=False,
                        name='∇×F',
                        opacity=0.7
                    ))
                
                fig_fields.update_layout(
                    title="Campo F (verde) y Rotacional ∇×F (rojo)",
                    scene=dict(
                        xaxis=dict(title='x'),
                        yaxis=dict(title='y'),
                        zaxis=dict(title='z'),
                        aspectmode='cube'
                    ),
                    width=900,
                    height=700
                )
                
                st.plotly_chart(fig_fields, use_container_width=True)
                st.info("💡 El rotacional ∇×F mide la 'rotación' del campo F en cada punto")
        
        except Exception as e:
            st.warning(f"⚠️ No se pudo generar la visualización: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Mostrar mapa de error si ya se calculó
    if st.session_state.get('stokes_computed', False):
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["🗺️ Mapa de Error", "📄 Exportar Informe"])
        
        with tab1:
            st.subheader("Visualización del Error")
            
            if st.button("🗺️ Mostrar Mapa de Error", key="show_error_map_btn"):
                with st.spinner("Generando mapa de calor..."):
                    resultado = st.session_state['stokes_result']
                    
                    error_grid = resultado['error_grid']
                    U = resultado['U']
                    V = resultado['V']
                    
                    # Calcular estadísticas
                    max_error = np.max(np.abs(error_grid))
                    min_error = np.min(np.abs(error_grid))
                    mean_error = np.mean(np.abs(error_grid))
                    std_error = np.std(np.abs(error_grid))
                    
                    st.markdown("#### Estadísticas del Error")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Máximo", f"{max_error:.3e}")
                    with col2:
                        st.metric("Mínimo", f"{min_error:.3e}")
                    with col3:
                        st.metric("Media", f"{mean_error:.3e}")
                    with col4:
                        st.metric("Desv. Std", f"{std_error:.3e}")
                    
                    # Generar heatmap
                    fig = cv.plot_error_heatmap(U, V, error_grid, title="Mapa de Error - Teorema de Stokes")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("""
                    **Interpretación del Mapa:**
                    - Colores claros: error mayor en la integración local
                    - Colores oscuros: mejor aproximación
                    - Distribución uniforme indica buen resultado numérico
                    """)
        
        with tab2:
            st.subheader("Exportar Informe PDF")
            
            if st.button("📄 Generar Informe Completo", key="export_stokes_pdf_btn"):
                with st.spinner("Generando informe PDF con gráficos..."):
                    import time
                    import os
                    
                    resultado = st.session_state['stokes_result']
                    
                    # Guardar el heatmap como imagen temporal
                    error_grid = resultado['error_grid']
                    U = resultado['U']
                    V = resultado['V']
                    fig = cv.plot_error_heatmap(U, V, error_grid)
                    
                    temp_dir = os.path.join(os.getcwd(), 'temp_figs')
                    os.makedirs(temp_dir, exist_ok=True)
                    img_path = os.path.join(temp_dir, f'stokes_heatmap_{int(time.time())}.png')
                    fig.write_image(img_path, width=800, height=600)
                    
                    # Crear reporte
                    report = {
                        'title': 'Verificación del Teorema de Stokes',
                        'author': 'Sistema de Cálculo Vectorial',
                        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'sections': [
                            {
                                'title': 'Teorema de Stokes',
                                'text': 'El teorema de Stokes relaciona la integral de línea de un campo vectorial F sobre una curva cerrada C con la integral de superficie del rotacional de F sobre una superficie S que tiene a C como frontera.',
                                'latex': [r'\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot \mathbf{n} \, dS'],
                                'fig_paths': [],
                                'data': {}
                            },
                            {
                                'title': 'Campo Vectorial',
                                'text': f'F = ({P_str}, {Q_str}, {R_str})',
                                'latex': [],
                                'fig_paths': [],
                                'data': {
                                    'Componente P': P_str,
                                    'Componente Q': Q_str,
                                    'Componente R': R_str
                                }
                            },
                            {
                                'title': 'Resultados de Verificación',
                                'text': f"El teorema {'SÍ' if resultado['stokes_holds'] else 'NO'} se verifica dentro de la tolerancia especificada.",
                                'latex': [],
                                'fig_paths': [],
                                'data': {
                                    'Integral de línea ∮F·dr': f"{resultado['line_integral']:.3f}",
                                    'Integral de superficie ∬(∇×F)·n dS': f"{resultado['surface_integral']:.3f}",
                                    'Diferencia absoluta': f"{resultado['difference']:.3e}",
                                    'Error relativo': f"{resultado['relative_error']:.2%}",
                                    'Tolerancia': f"{tol:.1e}",
                                    'Verificación': 'EXITOSA' if resultado['stokes_holds'] else 'FALLIDA'
                                }
                            },
                            {
                                'title': 'Mapa de Error',
                                'text': 'Visualización de la distribución del error en la superficie parametrizada.',
                                'latex': [],
                                'fig_paths': [img_path],
                                'data': {
                                    'Error máximo': f"{np.max(np.abs(error_grid)):.3e}",
                                    'Error mínimo': f"{np.min(np.abs(error_grid)):.3e}",
                                    'Error medio': f"{np.mean(np.abs(error_grid)):.3e}",
                                    'Desviación estándar': f"{np.std(np.abs(error_grid)):.3e}"
                                }
                            }
                        ]
                    }
                    
                    pdf_path = cv.export_report_pdf(report, f'stokes_{int(time.time())}')
                    
                    st.success(f"✅ PDF generado: {pdf_path}")
                    
                    # Leer y ofrecer descarga
                    with open(pdf_path, 'rb') as f:
                        pdf_bytes = f.read()
                    
                    st.download_button(
                        label="⬇️ Descargar Informe PDF",
                        data=pdf_bytes,
                        file_name='informe_stokes.pdf',
                        mime='application/pdf',
                        key="download_stokes_pdf"
                    )
                    
                    # Limpiar imagen temporal
                    try:
                        os.remove(img_path)
                    except:
                        pass

# ============================================================================
# OPCIÓN 6: GENERADOR DE EJERCICIOS
# ============================================================================

elif funcionalidad == "🎓 Generador de Ejercicios":
    st.header("🎓 Generador Inteligente de Ejercicios")
    
    st.info("""
    **✨ Sistema avanzado de generación de ejercicios** con:
    
    🎯 **Dificultad progresiva** (fácil → intermedio → difícil)
    
    💡 **Pistas multinivel** (4 niveles de ayuda)
    
    🔬 **Interpretación física** y geométrica
    
    📊 **Soluciones detalladas** con LaTeX
    
    📦 **Exportación a ZIP** (JSON + Markdown)
    """)
    
    st.markdown("---")
    
    # Instrucciones de uso
    with st.expander("📖 Instrucciones de Uso", expanded=False):
        st.markdown("""
        ### ¿Cómo usar este generador?
        
        1. **Selecciona el tipo de ejercicio** que quieres practicar:
           - **Gradiente**: Cálculo de ∇φ para campos escalares
           - **Divergencia/Rotacional**: Análisis de campos vectoriales (∇·F, ∇×F)
           - **Integral de Línea**: Cálculo de ∮ F·dr sobre curvas
           - **Teorema de Stokes**: Verificación ∮ F·dr = ∬ (∇×F)·n dS
        
        2. **Define la cantidad** de ejercicios (1-20)
           - Los primeros serán más fáciles
           - La dificultad aumenta gradualmente
        
        3. **Semilla aleatoria** (opcional):
           - Usa la misma semilla para generar los mismos ejercicios
           - Útil para compartir con compañeros
        
        4. **Ver pistas progresivas**:
           - Nivel 1: Concepto básico
           - Nivel 2: Contexto del problema
           - Nivel 3: Fórmulas específicas
           - Nivel 4: Pasos detallados
        
        5. **Exportar a ZIP**:
           - Archivo JSON para uso programático
           - Archivo Markdown legible
           - Soluciones completas incluidas
        """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tipo_ejercicio = st.selectbox(
            "📐 Tipo de ejercicio:",
            ["gradiente", "divergencia_rotacional", "line_integral", "stokes", "optimizacion"],
            format_func=lambda x: {
                "gradiente": "∇φ Gradiente",
                "divergencia_rotacional": "∇·F y ∇×F Div/Curl",
                "line_integral": "∮ F·dr Integral de Línea",
                "stokes": "🔄 Teorema de Stokes",
                "optimizacion": "📊 Optimización (Máx/Mín)"
            }[x],
            key="tipo_ejercicio",
            help="Selecciona qué concepto quieres practicar"
        )
    
    with col2:
        num_ejercicios = st.number_input(
            "🔢 Cantidad:", 
            min_value=1, 
            max_value=20, 
            value=5, 
            key="num_ejercicios",
            help="Número de ejercicios (dificultad progresiva)"
        )
    
    with col3:
        semilla = st.number_input(
            "🎲 Semilla:", 
            min_value=1, 
            max_value=10000, 
            value=42, 
            key="semilla",
            help="Para reproducibilidad (misma semilla = mismos ejercicios)"
        )
    
    if st.button("✨ Generar Ejercicios", type="primary", key="gen_exercises_btn"):
        with st.spinner(f"🔄 Generando {num_ejercicios} ejercicios de tipo '{tipo_ejercicio}'..."):
            try:
                ejercicios = cv.generate_exercises(
                    seed=int(semilla),
                    n=int(num_ejercicios),
                    tipo=tipo_ejercicio
                )
                
                # Guardar en session_state
                st.session_state['ejercicios'] = ejercicios
                st.session_state['ejercicios_generados'] = True
                
                st.success(f"✅ Se generaron **{len(ejercicios)}** ejercicio(s) de tipo **{tipo_ejercicio}**")
                
                # Estadísticas de dificultad
                dificultades = [ej.get('dificultad', 'desconocida') for ej in ejercicios]
                facil = dificultades.count('facil')
                intermedio = dificultades.count('intermedio')
                dificil = dificultades.count('dificil')
                
                cols = st.columns(3)
                with cols[0]:
                    st.metric("📗 Fácil", facil)
                with cols[1]:
                    st.metric("📙 Intermedio", intermedio)
                with cols[2]:
                    st.metric("📕 Difícil", dificil)
                
            except Exception as e:
                st.error(f"❌ Error al generar ejercicios: {str(e)}")
    
    # Mostrar ejercicios generados
    if st.session_state.get('ejercicios_generados', False):
        st.markdown("---")
        st.subheader(f"📚 Ejercicios Generados ({len(st.session_state['ejercicios'])})")
        
        ejercicios = st.session_state['ejercicios']
        
        for idx, ejercicio in enumerate(ejercicios):
            # Color según dificultad
            dificultad = ejercicio.get('dificultad', 'desconocida')
            emoji_dificultad = {
                'facil': '📗',
                'intermedio': '📙',
                'dificil': '📕'
            }.get(dificultad, '📘')
            
            with st.expander(f"{emoji_dificultad} Ejercicio {idx+1} - {dificultad.upper()} - {ejercicio['tipo']}", expanded=(idx==0)):
                st.markdown(f"### {ejercicio['instruccion']}")
                
                if 'descripcion' in ejercicio:
                    st.info(f"📌 {ejercicio['descripcion']}")
                
                # Datos del problema
                st.markdown("#### 📋 Datos del Problema")
                for key, value in ejercicio['inputs'].items():
                    st.code(f"{key}: {value}", language="")
                
                # Pistas progresivas (4 niveles)
                st.markdown("#### 💡 Pistas Progresivas")
                if ejercicio.get('hints'):
                    hint_tabs = st.tabs([f"Nivel {i+1}" for i in range(len(ejercicio['hints']))])
                    
                    for i, (tab, hint) in enumerate(zip(hint_tabs, ejercicio['hints'])):
                        with tab:
                            st.markdown(hint)
                
                # Solución completa (oculta por defecto)
                with st.expander("✅ Ver Solución Completa", expanded=False):
                    st.markdown("#### 🎯 Solución")
                    
                    # Mostrar solución formateada
                    sol = ejercicio['solution']
                    
                    # GRADIENTE
                    if 'symbolic_latex' in sol:
                        st.markdown("**∇φ (Gradiente general):**")
                        for i, latex_expr in enumerate(sol['symbolic_latex']):
                            var_names = ['x', 'y', 'z']
                            st.latex(f"\\frac{{\\partial \\phi}}{{\\partial {var_names[i]}}} = {latex_expr}")
                    
                    if 'grad_at_point_latex' in sol:
                        st.markdown("**∇φ evaluado en el punto dado:**")
                        st.latex(f"\\nabla\\phi = \\left( {', '.join(sol['grad_at_point_latex'])} \\right)")
                    
                    if 'magnitude_latex' in sol:
                        st.markdown("**Magnitud del gradiente:**")
                        st.latex(f"\\|\\nabla\\phi\\| = {sol['magnitude_latex']}")
                    
                    if 'unit_vector_latex' in sol:
                        st.markdown("**Vector unitario (dirección de máximo crecimiento):**")
                        st.latex(f"\\hat{{v}} = \\left( {', '.join(sol['unit_vector_latex'])} \\right)")
                    
                    if 'phi_at_point_latex' in sol:
                        st.markdown("**Valor de φ en el punto:**")
                        st.latex(f"\\phi = {sol['phi_at_point_latex']}")
                    
                    # DIVERGENCIA Y ROTACIONAL
                    if 'divergence_latex' in sol:
                        st.markdown("**Divergencia:**")
                        st.latex(f"\\nabla \\cdot F = {sol['divergence_latex']}")
                    
                    if 'curl_latex' in sol:
                        st.markdown("**Rotacional:**")
                        st.latex(f"\\nabla \\times F = \\left( {', '.join(sol['curl_latex'])} \\right)")
                    
                    if 'div_at_point_latex' in sol:
                        st.markdown("**Divergencia evaluada:**")
                        st.latex(f"\\nabla \\cdot F = {sol['div_at_point_latex']}")
                    
                    if 'curl_at_point_latex' in sol:
                        st.markdown("**Rotacional evaluado:**")
                        st.latex(f"\\nabla \\times F = \\left( {', '.join(sol['curl_at_point_latex'])} \\right)")
                    
                    # INTEGRAL DE LÍNEA
                    if 'integrand_latex' in sol:
                        st.markdown("**Integrando:**")
                        st.latex(f"F(r(t)) \\cdot r'(t) = {sol['integrand_latex']}")
                    
                    if ejercicio['tipo'] == 'line_integral' and 'symbolic_latex' in sol:
                        st.markdown("**Resultado de la integral:**")
                        st.latex(f"\\oint_C F \\cdot dr = {sol['symbolic_latex']}")
                    
                    # TEOREMA DE STOKES
                    if ejercicio['tipo'] == 'stokes':
                        if 'line_integral_latex' in sol:
                            st.markdown("**Integral de línea:**")
                            st.latex(f"\\oint_C F \\cdot dr = {sol['line_integral_latex']}")
                        
                        if 'surface_integral_latex' in sol:
                            st.markdown("**Integral de superficie:**")
                            st.latex(f"\\iint_S (\\nabla \\times F) \\cdot n \\, dS = {sol['surface_integral_latex']}")
                        
                        if sol.get('stokes_holds'):
                            st.success("✅ El teorema de Stokes se verifica (ambos lados son iguales)")
                    
                    # OPTIMIZACIÓN
                    if ejercicio['tipo'].startswith('optimizacion_'):
                        # PUNTOS CRÍTICOS
                        if 'gradient' in sol:
                            st.markdown("**Gradiente:**")
                            st.latex(f"\\nabla\\phi = \\left( {', '.join(sol['gradient'])} \\right)")
                        
                        if 'critical_point' in sol:
                            st.markdown("**Punto crítico:**")
                            st.latex(f"(x, y) = {sol['critical_point']}")
                        
                        if 'hessian' in sol:
                            st.markdown("**Matriz Hessiana:**")
                            st.latex(f"H = {sol['hessian']}")
                        
                        if 'eigenvalues' in sol:
                            st.markdown("**Valores propios de H:**")
                            st.latex(f"\\lambda = {sol['eigenvalues']}")
                        
                        if 'classification' in sol:
                            st.markdown(f"**Clasificación:** {sol['classification'].upper()}")
                            if sol['classification'] == 'mínimo local':
                                st.success("✅ MÍNIMO LOCAL (todos los eigenvalues > 0)")
                            elif sol['classification'] == 'máximo local':
                                st.error("⬆️ MÁXIMO LOCAL (todos los eigenvalues < 0)")
                            elif sol['classification'] == 'punto silla':
                                st.warning("⚠️ PUNTO SILLA (eigenvalues con signos mixtos)")
                        
                        if 'phi_at_point' in sol:
                            st.markdown(f"**Valor de φ en el punto crítico:** {sol['phi_at_point']:.6f}")
                        
                        # LAGRANGE
                        if 'lagrangian' in sol:
                            st.markdown("**Lagrangiano:**")
                            st.latex(f"L = {sol['lagrangian']}")
                        
                        if 'lambda' in sol:
                            st.markdown(f"**Multiplicador de Lagrange:** λ = {sol['lambda']:.6f}")
                        
                        if 'optimal_value' in sol:
                            st.markdown(f"**Valor óptimo de φ:** {sol['optimal_value']:.6f}")
                        
                        # REGIÓN
                        if 'max_point' in sol:
                            st.markdown(f"**Máximo global:** en {sol['max_point']} con valor {sol['max_value']:.6f}")
                        
                        if 'min_point' in sol:
                            st.markdown(f"**Mínimo global:** en {sol['min_point']} con valor {sol['min_value']:.6f}")
                        
                        if 'method' in sol:
                            st.info(f"Método usado: {sol['method']}")
                    
                    # Mostrar interpretación si existe
                    if 'interpretacion' in ejercicio:
                        st.markdown("#### 🔍 Interpretación")
                        st.markdown(ejercicio['interpretacion'])
                    
                    # Resto de datos de la solución
                    st.markdown("#### 📊 Datos Completos de la Solución")
                    st.json(sol)
                    
                    st.markdown(f"**Tolerancia numérica:** `{ejercicio['tolerance']}`")
        
        # Exportar a ZIP
        st.markdown("---")
        st.subheader("📦 Exportar Paquete de Ejercicios")
        
        if st.button("💾 Generar ZIP", key="export_exercises_zip_btn"):
            import json
            import zipfile
            import io
            import time
            
            with st.spinner("Creando archivo ZIP..."):
                # Crear buffer en memoria
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Archivo JSON con ejercicios
                    ejercicios_json = json.dumps(ejercicios, indent=2, ensure_ascii=False)
                    zipf.writestr('exercises.json', ejercicios_json)
                    
                    # Archivo answers.md con soluciones
                    md_content = f"# Soluciones de Ejercicios\n\n"
                    md_content += f"**Tipo:** {tipo_ejercicio}\n"
                    md_content += f"**Cantidad:** {len(ejercicios)}\n"
                    md_content += f"**Semilla:** {semilla}\n"
                    md_content += f"**Fecha:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    md_content += "---\n\n"
                    
                    for idx, ejercicio in enumerate(ejercicios):
                        md_content += f"## Ejercicio {idx+1}\n\n"
                        md_content += f"**Enunciado:** {ejercicio['instruccion']}\n\n"
                        md_content += "**Solución:**\n```json\n"
                        md_content += json.dumps(ejercicio['solution'], indent=2, ensure_ascii=False)
                        md_content += "\n```\n\n"
                        md_content += f"**Tolerancia:** {ejercicio['tolerance']}\n\n"
                        md_content += "---\n\n"
                    
                    zipf.writestr('answers.md', md_content)
                    
                    # Archivo README
                    readme = """# Paquete de Ejercicios de Cálculo Vectorial

Este paquete contiene ejercicios autogenerados para practicar cálculo vectorial.

## Archivos incluidos

- `exercises.json`: Ejercicios en formato JSON (programático)
- `answers.md`: Soluciones en formato Markdown (legible)

## Cómo usar

1. Abre `exercises.json` para ver los ejercicios en formato estructurado
2. Intenta resolver cada ejercicio
3. Verifica tu respuesta en `answers.md`
4. Usa la tolerancia especificada para comparar resultados numéricos

## Generado por

Sistema de Cálculo Vectorial 3D - Proyecto Profesional
"""
                    zipf.writestr('README.md', readme)
                
                zip_buffer.seek(0)
                
                st.success("✅ ZIP creado exitosamente!")
                
                st.download_button(
                    label="⬇️ Descargar Paquete ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f'ejercicios_{tipo_ejercicio}_{int(time.time())}.zip',
                    mime='application/zip',
                    key="download_exercises_zip"
                )

# ============================================================================
# OPCIÓN 7: VISUALIZADOR 3D AVANZADO
# ============================================================================

elif funcionalidad == "🎨 Visualizador 3D Avanzado":
    st.header("🎨 Visualizador 3D Avanzado - Three.js r160")
    
    st.markdown("""
    ### Visualizador WebGL de alta calidad con API completa
    
    **Características técnicas:**
    - 🎯 **Three.js r160** con OrbitControls + damping
    - 💡 **Iluminación física**: AmbientLight + DirectionalLight con sombras
    - 🎨 **MeshStandardMaterial** (PBR)
    - 📊 **Ejes 3D** con ticks y etiquetas numéricas
    - 🌐 **Grid 3D** configurable
    - 📍 **HUD overlay** mostrando coordenadas (x,y,z) con raycaster
    - 📦 **Exportación**: PNG (sin fondo) y OBJ (compatible Blender)
    - ⚙️ **Toggles individuales**: Superficie, vectores, gradiente, streamlines, ejes, labels, grid
    - 🚀 **API JS**: window.viewer con 6 métodos públicos
    - ⚡ **Optimizado**: 60 FPS, antialias, ACESFilmic tone mapping
    """)
    
    st.markdown("---")
    
    # Panel de entrada de funciones
    st.subheader("📝 Definir Funciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Campo Escalar φ(x,y)**")
        phi_str = st.text_input(
            "φ(x,y) =",
            value="sin(sqrt(x^2 + y^2))",
            help="Superficie 3D: z = φ(x,y)"
        )
        
        enable_surface = st.checkbox("✅ Mostrar superficie", value=True)
        enable_gradient = st.checkbox("📊 Mostrar gradiente ∇φ", value=False)
    
    with col2:
        st.markdown("**Campo Vectorial F(x,y,z)**")
        enable_field = st.checkbox("✅ Activar campo vectorial", value=False)
        
        if enable_field:
            Fx_str = st.text_input("F_x =", value="-y", help="Componente x")
            Fy_str = st.text_input("F_y =", value="x", help="Componente y")
            Fz_str = st.text_input("F_z =", value="0", help="Componente z")
            
            enable_streamlines = st.checkbox("🌊 Calcular streamlines", value=False)
            if enable_streamlines:
                num_seeds = st.slider("Semillas:", 3, 15, 5)
        else:
            Fx_str, Fy_str, Fz_str = "-y", "x", "0"
            enable_streamlines = False
            num_seeds = 5
    
    st.markdown("---")
    
    # Configuración de visualización
    st.subheader("⚙️ Configuración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dominio**")
        x_min = st.number_input("x mín:", value=-3.0, step=0.5)
        x_max = st.number_input("x máx:", value=3.0, step=0.5)
        y_min = st.number_input("y mín:", value=-3.0, step=0.5)
        y_max = st.number_input("y máx:", value=3.0, step=0.5)
    
    with col2:
        st.markdown("**Resolución**")
        surface_resolution = st.slider("Superficie:", 20, 100, 50)
        vector_density = st.slider("Vectores:", 5, 20, 10)
    
    st.markdown("---")
    
    # Botón de generación
    if st.button("🚀 Generar Visualización 3D", type="primary"):
        try:
            import json
            import sympy as sp
            
            x, y, z = sp.symbols('x y z')
            
            # ==== PARSEAR FUNCIONES ====
            phi_expr = cv.safe_parse(phi_str, [x, y, z])
            phi_lambda = sp.lambdify([x, y], phi_expr, 'numpy')
            
            # Generar malla de superficie
            x_arr = np.linspace(x_min, x_max, surface_resolution)
            y_arr = np.linspace(y_min, y_max, surface_resolution)
            X, Y = np.meshgrid(x_arr, y_arr)
            Z = phi_lambda(X, Y)
            
            # Exportar malla a JSON
            mesh_json = cv.export_mesh_json_from_surface(X, Y, Z)
            
            # ==== GRADIENTE ====
            grad_json = None
            if enable_gradient:
                grad_x_expr = sp.diff(phi_expr, x)
                grad_y_expr = sp.diff(phi_expr, y)
                grad_x_lambda = sp.lambdify([x, y], grad_x_expr, 'numpy')
                grad_y_lambda = sp.lambdify([x, y], grad_y_expr, 'numpy')
                
                # Crear campo vectorial del gradiente
                grad_Fx = lambda x, y, z: grad_x_lambda(x, y)
                grad_Fy = lambda x, y, z: grad_y_lambda(x, y)
                grad_Fz = lambda x, y, z: np.zeros_like(x)
                
                domain = {
                    'xmin': x_min, 'xmax': x_max,
                    'ymin': y_min, 'ymax': y_max,
                    'zmin': 0, 'zmax': 0
                }
                grad_json = cv.compute_vector_field_grid(
                    (grad_Fx, grad_Fy, grad_Fz),
                    domain,
                    density=vector_density
                )
                grad_json['type'] = 'gradient'  # Marcar como gradiente
            
            # ==== CAMPO VECTORIAL ====
            field_json = None
            if enable_field:
                Fx_expr = cv.safe_parse(Fx_str, [x, y, z])
                Fy_expr = cv.safe_parse(Fy_str, [x, y, z])
                Fz_expr = cv.safe_parse(Fz_str, [x, y, z])
                
                Fx_lambda = sp.lambdify([x, y, z], Fx_expr, 'numpy')
                Fy_lambda = sp.lambdify([x, y, z], Fy_expr, 'numpy')
                Fz_lambda = sp.lambdify([x, y, z], Fz_expr, 'numpy')
                
                F_num = lambda x, y, z: (Fx_lambda(x, y, z), Fy_lambda(x, y, z), Fz_lambda(x, y, z))
                
                domain = {
                    'xmin': x_min, 'xmax': x_max,
                    'ymin': y_min, 'ymax': y_max,
                    'zmin': 0, 'zmax': 0
                }
                field_json = cv.compute_vector_field_grid(F_num, domain, density=vector_density)
            
            # ==== STREAMLINES ====
            stream_json = None
            if enable_streamlines and enable_field:
                st.info("🌊 Calculando streamlines...")
                
                # Semillas en círculo
                theta = np.linspace(0, 2*np.pi, num_seeds)
                seeds = np.column_stack([
                    0.8 * np.cos(theta),
                    0.8 * np.sin(theta),
                    np.zeros(num_seeds)
                ])
                
                streamlines_data = cv.compute_streamlines(
                    F_num,
                    seeds,
                    step=0.05,
                    max_steps=500,
                    domain=(x_min, x_max, y_min, y_max, -5, 5),
                    both_directions=True
                )
                
                stream_json = cv.export_streamlines_json(streamlines_data, method='rk4', step=0.05)
                st.success(f"✅ {len(streamlines_data)} streamlines generadas")
            
            # ==== CARGAR HTML ====
            import os
            html_path = os.path.join("static", "threejs_viewer.html")
            
            if not os.path.exists(html_path):
                st.error(f"❌ No se encuentra {html_path}")
                st.stop()
            
            with open(html_path, 'r', encoding='utf-8') as f:
                html_template = f.read()
            
            # ==== INYECTAR DATOS EN HTML ====
            init_script = f"""
            <script>
            window.addEventListener('load', function() {{
                // Esperar a que window.viewer esté disponible
                const waitForViewer = setInterval(() => {{
                    if (window.viewer) {{
                        clearInterval(waitForViewer);
                        
                        // Cargar datos
                        console.log('🎨 Cargando visualización...');
                        
                        {"window.viewer.updateMesh(" + json.dumps(mesh_json) + ");" if enable_surface else ""}
                        {"window.viewer.updateVectorField(" + json.dumps(grad_json) + ");" if enable_gradient and grad_json else ""}
                        {"window.viewer.updateVectorField(" + json.dumps(field_json) + ");" if enable_field and field_json else ""}
                        {"window.viewer.updateStreamlines(" + json.dumps(stream_json) + ");" if enable_streamlines and stream_json else ""}
                        
                        console.log('✅ Datos cargados');
                    }}
                }}, 100);
            }});
            </script>
            """
            
            # Insertar script antes de </body>
            html_with_data = html_template.replace("</body>", init_script + "</body>")
            
            # ==== RENDERIZAR ====
            st.markdown("---")
            st.subheader("🎮 Visualizador Three.js")
            
            import streamlit.components.v1 as components
            components.html(html_with_data, height=900, scrolling=False)
            
            st.markdown("---")
            st.info("""
            **🎮 Controles:**
            - 🖱️ **Click izquierdo + arrastrar**: Rotar cámara
            - 🖱️ **Rueda del mouse**: Zoom
            - 🖱️ **Click derecho + arrastrar**: Pan
            - 📍 **Hover sobre superficie**: Ver coordenadas (x,y,z) en overlay superior
            - ⚙️ **Panel derecho**: Toggles para cada elemento
            - 📷 **Exportar PNG**: Imagen sin fondo negro
            - 📦 **Exportar OBJ**: Compatible con Blender/Maya
            - 🔄 **Reset Cámara**: Volver a vista inicial
            
            **📡 API JavaScript disponible:**
            ```javascript
            window.viewer.updateMesh(meshJson)
            window.viewer.updateVectorField(fieldJson)
            window.viewer.updateStreamlines(streamJson)
            window.viewer.resetCamera()
            window.viewer.exportPNG()
            window.viewer.exportOBJ()
            ```
            """)
            
        except Exception as e:
            import traceback
            st.error(f"❌ Error: {e}")
            with st.expander("🔍 Ver traceback completo"):
                st.code(traceback.format_exc())


# ============================================================================
# OPCIÓN 6: OPTIMIZACIÓN (MÁXIMOS/MÍNIMOS)
# ============================================================================

elif funcionalidad == "📊 Optimización (Máximos/Mínimos)":
    st.header("📊 Optimización de Funciones Multivariables")
    
    st.markdown("""
    ### Herramientas de Optimización Completas
    
    Encuentra máximos, mínimos y puntos silla de funciones de varias variables.
    Incluye:
    - **Gradiente y derivada direccional**
    - **Puntos críticos y clasificación (Hessiana)**
    - **Optimización sin restricciones**
    - **Multiplicadores de Lagrange** (con restricciones)
    - **Optimización sobre regiones** (triángulos, rectángulos, elipses)
    - **Visualizaciones 3D/2D estilo GeoGebra**
    """)
    
    # Importar módulo de optimización
    try:
        import optimizacion as opt
    except ImportError:
        st.error("❌ El módulo de optimización no está disponible. Asegúrate de que `optimizacion.py` esté en el directorio.")
        st.stop()
    
    # Sub-tabs para diferentes tipos de optimización
    opt_tab = st.tabs([
        "📐 Gradiente y Derivada Direccional",
        "🎯 Puntos Críticos",
        "🔓 Optimización Libre",
        "🔗 Multiplicadores de Lagrange",
        "📍 Optimización en Regiones",
        "⭐ Casos Especiales"
    ])
    
    # ========================================================================
    # TAB 1: GRADIENTE Y DERIVADA DIRECCIONAL
    # ========================================================================
    
    with opt_tab[0]:
        st.subheader("📐 Gradiente y Derivada Direccional")
        
        st.info("💡 Calcula el gradiente ∇φ y la derivada direccional D_u φ en un punto dado.")
        
        # Inputs
        col1, col2 = st.columns([2, 1])
        
        with col1:
            grad_phi_str = st.text_input(
                "Función φ(x,y) o φ(x,y,z):",
                value="x^2 + y^2",
                help="Ejemplos: x^2 + y^2, x*y*z, sin(x)*cos(y)",
                key="input_grad_phi"
            )
        
        with col2:
            grad_vars_str = st.text_input(
                "Variables (separadas por coma):",
                value="x, y",
                help="Ejemplo: x, y o x, y, z",
                key="input_grad_vars"
            )
        
        # Punto de evaluación
        col1, col2, col3 = st.columns(3)
        with col1:
            grad_px = st.number_input("Punto x₀:", value=1.0, step=0.1, key="grad_px")
        with col2:
            grad_py = st.number_input("Punto y₀:", value=1.0, step=0.1, key="grad_py")
        with col3:
            grad_pz = st.number_input("Punto z₀ (si aplica):", value=0.0, step=0.1, key="grad_pz")
        
        # Dirección
        st.markdown("**Vector dirección** (se normalizará automáticamente):")
        col1, col2, col3 = st.columns(3)
        with col1:
            grad_dx = st.number_input("Componente u:", value=1.0, step=0.1, key="grad_dx")
        with col2:
            grad_dy = st.number_input("Componente v:", value=0.0, step=0.1, key="grad_dy")
        with col3:
            grad_dz = st.number_input("Componente w (si aplica):", value=0.0, step=0.1, key="grad_dz")
        
        # Opciones de visualización
        grad_show_exact = st.checkbox("Mostrar valores exactos (raíces, fracciones)", value=True, key="grad_show_exact")
        
        if st.button("🔍 Calcular Gradiente y Derivada Direccional", type="primary", key="calc_grad"):
            try:
                # Parsear función
                vars_list = [v.strip() for v in grad_vars_str.split(',')]
                n_vars = len(vars_list)
                
                vars_sym = sp.symbols(' '.join(vars_list))
                if n_vars == 1:
                    vars_sym = (vars_sym,)
                
                phi = cv.safe_parse(grad_phi_str, vars_sym)
                
                # Construir punto y dirección
                if n_vars == 2:
                    point = (grad_px, grad_py)
                    direction = (grad_dx, grad_dy)
                else:
                    point = (grad_px, grad_py, grad_pz)
                    direction = (grad_dx, grad_dy, grad_dz)
                
                # Calcular derivada direccional
                result = opt.directional_derivative(phi, vars_sym, point, direction)
                
                # Guardar en session state
                st.session_state['grad_result_computed'] = result
                st.session_state['grad_phi_parsed'] = phi
                st.session_state['grad_vars_parsed'] = vars_sym
                st.session_state['grad_point'] = point
                st.session_state['grad_computed'] = True
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state['grad_computed'] = False
        
        # Mostrar resultados
        if st.session_state.get('grad_computed', False):
            result = st.session_state['grad_result_computed']
            phi = st.session_state['grad_phi_parsed']
            vars_sym = st.session_state['grad_vars_parsed']
            point = st.session_state['grad_point']
            
            st.success("✅ Cálculo completado")
            
            # Mostrar pasos en LaTeX
            st.markdown("### 📝 Paso a paso:")
            for step in result['latex_steps']:
                st.latex(step)
            
            # Resultados numéricos
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Derivada Direccional",
                    f"{result['directional_derivative']:.6f}"
                )
            
            with col2:
                st.metric(
                    "||∇φ||",
                    f"{result['gradient_magnitude']:.6f}"
                )
            
            with col3:
                if result['is_maximum_direction']:
                    st.metric("Dirección", "⬆️ MÁXIMO")
                elif result['is_minimum_direction']:
                    st.metric("Dirección", "⬇️ MÍNIMO")
                else:
                    angle = np.arccos(np.clip(
                        result['directional_derivative'] / result['gradient_magnitude'] if result['gradient_magnitude'] > 0 else 0,
                        -1, 1
                    )) * 180 / np.pi
                    st.metric("Ángulo con ∇φ", f"{angle:.1f}°")
            
            # Visualización
            if len(vars_sym) == 2:
                st.markdown("### 📊 Visualización")
                
                try:
                    fig = opt.visualize_contour_2d(
                        phi,
                        vars_sym,
                        critical_points=[],
                        bounds=(
                            (point[0] - 3, point[0] + 3),
                            (point[1] - 3, point[1] + 3)
                        ),
                        show_gradient=True,
                        resolution=80
                    )
                    
                    # Marcar el punto de evaluación
                    import plotly.graph_objects as go
                    fig.add_trace(go.Scatter(
                        x=[point[0]],
                        y=[point[1]],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='star'),
                        name='Punto de evaluación',
                        showlegend=True
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"No se pudo generar visualización: {e}")
    
    # ========================================================================
    # TAB 2: PUNTOS CRÍTICOS Y CLASIFICACIÓN
    # ========================================================================
    
    with opt_tab[1]:
        st.subheader("🎯 Puntos Críticos y Clasificación")
        
        st.info("💡 Encuentra todos los puntos donde ∇φ = 0 y clasifícalos usando la matriz Hessiana.")
        
        # Inputs
        crit_phi_str = st.text_input(
            "Función φ(x,y) o φ(x,y,z):",
            value="x^2 - y^2",
            help="Ejemplo clásico de punto silla: x^2 - y^2",
            key="input_crit_phi"
        )
        
        crit_vars_str = st.text_input(
            "Variables:",
            value="x, y",
            key="crit_vars"
        )
        
        if st.button("🔍 Encontrar Puntos Críticos", type="primary", key="calc_crit"):
            try:
                # Parsear
                vars_list = [v.strip() for v in crit_vars_str.split(',')]
                vars_sym = sp.symbols(' '.join(vars_list))
                if len(vars_list) == 1:
                    vars_sym = (vars_sym,)
                
                phi = cv.safe_parse(crit_phi_str, vars_sym)
                
                # Optimizar
                result = opt.optimize_unconstrained(phi, vars_sym)
                
                # Guardar
                st.session_state['crit_result_computed'] = result
                st.session_state['crit_phi_parsed'] = phi
                st.session_state['crit_vars_parsed'] = vars_sym
                st.session_state['crit_computed'] = True
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state['crit_computed'] = False
        
        # Mostrar resultados
        if st.session_state.get('crit_computed', False):
            result = st.session_state['crit_result_computed']
            phi = st.session_state['crit_phi_parsed']
            vars_sym = st.session_state['crit_vars_parsed']
            
            st.success(f"✅ Encontrados {len(result['critical_points'])} punto(s) crítico(s)")
            
            # Pasos LaTeX
            with st.expander("📝 Ver pasos completos", expanded=False):
                for step in result['latex_steps']:
                    st.latex(step)
            
            # Tabla de puntos críticos
            if result['critical_points']:
                st.markdown("### 📋 Resumen de Puntos Críticos")
                
                import pandas as pd
                
                data = []
                for i, cp in enumerate(result['critical_points'], 1):
                    point_str = f"({', '.join(f'{p:.4f}' for p in cp['point'])})"
                    data.append({
                        '#': i,
                        'Punto': point_str,
                        'φ(punto)': f"{cp['function_value']:.6f}",
                        'Clasificación': cp['classification']
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                
                # Métricas
                col1, col2, col3 = st.columns(3)
                
                minimos = [cp for cp in result['critical_points'] if cp['classification'] == 'mínimo local']
                maximos = [cp for cp in result['critical_points'] if cp['classification'] == 'máximo local']
                sillas = [cp for cp in result['critical_points'] if cp['classification'] == 'punto silla']
                
                with col1:
                    st.metric("Mínimos locales", len(minimos))
                with col2:
                    st.metric("Máximos locales", len(maximos))
                with col3:
                    st.metric("Puntos silla", len(sillas))
                
                # Visualización
                if len(vars_sym) == 2:
                    st.markdown("### 📊 Visualización 3D")
                    
                    # Determinar bounds basado en puntos críticos
                    all_x = [cp['point'][0] for cp in result['critical_points']]
                    all_y = [cp['point'][1] for cp in result['critical_points']]
                    
                    x_range = (min(all_x) - 2, max(all_x) + 2)
                    y_range = (min(all_y) - 2, max(all_y) + 2)
                    
                    try:
                        fig = opt.visualize_optimization_3d(
                            phi,
                            vars_sym,
                            critical_points=result['critical_points'],
                            bounds=(x_range, y_range),
                            resolution=60,
                            show_gradient=True,
                            gradient_density=12
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"No se pudo generar visualización 3D: {e}")
            else:
                st.warning("⚠️ No se encontraron puntos críticos")
    
    # ========================================================================
    # TAB 3: OPTIMIZACIÓN SIN RESTRICCIONES
    # ========================================================================
    
    with opt_tab[2]:
        st.subheader("🔓 Optimización Sin Restricciones")
        
        st.info("💡 Igual que Puntos Críticos, pero con enfoque en encontrar el máximo/mínimo global.")
        
        st.markdown("Consulta la pestaña **Puntos Críticos** para optimización sin restricciones.")
        st.markdown("Esta funcionalidad está integrada en ese módulo.")
    
    # ========================================================================
    # TAB 4: MULTIPLICADORES DE LAGRANGE
    # ========================================================================
    
    with opt_tab[3]:
        st.subheader("🔗 Multiplicadores de Lagrange")
        
        st.info("💡 Optimiza φ(x,y,...) sujeto a restricciones g₁=0, g₂=0, ...")
        
        # Función objetivo
        lag_phi_str = st.text_input(
            "Función objetivo φ:",
            value="x*y",
            help="Función a maximizar/minimizar",
            key="input_lag_phi"
        )
        
        lag_vars_str = st.text_input(
            "Variables:",
            value="x, y",
            key="lag_vars"
        )
        
        # Restricciones
        st.markdown("**Restricciones** (una por línea, en forma g=0):")
        lag_constraints_str = st.text_area(
            "Restricciones:",
            value="x + y - 10",
            help="Ejemplo: x + y - 10 (para x+y=10). Una restricción por línea.",
            key="lag_constraints",
            height=100
        )
        
        if st.button("🔍 Resolver con Lagrange", type="primary", key="calc_lag"):
            try:
                # Parsear
                vars_list = [v.strip() for v in lag_vars_str.split(',')]
                vars_sym = sp.symbols(' '.join(vars_list))
                if len(vars_list) == 1:
                    vars_sym = (vars_sym,)
                
                phi = cv.safe_parse(lag_phi_str, vars_sym)
                
                # Parsear restricciones
                constraints_lines = [line.strip() for line in lag_constraints_str.strip().split('\n') if line.strip()]
                constraints = [cv.safe_parse(line, vars_sym) for line in constraints_lines]
                
                # Resolver
                result = opt.solve_lagrange(phi, vars_sym, constraints)
                
                # Guardar
                st.session_state['lag_result_computed'] = result
                st.session_state['lag_phi_parsed'] = phi
                st.session_state['lag_vars_parsed'] = vars_sym
                st.session_state['lag_constraints_parsed'] = constraints
                st.session_state['lag_computed'] = True
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state['lag_computed'] = False
        
        # Mostrar resultados
        if st.session_state.get('lag_computed', False):
            result = st.session_state['lag_result_computed']
            phi = st.session_state['lag_phi_parsed']
            
            st.success(f"✅ Encontradas {len(result['solutions'])} solución(es)")
            
            # Pasos
            with st.expander("📝 Ver construcción del Lagrangiano y sistema", expanded=True):
                for step in result['latex_steps']:
                    st.latex(step)
            
            # Tabla de soluciones
            if result['solutions']:
                st.markdown("### 📋 Soluciones")
                
                import pandas as pd
                
                data = []
                for i, sol in enumerate(result['solutions'], 1):
                    point_str = f"({', '.join(f'{p:.4f}' for p in sol['point'])})"
                    lambda_str = f"({', '.join(f'{lv:.4f}' for lv in sol['lambda_values'])})"
                    
                    data.append({
                        '#': i,
                        'Punto': point_str,
                        'λ': lambda_str,
                        'φ(punto)': f"{sol['function_value']:.6f}"
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                
                # Valor óptimo
                if result['optimal_value'] is not None:
                    st.metric("Valor Óptimo", f"{result['optimal_value']:.6f}")
    
    # ========================================================================
    # TAB 5: OPTIMIZACIÓN EN REGIONES
    # ========================================================================
    
    with opt_tab[4]:
        st.subheader("📍 Optimización sobre Regiones")
        
        st.info("💡 Encuentra máximos/mínimos globales en una región acotada (triángulo, rectángulo, elipse).")
        
        # Función
        reg_phi_str = st.text_input(
            "Función φ(x,y):",
            value="x + y",
            key="input_reg_phi"
        )
        
        # Tipo de región
        region_type = st.selectbox(
            "Tipo de región:",
            ["Triángulo", "Rectángulo", "Elipse"],
            key="reg_type"
        )
        
        region_dict = {}
        
        if region_type == "Triángulo":
            st.markdown("**Vértices del triángulo:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                v1_x = st.number_input("V1 x:", value=0.0, step=0.5, key="v1x")
                v1_y = st.number_input("V1 y:", value=0.0, step=0.5, key="v1y")
            with col2:
                v2_x = st.number_input("V2 x:", value=0.0, step=0.5, key="v2x")
                v2_y = st.number_input("V2 y:", value=8.0, step=0.5, key="v2y")
            with col3:
                v3_x = st.number_input("V3 x:", value=4.0, step=0.5, key="v3x")
                v3_y = st.number_input("V3 y:", value=0.0, step=0.5, key="v3y")
            
            region_dict = {
                'type': 'triangle',
                'vertices': [(v1_x, v1_y), (v2_x, v2_y), (v3_x, v3_y)]
            }
        
        elif region_type == "Rectángulo":
            st.markdown("**Límites del rectángulo:**")
            col1, col2 = st.columns(2)
            
            with col1:
                x_min = st.number_input("x mín:", value=0.0, step=0.5, key="xmin")
                x_max = st.number_input("x máx:", value=4.0, step=0.5, key="xmax")
            with col2:
                y_min = st.number_input("y mín:", value=0.0, step=0.5, key="ymin")
                y_max = st.number_input("y máx:", value=8.0, step=0.5, key="ymax")
            
            region_dict = {
                'type': 'rectangle',
                'bounds': [(x_min, x_max), (y_min, y_max)]
            }
        
        elif region_type == "Elipse":
            st.markdown("**Parámetros de la elipse:**")
            col1, col2 = st.columns(2)
            
            with col1:
                ell_a = st.number_input("Semi-eje a:", value=3.0, step=0.5, key="ell_a")
                ell_b = st.number_input("Semi-eje b:", value=2.0, step=0.5, key="ell_b")
            with col2:
                ell_h = st.number_input("Centro h:", value=0.0, step=0.5, key="ell_h")
                ell_k = st.number_input("Centro k:", value=0.0, step=0.5, key="ell_k")
            
            region_dict = {
                'type': 'ellipse',
                'a': ell_a,
                'b': ell_b,
                'center': (ell_h, ell_k)
            }
        
        if st.button("🔍 Optimizar en Región", type="primary", key="calc_reg"):
            try:
                # Parsear
                x, y = sp.symbols('x y')
                phi = cv.safe_parse(reg_phi_str, (x, y))
                
                # Optimizar
                result = opt.optimize_on_region(phi, (x, y), region_dict)
                
                # Guardar
                st.session_state['reg_result_computed'] = result
                st.session_state['reg_phi_parsed'] = phi
                st.session_state['reg_region_parsed'] = region_dict
                st.session_state['reg_computed'] = True
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state['reg_computed'] = False
        
        # Mostrar resultados
        if st.session_state.get('reg_computed', False):
            result = st.session_state['reg_result_computed']
            phi = st.session_state['reg_phi_parsed']
            region_dict = st.session_state['reg_region_parsed']
            
            st.success("✅ Optimización completada")
            
            # Pasos
            with st.expander("📝 Ver análisis completo", expanded=False):
                for step in result['latex_steps']:
                    st.latex(step)
            
            # Resultados principales
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔺 Mínimo Global")
                min_point = result['global_minimum']['point']
                min_val = result['global_minimum']['value']
                st.latex(f"\\text{{Punto: }} ({min_point[0]:.4f}, {min_point[1]:.4f})")
                st.latex(f"\\phi = {min_val:.6f}")
                st.caption(f"Tipo: {result['global_minimum']['type']}")
            
            with col2:
                st.markdown("### 🔻 Máximo Global")
                max_point = result['global_maximum']['point']
                max_val = result['global_maximum']['value']
                st.latex(f"\\text{{Punto: }} ({max_point[0]:.4f}, {max_point[1]:.4f})")
                st.latex(f"\\phi = {max_val:.6f}")
                st.caption(f"Tipo: {result['global_maximum']['type']}")
            
            # Tabla comparativa completa
            with st.expander("📊 Ver tabla completa de candidatos", expanded=False):
                import pandas as pd
                
                data = []
                for cand in result['comparison_table']:
                    point_str = f"({cand['point'][0]:.4f}, {cand['point'][1]:.4f})"
                    data.append({
                        'Punto': point_str,
                        'φ': f"{cand['value']:.6f}",
                        'Tipo': cand['type']
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
            
            # Visualización 2D
            st.markdown("### 📊 Visualización")
            
            try:
                # Construir lista de todos los puntos para marcar
                all_points = []
                
                # Añadir puntos críticos interiores
                for cp in result['interior_critical_points']:
                    all_points.append({
                        'point': cp['point'],
                        'value': cp['function_value'],
                        'type': f"interior ({cp['classification']})"
                    })
                
                # Añadir puntos de borde
                for bp in result['boundary_critical_points']:
                    all_points.append(bp)
                
                # Añadir vértices
                for vp in result['vertex_values']:
                    vp_copy = vp.copy()
                    vp_copy['type'] = 'vértice'
                    all_points.append(vp_copy)
                
                # Determinar bounds
                all_x = [p['point'][0] for p in all_points]
                all_y = [p['point'][1] for p in all_points]
                
                margin = 1.0
                bounds = (
                    (min(all_x) - margin, max(all_x) + margin),
                    (min(all_y) - margin, max(all_y) + margin)
                )
                
                fig = opt.visualize_contour_2d(
                    phi,
                    (sp.Symbol('x'), sp.Symbol('y')),
                    critical_points=all_points,
                    bounds=bounds,
                    show_gradient=False,
                    region=region_dict,
                    resolution=100
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"No se pudo generar visualización: {e}")
    
    # ========================================================================
    # TAB 6: CASOS ESPECIALES
    # ========================================================================
    
    with opt_tab[5]:
        st.subheader("⭐ Casos Especiales Pre-Configurados")
        
        st.markdown("""
        Problemas clásicos de optimización con soluciones conocidas.
        Úsalos para verificar el funcionamiento del módulo.
        """)
        
        caso = st.selectbox(
            "Selecciona un caso:",
            [
                "Rectángulo inscrito en elipse",
                "Cobb-Douglas con restricción presupuestaria",
                "Integral de línea F·dr (círculo unitario)"
            ],
            key="caso_especial"
        )
        
        # ---- CASO 1: Rectángulo en elipse ----
        if caso == "Rectángulo inscrito en elipse":
            st.markdown("### Rectángulo Inscrito en Elipse")
            
            st.latex(r"\text{Maximizar área } A = 4xy \text{ sujeto a } \frac{x^2}{a^2} + \frac{y^2}{b^2} = 1")
            
            col1, col2 = st.columns(2)
            with col1:
                a_val = st.number_input("Semi-eje a:", value=3.0, step=0.5, key="rect_a")
            with col2:
                b_val = st.number_input("Semi-eje b:", value=2.0, step=0.5, key="rect_b")
            
            if st.button("✅ Resolver", key="solve_rect"):
                result = opt.max_rectangle_in_ellipse(a_val, b_val)
                
                st.success("✅ Solución analítica:")
                
                for step in result['latex_steps']:
                    st.latex(step)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("x óptimo", f"{result['optimal_point'][0]:.6f}")
                with col2:
                    st.metric("y óptimo", f"{result['optimal_point'][1]:.6f}")
                with col3:
                    st.metric("Área máxima", f"{result['maximum_area']:.6f}")
        
        # ---- CASO 2: Cobb-Douglas ----
        elif caso == "Cobb-Douglas con restricción presupuestaria":
            st.markdown("### Optimización Cobb-Douglas")
            
            st.latex(r"\text{Maximizar } f(x,y) = x^\alpha y^\beta \text{ sujeto a } p_x x + p_y y = M")
            
            col1, col2 = st.columns(2)
            with col1:
                alpha_val = st.number_input("α:", value=0.5, step=0.1, key="cd_alpha")
                beta_val = st.number_input("β:", value=0.5, step=0.1, key="cd_beta")
            with col2:
                px_val = st.number_input("Precio x:", value=150.0, step=10.0, key="cd_px")
                py_val = st.number_input("Precio y:", value=250.0, step=10.0, key="cd_py")
            
            M_val = st.number_input("Presupuesto M:", value=50000.0, step=1000.0, key="cd_M")
            
            if st.button("✅ Resolver", key="solve_cd"):
                result = opt.cobb_douglas_optimization(alpha_val, beta_val, px_val, py_val, M_val)
                
                st.success("✅ Solución con Lagrange:")
                
                with st.expander("📝 Ver pasos completos", expanded=True):
                    for step in result['latex_steps']:
                        st.latex(step)
                
                if result['result']['solutions']:
                    sol = result['result']['solutions'][0]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("x*", f"{sol['point'][0]:.4f}")
                    with col2:
                        st.metric("y*", f"{sol['point'][1]:.4f}")
                    with col3:
                        st.metric("f(x*, y*)", f"{sol['function_value']:.4f}")
        
        # ---- CASO 3: Integral de línea ----
        elif caso == "Integral de línea F·dr (círculo unitario)":
            st.markdown("### Integral de Línea Clásica")
            
            st.markdown("""
            Campo rotacional típico: F = (-y, x, 0)
            
            Curva: círculo unitario parametrizado como r(t) = (cos(t), sin(t), 0), t ∈ [0, 2π]
            
            **Resultado esperado:** ∫ F·dr = 2π
            """)
            
            st.info("💡 Este caso está implementado en la sección **Integral de Línea**. Ve allí para cálculo completo.")
            
            if st.button("📝 Mostrar derivación paso a paso", key="show_line_derivation"):
                st.markdown("### Derivación:")
                
                steps = [
                    r"\text{1. Campo: } \mathbf{F} = (-y, x, 0)",
                    r"\text{2. Curva: } \mathbf{r}(t) = (\cos t, \sin t, 0), \quad t \in [0, 2\pi]",
                    r"\text{3. Derivada: } \mathbf{r}'(t) = (-\sin t, \cos t, 0)",
                    r"\text{4. Sustituir en F: } \mathbf{F}(\mathbf{r}(t)) = (-\sin t, \cos t, 0)",
                    r"\text{5. Producto punto: } \mathbf{F} \cdot \mathbf{r}' = (-\sin t)(-\sin t) + (\cos t)(\cos t) + 0 = \sin^2 t + \cos^2 t = 1",
                    r"\text{6. Integrar: } \int_0^{2\pi} 1 \, dt = 2\pi"
                ]
                
                for step in steps:
                    st.latex(step)
                
                st.success("✅ Resultado: 2π ≈ 6.28318531")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🧮 Cálculo Vectorial 3D | Proyecto Profesional</p>
    <p>Módulo testeado (23 tests ✓) | CI/CD con GitHub Actions | Seguro y vectorizado</p>
    <p>Noviembre 2025</p>
</div>
""", unsafe_allow_html=True)
