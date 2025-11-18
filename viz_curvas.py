"""
Módulo de Visualización de Curvas e Integrales
==============================================

Funciones para visualizar curvas parametrizadas, campos vectoriales
a lo largo de curvas, y componentes tangenciales.

Autor: Sistema de Cálculo Vectorial
Versión: 1.0
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
from typing import Tuple, Optional


def ensure_array(func, *args):
    """Asegura que el resultado de una función lambdify sea siempre un array."""
    result = func(*args)
    if np.isscalar(result):
        # Si es escalar, crear array con el mismo shape que el primer argumento
        shape = args[0].shape if hasattr(args[0], 'shape') else (len(args[0]),)
        return np.full(shape, result)
    return result


def plot_parametric_curve_3d(
    r_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    t_var: sp.Symbol,
    t_range: Tuple[float, float],
    n_points: int = 100,
    show_tangent: bool = True,
    tangent_points: int = 10,
    title: str = "Curva Parametrizada r(t)"
) -> go.Figure:
    """
    Visualiza una curva parametrizada en 3D con vectores tangentes opcionales.
    
    Parámetros
    ----------
    r_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Curva parametrizada (x(t), y(t), z(t))
    t_var : sp.Symbol
        Parámetro t
    t_range : Tuple[float, float]
        Rango (t₀, t₁)
    n_points : int
        Puntos para graficar la curva
    show_tangent : bool
        Mostrar vectores tangentes r'(t)
    tangent_points : int
        Número de vectores tangentes
    title : str
        Título
    
    Retorna
    -------
    go.Figure
        Curva 3D con tangentes
    """
    x_t, y_t, z_t = r_sym
    t0, t1 = t_range
    
    # Convertir a funciones
    x_func = sp.lambdify(t_var, x_t, modules=['numpy'])
    y_func = sp.lambdify(t_var, y_t, modules=['numpy'])
    z_func = sp.lambdify(t_var, z_t, modules=['numpy'])
    
    # Calcular derivadas
    dx_dt = sp.diff(x_t, t_var)
    dy_dt = sp.diff(y_t, t_var)
    dz_dt = sp.diff(z_t, t_var)
    
    dx_func = sp.lambdify(t_var, dx_dt, modules=['numpy'])
    dy_func = sp.lambdify(t_var, dy_dt, modules=['numpy'])
    dz_func = sp.lambdify(t_var, dz_dt, modules=['numpy'])
    
    # Evaluar curva
    t_vals = np.linspace(t0, t1, n_points)
    x_vals = ensure_array(x_func, t_vals)
    y_vals = ensure_array(y_func, t_vals)
    z_vals = ensure_array(z_func, t_vals)
    
    # Crear figura
    fig = go.Figure()
    
    # Curva principal
    fig.add_trace(go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='lines',
        line=dict(color='blue', width=4),
        name='r(t)'
    ))
    
    # Puntos inicial y final
    fig.add_trace(go.Scatter3d(
        x=[x_vals[0], x_vals[-1]],
        y=[y_vals[0], y_vals[-1]],
        z=[z_vals[0], z_vals[-1]],
        mode='markers',
        marker=dict(size=[8, 8], color=['green', 'red']),
        name='Inicio/Fin'
    ))
    
    # Vectores tangentes
    if show_tangent:
        t_tangent = np.linspace(t0, t1, tangent_points)
        x_tangent = ensure_array(x_func, t_tangent)
        y_tangent = ensure_array(y_func, t_tangent)
        z_tangent = ensure_array(z_func, t_tangent)
        
        dx_tangent = ensure_array(dx_func, t_tangent)
        dy_tangent = ensure_array(dy_func, t_tangent)
        dz_tangent = ensure_array(dz_func, t_tangent)
        
        # Normalizar para visualización
        scale = 0.2
        
        fig.add_trace(go.Cone(
            x=x_tangent,
            y=y_tangent,
            z=z_tangent,
            u=dx_tangent,
            v=dy_tangent,
            w=dz_tangent,
            colorscale='Oranges',
            sizemode="scaled",
            sizeref=scale,
            showscale=False,
            name="r'(t)"
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title='x'),
            yaxis=dict(title='y'),
            zaxis=dict(title='z'),
            aspectmode='data'
        ),
        width=800,
        height=700,
        showlegend=True
    )
    
    return fig


def plot_line_integral_visualization(
    F_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    r_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    t_var: sp.Symbol,
    t_range: Tuple[float, float],
    vars_: Tuple[sp.Symbol, sp.Symbol, sp.Symbol],
    n_points: int = 100,
    n_field_points: int = 10,
    title: str = "Integral de Línea: ∫ F·dr"
) -> go.Figure:
    """
    Visualización completa de integral de línea con curva, campo y componente tangencial.
    
    Parámetros
    ----------
    F_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Campo vectorial (P, Q, R)
    r_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Curva parametrizada
    t_var : sp.Symbol
        Parámetro
    t_range : Tuple[float, float]
        Rango de t
    vars_ : Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
        Variables (x, y, z)
    n_points : int
        Puntos para la curva
    n_field_points : int
        Puntos para campo vectorial
    title : str
        Título
    
    Retorna
    -------
    go.Figure
        Visualización completa
    """
    P, Q, R = F_sym
    x_t, y_t, z_t = r_sym
    t0, t1 = t_range
    x_sym, y_sym, z_sym = vars_
    
    # Convertir curva a funciones
    x_func = sp.lambdify(t_var, x_t, modules=['numpy'])
    y_func = sp.lambdify(t_var, y_t, modules=['numpy'])
    z_func = sp.lambdify(t_var, z_t, modules=['numpy'])
    
    # Derivadas
    dx_dt = sp.diff(x_t, t_var)
    dy_dt = sp.diff(y_t, t_var)
    dz_dt = sp.diff(z_t, t_var)
    
    dx_func = sp.lambdify(t_var, dx_dt, modules=['numpy'])
    dy_func = sp.lambdify(t_var, dy_dt, modules=['numpy'])
    dz_func = sp.lambdify(t_var, dz_dt, modules=['numpy'])
    
    # Sustituir r(t) en F
    F_of_r_P = P.subs({x_sym: x_t, y_sym: y_t, z_sym: z_t})
    F_of_r_Q = Q.subs({x_sym: x_t, y_sym: y_t, z_sym: z_t})
    F_of_r_R = R.subs({x_sym: x_t, y_sym: y_t, z_sym: z_t})
    
    F_P_func = sp.lambdify(t_var, F_of_r_P, modules=['numpy'])
    F_Q_func = sp.lambdify(t_var, F_of_r_Q, modules=['numpy'])
    F_R_func = sp.lambdify(t_var, F_of_r_R, modules=['numpy'])
    
    # Producto punto F·dr/dt
    dot_product = F_of_r_P * dx_dt + F_of_r_Q * dy_dt + F_of_r_R * dz_dt
    dot_func = sp.lambdify(t_var, dot_product, modules=['numpy'])
    
    # Evaluar a lo largo de la curva
    t_vals = np.linspace(t0, t1, n_points)
    x_vals = ensure_array(x_func, t_vals)
    y_vals = ensure_array(y_func, t_vals)
    z_vals = ensure_array(z_func, t_vals)
    
    dx_vals = ensure_array(dx_func, t_vals)
    dy_vals = ensure_array(dy_func, t_vals)
    dz_vals = ensure_array(dz_func, t_vals)
    
    F_P_vals = ensure_array(F_P_func, t_vals)
    F_Q_vals = ensure_array(F_Q_func, t_vals)
    F_R_vals = ensure_array(F_R_func, t_vals)
    
    dot_vals = ensure_array(dot_func, t_vals)
    
    # Crear figura
    fig = go.Figure()
    
    # Curva con color según F·dr (trabajo)
    fig.add_trace(go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='lines',
        line=dict(
            color=dot_vals,
            colorscale='RdYlGn',  # Rojo=negativo, Verde=positivo
            width=6,
            colorbar=dict(title='F·dr/dt')
        ),
        name='Curva C'
    ))
    
    # Campo vectorial F a lo largo de C
    t_field = np.linspace(t0, t1, n_field_points)
    x_field = ensure_array(x_func, t_field)
    y_field = ensure_array(y_func, t_field)
    z_field = ensure_array(z_func, t_field)
    
    F_P_field = ensure_array(F_P_func, t_field)
    F_Q_field = ensure_array(F_Q_func, t_field)
    F_R_field = ensure_array(F_R_func, t_field)
    
    fig.add_trace(go.Cone(
        x=x_field,
        y=y_field,
        z=z_field,
        u=F_P_field,
        v=F_Q_field,
        w=F_R_field,
        colorscale='Blues',
        sizemode="scaled",
        sizeref=0.3,
        showscale=False,
        name='F'
    ))
    
    # Puntos inicio/fin
    fig.add_trace(go.Scatter3d(
        x=[x_vals[0], x_vals[-1]],
        y=[y_vals[0], y_vals[-1]],
        z=[z_vals[0], z_vals[-1]],
        mode='markers',
        marker=dict(size=[10, 10], color=['green', 'red']),
        name='Inicio/Fin'
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title='x'),
            yaxis=dict(title='y'),
            zaxis=dict(title='z'),
            aspectmode='data'
        ),
        width=900,
        height=700,
        showlegend=True
    )
    
    return fig


def plot_integrand_graph(
    F_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    r_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    t_var: sp.Symbol,
    t_range: Tuple[float, float],
    vars_: Tuple[sp.Symbol, sp.Symbol, sp.Symbol],
    n_points: int = 200,
    title: str = "Integrando: F·(dr/dt)"
) -> go.Figure:
    """
    Gráfica 2D del integrando F·(dr/dt) vs t con MÁXIMA PRECISIÓN.
    
    Criterios estrictos:
    1. Dominio completo del parámetro t
    2. Signo absolutamente correcto del integrando
    3. Áreas positivas y negativas claramente diferenciadas
    4. Valor numérico destacado con precisión
    
    Parámetros
    ----------
    F_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Campo vectorial
    r_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Curva
    t_var : sp.Symbol
        Parámetro
    t_range : Tuple[float, float]
        Rango COMPLETO
    vars_ : Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
        Variables
    n_points : int
        Puntos (más puntos = más precisión)
    title : str
        Título
    
    Retorna
    -------
    go.Figure
        Gráfica del integrando con precisión absoluta
    """
    P, Q, R = F_sym
    x_t, y_t, z_t = r_sym
    t0, t1 = t_range
    x_sym, y_sym, z_sym = vars_
    
    # ========== CÁLCULO DEL INTEGRANDO ==========
    # Derivadas de la curva
    dx_dt = sp.diff(x_t, t_var)
    dy_dt = sp.diff(y_t, t_var)
    dz_dt = sp.diff(z_t, t_var)
    
    # F(r(t)) - sustituir la curva en el campo
    F_of_r_P = P.subs({x_sym: x_t, y_sym: y_t, z_sym: z_t})
    F_of_r_Q = Q.subs({x_sym: x_t, y_sym: y_t, z_sym: z_t})
    F_of_r_R = R.subs({x_sym: x_t, y_sym: y_t, z_sym: z_t})
    
    # F·(dr/dt) - producto punto
    dot_product = F_of_r_P * dx_dt + F_of_r_Q * dy_dt + F_of_r_R * dz_dt
    dot_product = sp.simplify(dot_product)
    
    # Lambdify para evaluación numérica
    dot_func = sp.lambdify(t_var, dot_product, modules=['numpy'])
    
    # ========== EVALUACIÓN NUMÉRICA ==========
    # Usar alta resolución para capturar todos los detalles
    t_vals = np.linspace(t0, t1, n_points)
    integrand_vals = np.array([dot_func(t) for t in t_vals])
    
    # ========== CÁLCULO DE LA INTEGRAL ==========
    # Calcular el área total (integral numérica)
    try:
        from scipy import integrate as sci_integrate
        integral_value, error = sci_integrate.quad(dot_func, t0, t1, limit=100)
    except:
        # Si scipy falla, usar trapezoides
        integral_value = np.trapz(integrand_vals, t_vals)
        error = 0
    
    # ========== SEPARAR ÁREAS POSITIVAS Y NEGATIVAS ==========
    # Para colorear correctamente según el signo
    integrand_positive = np.where(integrand_vals >= 0, integrand_vals, 0)
    integrand_negative = np.where(integrand_vals < 0, integrand_vals, 0)
    
    # Calcular áreas separadas
    area_positive = np.trapz(integrand_positive, t_vals)
    area_negative = np.trapz(integrand_negative, t_vals)
    
    # ========== CREAR FIGURA ==========
    fig = go.Figure()
    
    # ÁREA POSITIVA (verde - trabajo a favor)
    fig.add_trace(go.Scatter(
        x=t_vals,
        y=integrand_positive,
        mode='lines',
        line=dict(width=0),
        fill='tozeroy',
        fillcolor='rgba(0, 200, 0, 0.4)',
        name=f'Área Positiva = {area_positive:.4f}',
        hovertemplate='t=%{x:.4f}<br>F·(dr/dt)=%{y:.4f}<extra></extra>',
        showlegend=True
    ))
    
    # ÁREA NEGATIVA (roja - trabajo en contra)
    fig.add_trace(go.Scatter(
        x=t_vals,
        y=integrand_negative,
        mode='lines',
        line=dict(width=0),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.4)',
        name=f'Área Negativa = {area_negative:.4f}',
        hovertemplate='t=%{x:.4f}<br>F·(dr/dt)=%{y:.4f}<extra></extra>',
        showlegend=True
    ))
    
    # LÍNEA DEL INTEGRANDO (azul oscuro)
    fig.add_trace(go.Scatter(
        x=t_vals,
        y=integrand_vals,
        mode='lines',
        line=dict(color='darkblue', width=3),
        name='F·(dr/dt)',
        hovertemplate='t=%{x:.4f}<br>F·(dr/dt)=%{y:.4f}<extra></extra>',
        showlegend=True
    ))
    
    # LÍNEA DE REFERENCIA EN y=0
    fig.add_hline(
        y=0, 
        line_dash="solid", 
        line_color="black", 
        line_width=2,
        annotation_text="y = 0 (eje t)",
        annotation_position="right"
    )
    
    # ========== AJUSTAR RANGOS DE EJES ==========
    # Asegurar que el rango vertical muestre claramente positivos Y negativos
    y_min = np.min(integrand_vals)
    y_max = np.max(integrand_vals)
    y_range = y_max - y_min
    y_padding = y_range * 0.15  # 15% de padding
    
    # Si todo es positivo o todo negativo, ajustar para mostrar el eje cero
    if y_min >= 0:
        y_min_plot = -y_max * 0.1  # Mostrar un poco bajo cero
        y_max_plot = y_max + y_padding
    elif y_max <= 0:
        y_max_plot = -y_min * 0.1  # Mostrar un poco sobre cero
        y_min_plot = y_min - y_padding
    else:
        y_min_plot = y_min - y_padding
        y_max_plot = y_max + y_padding
    
    # ========== ANOTACIONES ==========
    # Valor de la integral con signo correcto
    sign_symbol = "+" if integral_value >= 0 else ""
    sign_color = "green" if integral_value >= 0 else "red"
    
    fig.add_annotation(
        text=f"<b>∫ F·(dr/dt) dt = {sign_symbol}{integral_value:.8f}</b>",
        xref="paper", yref="paper",
        x=0.5, y=1.12,
        xanchor='center',
        showarrow=False,
        bgcolor=sign_color,
        opacity=0.9,
        font=dict(size=16, color='white'),
        bordercolor='black',
        borderwidth=2
    )
    
    # Información adicional
    fig.add_annotation(
        text=f"Área sobre eje (positiva): <b>{area_positive:.6f}</b><br>" +
             f"Área bajo eje (negativa): <b>{area_negative:.6f}</b><br>" +
             f"Suma algebraica: <b>{sign_symbol}{integral_value:.6f}</b>",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        showarrow=False,
        bgcolor="rgba(255, 255, 200, 0.9)",
        bordercolor='black',
        borderwidth=1,
        font=dict(size=11),
        align='left'
    )
    
    # ========== LAYOUT ==========
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Dominio: t ∈ [{t0:.4f}, {t1:.4f}]</sub>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis=dict(
            title=f'Parámetro t',
            range=[t0, t1],
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        ),
        yaxis=dict(
            title='F·(dr/dt)',
            range=[y_min_plot, y_max_plot],
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False
        ),
        width=900,
        height=550,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.02,
            xanchor='left',
            yanchor='bottom',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        hovermode='x unified',
        plot_bgcolor='white'
    )
    
    return fig


def plot_surface_parametric(
    r_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    u_var: sp.Symbol,
    v_var: sp.Symbol,
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    n_u: int = 30,
    n_v: int = 30,
    show_normal: bool = True,
    normal_points: int = 10,
    title: str = "Superficie Parametrizada r(u,v)"
) -> go.Figure:
    """
    Visualiza una superficie parametrizada con vectores normales.
    
    Parámetros
    ----------
    r_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Superficie (x(u,v), y(u,v), z(u,v))
    u_var, v_var : sp.Symbol
        Parámetros
    u_range, v_range : Tuple[float, float]
        Rangos
    n_u, n_v : int
        Resolución
    show_normal : bool
        Mostrar vectores normales
    normal_points : int
        Número de normales
    title : str
        Título
    
    Retorna
    -------
    go.Figure
        Superficie con normales
    """
    x_uv, y_uv, z_uv = r_sym
    u0, u1 = u_range
    v0, v1 = v_range
    
    # Convertir a funciones
    x_func = sp.lambdify((u_var, v_var), x_uv, modules=['numpy'])
    y_func = sp.lambdify((u_var, v_var), y_uv, modules=['numpy'])
    z_func = sp.lambdify((u_var, v_var), z_uv, modules=['numpy'])
    
    # Evaluar superficie
    u_vals = np.linspace(u0, u1, n_u)
    v_vals = np.linspace(v0, v1, n_v)
    U, V = np.meshgrid(u_vals, v_vals)
    
    X = ensure_array(x_func, U, V)
    Y = ensure_array(y_func, U, V)
    Z = ensure_array(z_func, U, V)
    
    # Crear figura
    fig = go.Figure()
    
    # Superficie
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        showscale=True,
        name='Superficie S'
    ))
    
    # Vectores normales
    if show_normal:
        # Calcular derivadas parciales
        r_u = (sp.diff(x_uv, u_var), sp.diff(y_uv, u_var), sp.diff(z_uv, u_var))
        r_v = (sp.diff(x_uv, v_var), sp.diff(y_uv, v_var), sp.diff(z_uv, v_var))
        
        # Producto cruz r_u × r_v
        normal_x = r_u[1]*r_v[2] - r_u[2]*r_v[1]
        normal_y = r_u[2]*r_v[0] - r_u[0]*r_v[2]
        normal_z = r_u[0]*r_v[1] - r_u[1]*r_v[0]
        
        nx_func = sp.lambdify((u_var, v_var), normal_x, modules=['numpy'])
        ny_func = sp.lambdify((u_var, v_var), normal_y, modules=['numpy'])
        nz_func = sp.lambdify((u_var, v_var), normal_z, modules=['numpy'])
        
        # Evaluar en puntos
        u_norm = np.linspace(u0, u1, normal_points)
        v_norm = np.linspace(v0, v1, normal_points)
        U_norm, V_norm = np.meshgrid(u_norm, v_norm)
        
        X_norm = ensure_array(x_func, U_norm, V_norm)
        Y_norm = ensure_array(y_func, U_norm, V_norm)
        Z_norm = ensure_array(z_func, U_norm, V_norm)
        
        NX = ensure_array(nx_func, U_norm, V_norm)
        NY = ensure_array(ny_func, U_norm, V_norm)
        NZ = ensure_array(nz_func, U_norm, V_norm)
        
        fig.add_trace(go.Cone(
            x=X_norm.flatten(),
            y=Y_norm.flatten(),
            z=Z_norm.flatten(),
            u=NX.flatten(),
            v=NY.flatten(),
            w=NZ.flatten(),
            colorscale='Reds',
            sizemode="scaled",
            sizeref=0.2,
            showscale=False,
            name='Normal n'
        ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title='x'),
            yaxis=dict(title='y'),
            zaxis=dict(title='z'),
            aspectmode='data'
        ),
        width=800,
        height=700,
        showlegend=True
    )
    
    return fig
