"""
Módulo de Visualización de Campos Vectoriales
==============================================

Proporciona funciones para visualizar campos vectoriales en 3D,
incluyendo divergencia, rotacional, y líneas de campo.

Autor: Sistema de Cálculo Vectorial
Versión: 1.0
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
from typing import Tuple, Optional, Dict, Any


def ensure_array(func, *args):
    """Asegura que el resultado de una función lambdify sea siempre un array."""
    result = func(*args)
    if np.isscalar(result):
        # Si es escalar, crear array con el mismo shape que el primer argumento
        shape = args[0].shape if hasattr(args[0], 'shape') else (len(args[0]),)
        return np.full(shape, result)
    return result


def plot_vector_field_3d(
    F_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    vars_: Tuple[sp.Symbol, sp.Symbol, sp.Symbol],
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-2, 2), (-2, 2), (-2, 2)),
    n_points: int = 8,
    scale: float = 0.3,
    show_divergence: bool = False,
    show_curl: bool = False,
    title: str = "Campo Vectorial 3D"
) -> go.Figure:
    """
    Crea una visualización 3D de un campo vectorial con flechas.
    
    Parámetros
    ----------
    F_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Campo vectorial simbólico (P, Q, R)
    vars_ : Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
        Variables (x, y, z)
    bounds : Tuple[Tuple[float, float], ...]
        Límites de visualización ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    n_points : int
        Número de puntos por eje (total = n_points³)
    scale : float
        Factor de escala para las flechas
    show_divergence : bool
        Si True, colorea las flechas según divergencia
    show_curl : bool
        Si True, muestra líneas de campo rotacional
    title : str
        Título de la gráfica
    
    Retorna
    -------
    go.Figure
        Figura de Plotly con el campo vectorial
    """
    x_sym, y_sym, z_sym = vars_
    P, Q, R = F_sym
    
    # Convertir a funciones numéricas
    P_func = sp.lambdify(vars_, P, modules=['numpy'])
    Q_func = sp.lambdify(vars_, Q, modules=['numpy'])
    R_func = sp.lambdify(vars_, R, modules=['numpy'])
    
    # Crear malla 3D
    x_range = np.linspace(bounds[0][0], bounds[0][1], n_points)
    y_range = np.linspace(bounds[1][0], bounds[1][1], n_points)
    z_range = np.linspace(bounds[2][0], bounds[2][1], n_points)
    
    X, Y, Z = np.meshgrid(x_range, y_range, z_range)
    
    # Evaluar campo vectorial
    U = ensure_array(P_func, X, Y, Z)
    V = ensure_array(Q_func, X, Y, Z)
    W = ensure_array(R_func, X, Y, Z)
    
    # Calcular magnitud para colorear
    magnitude = np.sqrt(U**2 + V**2 + W**2)
    
    # Si se pide divergencia, calcularla
    if show_divergence:
        div_F = sp.diff(P, x_sym) + sp.diff(Q, y_sym) + sp.diff(R, z_sym)
        div_func = sp.lambdify(vars_, div_F, modules=['numpy'])
        div_values = ensure_array(div_func, X, Y, Z)
        color_values = div_values.flatten()
        colorscale = 'RdBu'
        colorbar_title = 'Divergencia ∇·F'
    else:
        color_values = magnitude.flatten()
        colorscale = 'Viridis'
        colorbar_title = 'Magnitud ||F||'
    
    # Crear figura
    fig = go.Figure()
    
    # Añadir vectores como conos 3D
    fig.add_trace(go.Cone(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        u=U.flatten(),
        v=V.flatten(),
        w=W.flatten(),
        colorscale=colorscale,
        sizemode="scaled",
        sizeref=scale,
        showscale=True,
        colorbar=dict(title=colorbar_title),
        cmin=color_values.min(),
        cmax=color_values.max(),
        colorbar_x=1.0
    ))
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(title='x', range=bounds[0]),
            yaxis=dict(title='y', range=bounds[1]),
            zaxis=dict(title='z', range=bounds[2]),
            aspectmode='cube'
        ),
        width=800,
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def plot_divergence_heatmap(
    F_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    vars_: Tuple[sp.Symbol, sp.Symbol, sp.Symbol],
    z_plane: float = 0.0,
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-2, 2), (-2, 2)),
    n_points: int = 50,
    title: str = "Mapa de Calor: Divergencia ∇·F"
) -> go.Figure:
    """
    Crea un mapa de calor 2D de la divergencia en un plano z=constante.
    
    Parámetros
    ----------
    F_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Campo vectorial simbólico
    vars_ : Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
        Variables (x, y, z)
    z_plane : float
        Valor de z donde cortar el plano
    bounds : Tuple[Tuple[float, float], Tuple[float, float]]
        Límites en x e y
    n_points : int
        Resolución de la malla
    title : str
        Título de la gráfica
    
    Retorna
    -------
    go.Figure
        Mapa de calor de la divergencia
    """
    x_sym, y_sym, z_sym = vars_
    P, Q, R = F_sym
    
    # Calcular divergencia simbólicamente
    div_F = sp.diff(P, x_sym) + sp.diff(Q, y_sym) + sp.diff(R, z_sym)
    div_func = sp.lambdify(vars_, div_F, modules=['numpy'])
    
    # Crear malla 2D
    x_range = np.linspace(bounds[0][0], bounds[0][1], n_points)
    y_range = np.linspace(bounds[1][0], bounds[1][1], n_points)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.full_like(X, z_plane)
    
    # Evaluar divergencia
    div_values = ensure_array(div_func, X, Y, Z)
    
    # Crear figura
    fig = go.Figure(data=go.Heatmap(
        x=x_range,
        y=y_range,
        z=div_values,
        colorscale='RdBu',
        colorbar=dict(title='∇·F'),
        zmid=0  # Centro en 0 (rojo=positivo, azul=negativo)
    ))
    
    fig.update_layout(
        title=dict(text=f"{title} (z={z_plane})", x=0.5, xanchor='center'),
        xaxis=dict(title='x'),
        yaxis=dict(title='y', scaleanchor='x'),
        width=700,
        height=600
    )
    
    return fig


def plot_curl_field(
    F_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    vars_: Tuple[sp.Symbol, sp.Symbol, sp.Symbol],
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-2, 2), (-2, 2), (-2, 2)),
    n_points: int = 8,
    scale: float = 0.3,
    title: str = "Campo Rotacional (∇×F)"
) -> go.Figure:
    """
    Visualiza el campo rotacional ∇×F como vectores.
    
    Parámetros
    ----------
    F_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Campo vectorial original
    vars_ : Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
        Variables (x, y, z)
    bounds : Tuple[Tuple[float, float], ...]
        Límites de visualización
    n_points : int
        Puntos por eje
    scale : float
        Escala de flechas
    title : str
        Título
    
    Retorna
    -------
    go.Figure
        Campo rotacional visualizado
    """
    x_sym, y_sym, z_sym = vars_
    P, Q, R = F_sym
    
    # Calcular rotacional simbólicamente
    curl_x = sp.diff(R, y_sym) - sp.diff(Q, z_sym)
    curl_y = sp.diff(P, z_sym) - sp.diff(R, x_sym)
    curl_z = sp.diff(Q, x_sym) - sp.diff(P, y_sym)
    
    # Crear campo rotacional
    curl_F = (curl_x, curl_y, curl_z)
    
    # Usar la función de campo vectorial general
    fig = plot_vector_field_3d(
        curl_F,
        vars_,
        bounds=bounds,
        n_points=n_points,
        scale=scale,
        title=title
    )
    
    return fig


def plot_combined_field_analysis(
    F_sym: Tuple[sp.Expr, sp.Expr, sp.Expr],
    vars_: Tuple[sp.Symbol, sp.Symbol, sp.Symbol],
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((-2, 2), (-2, 2), (-2, 2)),
    n_points: int = 6
) -> go.Figure:
    """
    Crea una visualización combinada con 3 subplots:
    1. Campo original F
    2. Mapa de divergencia ∇·F
    3. Campo rotacional ∇×F
    
    Parámetros
    ----------
    F_sym : Tuple[sp.Expr, sp.Expr, sp.Expr]
        Campo vectorial
    vars_ : Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
        Variables
    bounds : Tuple[Tuple[float, float], ...]
        Límites
    n_points : int
        Puntos por eje
    
    Retorna
    -------
    go.Figure
        Figura con subplots
    """
    from plotly.subplots import make_subplots
    
    x_sym, y_sym, z_sym = vars_
    P, Q, R = F_sym
    
    # Calcular divergencia y rotacional
    div_F = sp.diff(P, x_sym) + sp.diff(Q, y_sym) + sp.diff(R, z_sym)
    curl_x = sp.diff(R, y_sym) - sp.diff(Q, z_sym)
    curl_y = sp.diff(P, z_sym) - sp.diff(R, x_sym)
    curl_z = sp.diff(Q, x_sym) - sp.diff(P, y_sym)
    
    # Convertir a funciones
    P_func = sp.lambdify(vars_, P, modules=['numpy'])
    Q_func = sp.lambdify(vars_, Q, modules=['numpy'])
    R_func = sp.lambdify(vars_, R, modules=['numpy'])
    div_func = sp.lambdify(vars_, div_F, modules=['numpy'])
    
    # Malla
    x_range = np.linspace(bounds[0][0], bounds[0][1], n_points)
    y_range = np.linspace(bounds[1][0], bounds[1][1], n_points)
    z_range = np.linspace(bounds[2][0], bounds[2][1], n_points)
    X, Y, Z = np.meshgrid(x_range, y_range, z_range)
    
    # Evaluar
    U = ensure_array(P_func, X, Y, Z)
    V = ensure_array(Q_func, X, Y, Z)
    W = ensure_array(R_func, X, Y, Z)
    div_vals = ensure_array(div_func, X, Y, Z)
    
    # Crear subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Campo Vectorial F", "Divergencia ∇·F"),
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]]
    )
    
    # Campo vectorial original
    fig.add_trace(
        go.Cone(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            u=U.flatten(), v=V.flatten(), w=W.flatten(),
            colorscale='Viridis',
            sizemode="scaled",
            sizeref=0.3,
            showscale=False
        ),
        row=1, col=1
    )
    
    # Divergencia coloreada
    fig.add_trace(
        go.Cone(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            u=U.flatten(), v=V.flatten(), w=W.flatten(),
            colorscale='RdBu',
            cmin=div_vals.min(),
            cmax=div_vals.max(),
            sizemode="scaled",
            sizeref=0.3,
            showscale=True,
            colorbar=dict(title="∇·F", x=1.1)
        ),
        row=1, col=2
    )
    
    # Layout
    fig.update_layout(
        height=500,
        width=1200,
        title_text="Análisis del Campo Vectorial",
        title_x=0.5
    )
    
    # Actualizar ejes
    for i in [1, 2]:
        fig.update_scenes(
            dict(
                xaxis=dict(title='x', range=bounds[0]),
                yaxis=dict(title='y', range=bounds[1]),
                zaxis=dict(title='z', range=bounds[2]),
                aspectmode='cube'
            ),
            row=1, col=i
        )
    
    return fig
