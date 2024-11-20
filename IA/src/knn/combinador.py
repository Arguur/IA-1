import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import pickle
from itertools import combinations, islice
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

def preseleccionar_caracteristicas(X, y, n_caracteristicas=24):
    # Calcula la información mutua entre las características y las etiquetas
    mi_scores = mutual_info_classif(X, y)
    # Ordena las características por relevancia (información mutua) en orden descendente
    caracteristicas_ranking = sorted(zip(mi_scores, X.columns), reverse=True)
    # Devuelve las n características más relevantes
    return [feature for _, feature in caracteristicas_ranking[:n_caracteristicas]]

def evaluar_separacion_clases(X, y, peso_compacidad=2.0):
    # Encuentra las clases únicas en el conjunto de etiquetas
    clases = np.unique(y)
    # Calcula los centroides para cada clase
    centroides = np.array([X[y == clase].mean(axis=0) for clase in clases])
    
    dispersiones = []
    for i, clase in enumerate(clases):
        # Calcula la distancia promedio de los puntos de cada clase a su centroide
        puntos_clase = X[y == clase]
        distancias_centroide = np.sqrt(np.sum((puntos_clase - centroides[i]) ** 2, axis=1))
        dispersiones.append(np.mean(distancias_centroide) ** peso_compacidad)  # Aumenta el peso de la compacidad
    
    # Calcula las distancias entre centroides de todas las clases
    distancias_entre_centroides = squareform(pdist(centroides))
    max_ratios = []
    for i in range(len(clases)):
        # Calcula la relación máxima entre dispersión y distancia intercentroides
        ratios = [(dispersiones[i] + dispersiones[j]) / (distancias_entre_centroides[i, j] + 1e-8)
                 for j in range(len(clases)) if i != j]
        max_ratios.append(max(ratios))
    
    # Retorna la inversa de la media de los ratios máximos (mejor si es mayor)
    return 1.0 / (np.mean(max_ratios) + 1e-8)

def generar_combinaciones(caracteristicas, min_features=3, max_features=5, max_por_tamano=25000):
    # Genera todas las combinaciones posibles de características dentro de un rango de tamaños
    todas_combinaciones = []
    print(f"Características disponibles: {len(caracteristicas)}")
    
    for n in range(min_features, max_features + 1):
        # Genera combinaciones de n características, limitando el número máximo por tamaño
        combinaciones_n = list(islice(combinations(caracteristicas, n), max_por_tamano))
        todas_combinaciones.extend(combinaciones_n)
        print(f"Tamaño {n}: {len(combinaciones_n):,} combinaciones")
    
    return todas_combinaciones

def encontrar_mejores_combinaciones(archivo_pkl, n_mejores=5, max_combinaciones=25000, 
                                  min_features=3, max_features=10, n_caracteristicas_preseleccion=24):
    # Carga el archivo pickle que contiene los datos preprocesados
    with open(archivo_pkl, 'rb') as f:
        df = pickle.load(f)
    
    # Filtra las filas con las clases objetivo
    df = df[df['palabra'].isin(['papa', 'berenjena', 'zanahoria', 'camote'])]
    # Selecciona las características excluyendo columnas de metadatos
    caracteristicas = [col for col in df.columns if col not in ['num_persona', 'palabra', 'num_muestra']]
    print(f"\nCaracterísticas totales: {len(caracteristicas)}")
    
    # Preselecciona las n características más relevantes según información mutua
    caracteristicas = preseleccionar_caracteristicas(df[caracteristicas], df['palabra'], 
                                                   n_caracteristicas_preseleccion)
    print(f"Características preseleccionadas: {len(caracteristicas)}")
    
    # Genera combinaciones de características para evaluación
    todas_combinaciones = generar_combinaciones(caracteristicas, min_features, max_features, max_por_tamano=25000)
    resultados = []
    total = len(todas_combinaciones)
    mejor_actual = float('-inf')
    
    print("\nEvaluando combinaciones:")
    for i, combo in enumerate(todas_combinaciones):
        # Extrae las combinaciones seleccionadas y normaliza las características
        X = df[list(combo)].values
        y = df['palabra'].values
        X_scaled = StandardScaler().fit_transform(X)
        # Evalúa la calidad de separación de clases de la combinación actual
        separacion = evaluar_separacion_clases(X_scaled, y)
        
        # Actualiza si se encuentra una mejor combinación
        if separacion > mejor_actual:
            mejor_actual = separacion
            print(f"\nNueva mejor combinación encontrada ({i+1}/{total}):")
            print(f"Separación: {separacion:.4f}")
            print("Características:", list(combo))
        
        if (i + 1) % 100 == 0:
            # Muestra el progreso en evaluaciones cada 100 iteraciones
            print(f"\rProgreso: {i+1}/{total} ({(i+1)/total*100:.1f}%)", end="")
        
        # Guarda los resultados de la evaluación
        resultados.append({
            'caracteristicas': combo, 
            'separacion': separacion, 
            'n_caracteristicas': len(combo)
        })
    
    # Ordena los resultados por la métrica de separación y selecciona los mejores
    mejores = sorted(resultados, key=lambda x: x['separacion'], reverse=True)[:n_mejores]
    print("\n\nMejores combinaciones finales:")
    for i, res in enumerate(mejores, 1):
        print(f"\n{i}. Separación: {res['separacion']:.4f}")
        print("Características:", list(res['caracteristicas']))
    
    return mejores

def visualizar_caracteristicas(caracteristicas, archivo_pkl, sufijo=""):
    # Visualiza las características seleccionadas en un espacio de 2D o 3D (si hay más de 2)
    with open(archivo_pkl, 'rb') as f:
        df = pickle.load(f)
    
    df = df[df['palabra'].isin(['papa', 'berenjena', 'zanahoria', 'camote'])]
    
    colores = {
        'papa': '#FF4B4B',
        'berenjena': '#9D4EDD',
        'zanahoria': '#FF9F1C',
        'camote': '#8B4513'
    }
    
    X_scaled = StandardScaler().fit_transform(df[list(caracteristicas)])
    fig = go.Figure()
    
    if len(caracteristicas) > 3:
        # Reduce dimensiones a 3 componentes principales con PCA
        X_pca = PCA(n_components=3).fit_transform(X_scaled)
        ejes = ['PC1', 'PC2', 'PC3']
        datos_viz = X_pca
    else:
        ejes = caracteristicas[:3]
        datos_viz = X_scaled
    
    # Traza las distribuciones de las clases en el espacio reducido
    for palabra in colores:
        mask = df['palabra'] == palabra
        fig.add_trace(go.Scatter3d(
            x=datos_viz[mask, 0],
            y=datos_viz[mask, 1],
            z=datos_viz[mask, 2] if len(caracteristicas) >= 3 else np.zeros(sum(mask)),
            mode='markers',
            name=palabra,
            marker=dict(size=6, color=colores[palabra], opacity=0.8)
        ))
    
    fig.update_layout(
        title=dict(text='Visualización de Características', x=0.5, y=0.95, font=dict(size=20)),
        scene=dict(
            xaxis_title=ejes[0],
            yaxis_title=ejes[1],
            zaxis_title=ejes[2] if len(caracteristicas) >= 3 else None,
            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=1000,
        height=800
    )
    
    ruta_salida = Path(r"C:\Users\juanf\OneDrive\Escritorio\IA\datos\caracteristicas") / f"viz_{len(caracteristicas)}D{sufijo}.html"
    fig.write_html(str(ruta_salida))

def visualizar_mejores_combinaciones(archivo_pkl, max_combinaciones=25000, min_features=3, max_features=20):
    # Encuentra y visualiza las mejores combinaciones de características
    mejores = encontrar_mejores_combinaciones(
        archivo_pkl, 
        n_mejores=10, 
        max_combinaciones=max_combinaciones,
        min_features=min_features,
        max_features=max_features
    )
    
    for i, resultado in enumerate(mejores, 1):
        print(f"\n{i}. Separación: {resultado['separacion']:.4f}")
        print(f"Características ({resultado['n_caracteristicas']}):")
        print('\n'.join(f"   - {caract}" for caract in resultado['caracteristicas']))
        visualizar_caracteristicas(resultado['caracteristicas'], archivo_pkl, f"_combo{i}")

if __name__ == "__main__":
    try:
        # Encuentra el archivo más reciente en el directorio especificado
        directorio = Path(r"C:\Users\juanf\OneDrive\Escritorio\IA\datos\caracteristicas")
        archivo = max(directorio.glob("caracteristicas_1seg_20241118_110444.pkl"), 
                     key=lambda x: x.stat().st_mtime)
        
        # Visualiza combinaciones con restricciones de tamaño
        visualizar_mejores_combinaciones(
            archivo,
            max_combinaciones=10000,
            min_features=12,
            max_features=12
        )
        
    except Exception as e:
        print(f"Error: {e}")