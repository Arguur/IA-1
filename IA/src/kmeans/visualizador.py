import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .normalizador import Normalizador

# Visualiza características de color LAB en diferentes espacios dimensionales
class VisualizadorCaracteristicas:
    def __init__(self):
        self.caracteristicas = []
        self.etiquetas = []
        self.nombres_caracteristicas = ['l_mean', 'a_mean', 'b_mean']
        self.normalizador = Normalizador()
        # Centroides predefinidos para cada tipo de verdura
        self.centroides = np.array([
            [0.94, 0.19, 0.8],  # Papa
            [0.07, 0.04, 0.04], # Berenjena
            [0.62, 0.9, 0.95],  # Zanahoria
            [0.47, 0.62, 0.4]   # Camote
        ])
        self.nombres_centroides = ['Papa', 'Berenjena', 'Zanahoria', 'Camote']

    def agregar_muestra(self, caracteristicas, etiqueta):
        # Almacena nuevas características y su etiqueta
        self.caracteristicas.append([
            caracteristicas[nombre] for nombre in self.nombres_caracteristicas
        ])
        self.etiquetas.append(etiqueta)

    def dibujar_lineas_centroides(self, ax, punto, dimension_3d=True):
        # Dibuja líneas desde un punto a cada centroide usando diferentes estilos
        estilos_linea = [':', '--', '-.', '-']
        for i, (centroide, nombre) in enumerate(zip(self.centroides, self.nombres_centroides)):
            if dimension_3d:
                ax.plot([punto[0], centroide[0]], 
                       [punto[1], centroide[1]], 
                       [punto[2], centroide[2]], 
                       linestyle=estilos_linea[i],
                       color='gray',
                       alpha=0.5,
                       label=f'Distancia a {nombre}')
            else:
                ax.plot([punto[0], centroide[0]], 
                       [punto[1], centroide[1]], 
                       linestyle=estilos_linea[i],
                       color='gray',
                       alpha=0.5,
                       label=f'Distancia a {nombre}')
        
    def visualizar_3d(self, nueva_muestra=None):
        if not self.caracteristicas:
            print("No hay datos para visualizar")
            return
            
        # Normaliza todas las características
        X = np.array(self.caracteristicas)
        self.normalizador.ajustar(X, self.nombres_caracteristicas)
        X_norm = self.normalizador.normalizar(X, self.nombres_caracteristicas)

        # Crea visualización 3D y proyección 2D (a-b)
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(121, projection='3d')
        
        # Grafica puntos por clase con diferentes colores
        etiquetas_unicas = list(set(self.etiquetas))
        colores = plt.cm.tab10(np.linspace(0, 1, len(etiquetas_unicas)))
        for etiqueta, color in zip(etiquetas_unicas, colores):
            mask = np.array(self.etiquetas) == etiqueta
            datos = X_norm[mask]
            ax.scatter(datos[:, 0], datos[:, 1], datos[:, 2], 
                      c=[color], label=etiqueta, alpha=0.6)

        # Agrega centroides y sus etiquetas
        ax.scatter(self.centroides[:, 0], self.centroides[:, 1], self.centroides[:, 2],
                  c='black', marker='x', s=100, label='Centroides')
        for centroide, nombre in zip(self.centroides, self.nombres_centroides):
            ax.text(centroide[0], centroide[1], centroide[2], nombre,
                   fontsize=8, color='black')

        # Si hay nueva muestra, la agrega y dibuja líneas a centroides
        if nueva_muestra is not None:
            nueva_muestra_array = np.array([[
                nueva_muestra[nombre] for nombre in self.nombres_caracteristicas
            ]])
            nueva_muestra_norm = self.normalizador.normalizar(
                nueva_muestra_array, self.nombres_caracteristicas)
            ax.scatter(nueva_muestra_norm[0, 0], nueva_muestra_norm[0, 1], 
                      nueva_muestra_norm[0, 2],
                      c='red', marker='*', s=200, label='Nueva muestra',
                      edgecolors='black', linewidth=1)
            self.dibujar_lineas_centroides(ax, nueva_muestra_norm[0])

        # Configura ejes y títulos
        ax.set_xlabel('L (Luminosidad) normalizado')
        ax.set_ylabel('a (Verde-Rojo) normalizado')
        ax.set_zlabel('b (Azul-Amarillo) normalizado')
        ax.set_title('Características LAB normalizadas en 3D')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Crea visualizaciones 2D para cada par de características
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        pares_caracteristicas = [
            (0, 1, 'L', 'a'),
            (1, 2, 'a', 'b'),
            (0, 2, 'L', 'b')
        ]
        
        for (idx1, idx2, nombre1, nombre2), ax in zip(pares_caracteristicas, axs):
            
            # Grafica datos, centroides y nueva muestra en cada proyección 2D
            for etiqueta, color in zip(etiquetas_unicas, colores):
                mask = np.array(self.etiquetas) == etiqueta
                datos = X_norm[mask]
                ax.scatter(datos[:, idx1], datos[:, idx2],
                          c=[color], label=etiqueta, alpha=0.6)
                          
            ax.scatter(self.centroides[:, idx1], self.centroides[:, idx2],
                      c='black', marker='x', s=100, label='Centroides')
                      
            if nueva_muestra is not None:
                ax.scatter(nueva_muestra_norm[0, idx1], nueva_muestra_norm[0, idx2],
                          c='red', marker='*', s=200, label='Nueva muestra',
                          edgecolors='black', linewidth=1)
                punto_2d = [nueva_muestra_norm[0, idx1], nueva_muestra_norm[0, idx2]]
                centroides_2d = self.centroides[:, [idx1, idx2]]
                for i, (centroide, nombre) in enumerate(zip(centroides_2d, self.nombres_centroides)):
                    ax.plot([punto_2d[0], centroide[0]], 
                           [punto_2d[1], centroide[1]], 
                           linestyle=[':', '--', '-.', '-'][i],
                           color='gray',
                           alpha=0.5,
                           label=f'Distancia a {nombre}')
                           
            ax.set_xlabel(f'{nombre1} normalizado')
            ax.set_ylabel(f'{nombre2} normalizado')
            ax.set_title(f'{nombre1} vs {nombre2}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.tight_layout()
        plt.show()