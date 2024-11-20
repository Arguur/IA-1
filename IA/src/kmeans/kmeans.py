import numpy as np
from .normalizador import Normalizador

# Implementación personalizada de K-means que mantiene correspondencia fija entre clusters y etiquetas
class KMeansPersonalizado:
    def __init__(self, k=4, max_iter=5):
        self.k = k
        self.max_iter = max_iter
        self.centroides = None
        self.etiquetas = ['papa', 'berenjena', 'zanahoria', 'camote']
        self.etiquetas_cluster = None
        self.centroides_ajustados = False
        
    def calcular_distancias(self, X):
        # Calcula distancia euclidiana entre cada punto y cada centroide
        distancias = []
        for centroide in self.centroides:
            dist = np.sqrt(np.sum((X - centroide) ** 2, axis=1))
            distancias.append(dist)
        return np.array(distancias)

    def ajustar_centroides(self, X, etiquetas_verdaderas):
        # Inicializa los centroides usando la media de las muestras de cada clase
        if self.centroides_ajustados:
            return
            
        print("\nAjustando centroides iniciales...")
        self.centroides = np.zeros((self.k, X.shape[1]))
        
        for i, etiqueta in enumerate(self.etiquetas):
            muestras_clase = X[np.array(etiquetas_verdaderas) == etiqueta]
            if len(muestras_clase) > 0:
                self.centroides[i] = np.mean(muestras_clase, axis=0)
                print(f"Centroide para {etiqueta}: L={self.centroides[i,0]:.2f}, a={self.centroides[i,1]:.2f}, b={self.centroides[i,2]:.2f}")
            
        self.centroides_ajustados = True
        
    def entrenar(self, X, etiquetas_verdaderas):
        # Asigna cada muestra al cluster más cercano y calcula precisión del modelo
        self.ajustar_centroides(X, etiquetas_verdaderas)
        distancias = self.calcular_distancias(X)
        self.etiquetas_cluster = np.argmin(distancias, axis=0)
        predicciones = [self.etiquetas[cluster] for cluster in self.etiquetas_cluster]
        precision = np.mean(np.array(predicciones) == np.array(etiquetas_verdaderas))
        print(f"\nPrecisión en el conjunto de entrenamiento: {precision*100:.2f}%")
        print("\nAsignaciones finales:")
        for i, (etiqueta, cluster) in enumerate(zip(etiquetas_verdaderas, self.etiquetas_cluster)):
            verdura_asignada = self.etiquetas[cluster]
            print(f"Muestra {i+1} ({etiqueta}) -> Asignada a {verdura_asignada}")
        return self
    
    def predecir(self, X):
        if not self.centroides_ajustados:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        distancias = self.calcular_distancias(X)
        return np.argmin(distancias, axis=0)