import numpy as np
from .procesamiento_imagen import ProcesadorImagen
from .kmeans import KMeansPersonalizado
from .visualizador import VisualizadorCaracteristicas
from .normalizador import Normalizador

# Implementa un clasificador de verduras basado en características de color usando K-means
class ClasificadorVerduras:
    def __init__(self):
        self.kmeans = KMeansPersonalizado()
        self.procesador = None
        self.etiquetas = ['papa', 'berenjena', 'zanahoria', 'camote']
        self.visualizador = VisualizadorCaracteristicas()
        self.normalizador = Normalizador()
        self.modo_debug = False
        
    def activar_debug(self):
        self.modo_debug = True
        
    def desactivar_debug(self):
        self.modo_debug = False
        
    def procesar_imagen(self, ruta_imagen, mostrar_proceso=False):
        # Extrae características de color en espacio LAB de la imagen
        self.procesador = ProcesadorImagen(ruta_imagen)
        self.procesador.procesar_imagen()
        if mostrar_proceso:
            self.procesador.mostrar_proceso()
        return self.procesador.extraer_caracteristicas()
        
    def entrenar(self, rutas_imagenes, etiquetas_verdaderas):
        X = []
        print("\nProcesando imágenes de entrenamiento...")
        
        # Extrae características de color de cada imagen de entrenamiento
        for i, (ruta, etiqueta) in enumerate(zip(rutas_imagenes, etiquetas_verdaderas)):
            print(f"\nProcesando imagen {i+1}/{len(rutas_imagenes)}")
            caracteristicas = self.procesar_imagen(ruta, mostrar_proceso=self.modo_debug)
            
            if caracteristicas is not None:
                self.visualizador.agregar_muestra(caracteristicas, etiqueta)
                valores = [
                    caracteristicas['l_mean'],
                    caracteristicas['a_mean'],
                    caracteristicas['b_mean']
                ]
                X.append(valores)
                print(f"Características extraídas para {etiqueta}:")
                print(f"L={valores[0]:.2f}, a={valores[1]:.2f}, b={valores[2]:.2f}")
        
        X = np.array(X)
        nombres_caracteristicas = ['l_mean', 'a_mean', 'b_mean']
        
        # Normaliza las características para mejorar el rendimiento del clustering
        self.normalizador.ajustar(X, nombres_caracteristicas)
        X_norm = self.normalizador.normalizar(X, nombres_caracteristicas)
        
        print("\nDatos normalizados:")
        for i, (x_norm, etiqueta) in enumerate(zip(X_norm, etiquetas_verdaderas)):
            print(f"{etiqueta}: L={x_norm[0]:.2f}, a={x_norm[1]:.2f}, b={x_norm[2]:.2f}")
        
        self.kmeans.entrenar(X_norm, etiquetas_verdaderas)
        
    def predecir(self, ruta_imagen):
        print("\nProcesando nueva imagen...")
        
        caracteristicas = self.procesar_imagen(ruta_imagen, mostrar_proceso=True)
        
        if caracteristicas is None:
            return None
            
        nombres_caracteristicas = ['l_mean', 'a_mean', 'b_mean']
        
        # Prepara las características para la predicción
        X = np.array([[caracteristicas['l_mean'], caracteristicas['a_mean'], caracteristicas['b_mean']]])
        X_norm = self.normalizador.normalizar(X, nombres_caracteristicas)
        
        # Predice la clase usando el modelo K-means entrenado
        cluster = self.kmeans.predecir(X_norm)[0]
        prediccion = self.etiquetas[cluster]
        
        print("\nCaracterísticas extraídas:")
        print(f"L (Luminosidad): {caracteristicas['l_mean']:.2f}")
        print(f"a (Verde-Rojo): {caracteristicas['a_mean']:.2f}")
        print(f"b (Azul-Amarillo): {caracteristicas['b_mean']:.2f}")
        
        print("\nMostrando gráfico de características con la nueva muestra...")
        self.visualizador.visualizar_3d(caracteristicas)
        
        return prediccion