import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import graycomatrix, graycoprops

# Procesa imágenes para segmentar verduras y extraer características de color
class ProcesadorImagen:
    def __init__(self, ruta_imagen):
        self.imagen = cv2.imread(ruta_imagen)
        if self.imagen is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}")
        self.imagen_original = self.imagen.copy()
        self.imagen_lab = None
        self.mascara = None
        self.contorno = None
        self.imagen_contorno = None
        self.imagen_mascara = None
        
    def procesar_imagen(self):
        # Convierte a LAB para análisis de color y HSV para segmentación
        self.imagen_lab = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2LAB)
        imagen_hsv = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2HSV)
        
        # Segmenta la verdura del fondo usando umbralización
        fondo = cv2.inRange(imagen_hsv, (0, 0, 0), (180, 30, 255))
        self.mascara = cv2.bitwise_not(fondo)
        
        # Limpia la máscara usando operaciones morfológicas
        kernel = np.ones((5,5), np.uint8)
        self.mascara = cv2.morphologyEx(self.mascara, cv2.MORPH_CLOSE, kernel)
        self.mascara = cv2.morphologyEx(self.mascara, cv2.MORPH_OPEN, kernel)
        
        # Encuentra el contorno más grande (la verdura)
        contornos, _ = cv2.findContours(self.mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return None
        self.contorno = max(contornos, key=cv2.contourArea)
        
        # Genera imágenes de visualización
        self.imagen_contorno = self.imagen.copy()
        cv2.drawContours(self.imagen_contorno, [self.contorno], -1, (0, 255, 0), 2)
        self.imagen_mascara = np.zeros_like(self.imagen)
        cv2.drawContours(self.imagen_mascara, [self.contorno], -1, (255, 255, 255), -1)
        
    def mostrar_proceso(self):
        # Visualiza los pasos del proceso de segmentación
        if self.contorno is None:
            print("Primero debe ejecutar procesar_imagen()")
            return
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Proceso de Segmentación', fontsize=16)
        axs[0, 0].imshow(cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('Imagen Original')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(self.imagen_lab)
        axs[0, 1].set_title('Espacio LAB')
        axs[0, 1].axis('off')
        axs[0, 2].imshow(self.mascara, cmap='gray')
        axs[0, 2].set_title('Máscara')
        axs[0, 2].axis('off')
        axs[1, 0].imshow(cv2.cvtColor(self.imagen_contorno, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title('Contorno Detectado')
        axs[1, 0].axis('off')
        axs[1, 1].imshow(cv2.cvtColor(self.imagen_mascara, cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title('Máscara Final')
        axs[1, 1].axis('off')
        resultado = cv2.bitwise_and(self.imagen_original, self.imagen_mascara)
        axs[1, 2].imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
        axs[1, 2].set_title('Resultado Final')
        axs[1, 2].axis('off')
        plt.tight_layout()
        plt.show()
        
    def extraer_caracteristicas(self):
        # Extrae valores medios de color en espacio LAB para la región segmentada
        if self.contorno is None:
            self.procesar_imagen()
        mascara_objeto = np.zeros_like(self.mascara)
        cv2.drawContours(mascara_objeto, [self.contorno], -1, 255, -1)
        mascara_objeto_3c = cv2.cvtColor(mascara_objeto, cv2.COLOR_GRAY2BGR)
        valores_medios = cv2.mean(self.imagen_lab, mask=mascara_objeto)
        return {
            'l_mean': valores_medios[0],
            'a_mean': valores_medios[1],
            'b_mean': valores_medios[2],
        }