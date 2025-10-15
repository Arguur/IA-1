#  Sistema de Clasificación de Verduras con IA
### Visión Artificial y Reconocimiento de Voz

Sistema dual de clasificación automática que combina **K-Means** para identificación de imágenes y **KNN** para reconocimiento de comandos de voz.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Accuracy](https://img.shields.io/badge/Accuracy-97.92%25-success)

##  Características

- **Reconocimiento de Voz**: 95.83% de precisión usando KNN con MFCC
- **Visión Artificial**: 100% de precisión con K-means en espacio LAB
- **Tiempo Real**: Interfaz gráfica con Tkinter para clasificación en vivo
- **Dataset**: 88 audios de 5 personas + 70 imágenes de verduras

##  Arquitectura

### Clasificación de Audio
1. Preprocesamiento (normalización, filtrado 80Hz-6.5kHz)
2. Extracción de 92 características (MFCC, ZCR, características espectrales)
3. Clasificación KNN en 2 etapas (k=7)

### Clasificación de Imágenes
1. Segmentación en espacio HSV
2. Conversión a espacio LAB
3. Extracción de características (L, a, b)
4. Clustering con K-means supervisado

##  Resultados

| Modalidad | Precisión | Dataset Prueba |
|-----------|-----------|----------------|
| Reconocimiento de Voz | 95.83% | 24 audios (6 personas) |
| Visión Artificial | 100% | 24 imágenes nuevas |
| **Sistema Completo** | **97.92%** | **48 muestras totales** |

##  Instalación
```bash
# Clonar repositorio
git clone https://github.com/Arguur/IA-1.git
cd IA-1

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

##  Uso
```bash
# Ejecutar interfaz principal
python main.py

# Entrenar modelos (opcional)
python train_audio_model.py
python train_image_model.py
```

##  Estructura del Proyecto
```
IA-1/
├── data/
│   ├── audios/          # Dataset de audios por persona
│   ├── images/          # Dataset de imágenes
│   └── models/          # Modelos entrenados (.pkl)
├── src/
│   ├── audio/
│   │   ├── preprocessor.py
│   │   ├── feature_extractor.py
│   │   └── knn_classifier.py
│   ├── vision/
│   │   ├── segmentation.py
│   │   ├── feature_extractor.py
│   │   └── kmeans_classifier.py
│   └── gui/
│       └── main_interface.py
├── notebooks/           # Análisis exploratorio
├── docs/
│   └── Informe_Final.pdf
├── requirements.txt
└── README.md
```

##  Tecnologías

**Procesamiento de Audio**
- `librosa` - Análisis de señales
- `sounddevice` - Captura de audio

**Visión Artificial**
- `OpenCV` - Procesamiento de imágenes
- `scikit-image` - Extracción de características

**Machine Learning**
- `NumPy` - Operaciones matemáticas
- `scikit-learn` - Métricas y validación
- Implementación custom de KNN y K-means

**Interfaz**
- `Tkinter` - GUI
- `Matplotlib/Plotly` - Visualizaciones

## 📈 Proceso de Desarrollo

### Mejoras Implementadas
- Sistema de dos etapas para distinguir papa/camote
- Optimización de k mediante validación cruzada
- Normalización de características LAB para robustez ante iluminación
- Selección de 8 características clave de 92 posibles

### Matriz de Confusión - Audio
[Incluir imagen de la matriz]

### Visualización PCA
[Incluir imagen 3D de los clusters]

##  Contexto Académico

**Materia**: Inteligencia Artificial I - 2024  
**Carrera**: Ingeniería en Mecatrónica  
**Institución**: [Tu Universidad]

##  Documentación Completa

Ver Informe para detalles sobre:
- Especificación del agente inteligente (Tabla REAS)
- Análisis matemático de los algoritmos
- Proceso de optimización de hiperparámetros
- Estudios de ablación

##  Mejoras Futuras

- [ ] Expandir dataset con más variabilidad
- [ ] Implementar data augmentation
- [ ] Agregar más categorías de verduras
- [ ] Deploy como aplicación web
- [ ] Optimización para edge devices (Raspberry Pi)


##  Autor

**Juan Francisco Huertas Coppo**
- GitHub: [@Arguur](https://github.com/Arguur)
- LinkedIn: www.linkedin.com/in/juan-francisco-huertas-coppo
