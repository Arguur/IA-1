import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from .grabador_audio import GrabadorAudio
from .preprocesamiento import procesar_audio
from .exctractor_caracteristicas import ExtractorCaracteristicas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import librosa

# Implementa clasificación KNN para muestras de audio de tubérculos usando características MFCC
class ClasificadorKNN:
    def __init__(self, k=7):
        self.k = k
        # Características seleccionadas basadas en análisis de importancia
        self.caracteristicas_objetivo = [
            'mfcc3_media',
            'ancho_banda_std',
            'zcr_max',
            'mfcc_delta20_media',
            'mfcc2_std',
            'mfcc5_media',
            'mfcc_delta5_std',
            'mfcc_delta3_std'
        ]
        # Esquema de colores para visualización
        self.colores = {
            'papa': '#FF4B4B',
            'berenjena': '#9D4EDD',
            'zanahoria': '#FF9F1C',
            'camote': '#8B4513'
        }
        self.cargar_datos_entrenamiento()
        
    def cargar_datos_entrenamiento(self):
        ruta_base = Path(r"C:\Users\juanf\OneDrive\Escritorio\IA\datos\caracteristicas")
        # Carga datasets con características extraídas para diferentes duraciones de audio
        with open(ruta_base / "caracteristicas_3seg_20241118_105442.pkl", 'rb') as f:
            self.df_3seg = pickle.load(f)
        with open(ruta_base / "caracteristicas_1seg_20241118_110444.pkl", 'rb') as f:
            self.df_1seg = pickle.load(f)

    def visualizar_pca(self, X_train, y_train, x_test, prediccion, vecinos_indices=None, es_binario=False):
        X_completo = np.vstack([X_train, x_test])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_completo)
        
        # Calcula índices de vecinos si no se proporcionaron
        if vecinos_indices is None:
            X_train_scaled = X_scaled[:-1]
            x_test_scaled = X_scaled[-1]
            distancias = [(np.sqrt(np.sum((X_train_scaled[i] - x_test_scaled) ** 2)), i) 
                         for i in range(len(X_train_scaled))]
            distancias.sort(key=lambda x: x[0])
            vecinos_indices = [d[1] for d in distancias[:self.k]]
        
        # Reduce dimensionalidad para visualización 3D
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        X_train_pca = X_pca[:-1]
        x_test_pca = X_pca[-1]
        
        varianza_explicada = pca.explained_variance_ratio_
        
        # Evalúa calidad de la proyección PCA
        distancias_originales = [np.linalg.norm(X_scaled[idx] - X_scaled[-1]) for idx in vecinos_indices]
        distancias_pca = [np.linalg.norm(X_train_pca[idx] - x_test_pca) for idx in vecinos_indices]
        correlacion = np.corrcoef(distancias_originales, distancias_pca)[0,1]
        
        # Genera visualización interactiva
        fig = go.Figure()
        clases_a_mostrar = ['papa', 'camote'] if es_binario else ['berenjena', 'zanahoria', 'papa', 'camote']
        y_train_array = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
        
        for clase in clases_a_mostrar:
            mask = y_train_array == clase
            if mask.any():
                mask_sin_vecinos = np.array([i not in vecinos_indices for i in range(len(mask)) if mask[i]])
                indices = np.where(mask)[0][mask_sin_vecinos]
                if len(indices) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=X_train_pca[indices, 0],
                        y=X_train_pca[indices, 1],
                        z=X_train_pca[indices, 2],
                        mode='markers',
                        name=clase,
                        marker=dict(size=8, color=self.colores[clase], opacity=0.8)
                    ))
        
        # Resalta los k vecinos más cercanos
        if vecinos_indices is not None:
            fig.add_trace(go.Scatter3d(
                x=X_train_pca[vecinos_indices, 0],
                y=X_train_pca[vecinos_indices, 1],
                z=X_train_pca[vecinos_indices, 2],
                mode='markers',
                name='K vecinos más cercanos',
                marker=dict(
                    size=12,
                    color=[self.colores[y_train_array[i]] for i in vecinos_indices],
                    opacity=1,
                    line=dict(color='white', width=2)
                )
            ))
            
            # Dibuja líneas entre muestra nueva y sus vecinos
            for idx in vecinos_indices:
                fig.add_trace(go.Scatter3d(
                    x=[x_test_pca[0], X_train_pca[idx, 0]],
                    y=[x_test_pca[1], X_train_pca[idx, 1]],
                    z=[x_test_pca[2], X_train_pca[idx, 2]],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False
                ))
        
        # Muestra la nueva muestra a clasificar
        fig.add_trace(go.Scatter3d(
            x=[x_test_pca[0]],
            y=[x_test_pca[1]],
            z=[x_test_pca[2]],
            mode='markers',
            name='Muestra nueva',
            marker=dict(
                size=15,
                color='black',
                symbol='diamond',
                line=dict(color='white', width=2)
            )
        ))
        
        # Configura título y layout
        titulo = "PCA Clasificación Papa/Camote" if es_binario else "PCA Clasificación General"
        fig.update_layout(
            title=dict(
                text=f'{titulo}<br>Predicción: {prediccion}<br>Calidad de proyección: {correlacion:.2f}<br>'
                     f'Varianza explicada: PC1={varianza_explicada[0]:.1f}%, PC2={varianza_explicada[1]:.1f}%, '
                     f'PC3={varianza_explicada[2]:.1f}%',
                x=0.5,
                y=0.95,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title=f'PC1 ({varianza_explicada[0]:.1f}%)',
                yaxis_title=f'PC2 ({varianza_explicada[1]:.1f}%)',
                zaxis_title=f'PC3 ({varianza_explicada[2]:.1f}%)'
            ),
            width=1000,
            height=800
        )
        
        nombre_archivo = "pca_papa_camote.html" if es_binario else "pca_general.html"
        ruta_salida = Path(r"C:\Users\juanf\OneDrive\Escritorio\IA\datos\caracteristicas") / nombre_archivo
        fig.write_html(str(ruta_salida))
        
        return {
            'correlacion_distancias': correlacion,
            'varianza_explicada': varianza_explicada,
            'distancias_originales': distancias_originales,
            'distancias_pca': distancias_pca
        }

    # Extrae características del primer segmento de audio
    def obtener_caracteristicas_segmento0(self, df):
        caracteristicas = [df[f"seg0_{caract}"].values for caract in self.caracteristicas_objetivo]
        return np.array(caracteristicas).T
    
    # Extrae características de todo el audio
    def obtener_caracteristicas_completas(self, df):
        caracteristicas = []
        for caract in self.caracteristicas_objetivo:
            if caract not in df.columns:
                columnas_coincidentes = [col for col in df.columns if caract in col]
                if columnas_coincidentes:
                    caracteristicas.append(df[columnas_coincidentes[0]].values)
                else:
                    raise ValueError(f"No se encontró la característica {caract}")
            else:
                caracteristicas.append(df[caract].values)
        return np.array(caracteristicas).T
    
    # Encuentra los k vecinos más cercanos en espacio PCA
    def encontrar_k_vecinos(self, X, x_test, y):
        X_completo = np.vstack([X, x_test.reshape(1, -1)])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_completo)
        
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        X_train_pca = X_pca[:-1]
        x_test_pca = X_pca[-1]
        
        diferencias = X_train_pca - x_test_pca
        distancias = np.sqrt(np.sum(diferencias**2, axis=1))
        
        indices_ordenados = np.argsort(distancias)
        k_indices = indices_ordenados[:self.k]
        etiquetas = [y.iloc[i] for i in k_indices]
        
        return etiquetas, k_indices.tolist()
        
    # Determina la clase por votación mayoritaria
    def predecir_clase(self, vecinos):
        return max(set(vecinos), key=vecinos.count)
    
    # Graba y procesa nuevo audio para clasificación
    def procesar_nuevo_audio(self):
        grabador = GrabadorAudio()
        audio_raw = grabador.grabar_y_guardar("temp_muestra")
        if audio_raw is None:
            raise Exception("Error en la grabación")
            
        ruta_entrada = Path(r"C:\Users\juanf\OneDrive\Escritorio\IA\datos\muestras_audio\temp_muestra.wav")
        directorio_salida = Path(r"C:\Users\juanf\OneDrive\Escritorio\IA\datos\audio_procesado")
        
        exito, ruta_procesada = procesar_audio(ruta_entrada, directorio_salida)
        if not exito:
            raise Exception("Error en el procesamiento")
        
        # Divide audio en segmentos para análisis
        audio, _ = librosa.load(str(ruta_procesada), sr=16000)
        longitud_segmento = len(audio) // 3
        segmentos = [audio[i*longitud_segmento:(i+1)*longitud_segmento] for i in range(3)]
        
        extractor = ExtractorCaracteristicas(freq_muestreo=16000, num_segmentos=3)
        caracteristicas_3seg = {}
        
        for i, segmento in enumerate(segmentos):
            if caract_seg := extractor.extraer_caracteristicas_segmento(segmento, f"seg{i}_"):
                caracteristicas_3seg.update(caract_seg)
        
        caracteristicas_1seg = extractor.extraer_caracteristicas_segmento(audio)
        return caracteristicas_3seg, caracteristicas_1seg
    
    # Realiza clasificación en dos etapas
    def clasificar(self, caracteristicas_3seg, caracteristicas_1seg):
        # Primera etapa: clasificación general usando segmentos de 3 segundos
        X_train_3seg = self.obtener_caracteristicas_segmento0(self.df_3seg)
        x_test_3seg = self.obtener_caracteristicas_segmento0(pd.DataFrame([caracteristicas_3seg]))
        
        vecinos, indices_vecinos = self.encontrar_k_vecinos(
            X_train_3seg,
            x_test_3seg[0],
            self.df_3seg['palabra']
        )
        
        prediccion = self.predecir_clase(vecinos)
        print("\nVecinos encontrados [palabra]:")
        print(f"{vecinos} -> {prediccion}")
        
        self.visualizar_pca(X_train_3seg, self.df_3seg['palabra'], x_test_3seg[0], prediccion, indices_vecinos)
        
        # Segunda etapa: clasificación específica papa/camote usando audio completo
        if prediccion in ['papa', 'camote']:
            df_filtrado = self.df_1seg[self.df_1seg['palabra'].isin(['papa', 'camote'])]
            X_train_1seg = self.obtener_caracteristicas_completas(df_filtrado)
            x_test_1seg = self.obtener_caracteristicas_completas(pd.DataFrame([caracteristicas_1seg]))
            
            vecinos_2, indices_vecinos_2 = self.encontrar_k_vecinos(
                X_train_1seg,
                x_test_1seg[0],
                df_filtrado['palabra']
            )
            
            prediccion_final = self.predecir_clase(vecinos_2)
            print("\nVecinos segunda etapa [papa/camote]:")
            print(f"{vecinos_2} -> {prediccion_final}")
            
            self.visualizar_pca(
                X_train_1seg,
                df_filtrado['palabra'],
                x_test_1seg[0],
                prediccion_final,
                indices_vecinos_2,
                es_binario=True
            )
            return prediccion_final
        
        return prediccion
    
    # Pipeline completo de clasificación
    def clasificar_audio(self):
        try:
            caracteristicas_3seg, caracteristicas_1seg = self.procesar_nuevo_audio()
            return self.clasificar(caracteristicas_3seg, caracteristicas_1seg)
        except Exception as e:
            print(f"Error en clasificación: {str(e)}")
            return None

if __name__ == "__main__":
    try:
        clasificador = ClasificadorKNN(k=7)
        if resultado := clasificador.clasificar_audio():
            print(f"\nClasificación exitosa: {resultado}")
    except Exception as e:
        print(f"Error: {str(e)}")