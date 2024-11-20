import numpy as np
import librosa
from scipy.signal import hilbert
import pandas as pd
from pathlib import Path
import pickle
import re
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Extrae y analiza características espectrales y temporales del audio
class ExtractorCaracteristicas:
    def __init__(self, freq_muestreo=16000, num_segmentos=8):
        self.freq_muestreo = freq_muestreo
        self.num_segmentos = num_segmentos
        self.caracteristicas = {}
        
    # Calcula tasa de cruces por cero para detección de transiciones
    def calcular_zcr(self, audio):
        try:
            audio_limpio = np.nan_to_num(audio, 0)
            if np.all(audio_limpio == 0):
                return {'zcr_media': 0, 'zcr_max': 0, 'zcr_std': 0}
            
            zcr = librosa.feature.zero_crossing_rate(audio_limpio)[0]
            return {
                'zcr_media': float(np.mean(zcr)),
                'zcr_max': float(np.max(zcr)),
                'zcr_std': float(np.std(zcr))
            }
        except Exception as e:
            print(f"Error en ZCR: {str(e)}")
            return {'zcr_media': 0, 'zcr_max': 0, 'zcr_std': 0}

    # Analiza amplitud y envolvente de la señal
    def calcular_amplitud(self, audio):
        try:
            audio_limpio = np.nan_to_num(audio, 0)
            if np.all(audio_limpio == 0):
                return {
                    'amplitud_media': 0, 'amplitud_max': 0, 'amplitud_std': 0,
                    'envolvente_media': 0, 'envolvente_max': 0
                }
            
            try:
                envolvente = np.abs(hilbert(audio_limpio))
            except:
                envolvente = np.abs(audio_limpio)
            
            return {
                'amplitud_media': float(np.mean(np.abs(audio_limpio))),
                'amplitud_max': float(np.max(np.abs(audio_limpio))),
                'amplitud_std': float(np.std(np.abs(audio_limpio))),
                'envolvente_media': float(np.mean(envolvente)),
                'envolvente_max': float(np.max(envolvente))
            }
        except Exception as e:
            print(f"Error en amplitud: {str(e)}")
            return {
                'amplitud_media': 0, 'amplitud_max': 0, 'amplitud_std': 0,
                'envolvente_media': 0, 'envolvente_max': 0
            }
        
    # Divide el audio en segmentos para análisis localizado
    def segmentar_audio(self, audio):
        longitud_segmento = len(audio) // self.num_segmentos
        return [audio[i * longitud_segmento:(i + 1) * longitud_segmento] 
                for i in range(self.num_segmentos)]

    # Calcula derivadas de características para capturar cambios temporales
    def calcular_delta(self, data, width=3, order=1):
        try:
            return librosa.feature.delta(data, width=width, order=order)
        except librosa.util.exceptions.ParameterError:
            if width > 3:
                return self.calcular_delta(data, width=width-2, order=order)
            return np.zeros_like(data)
        except:
            return np.zeros_like(data)

    # Extrae conjunto completo de características espectrales y temporales
    def extraer_caracteristicas_segmento(self, audio, prefijo=''):
        try:
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
                
            audio = np.nan_to_num(audio, 0)
            max_abs = np.max(np.abs(audio))
            if max_abs > 0:
                audio = audio / max_abs
            else:
                return None
            
            caract = {}
            
            # Extrae MFCCs y sus derivadas
            mfccs = librosa.feature.mfcc(y=audio, sr=self.freq_muestreo, n_mfcc=13)
            mfcc_delta = self.calcular_delta(mfccs, order=1)
            mfcc_delta2 = self.calcular_delta(mfccs, order=2)
            
            for i in range(13):
                caract.update({
                    f'{prefijo}mfcc{i}_media': float(np.mean(mfccs[i])),
                    f'{prefijo}mfcc{i}_std': float(np.std(mfccs[i])),
                    f'{prefijo}mfcc_delta{i}_media': float(np.mean(mfcc_delta[i])),
                    f'{prefijo}mfcc_delta{i}_std': float(np.std(mfcc_delta[i])),
                    f'{prefijo}mfcc_delta2{i}_media': float(np.mean(mfcc_delta2[i])),
                    f'{prefijo}mfcc_delta2{i}_std': float(np.std(mfcc_delta2[i]))
                })
            
            # Calcula características espectrales adicionales
            espectral = {
                'centroide': librosa.feature.spectral_centroid(y=audio, sr=self.freq_muestreo)[0],
                'ancho_banda': librosa.feature.spectral_bandwidth(y=audio, sr=self.freq_muestreo)[0],
                'rolloff': librosa.feature.spectral_rolloff(y=audio, sr=self.freq_muestreo)[0]
            }
            
            for nombre, valor in espectral.items():
                caract.update({
                    f'{prefijo}{nombre}_media': float(np.mean(valor)),
                    f'{prefijo}{nombre}_std': float(np.std(valor))
                })
            
            caract.update({f'{prefijo}{k}': v for k, v in self.calcular_zcr(audio).items()})
            caract.update({f'{prefijo}{k}': v for k, v in self.calcular_amplitud(audio).items()})
            
            return caract
                
        except Exception as e:
            print(f"Error en extracción: {str(e)}")
            return None

    # Procesa y extrae características de todo el audio
    def extraer_todas_caracteristicas(self, audio, metadata=None):
        try:
            # Preprocesamiento del audio
            audio = audio - np.mean(audio)
            audio = (audio - np.mean(audio)) / np.std(audio)
            audio = librosa.effects.preemphasis(audio, coef=0.95)
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            self.caracteristicas = metadata or {}
            
            for i, segmento in enumerate(self.segmentar_audio(audio)):
                caracteristicas_segmento = self.extraer_caracteristicas_segmento(segmento, f"seg{i}_")
                if caracteristicas_segmento:
                    self.caracteristicas.update(caracteristicas_segmento)
                else:
                    self.caracteristicas[f'seg{i}_error'] = 1
                
            return self.caracteristicas
            
        except Exception as e:
            print(f"Error en extracción: {str(e)}")
            self.caracteristicas['error'] = 1
            return self.caracteristicas

# Gestiona el proceso de extracción y evaluación de características
class GestorCaracteristicas:
    def __init__(self, num_segmentos=4):
        self.directorio_entrada = Path(r"C:\Users\juanf\OneDrive\Escritorio\IA\datos\audio_procesado")
        self.directorio_caracteristicas = Path(r"C:\Users\juanf\OneDrive\Escritorio\IA\datos\caracteristicas")
        self.directorio_caracteristicas.mkdir(parents=True, exist_ok=True)
        self.num_segmentos = num_segmentos
        
    # Extrae metadatos del nombre del archivo
    def extraer_informacion_nombre(self, nombre_archivo):
        patron = r"proc_persona(\d+)_(\w+)_muestra(\d+)"
        coincidencia = re.match(patron, nombre_archivo)
        return {
            'num_persona': int(coincidencia.group(1)),
            'palabra': coincidencia.group(2),
            'num_muestra': int(coincidencia.group(3))
        } if coincidencia else None

    # Procesa todos los archivos de audio del directorio
    def procesar_directorio(self):
        archivos = list(self.directorio_entrada.glob("*.wav"))
        todas_caracteristicas = []
        
        for i, archivo in enumerate(archivos, 1):
            print(f"Procesando archivo {i}/{len(archivos)}: {archivo.name}")
            if info := self.extraer_informacion_nombre(archivo.stem):
                try:
                    audio, sr = librosa.load(archivo, sr=16000)
                    extractor = ExtractorCaracteristicas(sr, self.num_segmentos)
                    if caracteristicas := extractor.extraer_todas_caracteristicas(audio, info):
                        if 'error' not in caracteristicas:
                            todas_caracteristicas.append(caracteristicas)
                except Exception as e:
                    print(f"Error en {archivo.name}: {str(e)}")
        
        if not todas_caracteristicas:
            raise ValueError("No se pudieron extraer características")
            
        df = pd.DataFrame(todas_caracteristicas)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        nombre_archivo = f"caracteristicas_{self.num_segmentos}seg_{timestamp}"
        
        df.to_csv(self.directorio_caracteristicas / f"{nombre_archivo}.csv", index=False)
        with open(self.directorio_caracteristicas / f"{nombre_archivo}.pkl", 'wb') as f:
            pickle.dump(df, f)
            
        return df
    
    # Evalúa la calidad de las características extraídas usando KNN
    def evaluar_segmentacion(self, df):
        X = df.drop(['num_persona', 'palabra', 'num_muestra'], axis=1)
        y = df['palabra']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }

# Compara diferentes configuraciones de segmentación
def comparar_segmentaciones(segmentaciones=[4, 5, 6, 7, 8]):
    resultados = {}
    for num_seg in segmentaciones:
        gestor = GestorCaracteristicas(num_segmentos=num_seg)
        df = gestor.procesar_directorio()
        resultados[num_seg] = gestor.evaluar_segmentacion(df)
    
    mejor_seg = max(resultados.keys(), key=lambda k: resultados[k]['accuracy'])
    return resultados, mejor_seg

if __name__ == "__main__":
    try:
        resultados, mejor_seg = comparar_segmentaciones(range(1, 2))
        print(f"\nMejor configuración: {mejor_seg} segmentos")
        print(f"Accuracy: {resultados[mejor_seg]['accuracy']:.4f}")
        print("\nMatriz de confusión:")
        print(resultados[mejor_seg]['confusion_matrix'])
        print("\nReporte de clasificación:")
        print(resultados[mejor_seg]['classification_report'])
    except Exception as e:
        print(f"\nError: {e}")