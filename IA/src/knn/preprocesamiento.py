import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import librosa
from pathlib import Path
import warnings
import soundfile as sf

# Procesamiento y normalización de señales de audio
class ProcesadorAudio:
    def __init__(self):
        self.freq_muestreo_objetivo = 44100
        self.audio = None
        self.freq_muestreo_original = None

    # Carga y normaliza audio a valores entre -1 y 1
    def cargar_audio(self, ruta_archivo):
        try:
            self.freq_muestreo_original, audio = wavfile.read(str(ruta_archivo))
            audio = audio.astype(np.float32)
            if audio.dtype == np.int16:
                audio = audio / 32768.0
            elif audio.dtype == np.int32:
                audio = audio / 2147483648.0
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            self.audio = audio
        except Exception as e:
            raise RuntimeError(f"Error cargando audio {ruta_archivo}: {e}")

    # Ajusta frecuencia de muestreo al valor objetivo
    def resamplear(self):
        if self.freq_muestreo_original != self.freq_muestreo_objetivo:
            self.audio = librosa.resample(
                y=self.audio,
                orig_sr=self.freq_muestreo_original,
                target_sr=self.freq_muestreo_objetivo,
                res_type='kaiser_fast'
            )
            self.freq_muestreo_original = self.freq_muestreo_objetivo

    # Aplica filtro para preservar frecuencias relevantes para voz
    def aplicar_filtro_pasa_banda(self, freq_baja=80, freq_alta=6500):
        nyquist = self.freq_muestreo_objetivo * 0.5
        orden = 4
        frecuencias = [freq_baja/nyquist, freq_alta/nyquist]
        b, a = butter(orden, frecuencias, btype='band')
        self.audio = filtfilt(b, a, self.audio, padtype='odd', padlen=None)

    # Extrae segmento principal basado en energía de la señal
    def eliminar_silencios(self, umbral_rms=0.005, longitud_ventana=1024):
        if len(self.audio) == 0:
            return
        energia = librosa.feature.rms(
            y=self.audio,
            frame_length=longitud_ventana,
            hop_length=longitud_ventana//4
        )[0]
        umbral = max(umbral_rms, np.mean(energia) * 0.1)
        segmentos = librosa.effects.split(
            self.audio,
            top_db=55,
            frame_length=longitud_ventana,
            hop_length=longitud_ventana//4
        )
        if len(segmentos) == 0:
            return
        duraciones = [end-start for start, end in segmentos]
        idx_max = np.argmax(duraciones)
        start, end = segmentos[idx_max]
        self.audio = self.audio[start:end]

    # Normaliza amplitud a rango estándar
    def normalizar_audio(self):
        if len(self.audio) == 0:
            return
        max_abs = np.max(np.abs(self.audio))
        if max_abs > 0:
            self.audio = self.audio / max_abs * 0.9

    # Pipeline completo de procesamiento
    def procesar(self, ruta_entrada, ruta_salida):
        try:
            self.cargar_audio(ruta_entrada)
            self.resamplear()
            self.aplicar_filtro_pasa_banda()
            self.eliminar_silencios()
            self.normalizar_audio()
            wavfile.write(
                str(ruta_salida),
                self.freq_muestreo_objetivo,
                (self.audio * 32767).astype(np.int16)
            )
            return True
        except Exception as e:
            warnings.warn(f"Error procesando {ruta_entrada}: {e}")
            return False

# Procesa audio con preprocesamiento específico para extracción de características
def procesar_audio(archivo_entrada, directorio_salida):
   try:
       audio, _ = librosa.load(archivo_entrada, sr=16000)
       audio = (audio - np.mean(audio)) / np.std(audio) 
       audio = librosa.effects.preemphasis(audio, coef=0.95)
       audio, _ = librosa.effects.trim(audio, top_db=20)
       nombre_archivo = Path(archivo_entrada).stem
       archivo_salida = directorio_salida / f"proc_{nombre_archivo}.wav"
       sf.write(archivo_salida, audio, 16000)
       return True, archivo_salida
   except Exception as e:
       print(f"Error procesando {archivo_entrada}: {str(e)}")
       return False, None

# Procesa todos los archivos de audio en el directorio
def procesar_directorio():
    directorio_entrada = Path(r"C:\Users\juanf\OneDrive\Escritorio\IA\datos\muestras_audio")
    directorio_salida = Path(r"C:\Users\juanf\OneDrive\Escritorio\IA\datos\audio_procesado")
    directorio_salida.mkdir(parents=True, exist_ok=True)
    
    archivos = []
    for extension in ['*.wav']:
        archivos.extend(directorio_entrada.glob(extension))
    
    print(f"Encontrados {len(archivos)} archivos de audio")
    
    procesados = 0
    errores = 0
    
    for archivo in archivos:
        print(f"\nProcesando {archivo.name}...")
        exito, archivo_salida = procesar_audio(archivo, directorio_salida)
        
        if exito:
            procesados += 1
            print(f"✓ Guardado como: {archivo_salida.name}")
        else:
            errores += 1
            print(f"✗ Error procesando {archivo.name}")
    
    print(f"\nProcesamiento completado:")
    print(f"- Archivos procesados: {procesados}")
    print(f"- Errores: {errores}")

if __name__ == "__main__":
    procesar_directorio()