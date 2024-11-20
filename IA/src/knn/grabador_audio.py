import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from pathlib import Path
import librosa
import queue
import time
import sys
import threading

class GrabadorAudio:
    def __init__(self):
        self.frecuencia_grabacion = 44100  # Mantenemos frecuencia consistente
        self.frecuencia_objetivo = 44100
        self.directorio_muestras = Path(__file__).parent.parent.parent / 'datos' / 'muestras_audio'
        self.directorio_muestras.mkdir(parents=True, exist_ok=True)
        
        self.blocksize = 4096
        self.channels = 1
        self.latency = 'low'
        
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.duracion_grabacion = 2
        
        # Pre-configurar el dispositivo
        self.configurar_dispositivo()

    def configurar_dispositivo(self):
        """Configura el dispositivo de audio"""
        try:
            device_info = sd.query_devices(kind='input')
            sd.default.device = device_info['index']
            sd.default.channels = self.channels
            sd.default.dtype = 'float32'
            sd.default.latency = self.latency
            sd.default.blocksize = self.blocksize
        except Exception as e:
            print(f"Error en configuración: {e}")
            raise

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def grabar_automatico(self, callback_cuenta_regresiva=None):
        """Maneja el proceso completo de grabación"""
        try:
            # Limpiar la cola
            while not self.audio_queue.empty():
                self.audio_queue.get()
            
            # Preparar el stream antes de la cuenta regresiva
            stream = sd.InputStream(
                samplerate=self.frecuencia_grabacion,
                channels=self.channels,
                blocksize=self.blocksize,
                callback=self.callback
            )
            
            with stream:
                # Cuenta regresiva
                for i in range(3, 0, -1):
                    if callback_cuenta_regresiva:
                        callback_cuenta_regresiva(i)
                    time.sleep(0.8)
                
                # Pequeña pausa para sincronización
                time.sleep(0.2)
                
                # Iniciar grabación
                if callback_cuenta_regresiva:
                    callback_cuenta_regresiva("¡Grabando!")
                    
                self.is_recording = True
                time.sleep(self.duracion_grabacion)  # Grabar exactamente por la duración especificada
                self.is_recording = False
            
            if callback_cuenta_regresiva:
                callback_cuenta_regresiva("Procesando...")
            
            # Procesar audio
            audio_frames = []
            while not self.audio_queue.empty():
                audio_frames.append(self.audio_queue.get())
            
            if not audio_frames or len(audio_frames) == 0:
                raise Exception("No se grabó audio")
            
            audio_buffer = np.concatenate(audio_frames)
            audio_buffer = audio_buffer.flatten()
            
            # Ya no necesitamos resamplear
            audio_procesado = audio_buffer
            
            # Aplicar normalización y fade
            fade_length = int(0.05 * self.frecuencia_objetivo)
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)
            
            audio_procesado[:fade_length] *= fade_in
            audio_procesado[-fade_length:] *= fade_out
            
            max_val = np.max(np.abs(audio_procesado))
            if max_val > 0:
                audio_procesado = audio_procesado / max_val * 0.9
            
            ruta_archivo = self.directorio_muestras / "temp_muestra.wav"
            wavfile.write(
                str(ruta_archivo), 
                self.frecuencia_objetivo,
                (audio_procesado * np.iinfo(np.int16).max).astype(np.int16)
            )
            
            return audio_procesado
            
        except Exception as e:
            print(f"Error en grabación: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self.is_recording = False

    def grabar_y_guardar(self, nombre_archivo="temp_muestra.wav"):
        """Graba y guarda el audio"""
        try:
            audio_grabado = self.grabar_automatico()
            if audio_grabado is not None:
                ruta_archivo = self.directorio_muestras / nombre_archivo
                wavfile.write(
                    str(ruta_archivo),
                    self.frecuencia_objetivo,
                    (audio_grabado * np.iinfo(np.int16).max).astype(np.int16)
                )
                return str(ruta_archivo)
            raise Exception("No se pudo grabar el audio")
        except Exception as e:
            print(f"Error en grabar_y_guardar: {e}")
            raise