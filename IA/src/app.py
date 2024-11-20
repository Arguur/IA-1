import os
import sys
import threading
from pathlib import Path
import time

# Obtener la ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Agregar las rutas al path
sys.path.append(str(BASE_DIR / "src"))

# Importar las clases necesarias
from knn.clasificadorknn import ClasificadorKNN
from kmeans.clasificador import ClasificadorVerduras
from knn.grabador_audio import GrabadorAudio
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

class ClasificadorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Clasificación de Verduras")
        self.root.geometry("1200x900")  # Aumentamos un poco la altura
        
        # Definir rutas base
        self.base_dir = BASE_DIR
        self.datos_dir = self.base_dir / "datos"
        self.muestras_dir = self.datos_dir / "imagenes" / "muestra"
        
        # Variables de estado
        self.ultimo_resultado_audio = None
        self.ultima_imagen = None
        self.modelo_entrenado = False
        
        # Cargar imágenes de muestra
        self.imagenes_muestra = {}
        self.cargar_imagenes_muestra()
        
        # Crear la interfaz primero
        self.crear_interfaz()
        
        # Inicializar clasificadores
        self.clasificador_audio = ClasificadorKNN(k=7)
        self.clasificador_imagen = ClasificadorVerduras()
        
        # Entrenar clasificador de imágenes después de crear la interfaz
        self.entrenar_clasificador_imagen()

        # Agregar protocolo de cierre
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup_and_close)

    def cargar_imagenes_muestra(self):
        """Carga las imágenes de muestra en memoria"""
        for verdura in ['papa', 'berenjena', 'zanahoria', 'camote']:
            ruta_imagen = self.muestras_dir / f"{verdura}.jpg"
            if ruta_imagen.exists():
                imagen = Image.open(ruta_imagen)
                imagen = imagen.resize((200, 200), Image.Resampling.LANCZOS)
                self.imagenes_muestra[verdura] = ImageTk.PhotoImage(imagen)

    def cleanup_and_close(self):
        """Limpia recursos y cierra la aplicación"""
        # Detener cualquier proceso en curso
        try:
            if hasattr(self, 'clasificador_audio'):
                # Limpiar recursos del clasificador de audio si es necesario
                pass
            if hasattr(self, 'clasificador_imagen'):
                # Limpiar recursos del clasificador de imagen si es necesario
                pass
        except:
            pass
        finally:
            # Cerrar la ventana
            self.root.destroy()

    def entrenar_clasificador_imagen(self):
        """Entrena el clasificador de imágenes con el dataset"""
        try:
            directorio_imagenes = self.datos_dir / "imagenes"
            rutas_imagenes = []
            etiquetas = []
            
            # Lista de extensiones de imagen soportadas
            extensiones = ["*.jpg", "*.jpeg", "*.png"]
            
            # Verificar que el directorio existe y mostrar la ruta completa
            print(f"\nBuscando imágenes en: {directorio_imagenes}")
            if not directorio_imagenes.exists():
                raise Exception(f"El directorio de imágenes no existe: {directorio_imagenes}")
            
            # Listar todo el contenido del directorio
            print("\nContenido del directorio imagenes:")
            for item in directorio_imagenes.iterdir():
                print(f"- {item.name} ({'directorio' if item.is_dir() else 'archivo'})")
            
            # Contar imágenes por clase
            for verdura in ['papa', 'berenjena', 'zanahoria', 'camote']:
                directorio_verdura = directorio_imagenes / verdura
                print(f"\nRevisando directorio {verdura}:")
                
                if not directorio_verdura.exists():
                    print(f"  ¡Advertencia! No existe el directorio: {directorio_verdura}")
                    continue
                
                # Buscar imágenes con todas las extensiones soportadas
                imagenes = []
                for extension in extensiones:
                    imagenes.extend(list(directorio_verdura.glob(extension)))
                
                print(f"  Encontradas {len(imagenes)} imágenes")
                
                for imagen in imagenes:
                    print(f"  - {imagen.name}")
                    rutas_imagenes.append(str(imagen))
                    etiquetas.append(verdura)
            
            # Verificar que hay suficientes imágenes
            if not rutas_imagenes:
                raise Exception("No se encontraron imágenes para el entrenamiento")
                
            print(f"\nTotal de imágenes encontradas: {len(rutas_imagenes)}")
            print("Distribución por clase:")
            for verdura in set(etiquetas):
                cantidad = etiquetas.count(verdura)
                print(f"- {verdura}: {cantidad} imágenes")
                
            # Entrenar el modelo
            print("\nIniciando entrenamiento...")
            self.clasificador_imagen.entrenar(rutas_imagenes, etiquetas)
            print("Entrenamiento completado con éxito")
            self.modelo_entrenado = True
                
        except Exception as e:
            mensaje_error = f"Error al entrenar el clasificador de imágenes: {str(e)}"
            print(mensaje_error)
            messagebox.showerror("Error de Entrenamiento", mensaje_error)
            self.modelo_entrenado = False
            if hasattr(self, 'boton_clasificar'):
                self.boton_clasificar.config(state='disabled')
            if hasattr(self, 'resultado_imagen'):
                self.resultado_imagen.config(
                    text="Error: El clasificador no está entrenado",
                    foreground='red'
                )

        
    def crear_interfaz(self):
        # Frame principal con scroll
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configurar scroll
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Título
        titulo = ttk.Label(main_frame, 
                          text="Clasificador de Verduras por Voz e Imagen",
                          font=('Helvetica', 16, 'bold'))
        titulo.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Frame izquierdo para audio
        self.audio_frame = ttk.LabelFrame(main_frame, text="Clasificación por Voz", padding="10")
        self.audio_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        
        # Contenido del frame de audio
        self.estado_audio = ttk.Label(self.audio_frame, 
                                    text="Estado: Listo para grabar\n\n" +
                                    "Instrucciones:\n" +
                                    "1. Presione el botón para iniciar la grabación\n" +
                                    "2. Pronuncie el nombre de una verdura\n" +
                                    "3. Espere el resultado de la clasificación")
        self.estado_audio.grid(row=0, column=0, columnspan=2, pady=10)
        
        self.boton_grabar = ttk.Button(self.audio_frame, 
                                      text="Iniciar Grabación",
                                      command=self.manejar_grabacion)
        self.boton_grabar.grid(row=1, column=0, pady=5, padx=5)
        
        self.resultado_audio = ttk.Label(self.audio_frame, 
                                       text="Resultado: -",
                                       font=('Helvetica', 12))
        self.resultado_audio.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Label para mostrar la imagen de muestra del audio
        self.label_muestra_audio = ttk.Label(self.audio_frame)
        self.label_muestra_audio.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Frame derecho para imagen
        self.imagen_frame = ttk.LabelFrame(main_frame, text="Clasificación por Imagen", padding="10")
        self.imagen_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        
        # Instrucciones para imagen
        ttk.Label(self.imagen_frame, 
                 text="Instrucciones:\n" +
                 "1. Seleccione una imagen de una verdura\n" +
                 "2. Presione 'Clasificar Imagen'\n" +
                 "3. Observe el proceso de clasificación").grid(row=0, column=0, columnspan=2, pady=5)
        
        # Área de visualización de imagen
        self.label_imagen = ttk.Label(self.imagen_frame)
        self.label_imagen.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Label para mostrar la imagen de muestra
        self.label_muestra_imagen = ttk.Label(self.imagen_frame)
        self.label_muestra_imagen.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Botones para imagen
        frame_botones = ttk.Frame(self.imagen_frame)
        frame_botones.grid(row=3, column=0, columnspan=2, pady=5)
        
        self.boton_imagen = ttk.Button(frame_botones, 
                                     text="Seleccionar Imagen",
                                     command=self.seleccionar_imagen)
        self.boton_imagen.grid(row=0, column=0, padx=5)
        
        self.boton_clasificar = ttk.Button(frame_botones, 
                                         text="Clasificar Imagen",
                                         command=self.clasificar_imagen,
                                         state='disabled')
        self.boton_clasificar.grid(row=0, column=1, padx=5)
        
        self.resultado_imagen = ttk.Label(self.imagen_frame, 
                                        text="Resultado: -",
                                        font=('Helvetica', 12))
        self.resultado_imagen.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Configurar pesos de las columnas
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def mostrar_imagen_muestra(self, verdura, label_destino):
        """Muestra la imagen de muestra en el label especificado"""
        if verdura in self.imagenes_muestra:
            label_destino.configure(image=self.imagenes_muestra[verdura])
        else:
            label_destino.configure(image='')

    def manejar_grabacion(self):
        """Maneja el proceso de grabación y clasificación de audio"""
        try:
            self.boton_grabar.config(state='disabled')
            grabador = GrabadorAudio()
            
            def actualizar_estado(mensaje):
                """Actualiza el estado de forma segura desde cualquier hilo"""
                def update():
                    if isinstance(mensaje, int):
                        self.estado_audio.config(
                            text=f"Preparándose para grabar...\n\n{mensaje}",
                            font=('Helvetica', 24, 'bold')
                        )
                    elif mensaje == "¡Grabando!":
                        def secuencia_visual():
                            time.sleep(4.4)
                            self.root.after(0, lambda: self.estado_audio.config(
                                text="¡Grabando!",
                                font=('Helvetica', 24, 'bold')
                            ))
                        threading.Thread(target=secuencia_visual, daemon=True).start()
                        
                        self.estado_audio.config(
                            text="Preparando grabación...",
                            font=('Helvetica', 12)
                        )
                    else:
                        self.estado_audio.config(
                            text=mensaje,
                            font=('Helvetica', 12)
                        )
                
                if threading.current_thread() is threading.main_thread():
                    update()
                else:
                    self.root.after(0, update)
            
            def ejecutar_grabacion():
                try:
                    grabador.grabar_automatico(callback_cuenta_regresiva=actualizar_estado)
                    
                    def actualizar_ui():
                        self.estado_audio.config(text="Estado: Un poquito más...")
                    
                    self.root.after(0, actualizar_ui)
                    resultado = self.clasificador_audio.clasificar_audio()
                    
                    def mostrar_resultado():
                        if resultado:
                            self.ultimo_resultado_audio = resultado
                            self.resultado_audio.config(
                                text=f"Resultado: {resultado.capitalize()}")
                            self.mostrar_imagen_muestra(resultado, self.label_muestra_audio)
                            messagebox.showinfo("Éxito", 
                                            f"Verdura identificada: {resultado.capitalize()}")
                        else:
                            messagebox.showerror("Error", 
                                            "No se pudo clasificar el audio")
                        
                        # Restablecer estado
                        self.estado_audio.config(
                            text="Estado: Listo para grabar\n\n" +
                                "Instrucciones:\n" +
                                "1. Presione el botón para iniciar la grabación\n" +
                                "2. Pronuncie el nombre de una verdura\n" +
                                "3. Espere el resultado de la clasificación",
                            font=('Helvetica', 12)
                        )
                        self.boton_grabar.config(state='normal')
                    
                    self.root.after(0, mostrar_resultado)
                    
                except Exception as e:
                    def mostrar_error():
                        messagebox.showerror("Error", f"Error en la grabación: {str(e)}")
                        self.estado_audio.config(
                            text="Estado: Listo para grabar\n\n" +
                                "Instrucciones:\n" +
                                "1. Presione el botón para iniciar la grabación\n" +
                                "2. Pronuncie el nombre de una verdura\n" +
                                "3. Espere el resultado de la clasificación",
                            font=('Helvetica', 12)
                        )
                        self.boton_grabar.config(state='normal')
                    
                    self.root.after(0, mostrar_error)
            
            # Ejecutar la grabación en un thread separado
            thread_grabacion = threading.Thread(target=ejecutar_grabacion, daemon=True)
            thread_grabacion.start()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al iniciar la grabación: {str(e)}")
            self.estado_audio.config(
                text="Estado: Listo para grabar\n\n" +
                    "Instrucciones:\n" +
                    "1. Presione el botón para iniciar la grabación\n" +
                    "2. Pronuncie el nombre de una verdura\n" +
                    "3. Espere el resultado de la clasificación",
                font=('Helvetica', 12)
            )
            self.boton_grabar.config(state='normal')

    def mostrar_imagen_muestra(self, verdura, label_destino):
        """Muestra la imagen de muestra en el label especificado de forma segura"""
        def actualizar_imagen():
            if verdura in self.imagenes_muestra:
                label_destino.configure(image=self.imagenes_muestra[verdura])
            else:
                label_destino.configure(image='')
        
        if threading.current_thread() is threading.main_thread():
            actualizar_imagen()
        else:
            self.root.after(0, actualizar_imagen)
            
    def seleccionar_imagen(self):
        """Permite al usuario seleccionar una imagen para clasificar"""
        ruta = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG"),
                ("Todos los archivos", "*.*")
            ]
        )
        if ruta:
            try:
                # Cargar y mostrar la imagen
                imagen = Image.open(ruta)
                imagen = imagen.resize((300, 300), Image.Resampling.LANCZOS)
                foto = ImageTk.PhotoImage(imagen)
                self.label_imagen.config(image=foto)
                self.label_imagen.image = foto
                self.ultima_imagen = ruta
                self.boton_clasificar.config(state='normal')
            except Exception as e:
                messagebox.showerror("Error", 
                                f"Error al cargar la imagen: {str(e)}")
                
    def clasificar_imagen(self):
        """Clasifica la imagen seleccionada y renombra el archivo con la predicción"""
        if self.ultima_imagen is None:
            messagebox.showwarning("Advertencia", 
                                "Por favor seleccione una imagen primero")
            return
            
        try:
            prediccion = self.clasificador_imagen.predecir(self.ultima_imagen)
            
            if prediccion:
                self.resultado_imagen.config(
                    text=f"Resultado: {prediccion.capitalize()}")
                # Mostrar imagen de muestra
                self.mostrar_imagen_muestra(prediccion, self.label_muestra_imagen)
                
                # Renombrar el archivo con la predicción
                ruta_original = Path(self.ultima_imagen)
                nueva_ruta = ruta_original.parent / f"{prediccion}{ruta_original.suffix}"
                
                try:
                    # Si ya existe un archivo con ese nombre, agregar un número
                    contador = 1
                    while nueva_ruta.exists():
                        nueva_ruta = ruta_original.parent / f"{prediccion}_{contador}{ruta_original.suffix}"
                        contador += 1
                    
                    ruta_original.rename(nueva_ruta)
                    self.ultima_imagen = str(nueva_ruta)
                    messagebox.showinfo("Éxito", 
                                    f"Verdura identificada: {prediccion.capitalize()}\n"
                                    f"Archivo renombrado a: {nueva_ruta.name}")
                except Exception as e:
                    messagebox.showwarning("Advertencia",
                                        f"Se clasificó la imagen como {prediccion} pero "
                                        f"no se pudo renombrar el archivo: {str(e)}")
            else:
                messagebox.showerror("Error", 
                                "No se pudo clasificar la imagen")
                
        except Exception as e:
            messagebox.showerror("Error", 
                            f"Error al clasificar la imagen: {str(e)}")
        
def main():
    root = tk.Tk()
    app = ClasificadorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()