#crear_base_datos.py
from grabador_audio import GrabadorAudio
import os
from time import sleep
from pathlib import Path

def cuenta_regresiva(segundos=3, mensaje_preparacion="Prepárese para hablar en:"):
    print(f"\n{mensaje_preparacion}")
    for i in range(segundos, 0, -1):
        print(f"{i}...")
        sleep(1)
    print("¡Ya!")

def crear_base_datos_audio():
    # Creación de múltiples muestras sin tiempo de recorte
    grabador = GrabadorAudio(duracion=2)  # Removido tiempo_recorte_inicio
    palabras = ["papa", "berenjena", "zanahoria", "camote"]
    
    # Solicitar número de persona
    while True:
        try:
            numero_persona = int(input("Ingrese el número de persona (1-10): "))
            if 1 <= numero_persona <= 10:
                break
            print("Por favor, ingrese un número entre 1 y 5.")
        except ValueError:
            print("Por favor, ingrese un número válido.")
    
    print("\n=== INSTRUCCIONES ===")
    print("1. Hable claro y a una distancia constante del micrófono (aprox. 20 cm)")
    print("2. Pronuncie la palabra cuando vea '¡Ya!'")
    print("3. Se grabarán 6 muestras de cada palabra")
    print("4. Dispone de 2 segundos para pronunciar cada palabra")  # Actualizado al quitar recorte
    print("5. Espere la cuenta regresiva antes de cada grabación")
    print("\nIMPORTANTE:")
    print("- Varíe ligeramente el tono y la velocidad entre muestras")
    print("- Mantenga silencio durante la cuenta regresiva")
    print("- Evite ruidos externos durante la grabación")
    
    input("\nPresione Enter cuando esté listo para comenzar...")
    

    
    # Crear directorio si no existe
    directorio_proyecto = Path(__file__).parent.parent.parent
    directorio_muestras = directorio_proyecto / 'datos' / 'muestras_audio'
    directorio_muestras.mkdir(parents=True, exist_ok=True)
    
    total_grabaciones = len(palabras) * 6
    grabaciones_completadas = 0
    
    for palabra in palabras:
        print(f"\n=== Grabando muestras para '{palabra}' ===")
        for i in range(6):  # 6 muestras por palabra
            grabaciones_completadas += 1
            print(f"\nMuestra {i+1}/6 de '{palabra}' (Progreso total: {grabaciones_completadas}/{total_grabaciones})")
            
            while True:
                input("Presione Enter cuando esté listo...")
                # Añadir pequeña pausa para asegurar silencio inicial
                sleep(0.5)
                cuenta_regresiva(3, f"Prepárese para decir '{palabra}' en:")
                
                try:
                    nombre_archivo = f"persona{numero_persona}_{palabra}_muestra{i+1}"
                    datos_audio = grabador.grabar_y_guardar(nombre_archivo)
                    
                    # Verificar calidad del audio
                    if hasattr(grabador, 'verificar_calidad_audio'):
                        calidad_ok, mensaje = grabador.verificar_calidad_audio(datos_audio)
                        if not calidad_ok:
                            print(f"\nProblema con la grabación: {mensaje}")
                            print("Volviendo a grabar esta muestra...")
                            continue
                    
                    # Preguntar si la grabación fue correcta
                    respuesta = input("\n¿La grabación fue correcta? (s/n): ").lower()
                    if respuesta == 's':
                        break
                    else:
                        print("Volviendo a grabar esta muestra...")
                        continue
                        
                except Exception as e:
                    print(f"\nError durante la grabación: {e}")
                    print("Intentando de nuevo...")
            
            sleep(0.5)

if __name__ == "__main__":
    print("=== CREACIÓN DE BASE DE DATOS DE AUDIO ===")
    print("Este programa grabará muestras de voz para el sistema de reconocimiento")
    
    try:
        crear_base_datos_audio()
        print("\n¡Base de datos creada exitosamente!")
        directorio_proyecto = Path(__file__).parent.parent.parent
        directorio_muestras = directorio_proyecto / 'datos' / 'muestras_audio'
        print(f"Las grabaciones se guardaron en: {directorio_muestras.absolute()}")
    except Exception as e:
        print(f"\nError durante la grabación: {e}")
        print("Por favor, verifique que su micrófono esté conectado y funcionando correctamente.")