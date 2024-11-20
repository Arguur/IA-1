# main.py
import os
from .clasificador import ClasificadorVerduras

#main antiguo usado antes de tener la app
def main():
    directorio_base = r"C:\Users\juanf\OneDrive\Escritorio\IA\datos\imagenes"
    
    print("Iniciando sistema de clasificación de verduras...")
    
    # Cargar dataset
    rutas_imagenes = []
    etiquetas = []
    for verdura in ['papa', 'berenjena', 'zanahoria', 'camote']:
        directorio_verdura = os.path.join(directorio_base, verdura)
        if os.path.exists(directorio_verdura):
            for imagen in os.listdir(directorio_verdura):
                if imagen.lower().endswith(('.png', '.jpg', '.jpeg')):
                    ruta_completa = os.path.join(directorio_verdura, imagen)
                    rutas_imagenes.append(ruta_completa)
                    etiquetas.append(verdura)
    
    if not rutas_imagenes:
        print("No se encontraron imágenes para procesar.")
        return
        
    print(f"\nSe encontraron {len(rutas_imagenes)} imágenes para entrenamiento")
    
    # Crear clasificador
    clasificador = ClasificadorVerduras()
    
    while True:
        print("\nOpciones:")
        print("1. Entrenar modelo")
        print("2. Clasificar una nueva imagen")
        print("3. Activar/Desactivar visualización en entrenamiento")
        print("4. Salir")
        
        opcion = input("\nSeleccione una opción (1-4): ")
        
        if opcion == "1":
            print("\nEntrenando modelo...")
            clasificador.entrenar(rutas_imagenes, etiquetas)
            
        elif opcion == "2":
            if not hasattr(clasificador.kmeans, 'centroides') or clasificador.kmeans.centroides is None:
                print("\nPrimero debe entrenar el modelo (opción 1)")
                continue
                
            ruta_imagen = input("\nIngrese la ruta completa de la imagen a clasificar: ").strip('"')
            
            if os.path.exists(ruta_imagen):
                prediccion = clasificador.predecir(ruta_imagen)
                if prediccion:
                    print(f"\nLa verdura en la imagen es: {prediccion}")
                else:
                    print("\nNo se pudo clasificar la imagen")
            else:
                print("\nLa ruta de la imagen no existe")
                
        elif opcion == "3":
            if clasificador.modo_debug:
                clasificador.desactivar_debug()
                print("\nVisualización en entrenamiento desactivada")
            else:
                clasificador.activar_debug()
                print("\nVisualización en entrenamiento activada")
                
        elif opcion == "4":
            print("\nSaliendo del programa...")
            break
        else:
            print("\nOpción no válida")

if __name__ == "__main__":
    main()