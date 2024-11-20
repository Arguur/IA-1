import os

# Ruta de la carpeta que contiene las imágenes
carpeta = r"C:\Users\juanf\OneDrive\Escritorio\IA\datos\imagenes\berenjena"

# Listar los archivos en la carpeta
archivos = os.listdir(carpeta)

# Filtrar solo los archivos de imagen que deseas renombrar
imagenes = [f for f in archivos if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# Renombrar cada imagen con el formato berenjena1, berenjena2, etc.
for i, nombre_original in enumerate(imagenes, start=1):
    # Crear el nuevo nombre con el formato berenjenaX.ext
    extension = os.path.splitext(nombre_original)[1]  # Obtener la extensión del archivo
    nuevo_nombre = f"berenjena{i}{extension}"
    
    # Ruta completa de los archivos
    ruta_original = os.path.join(carpeta, nombre_original)
    ruta_nueva = os.path.join(carpeta, nuevo_nombre)
    
    # Renombrar el archivo
    os.rename(ruta_original, ruta_nueva)

print("Renombrado completado con éxito.")
