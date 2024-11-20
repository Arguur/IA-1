import numpy as np

# Normaliza características al rango [0,1] usando min-max scaling
class Normalizador:
    def __init__(self):
        self.parametros = {}
        
    def ajustar(self, X, nombres_caracteristicas):
        # Almacena valores mínimos y máximos de cada característica
        for i, nombre in enumerate(nombres_caracteristicas):
            self.parametros[nombre] = {
                'min': np.min(X[:, i]),
                'max': np.max(X[:, i])
            }
            
    def normalizar(self, X, nombres_caracteristicas):
        if not self.parametros:
            self.ajustar(X, nombres_caracteristicas)
            
        X_norm = np.zeros_like(X, dtype=np.float64)
        
        # Aplica normalización min-max por característica
        for i, nombre in enumerate(nombres_caracteristicas):
            params = self.parametros[nombre]
            denominador = (params['max'] - params['min'])
            if denominador != 0:
                X_norm[:, i] = (X[:, i] - params['min']) / denominador
            else:
                X_norm[:, i] = 0
                
        return X_norm