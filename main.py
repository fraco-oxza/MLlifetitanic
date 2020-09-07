import pandas as pd
import numpy as np
#Creamos un menu para seleccionar lo que la persona quiera hacer
print("Bienvenido, seleccione lo que desee:\n")
print(" 1. Reentrenar la red neuronal")
print(" 2. Introducir datos para predecir")
print(" 0. salir")
opcion = input("\nIntrodusca un numero: ")

if opcion == "1":
    from lib import train_weights
elif opcion == "2":
    from lib import predict
elif opcion == "0":
    exit()
else:
     print("\nPor favor seleccione una opcion valida\n")

#Prueba