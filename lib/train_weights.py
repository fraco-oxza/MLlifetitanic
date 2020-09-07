efect_v = 0
while True: 
    """
    A partir de aqui se entrenara Theta_1,2,3 respectivamente
    para poder utilizarlo posterirmente en una red neuronal
    con 4 capas, 1 de entrada con 7 nodos(8 mas bios), 2 
    ocultas con 11 y 4 nodos(12 y 5 mas bios), y 1 de salida con 1 nodo.
    """

    #================================================
    """
    En esta parte llamaremos a todas las librerias python que
    necesitaremos para la red
    """
    import numpy as np  #Numpy lo utilizaremos para calculos con matrices y vectores
    import pandas as pd #Pandas lo utilizaremos para abrir los datos y posterirmente guardarlos
    import NNetwork as NN   #NNetwork lo utilizaremos para ejcutar algunas lineas de codigo personalisada
    from scipy.optimize import minimize #Minimize nos permitira reducir theta hasta el mejor valor
    from time import sleep
    #================================================            
    """
    En esta parte abrimeremos todos los archivos que necesitamos
    para nuestra red neuronal 
    """
    Theta_1 = np.matrix(pd.read_csv("data/Weights/Theta_1.csv"))    #Dentro de estos 3 archivos se encuentra cada operacion que se debe
    Theta_2 = np.matrix(pd.read_csv("data/Weights/Theta_2.csv"))    #hacer para tener una red neuronal con los tamaños
    train_set = np.matrix(pd.read_csv("data/Sets/train_set.csv"))   #En el train_set se almacenan todos los ejemplos que le daremos a la red
    Theta_3 = np.matrix(pd.read_csv("data/Weights/Theta_3.csv"))    #indicados anteriormente
    #================================================
    """
    En este espacio se iniciaran todas las variables que utilizara el programa
    en su ejecucion
    """
    Theta = NN.flatten(Theta_1,Theta_2,Theta_3) #Usando esta funcion podemos "Aplanar" muy rapido Theta y dejarlo listo para usar
    epsilon = 1 #Esta variable define el rango en el que se encontrara theta
    Theta = NN.random_theta(Theta, epsilon)  #inicializamos theta con valores aleatorios
    lambda_ = 0 #Con esta variable podremos regularizar
    X = train_set[:,0:7]    #Creamos X con todos los ejemplos de entrenamiento
    y = np.array(train_set[:,7:8])  #Creamos y con todas las respuesta correctas para x
    #================================================
    """
    Ahora ejecutaremos el descenso de gradiente hasta encontrar
    el minimo para theta
    """
    def short_cost_function(x0):
        return NN.cost_function(X,y,x0,lambda_)
    def short_gradient(x0):
        return NN.background_propagation(X,y,x0, lambda_)
    res = minimize(short_cost_function, Theta, method="BFGS", jac=short_gradient, options={"disp":1})
    Theta = res.x

    #================================================
    """
    Ahora almacenamos los valores que acaba de aprender
    nuestra red en theta
    """
    Theta1, Theta2, Theta3 = NN.inflate(Theta)    #Le damos nueva forma Theta

    efect_a = NN.efex(Theta1,Theta2,Theta3)
    print("Efectividad:  ",efect_a)
    if efect_a > efect_v:
        efect_v = efect_a
        Theta_1 = pd.core.frame.DataFrame(Theta1)  #Transformamos a un tipo
        Theta_2 = pd.core.frame.DataFrame(Theta2)  #De dato exportable
        Theta_3 = pd.core.frame.DataFrame(Theta3)

        Theta_1.to_csv("data/Weights/Theta_1.csv", index=False)
        Theta_2.to_csv("data/Weights/Theta_2.csv", index=False)
        Theta_3.to_csv("data/Weights/Theta_3.csv", index=False)

        print("Nuevos valores para theta exportados de manera exitosa")

    else:
        print("Nuevos valores no almacenados")

