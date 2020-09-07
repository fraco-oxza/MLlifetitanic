def efex():
    if __name__ == "__main__":
        import NNetwork
    else:
        try:
            from lib import NNetwork
        except:
            import NNetwork
    import numpy as np
    import pandas as pd

    test_set = np.matrix(pd.read_csv("data/Sets/test_set.csv"))   #En el train_set se almacenan todos los ejemplos que le dar
    Theta_1 = np.matrix(pd.read_csv("data/Weights/Theta_1.csv"))    #Dentro de estos 3 archivos se encuentra cada operacion que se debe
    Theta_2 = np.matrix(pd.read_csv("data/Weights/Theta_2.csv"))    #hacer para tener una red neuronal con los tamaños
    Theta_3 = np.matrix(pd.read_csv("data/Weights/Theta_3.csv"))    #indicados anteriormente
    X = test_set[:,0:7]    #Creamos X con todos los ejemplos de entrenamiento
    m = np.size(X,0)    #Obtenemos la cantidad de ejemplos y la guardamos en m
    X = np.concatenate((np.ones((m,1)), X), axis=1) #agregamos la bios unity
    yR = np.array(test_set[:,7:8])  #Creamos y con todas las respuesta correctas para x


    y = NNetwork.forward_propagation(X, Theta_1, Theta_2, Theta_3)

    for i in range(len(y)):
        if y[i] <= 0.5:
            y[i] = 0
        else:
            y[i] = 1

    Tpositive = 0 
    Fnegative = 0
    Fpositive = 0
    for i in range(len(y)):
        if y[i] == yR[i]:
            if y[i] == 1:
                Tpositive += 1
        if yR[i] == 1:
            if y[i] != 1:
                Fnegative += 1
        else:
            if y[i] == 1:
                Fpositive += 1

    presision = Tpositive/(Tpositive + Fnegative)
    recall = Tpositive/(Tpositive + Fpositive)
    Efectividad = ((presision+recall)/2)*100
    print("Efectividad: ",2 * ((presision*recall)/(presision + recall)))


    test_set = pd.read_csv("data/Sets/test_set.csv")   #En el train_set se almacenan todos los ejemplos que le daremos a la red
    for i in range(len(test_set["Survived"])):
        test_set["Survived"][i] = y[i]

    test_set.to_csv("data/Sets/predict.csv", index=False)
    print("Todo salio bien")

    return Efectividad

if __name__ == "__main__":
    print(efex())

