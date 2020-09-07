"""
Esta libreria contendra componentes necesarios
para la ejecucion del Machine Learning con una
red neuronal para el caso del programa "Vive o
muere". NO MULTIPROPOCITOS!!!!!!!!!!!!!!!!!!!!
"""

#================================================
"""
En esta parte llamaremos a todas las librerias python que
necesitaremos para la ejecucion de este archivo
"""
import numpy as np  #Numpy lo utilizaremos para calculos con matrices y vectores
import pandas as pd #Pandas lo utilizaremos para abrir los datos y posterirmente guardarlos

#================================================
"""
Con estas funciones convertiremos theta en matriz
o vector
"""
def flatten(Theta_1, Theta_2, Theta_3):    #Aplanadora de Theta
    try:
        Theta_1 = Theta_1.flatten() #Lo que hacemos aqui es "aplanar" cada
        Theta_2 = Theta_2.flatten() #matriz para formar Arrays
        Theta_3 = Theta_3.flatten() 
        Theta = np.array(np.concatenate((Theta_1, Theta_2, Theta_3), axis=1))   #Y por ultimo unimos todo theta en un gran array
        return Theta    #Devolvemos un array con todo theta
    except:
        Theta_1 = Theta_1.flatten() #Lo que hacemos aqui es "aplanar" cada
        Theta_2 = Theta_2.flatten() #matriz para formar Arrays
        Theta_3 = Theta_3.flatten() 
        Theta = np.array(np.concatenate((Theta_1, Theta_2, Theta_3), axis=0))   #Y por ultimo unimos todo theta en un gran array
        return Theta    #Devolvemos un array con todo theta

def inflate(Theta): #Infladora de Theta
    try:
        Theta_1 = np.reshape(Theta[0,0:88], (11,8))     #Tomamos secciones de Theta y las
        Theta_2 = np.reshape(Theta[0,88:136], (4,12))   #Convertimos en matrices con las
        Theta_3 = np.reshape(Theta[0,136:141], (1,5))   #dimenciones preseleccionadas
        return Theta_1, Theta_2, Theta_3    #devolvemos los 3 elementos
    except:
        Theta_1 = np.reshape(Theta[0:88], (11,8))     #Tomamos secciones de Theta y las
        Theta_2 = np.reshape(Theta[88:136], (4,12))   #Convertimos en matrices con las
        Theta_3 = np.reshape(Theta[136:141], (1,5))   #dimenciones preseleccionadas
        return Theta_1, Theta_2, Theta_3    #devolvemos los 3 elementos


#=================================================
"""
Aqui estan las formulas necesarias para los calculos
que se tengan que utilizar multiples veces
"""
def sigmoid(z): #Funcion sigmidea, que solo entrega valores entre 0 y 1
    g = 1 / (1 + np.exp(-z))
    return g

def grad_sigmoid(z): #Funcion sigmoidea modificada para entregar la derivada
    gs =  np.multiply(sigmoid(z), (1 - sigmoid(z)))  
    return gs
#=================================================
"""
Con esta funcion iniciaremos theta en valores alea-
torios para que la red neuronal funcione de la mejor
manera posible
"""
def random_theta(theta,epsilon):    #Creadora de inteligencia
    row, column = theta.shape   #Tomamos las dimenciones de theta
    rand_theta = np.random.rand(row,column) * (2*epsilon) - epsilon #Creamos una matriz del mismo tamaño
    return rand_theta   #y devolvemos numeros al azar

#=================================================
"""
Aqui esta la funcion de costo la cual nos puede decir
que tan bien ajustada esta nuestra hipotesis en cuanto 
un conjunto de datos
"""
def cost_function(X,y,theta,lambda_):   #Funcion de costo
    # try:
    theta1, theta2, theta3 = inflate(theta) #Convertimos el vector theta en theta 1,2 y3
    m = np.size(X,0)    #Obtenemos la cantidad de ejemplos y la guardamos en m
    X = np.concatenate((np.ones((m,1)), X), axis=1) #agregamos la bios unity
    h_theta = forward_propagation(X,theta1,theta2,theta3)   #Ejecutamos la red para poder evaluar su costo
    reg = (lambda_/(2*m)) * (np.sum(np.sum(np.power(theta1[:,1:],2))) + np.sum(np.sum(np.power(theta2[:,1:],2))) + np.sum(np.sum(np.power(theta3[:,1:],2))))
    J = (-1/m) * sum(np.multiply(y, np.log(h_theta)) + np.multiply((1-y), np.log(1-h_theta)))   #Ejecutamos la funcion de costo
    J += reg    #Añadimos la regularizacion
    print(J, end="\r")
    return J
    # except:
    #     print(theta,"\n\n",theta1,"\n\n",theta2,"\n\n",theta3)
    #     exit()
#=================================================
"""
Aqui encontramos la propagacion hacia adelante que nos
permite calcular la hipotesis avanzando por cada neurona
de la red
"""
def forward_propagation(X,theta1,theta2,theta3):
    m = np.size(X,0)    #Obtenemos la cantidad de ejemplos en la base de datos
    #----------------------------
    #Capa de entrada
    #layer 1
    a_1 = X # ingresamos los datos de entrada

    #----------------------------
    #Capa Oculta
    #layer 2
    z_2 = theta1 * a_1.T    #Calculamos z para la segunda capa
    a_2 = sigmoid(z_2)  #Calculamos el valor de las neuronas de la segunda capa
    a_2 = np.concatenate((np.ones((1,m)), a_2), axis=0) #Añadimos una fila de 1's a la matriz, por la bios

    #layer 3
    z_3 = theta2 * a_2  #Calculamos z para la tercera capa 
    a_3 = sigmoid(z_3)  #Calculamos el valor de las neuronas de la tercera capa
    a_3 = np.concatenate((np.ones((1,m)), a_3), axis=0) #Añadimos una fila de 1's a la matriz, por la bios

    #Capa de salida
    #layer 4
    z_4 = theta3 * a_3 #Calculamos z para la ultima capa
    a_4 = sigmoid(z_4)  #Calculamos h de theta y obtenemos el resultado
    return a_4.T

#=================================================
"""
En esta funcion calcularemos una derivada de forma
mas sencilla para revisar si el desenso de gradiente 
esta bien.
"""
def grad_check(theta, epsilon_,X,y,lambda_):    #Gradiente numerico
    try:    #primero intentamos asi para una matriz
        numgrad = np.zeros(theta.shape) #Creamos la variable para guardar el resultado
        perturb = np.zeros(theta.shape) #Creamos la variable para guardar la diferencia en cada calulo
        e = epsilon_    #establecemos la distancia que tendra cada calculo
 
        for p in range(np.size(theta)):
            perturb[0,p] = e
            loss1 = cost_function(X,y,(theta - perturb), lambda_) #Con este calculo obtenemos la primera deribada
            loss2 = cost_function(X,y,(theta + perturb), lambda_)   #con este la segunda

            numgrad[0,p] = (loss2 - loss1) / (2*e)  #Calculamos la pendiente
            perturb[0,p] = 0    #Desasemos uno de los cambios
        return numgrad
    except: #este caso es para arrays
        numgrad = np.zeros(theta.shape)
        perturb = np.zeros(theta.shape)
        e = epsilon_

        for p in range(np.size(theta)):
            perturb[p] = e
            loss1 = cost_function(X,y,(theta - perturb), lambda_)
            loss2 = cost_function(X,y,(theta + perturb), lambda_)

            numgrad[p] = (loss2 - loss1) / (2*e)
            perturb[p] = 0
        return numgrad
        
#=================================================
"""
En esta parte desarrollaremos el background
propagation para calcular el gradiente de J
"""
def background_propagation(X,y,theta, reg_ter):
    theta1, theta2, theta3 = inflate(theta) #Convertimos el vector theta en theta 1,2 y3
    #Primero hacemos la propagacion hacia adelante
    m = np.size(X,0)    #Obtenemos la cantidad de ejemplos en la base de datos

    #----------------------------
    #Capa de entrada
    #layer 1
    a_1 = X # ingresamos los datos de entrada
    a_1 = np.concatenate((np.ones((m,1)), a_1), axis=1) #agregamos la bios unity


    #----------------------------
    #Capa Oculta
    #layer 2
    z_2 = theta1 * a_1.T    #Calculamos z para la segunda capa
    a_2 = sigmoid(z_2)  #Calculamos el valor de las neuronas de la segunda capa
    a_2 = np.concatenate((np.ones((1,m)), a_2), axis=0) #Añadimos una fila de 1's a la matriz, por la bios

    #layer 3
    z_3 = theta2 * a_2  #Calculamos z para la tercera capa 
    a_3 = sigmoid(z_3)  #Calculamos el valor de las neuronas de la tercera capa
    a_3 = np.concatenate((np.ones((1,m)), a_3), axis=0) #Añadimos una fila de 1's a la matriz, por la bios

    #Capa de salida
    #layer 4
    z_4 = theta3 * a_3 #Calculamos z para la ultima capa
    a_4 = sigmoid(z_4)  #Calculamos h de theta y obtenemos el resultado
    a_4 = a_4.T
    
    
    #BACKPROPAGATION
    #Luego hacemos la propagacion hacia atras
    d_4 = a_4 - y   #Objetenemos el error de cada capa y neurona correspondiente
    d_3 = np.multiply(((theta3.T) * d_4.T), np.concatenate((np.ones((1,m)),grad_sigmoid(z_3))))
    d_3 = d_3[1:,:].T   #Eliminamos la Bios unity para poder hacer el resto de calculos
    d_2 = np.multiply(((theta2.T) * d_3.T), np.concatenate((np.ones((1,m)),grad_sigmoid(z_2))))
    d_2 = d_2[1:,:].T 
    

    Delta_1 = d_2.T * a_1   #Calculamons nuevamente
    Delta_2 = d_3.T * a_2.T #El valor para las neuronas
    Delta_3 = d_4.T * a_3.T #Y lo almacenamos por separado
    
    if reg_ter == 0:
        Theta1_grad = (1/m) * Delta_1
        Theta2_grad = (1/m) * Delta_2
        Theta3_grad = (1/m) * Delta_3
    else:
        dim_1 = Delta_1.shape
        dim_2 = Delta_2.shape
        dim_3 = Delta_3.shape

        Theta1_grad = np.zeros(dim_1)
        Theta2_grad = np.zeros(dim_2)
        Theta3_grad = np.zeros(dim_3)

        for i in range(dim_1[0]):
            Theta1_grad[i,0] = (1/m) * Delta_1[i,0]
            for j in range(dim_1[1]):
               Theta1_grad[i,j] = (1/m) * Delta_1[i,j] + (reg_ter/m) * theta1[i,j]
    
        for i in range(dim_2[0]):
            Theta2_grad[i,0] = (1/m) * Delta_2[i,0]
            for j in range(dim_2[1]):
               Theta2_grad[i,j] = (1/m) * Delta_2[i,j] + (reg_ter/m) * theta2[i,j]
    
        for i in range(dim_3[0]):
            Theta3_grad[i,0] = (1/m) * Delta_3[i,0]
            for j in range(dim_3[1]):
               Theta3_grad[i,j] = (1/m) * Delta_3[i,j] + (reg_ter/m) * theta3[i,j]



    Theta_grad = flatten(Theta1_grad,Theta2_grad,Theta3_grad)
    Theta_grad_ = np.zeros(np.size(Theta_grad))
    try:
        for i in range(len(Theta_grad_)):
            Theta_grad_[i] = Theta_grad[0,i]
    except:
         for i in range(len(Theta_grad_)):
            Theta_grad_[i] = Theta_grad[i]
    
    return Theta_grad_
#=================================================
def efex(t1,t2,t3):
    test_set = np.matrix(pd.read_csv("data/Sets/cross_set.csv"))   #En el train_set se almacenan todos los ejemplos que le daros a la red
    X = test_set[:,0:7]    #Creamos X con todos los ejemplos de entrenamiento
    m = np.size(X,0)    #Obtenemos la cantidad de ejemplos en la base de datos
    X = np.concatenate((np.ones((m,1)), X), axis=1) #agregamos la bios unity
    yP = forward_propagation(X,t1,t2,t3)
    yR = np.array(test_set[:,7:8])  #Creamos y con todas las respuesta correctas para x
    
    for i in range(len(yP)):
        if yP[i] <= 0.5:
            yP[i] = 0
        else:
            yP[i] = 1
    
    Tpositive = 0 
    Fnegative = 0
    Fpositive = 0
    for i in range(len(yP)):
        if yP[i] == yR[i]:
            if yP[i] == 1:
                Tpositive += 1
        if yR[i] == 1:
            if yP[i] != 1:
                Fnegative += 1
        else:
            if yP[i] == 1:
                Fpositive += 1

    presision = Tpositive/(Tpositive + Fnegative)
    recall = Tpositive/(Tpositive + Fpositive)
    return 2 * ((presision*recall)/(presision + recall))


