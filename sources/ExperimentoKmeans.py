# -*- coding: utf-8 -*-
"""
Created on Sat May 19 08:52:36 2018

@author: Jonatan Alberto Afonso
"""

import kmeansMy as my
from sklearn.cluster import KMeans  as sk
from distances import Distance
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import calinski_harabaz_score as cal
from matplotlib import pyplot as plt 
import numpy

def main():
    #Inicialização dos vetores
    silMineVector = []
    silSkVector = []
    calMineVector = []
    calSkVector = []
    iterationMine = []
    iterationSk = []
    num_it = 1000
    for i in range(0,num_it):
        print(i)
        #leitura do dataset no modo aleatorio
        data = my.get_data_lc('../dataset/ionosphere.csv', range(350), range(34), randomize=True)
    
        #Executção do nosso Kmeans
        kmeansMine = my.Kmeans(k=2, max_iterations=500)
        kmeansMine.fit(data, distance_method=Distance.Type.euclidean)
        
        #Execução do kmeans Scikit Learn
        kmeansSk = sk(n_clusters=2,algorithm='auto',tol=1e-10,max_iter=5000,init='random')#elkan
        kmeansSk.fit_predict(data)
        
  
        #print("Pontos Nosso")
        #print("Pontos do tipo 1: %d" % len(kmeansMine.clusters[0]))
        #print("Pontos do tipo 2: %d" % len(kmeansMine.clusters[1]))
        
        #Guardando os numeros de iterações
        iterationMine.append(kmeansMine.iteration)
        iterationSk.append(kmeansSk.n_iter_)
        
        #Criação dos clusters do Scikit Learn
        clusterSK1 = []
        clusterSK0 = []
        for i in range(len(kmeansSk.labels_)):
            value = kmeansSk.labels_[i]
            if value == 0 :
                clusterSK0.append(data[i])
            else :
                clusterSK1.append(data[i])
                    
        #print("Pontos Scikit")
        #print("Pontos do tipo 1: %d" % len(clusterSK0))
        #print("Pontos do tipo 2: %d" % len(clusterSK1))
        
        #Criação dos labels do nosso Kmeans
        labels = kmeansMine.labels_(data)
        #Calculo da metrica Silhouette Score e armazenamento dos valores
        silMine = sil(data,labels,metric='euclidean')
        silSk = sil(data,labels,metric='euclidean')
        silMineVector.append(silMine)
        silSkVector.append(silSk)
        #print("Silhouette Score")
        #print("Kmeans Nosso")
        #print(silMine)
        #print("Kmeans Scikit")
        #print(silSk)
    
        #Calculo da metrica Calinski and Harabaz Score e armazenamento dos valores
        calMine = cal(data,labels)
        calSk = cal(data,kmeansSk.labels_)
        calMineVector.append(calMine)
        calSkVector.append(calSk)
        #print("Calinski and Harabaz Score")
        #print("Kmeans Nosso")
        #print(calMine)
        #print("Kmeans Scikit")
        #print(calSk)
    
    #PLot dos graficos
    plt.figure(1)
    
    
   
    plt.title('Iterações')
    plt.plot(range(0,num_it),iterationMine)
    plt.plot(range(0,num_it),iterationSk)
    plt.legend(['Nosso','Scikit'],loc=3)
    plt.show()
    
    plt.title('Silhouette Score')
    
    plt.plot(range(0,num_it),silMineVector)
    
    plt.plot(range(0,num_it),silSkVector)
    plt.legend(['Nosso','Scikit'],loc=3)
    plt.show()
    
    plt.title('Calinski Harabaz Score')
    
    plt.plot(range(0,num_it),calMineVector)
    
    plt.plot(range(0,num_it),calSkVector)
    plt.legend(['Nosso','Scikit'],loc=3)
    plt.show()
    
    #Medias e desvio Padrão

    print("Iterações")
    print("Media Nosso: %f" % numpy.mean(iterationMine))
    print("Media Scikit: %f" % numpy.mean(iterationSk))
    print("Desvio Padrão Nosso: %f" % numpy.std(iterationMine))
    print("Desvio Padrão Scikit: %f" % numpy.std(iterationSk))
    
    print("Silhouette Score")
    print("Media Nosso: %f" % numpy.mean(silMineVector))
    print("Media Scikit: %f" % numpy.mean(silSkVector))
    print("Desvio Padrão Nosso: %f" % numpy.std(silMineVector))
    print("Desvio Padrão Scikit: %f" % numpy.std(silSkVector))
    
    print("Calinski Harabaz Score")
    print("Media Nosso: %f" % numpy.mean(calMineVector))
    print("Media Scikit: %f" % numpy.mean(calSkVector))
    print("Desvio Padrão Nosso: %f" % numpy.std(calMineVector))
    print("Desvio Padrão Scikit: %f" % numpy.std(calSkVector))
    
if __name__ == '__main__':
    main()