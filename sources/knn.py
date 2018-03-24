#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Implementação do KNN 
 
# Author        :   Vitor Rodrigues Di Toro
# E-Mail        :   vitorrditoro@gmail.com
# Date          :   19/03/2018
# Last Update   :   22/03/2018


import csv
import distances as dist
from random import shuffle

euclidean = 1
manhattan = 2
minkowski = 3



def num(s):
    try:
        return float(s)
    except ValueError:
        return s


    
def getData(datasetName = 'ionosphere.csv', percentToTraining = 60, randomize = True, verbose = True):
    
    count = 0
    group_g = 0
    test_data = []
    training_data = []
    
    
    with open(datasetName,'r') as File:
        reader = csv.reader(File, delimiter=',', quotechar=',',
                            quoting=csv.QUOTE_MINIMAL)

        l_reader = list(reader)
        limit = int(len(l_reader) * (percentToTraining/100))
        
        if randomize:
            shuffle(l_reader)

        for row in l_reader:

            if row[-1] == "g" :
                group_g += 1

            if count < limit:
                training_data.append([num(i) for i in row])
            else:
                test_data.append([num(i) for i in row])

            count += 1
            
    if verbose:
        print("Total de amostras: %d:" %len(l_reader))
        print("    - %d amostra do tipo \"Good\"" % group_g)
        print("    - %d amostra do tipo \"Bad\"" % (len(l_reader) - group_g))
        print("")
        print("%d %% das amostras separadas para treino." %percentToTraining)
        print("    - %d amostras para treino" % (len(training_data)))
        print("    - %d amostras para teste" % (len(test_data)))
            
    return training_data, test_data



def knn(training, test, k, distance = euclidean, distanceOrder = 0.5 ):
    
    result = []
	
    print("Method: %d" % distance)
    
    for i in range(len(test)):
        distances = {}
        
        for j in range(len(training)):
            if distance == euclidean:
                distances[j] = dist.euclideanDistance(test[i], training[j])
            elif distance == manhattan:
                distances[j] = dist.manhattanDistance(test[i], training[j])
            elif distance == minkowski:
                distances[j] = dist.minkowskiDistance(test[i], training[j], distanceOrder)
                
        k_neighbors = sorted(distances, key=distances.get)[:k]
        
        g_count, b_count = 0, 0
        
        for index in k_neighbors:
            if training[index][-1] == 'g':
                g_count += 1
            else:
                b_count += 1
        
        if g_count > b_count:
            result.append('g')
        else:
            result.append('b')
            
        
    acertos = 0
    for i in range(len(test)):
        #print("Obtido : " + result[i])
        #print("Correto: " + test[i][-1])
        
        if(result[i] == test[i][-1]):
            acertos += 1
        
    print("Porcentagem de acertos: %.4f %%" % (100*acertos/len(test)))


def main():
    training_data, test_data = getData('ionosphere.csv',60,randomize=True,verbose=False,)

    print("\nEuclidean distance:")
    knn(training_data, test_data, k=13, distance=euclidean)

    print("\nManhattan distance:")
    knn(training_data, test_data, k=13, distance=manhattan)

    print("\nMinkowski distance:")
    knn(training_data, test_data, k=13, distance=minkowski, distanceOrder = 0.2)
    

if __name__ == '__main__':
    main()
