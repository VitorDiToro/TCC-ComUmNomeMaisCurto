#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Implementação do KNN

# Author        :   Vitor Rodrigues Di Toro
# E-Mail        :   vitorrditoro@gmail.com
# Date          :   23/03/2018
# Last Update   :   23/03/2018

with open('ionosphere.data','r') as File:
    reader = csv.reader(File, delimiter=',', quotechar=',',
                        quoting=csv.QUOTE_MINIMAL)

    l_reader = list(reader)

mycsv_list = []
for i in range(len(l_reader)):
    if not(l_reader[i] in l_reader[i+1:len(l_reader)]):
        mycsv_list.append(l_reader[i])


print(len(mycsv_list))
print(type(mycsv_list))

with open("ionosphere.csv", "w") as File:
    writer = csv.writer(File, lineterminator='\n')
    writer.writerows(mycsv_list)
