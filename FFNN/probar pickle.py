# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 10:15:09 2022

@author: luisz
"""

import pickle
import csv


with open("lista de valores para entrenar.pickle", "rb") as f:
    cierre = pickle.load(f)
# Imprime [1, 2, 3, 4].
print(len(cierre))


tiempo=[]
for i in range (0,6000,5):
    tiempo.append(i)
    
valor=[]
for i in range (0,1000,1):
    valor.append(cierre[i])

    

with open('prueba1000.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Tiempo","Precio"])
    for i in range (0,1000,1):
        writer.writerow([tiempo[i],valor[i]])
