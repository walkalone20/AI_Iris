import numpy as np
import csv

Iriskind = dict()
IdName = dict()
ind = 0
IrisData = []
species = []

with open("./Iris.csv", newline='') as csvfile:
    Irisreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    head = 0
    for row in Irisreader:
        if not head:
            head = 1
            continue
        CurFlower = []
        for i in range(1, 5):
            CurFlower.append(float(row[i]))
        IrisData.append(CurFlower)
        if Iriskind.get(row[5], -1) == -1:
            ind += 1
            Iriskind[row[5]] = ind
            IdName[ind] = row[5]
        species.append(Iriskind.get(row[5]))

for i in range(len(IrisData)):
    print(IrisData[i], " ")
    print(species[i])

for row in IrisData:
    row = np.array(row)
species = np.array(species)