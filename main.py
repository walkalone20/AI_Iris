import numpy as np
import csv

Iriskind = dict()
IdName = dict()
ind = 0
IrisData = []
species = []
sigmas = np.array([])
mus = np.array([])


def normalization(id):
    cur = []
    for row in IrisData:
        cur.append(row[id])
    mus[id] = np.mean(cur)
    sigmas[id] = np.std(cur)
    for row in IrisData:
        row[id] = (row[id] - mus[id]) / sigmas[id]

def print_data():
    for i in range(len(IrisData)):
        print(IrisData[i], end = " ")
        print(species[i])

with open("./Train/Iris.csv", newline='') as csvfile:
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

# sigmas = np.arange(4)
# mus = np.arange(4)

sigmas = [0.0] * 4
mus = [0.0] * 4

# print_data()

for i in range(0, 4):
    normalization(i)
# print_data()

w = [-0.0806375021419064, -0.022860578827859995,
     0.38650427483942257, 0.4650984917666979]
b = 1.9933554817275565
alpha = 0.003
beta = 0.99999
eps = 1e-8
m = len(IrisData)

while 1:
    change = 0
    neww = [0.0] * 4
    newb = 0
    J = 0
    for i in range(len(IrisData)):
        J += np.square(np.dot(IrisData[i], w) + b - species[i])
    for j in range(0, 4):
        cur = 0
        for i in range(len(IrisData)):
            cur += (np.dot(IrisData[i], w) + b - species[i]) * IrisData[i][j]
        neww[j] = beta * w[j] - alpha / m * cur
    curb = 0
    for i in range(len(IrisData)):
        curb += (np.dot(IrisData[i], w) + b - species[i])
    newb = beta * b - alpha / m * curb
    for i in range(len(w)):
        if np.abs(neww[i] - w[i]) > eps:
            change = 1
            break
    if change == 0:
        break
    print(J)
    w = neww
    b = newb

print(w)
print(b)

def predict(para):
    for i in range(len(para)):
        para[i] = (para[i] - mus[i]) / sigmas[i]
    print("CUR", para)
    res = np.dot(w, para) + b
    mindif = 1e12
    minid = 0
    for i in range(1, ind + 1):
        if np.abs(res - i) < mindif:
            mindif = np.abs(res - i)
            minid = i
    print(res, minid)
    return IdName.get(minid)

print(predict([6.0,2.2,4.0,1.0]))