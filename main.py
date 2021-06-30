from hmm import HMM
from dwt import DWT

dwt = DWT(False)
dwt2 = DWT(True)  # with properties

hmm = HMM(False)
hmm2 = HMM(True)  # with properties

# hmm learning data files address : Safarmohammadloo
hmml = []

for i in range(1, 70):
    hmml.append("./hmml/" + str(i) + ".wav")

hmm.learn(hmml)
hmm2.learn(hmml)


def compare(x, y):
    print("DWT :")
    temp = []
    for i in x:
        results = []
        for j in y:
            if (str(i) + str(j) not in temp) or (str(j) + str(i) not in temp):
                temp.append(str(i) + str(j))
                temp.append(str(j) + str(i))
                results.append(dwt.against(i, j))
            else:
                results.append(0)
        print(results)

    print("DWT with properties :")
    temp = []
    for i in x:
        results = []
        for j in y:
            if (str(i) + str(j) not in temp) or (str(j) + str(i) not in temp):
                temp.append(str(i) + str(j))
                temp.append(str(j) + str(i))
                results.append(dwt2.against(i, j))
            else:
                results.append(0)
        print(results)

    print("HMM :")
    temp = []
    for i in x:
        results = []
        for j in y:
            if (str(i) + str(j) not in temp) or (str(j) + str(i) not in temp):
                temp.append(str(i) + str(j))
                temp.append(str(j) + str(i))
                results.append(hmm.against(i, j))
            else:
                results.append(0)
        print(results)

    print("HMM with properties :")
    temp = []
    for i in x:
        results = []
        for j in y:
            if (str(i) + str(j) not in temp) or (str(j) + str(i) not in temp):
                temp.append(str(i) + str(j))
                temp.append(str(j) + str(i))
                results.append(hmm2.against(i, j))
            else:
                results.append(0)
        print(results)


# -----------------
# A
print("")
print("A :")

# last name files address : Safarmohammadloo
last_name = []

for i in range(1, 12):
    last_name.append("./l/" + str(i) + ".wav")

compare(last_name, last_name)

# -----------------
# B
print("")
print("B :")

print("")
print("Mahdi :")

# first name files address : Mahdi
first_name = []

for i in range(1, 4):
    first_name.append("./f/" + str(i) + ".wav")

compare(last_name, first_name)

print("")
print("Mohammad Mahdi :")

# first name 2 files address : Mohammad Mahdi
first_name2 = []

for i in range(1, 4):
    first_name2.append("./fs/" + str(i) + ".wav")

compare(last_name, first_name2)

# -----------------
# C
print("")
print("C :")

# last name files address other person : Safarmohammadloo
last_name2 = []

for i in range(1, 12):
    last_name2.append("./ls/" + str(i) + ".wav")

compare(last_name, last_name2)
