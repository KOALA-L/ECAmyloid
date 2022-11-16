#转换二级结构信息为数值向量


import math
import numpy as np
import os

#seq：特征序列
#num_C,num_H,num_E：对应C，H，E的数量
#num_cp：类型组合数量
#address：类型所在位置
#composition：成分信息
#transition：转换信息
#position：位置信息
path = 'G:/Protein Seconday Structure/RFAmyloid_neg/'
path_out = 'G:/Protein Seconday Structure/RFAmyloid_neg/output'

for name in range(334,335):
    if name < 9:
        name = 'neg_' + '00' + str(name + 1) + '.txt'
    elif name < 99:
        name = 'neg_' + '0' + str(name + 1) + '.txt'
    else:
        name = 'neg_' + str(name + 1) + '.txt'

    # if name < 9:
    #     name = '00' + str(name + 1) + '.txt'
    # elif name < 99:
    #     name = '0' + str(name + 1) + '.txt'
    # else:
    #     name = str(name + 1) + '.txt'

    f = open(os.path.join(path,name))
    content = f.read()
    print("type:", type(content))
    print("content:", content)
    a = list(content)
    seq = np.array(a)
    print(type(seq))
    print(seq)
    print(len(seq))
    # seq = np.array(['B','e','B','E','b','e'])
    print("序列长度为：", len(seq))
    num_C = 0
    num_H = 0
    num_E = 0

    # composition
    for i in range(len(seq)):
        if seq[i] == 'C':
            num_C = num_C + 1
        elif seq[i] == 'H':
            num_H = num_H + 1
        else:
            num_E = num_E + 1
    C = num_C / len(seq)
    H = num_H / len(seq)
    E = num_E / len(seq)
    # print(b)
    compositon = np.array([C, H, E])
    print("compositon:", compositon.round(2))

    # transition
    # cp[CC,CH,CE,HC,HH,HE,EC,EH,EE]
    cp = np.zeros(9, dtype=int)
    transtion = np.zeros(9)
    # print(seq[0:2])
    for i in range(len(seq) - 1):
        if seq[i] == 'C':
            if seq[i + 1] == 'C':
                cp[0] = cp[0] + 1
            elif seq[i + 1] == 'H':
                cp[1] = cp[1] + 1
            else:
                cp[2] = cp[2] + 1
        elif seq[i] == 'H':
            if seq[i + 1] == 'C':
                cp[3] = cp[3] + 1

            elif seq[i + 1] == 'H':
                cp[4] = cp[4] + 1

            else:
                cp[5] = cp[5] + 1


        elif seq[i] == 'E':
            if seq[i + 1] == 'C':
                cp[6] = cp[6] + 1
            elif seq[i + 1] == 'H':
                cp[7] = cp[7] + 1
            else:
                cp[8] = cp[8] + 1
    # print(cp)
    a = len(seq) - 1
    # print(a)
    for i in range(len(cp)):
        transtion[i] = cp[i] / a
    print("transtion:", transtion.round(2))

    # position
    n_C = n_H = n_E = 0
    for j in range(len(seq)):
        if seq[j] == 'C':
            n_C = j + n_C + 1
        elif seq[j] == 'H':
            n_H = j + n_H + 1
        else:
            n_E = j + n_E + 1
    b = a * (a + 1)
    C1 = n_C / b
    H1 = n_H / b
    E1 = n_E / b
    positon = np.array([C1, H1, E1])
    print("positon:", positon.round(2))
    final = np.concatenate((compositon.round(2), transtion.round(2), positon.round(2)), axis=0)
    # print(final)
    f = open(os.path.join(path_out,name), "w")
    print(final, file=f)
    f.close()
