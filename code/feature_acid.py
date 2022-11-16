#转换溶剂可及性结果为数值向量


import math
import numpy as np
import os
#seq：特征序列
#num_B,num_b,num_E,num_e：对应B，b，E，e的数量
#num_cp：类型组合数量
#address：类型所在位置
#composition：成分信息
#transition：转换信息
#position：位置信息

#content：读取文件内容的字符串
#a：分离字符串
#b：单个字母分离后的数组
path = 'G:/solvent accessibility/RFAmyloid_neg/'
path_out = 'G:/solvent accessibility/RFAmyloid_neg/output'

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
    num_B = 0
    num_b = 0
    num_E = 0
    num_e = 0

    # composition
    for i in range(len(seq)):
        if seq[i] == 'B':
            num_B = num_B + 1
        elif seq[i] == 'b':
            num_b = num_b + 1
        elif seq[i] == 'E':
            num_E = num_E + 1
        else:
            num_e = num_e + 1
    B = num_B / len(seq)
    b = num_b / len(seq)
    E = num_E / len(seq)
    e = num_e / len(seq)
    # print(b)
    compositon = np.array([B, b, E, e])
    print("compositon:", compositon.round(2))

    # transition
    # cp[BB,Bb,BE,Be,bB,bb,bE,be,EB,Eb,EE,Ee,eB,eb,eE,ee]
    cp = np.zeros(16, dtype=int)
    transtion = np.zeros(16)
    # print(seq[0:2])
    for i in range(len(seq) - 1):
        # if seq[i:i+2] == ['B''B']:
        if seq[i] == 'B':
            if seq[i + 1] == 'B':
                cp[0] = cp[0] + 1
            elif seq[i + 1] == 'b':
                cp[1] = cp[1] + 1
            elif seq[i + 1] == 'E':
                cp[2] = cp[2] + 1
            else:
                cp[3] = cp[3] + 1
        elif seq[i] == 'b':
            if seq[i + 1] == 'B':
                cp[4] = cp[4] + 1
            elif seq[i + 1] == 'b':
                cp[5] = cp[5] + 1
            elif seq[i + 1] == 'E':
                cp[6] = cp[6] + 1
            else:
                cp[7] = cp[7] + 1
        elif seq[i] == 'E':
            if seq[i + 1] == 'B':
                cp[8] = cp[8] + 1
            elif seq[i + 1] == 'b':
                cp[9] = cp[9] + 1
            elif seq[i + 1] == 'E':
                cp[10] = cp[10] + 1
            else:
                cp[11] = cp[11] + 1
        elif seq[i] == 'e':
            if seq[i + 1] == 'B':
                cp[12] = cp[12] + 1
            elif seq[i + 1] == 'b':
                cp[13] = cp[13] + 1
            elif seq[i + 1] == 'E':
                cp[14] = cp[14] + 1
            else:
                cp[15] = cp[15] + 1
    # print(cp)
    a = len(seq) - 1
    # print(a)
    for i in range(len(cp)):
        transtion[i] = cp[i] / a
    print("transtion:", transtion.round(2))

    # position
    n_B = n_b = n_E = n_e = 0
    for j in range(len(seq)):
        if seq[j] == 'B':
            n_B = j + n_B + 1
        elif seq[j] == 'b':
            n_b = j + n_b + 1
        elif seq[j] == 'E':
            n_E = j + n_E + 1
        else:
            n_e = j + n_e + 1
    b = a * (a + 1)
    B1 = n_B / b
    b1 = n_b / b
    E1 = n_E / b
    e1 = n_e / b
    positon = np.array([B1, b1, E1, e1])
    print("positon:", positon.round(2))
    final = np.concatenate((compositon.round(2), transtion.round(2), positon.round(2)), axis=0)
    # print(final)
    f = open(os.path.join(path_out,name), "w")
    print(final, file=f)
    f.close()
