#-*- coding: utf-8 -*-
#!/usr/bin/env python

import sys
import numpy

def editDistance(r, h):
    '''
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.

    Main algorithm used is dynamic programming.

    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    '''
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def getStepList(r, h, d):
    '''
    This function is to get the list of steps in the process of dynamic programming.

    Attributes: 
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calulating the editting distance of h and r.
    '''
    x = len(r)
    y = len(h)
    list = []
    while True:
        if x == 0 and y == 0: 
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1] and r[x-1] == h[y-1]: 
            list.append("e")
            x = x - 1
            y = y - 1
        elif y >= 1 and d[x][y] == d[x][y-1]+1:
            list.append("i")
            x = x
            y = y - 1
        elif x >= 1 and y >= 1 and d[x][y] == d[x-1][y-1]+1:
            list.append("s")
            x = x - 1
            y = y - 1
        else:
            list.append("d")
            x = x - 1
            y = y
    return list[::-1]

def alignedPrint(list, r, h):#, result):
    '''
    This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.
    
    Attributes:
        list   -> the list of steps.
        r      -> the list of words produced by splitting reference sentence.
        h      -> the list of words produced by splitting hypothesis sentence.
        result -> the rate calculated based on edit distance.
    '''
    ref_list = []
    hypo_list = []

    #print("REF:", end=" ")
    for i in range(len(list)):
        if list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            #print(" "*(len(h[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            '''
            if len(r[index1]) < len(h[index2]):
                print(r[index1] + " " * (len(h[index2])-len(r[index1])), end=" ")
            else:
                print(r[index1], end=" ")
            '''
            ref_list.append(r[index1])
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            #print(r[index], end=" ")
    #print("\nHYP:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            #print(" " * (len(r[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            '''
            if len(r[index1]) > len(h[index2]):
                print(h[index2] + " " * (len(r[index1])-len(h[index2])), end=" ")
            else:
                print(h[index2], end=" ")
            '''
            hypo_list.append(h[index2])
        else:
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            #print(h[index], end=" ")
    #print("\nEVA:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            #print("D" + " " * (len(r[index])-1), end=" ")
        elif list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            #print("I" + " " * (len(h[index])-1), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            '''
            if len(r[index1]) > len(h[index2]):
                print("S" + " " * (len(r[index1])-1), end=" ")
            else:
                print("S" + " " * (len(h[index2])-1), end=" ")
            '''
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            #print(" " * (len(r[index])), end=" ")

    return ref_list, hypo_list
    #print("\nWER: " + result)

def wer(r, h):
    """
    This is a function that calculate the word error rate in ASR.
    You can use it like this: wer("what is it".split(), "what is".split()) 
    """
    # build the matrix
    d = editDistance(r, h)

    # find out the manipulation steps
    list = getStepList(r, h, d)

    # print the result in aligned way
    result = float(d[len(r)][len(h)]) / len(r) * 100
    result = str("%.2f" % result) + "%"
    #alignedPrint(list, r, h, result)
    alignedPrint(list, r, h)

if __name__ == '__main__':
    '''
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    with open(filename1, 'r', encoding="utf8") as ref:
        r = ref.read().split()
    with open(filename2, 'r', encoding="utf8") as hyp:
        h = hyp.read().split()
    '''
    #h = "wadiz it".split()
    #r = "what is it".split()
    #h = 'WIL YOUO FOR GIVE ME NOW here'.split()
    #r = 'WILL YOU FORGIVE ME NOW here here here'.split()
    #h = 'YESTERDY YOU WERE TREMBLING FOR A HELTH THAT IS DER TO YO TO DAY YOU FERE FOR YORE ON TO MAROW IT WL BE ANGSADY ABOUT MONEY THE DAY AFTER TOMAROW THE DI AF TRIB OF A SLANDER THE DAY AFFTTER THAT THE MIS FORCTIUON OFE SOME FREND THEN THE PREVAILING WHETHER THEN SOMETHING THAT HAS BEEN BROKOAN OR LOSED THEN A PLESUR WITH WHICH YORE CONCHIONCS AND YOR VERTABRIL CALLME REPROCH O AGEN THE CORS OF POUBLICK AFFAIRS'.split()
    #r = 'YESTERDAY YOU WERE TREMBLING FOR A HEALTH THAT IS DEAR TO YOU TO DAY YOU FEAR FOR YOUR OWN TO MORROW IT WILL BE ANXIETY ABOUT MONEY THE DAY AFTER TO MORROW THE DIATRIBE OF A SLANDERER THE DAY AFTER THAT THE MISFORTUNE OF SOME FRIEND THEN THE PREVAILING WEATHER THEN SOMETHING THAT HAS BEEN BROKEN OR LOST THEN A PLEASURE WITH WHICH YOUR CONSCIENCE AND YOUR VERTEBRAL COLUMN REPROACH YOU AGAIN THE COURSE OF PUBLIC AFFAIRS'.split()
    h = 'YESTERDY YOU WERE TREMBLING FOR A HELTH THAT IS DER TO YO TO DAY YOU FERE FOR YORE ON TO MAROW IT WL BE ANGSADY ABOUT MONEY THE DAY AFTER TOMAROW THE DI AF TRIB OF A SLANDER THE DAY AFFTTER THAT THE MIS FORCTIUON OFE SOME FREND THEN THE PREVAILING WHETHER THEN SOMETHING THAT HAS BEEN BROKOAN OR LOSED THEN A PLESUR WITH WHICH YORE CONCHIONCS AND YOR VERTABRIL CALLME REPROCH O AGEN THE CORS OF POUBLICK AFFAIRS'.split()
    r = 'YESTERDAY YOU WERE TREMBLING FOR A HEALTH THAT IS DEAR TO YOU TO DAY YOU FEAR FOR YOUR OWN TO MORROW IT WILL BE ANXIETY ABOUT MONEY THE DAY AFTER TO MORROW THE DIATRIBE OF A SLANDERER THE DAY AFTER THAT THE MISFORTUNE OF SOME FRIEND THEN THE PREVAILING WEATHER THEN SOMETHING THAT HAS BEEN BROKEN OR LOST THEN A PLEASURE WITH WHICH YOUR CONSCIENCE AND YOUR VERTEBRAL COLUMN REPROACH YOU AGAIN THE COURSE OF PUBLIC AFFAIRS'.split()

    wer(r, h)
