from math import sqrt

import scipy
from scipy.stats import pearsonr
from scipy.stats.stats import spearmanr

def cosine(dataA, dataB):
    if type(dataA) is list and type(dataB) is list:
        if len(dataA) != len(dataB):
            print("Error: the length of two input lists are not same.")
            return -1
        AB = sum([dataA[i] * dataB[i] for i in range(len(dataA))])
        normA = sqrt(sum([dataA[i] ** 2 for i in range(len(dataA))]))
        normB = sqrt(sum([dataB[i] ** 2 for i in range(len(dataB))]))
        denominator = normA * normB
        if denominator == 0:
            return 0
        return AB / denominator
    elif type(dataA) is dict and type(dataB) is dict:
        interSet = [obj for obj in dataA if obj in dataB]
        if len(interSet) == 0:
            return 0
        AB = sum([dataA[obj] * dataB[obj] for obj in interSet])
        normA = sqrt(sum([dataA[obj] ** 2 for obj in interSet]))
        normB = sqrt(sum([dataB[obj] ** 2 for obj in interSet]))
        denominator = normA * normB
        if denominator == 0:
            return -1
        return AB / denominator
    else:
        print("Error: input data type is invalid.")
        return -1
    
def jaccard(dataA, dataB):
    # Jaccard similarity is applicable to both list type and dictionary type.
    interSet = sum([1 for obj in dataA if obj in dataB])
    unionSet = len(dataA) + len(dataB) - interSet
    if unionSet == 0:
        return -1
    return interSet / unionSet

def pearsonr(listA, listB):
    r_row, p_value = pearsonr(scipy.array(listA), scipy.array(listB))
    return r_row

def spearmanr(listX, listY):
    r_row, p_value = spearmanr(scipy.array(listX), scipy.array(listY))
    return r_row
