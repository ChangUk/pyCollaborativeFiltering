from math import sqrt

import numpy as np


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
        normA = sqrt(sum([dataA[obj] ** 2 for obj in dataA]))
        normB = sqrt(sum([dataB[obj] ** 2 for obj in dataB]))
        denominator = normA * normB
        if denominator == 0:
            return -1
        return AB / denominator
    else:
        print("Error: input data type is invalid.")
        return -1

def cosineForInterSet(dataA, dataB):
    if type(dataA) is list and type(dataB) is list:
        if len(dataA) != len(dataB):
            print("Error: the length of two input lists are not same.")
            return -1
        interSet = [i for i in range(len(dataA)) if dataA[i] * dataB[i] != 0]
        if len(interSet) == 0:
            return 0
        AB = sum([dataA[i] * dataB[i] for i in range(interSet)])
        normA = sqrt(sum([dataA[i] ** 2 for i in range(interSet)]))
        normB = sqrt(sum([dataB[i] ** 2 for i in range(interSet)]))
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
    
def pearson(dataA, dataB):
    if type(dataA) is list and type(dataB) is list:
        if len(dataA) != len(dataB):
            print("Error: the length of two input lists are not same.")
            return -1
        length = len(dataA)
        interSet = [i for i in range(length) if dataA[i] != 0 and dataB[i] != 0]    # Contains indices of co-rated items
        if len(interSet) == 0:
            return 0
        meanA = np.mean([dataA[i] for i in range(length) if dataA[i] != 0])
        meanB = np.mean([dataB[i] for i in range(length) if dataB[i] != 0])
        numerator = sum([(dataA[i] - meanA) * (dataB[i] - meanB) for i in interSet])
        deviationA = sqrt(sum([(dataA[i] - meanA) ** 2 for i in interSet]))
        deviationB = sqrt(sum([(dataB[i] - meanB) ** 2 for i in interSet]))
        if (deviationA * deviationB) == 0:
            return 0
        return numerator / (deviationA * deviationB)
    elif type(dataA) is dict and type(dataB) is dict:
        interSet = [obj for obj in dataA if obj in dataB]
        if len(interSet) == 0:
            return 0
        meanA = np.mean([dataA[obj] for obj in dataA.keys()])
        meanB = np.mean([dataB[obj] for obj in dataB.keys()])
        numerator = sum([(dataA[obj] - meanA) * (dataB[obj] - meanB) for obj in interSet])
        deviationA = sqrt(sum([(dataA[obj] - meanA) ** 2 for obj in interSet]))
        deviationB = sqrt(sum([(dataB[obj] - meanB) ** 2 for obj in interSet]))
        if (deviationA * deviationB) == 0:
            return 0
        correlation = numerator / (deviationA * deviationB)
        
        # Correlation significance weighting
        if len(interSet) < 50:
            correlation *= (len(interSet) / 50)
        
        return correlation
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
