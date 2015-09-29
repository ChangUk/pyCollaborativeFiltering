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

def cosine_intersection(dataA, dataB):
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
    
def pearson(dataA, dataB, significanceWeighting = False):
    if type(dataA) is list and type(dataB) is list:
        if len(dataA) != len(dataB):
            print("Error: the length of two input lists are not same.")
            return -1
        length = len(dataA)
        intersection = [i for i in range(length) if dataA[i] != 0 and dataB[i] != 0]    # Contains indices of co-rated items
        if len(intersection) == 0:
            return 0
        meanA = np.mean([dataA[i] for i in range(length) if dataA[i] != 0])
        meanB = np.mean([dataB[i] for i in range(length) if dataB[i] != 0])
        numerator = sum([(dataA[i] - meanA) * (dataB[i] - meanB) for i in intersection])
        deviationA = sqrt(sum([(dataA[i] - meanA) ** 2 for i in intersection]))
        deviationB = sqrt(sum([(dataB[i] - meanB) ** 2 for i in intersection]))
        if (deviationA * deviationB) == 0:
            return 0
        correlation = numerator / (deviationA * deviationB)
    elif type(dataA) is dict and type(dataB) is dict:
        intersection = [obj for obj in dataA if obj in dataB]
        if len(intersection) == 0:
            return 0
        meanA = np.mean([dataA[obj] for obj in dataA.keys()])
        meanB = np.mean([dataB[obj] for obj in dataB.keys()])
        numerator = sum([(dataA[obj] - meanA) * (dataB[obj] - meanB) for obj in intersection])
        deviationA = sqrt(sum([(dataA[obj] - meanA) ** 2 for obj in intersection]))
        deviationB = sqrt(sum([(dataB[obj] - meanB) ** 2 for obj in intersection]))
        if (deviationA * deviationB) == 0:
            return 0
        correlation = numerator / (deviationA * deviationB)
    else:
        print("Error: input data type is invalid.")
        return -1
    
    # Correlation significance weighting
    # Reference: An Algorithmic Framework for Performing Collaborative Filtering (SIGIR 1999)
    if significanceWeighting == True:
        if len(intersection) < 50:
            correlation *= (len(intersection) / 50)
    
    return correlation
    
def jaccard(dataA, dataB):
    # Jaccard similarity is applicable to both list type and dictionary type.
    nIntersection = sum([1 for obj in dataA if obj in dataB])
    nUnion = len(dataA) + len(dataB) - nIntersection
    if nUnion == 0:
        return -1
    return nIntersection / nUnion
