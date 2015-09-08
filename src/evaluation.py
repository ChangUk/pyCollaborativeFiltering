import os

import tool


def LeaveOneOutValidation(recommender, pathData, topN = 10, nNeighbors = 20):
    # Data split
    trainSet, testSet = tool.LeaveOneOutSplit(pathData)
    
    # File path
    curDir = os.path.dirname(os.path.abspath(pathData)) + "/"
    basename = os.path.basename(pathData)
    pathModel = curDir + os.path.splitext(basename)[0] + "_" + recommender.title + "_model.pickle"
    
    # Build or load recommendation model
    recommender.loadData(trainSet)
    model = recommender.loadExtModel(pathModel)
    if model == None:
        model = recommender.buildModel(nNeighbors, pathModel)
    
    print("Leave-one-out validation...")
    precision = 0
    recall = 0
    totalHits = 0
    nTrials = 0
    for user in testSet.keys():
        recommendation = recommender.Recommendation(model, user, topN)
        hit = sum([1 for item in testSet[user] if item in recommendation])
        precision += hit / topN
        recall += hit / len(testSet[user])
        totalHits += hit
        nTrials += 1
    precision /= nTrials
    recall /= nTrials
    hitrate = totalHits / nTrials
    print("The result:")
    print("\t* Precision: " + str(precision))
    print("\t* Recall: " + str(recall))
    print("\t* HitRate: " + str(hitrate))
    return precision, recall, hitrate

def evaluation(recommender, trainSet, testSet, topN = 10, nNeighbors = 20):
    # Build or load recommendation model
    recommender.loadData(trainSet)
    model = recommender.buildModel(nNeighbors)
    
    print("Recommender Evaluation...")
    precision = 0
    recall = 0
    totalHits = 0
    nTrials = 0
    for user in testSet:
        recommendation = recommender.Recommendation(model, user, topN)
        hit = sum([1 for item in testSet[user] if item in recommendation])
        precision += hit / topN
        recall += hit / len(testSet[user])
        totalHits += hit
        nTrials += 1
    precision /= nTrials
    recall /= nTrials
    hitrate = totalHits / nTrials
    print("The result:")
    print("\t* Precision: " + str(precision))
    print("\t* Recall: " + str(recall))
    print("\t* HitRate: " + str(hitrate))
    return precision, recall
