from copy import deepcopy
from datetime import datetime


def evaluateRecommender(testSet, recommender, simMeasure = None, nNeighbors = None, model = None, topN = None):
    # Evaluation metrics
    totalPrecision = 0
    totalRecall = 0
    totalF1score = 0
    totalHit = 0
    
    for user in testSet:
        recommendation = recommender.Recommendation(user, simMeasure = simMeasure, nNeighbors = nNeighbors, model = model, topN = topN)
        hit = sum([1 for item in testSet[user] if item in recommendation])
        precision = hit / topN
        recall = hit / len(testSet[user])
        f1score = 0 if hit == 0 else 2 * precision * recall / (precision + recall)
        
        totalPrecision += precision
        totalRecall += recall
        totalF1score += f1score
        totalHit += hit
    
    # Find final results
    result = {}
    result["Precision"] = totalPrecision / len(testSet)
    result["Recall"] = totalRecall / len(testSet)
    result["F1-score"] = totalF1score / len(testSet)
    result["Hit-rate"] = totalHit / len(testSet)
    return result

class CrossValidation(object):
    def KFoldSplit(self, data, fold, nFolds):           # fold: 0~4 when 5-Fold validation
        trainSet = deepcopy(data)                       # data = {user: {item: rating, ...}, ...}
        testSet = {}
        for user in data:
            testSet.setdefault(user, {})
            unitLength = int(len(data[user]) / nFolds)  # data[user] = {item: rating, ...}
            lowerbound = unitLength * fold
            upperbound = unitLength * (fold + 1) if fold < nFolds - 1 else len(data[user])
            testItems = {}
            for i, item in enumerate(data[user]):
                if lowerbound <= i and i < upperbound:
                    testItems[item] = float(trainSet[user].pop(item))
            testSet[user] = testItems
        return trainSet, testSet
    
    def KFold(self, data, recommender, simMeasure = None, nNeighbors = None, model = None, topN = 10, nFolds = 5):
        start_time = datetime.now()
        
        # Evaluation metrics
        totalPrecision = 0
        totalRecall = 0
        totalF1score = 0
        totalHitrate = 0
        
        for fold in range(nFolds):
            trainSet, testSet = self.KFoldSplit(data, fold, nFolds)
            recommender.loadData(trainSet)
            evaluation = evaluateRecommender(testSet, recommender, simMeasure = simMeasure, nNeighbors = nNeighbors, model = model, topN = topN)
                
            totalPrecision += evaluation["Precision"]
            totalRecall += evaluation["Recall"]
            totalF1score += evaluation["F1-score"]
            totalHitrate += evaluation["Hit-rate"]
            
            del(trainSet)
            del(testSet)
        
        # Find final results
        result = {}
        result["Precision"] = totalPrecision / nFolds
        result["Recall"] = totalRecall / nFolds
        result["F1-score"] = totalF1score / nFolds
        result["Hit-rate"] = totalHitrate / nFolds
        
        print("Execution time: {}".format(datetime.now() - start_time))
        return result
    
    def LeaveKOutSplit(self, data, user, items):        # `user` should have rating scores on `items` in `data`
        trainSet = deepcopy(data)                       # To prevent original input data from being modified
        testSet = {}
        testSet.setdefault(user, {})
        for item in items:
            testSet[user][item] = float(trainSet[user].pop(item))
        return trainSet, testSet
    
    def LeaveOneOut(self, data, recommender, simMeasure = None, nNeighbors = None, model = None, topN = 10):
        start_time = datetime.now()
        
        # Evaluation metrics
        totalPrecision = 0
        totalRecall = 0
        totalF1score = 0
        totalHitrate = 0
        
        nTrials = 0
        for user in data:
            for item in data[user]:
                trainSet, testSet = self.LeaveKOutSplit(data, user, [item])
                recommender.loadData(trainSet)
                evaluation = evaluateRecommender(testSet, recommender, simMeasure = simMeasure, nNeighbors = nNeighbors, model = model, topN = topN)
                
                totalPrecision += evaluation["Precision"]
                totalRecall += evaluation["Recall"]
                totalF1score += evaluation["F1-score"]
                totalHitrate += evaluation["Hit-rate"]
                nTrials += 1
                
                del(trainSet)
                del(testSet)
        
        # Find final results
        result = {}
        result["Precision"] = totalPrecision / nTrials
        result["Recall"] = totalRecall / nTrials
        result["F1-score"] = totalF1score / nTrials
        result["Hit-rate"] = totalHitrate / nTrials
        
        print("Execution time: {}".format(datetime.now() - start_time))
        return result
