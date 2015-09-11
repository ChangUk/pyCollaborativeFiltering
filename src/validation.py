from copy import deepcopy

import similarity


def evaluateRecommender(recommender, model, testSet, topN = 10):
    # Evaluation metrics
    precision = 0
    recall = 0
    hitrate = 0
    
    nTrials = 0
    for user in testSet:
        recommendation = recommender.Recommendation(model, user, topN)
        hit = sum([1 for item in testSet[user] if item in recommendation])
        precision += hit / topN
        recall += hit / len(testSet[user])
        hitrate += hit
        nTrials += 1
    
    # Find final results
    result = {}
    result["precision"] = precision / nTrials
    result["recall"] = recall / nTrials
    result["hitrate"] = hitrate / nTrials
    return result

class CrossValidation(object):
    def LeaveOneOut(self, recommender, similarityMeasure = similarity.cosineForInterSet, topN = 10, nNeighbors = 20):
        # Evaluation metrics
        precision = 0
        recall = 0
        hitrate = 0
        
        nTrials = 0
        for user in recommender.prefs.keys():
            # The number of subjects for each object is more than 1 at least in order to leave one out.
            if len(recommender.prefs[user]) <= 1: continue
            
            for heldOutRecord in recommender.prefs[user].items():
                # Make training set and test set
                trainSet = deepcopy(recommender.prefs)
                testSet = {}
                testSet.setdefault(user, {})
                testSet[user][heldOutRecord[0]] = float(trainSet[user].pop(heldOutRecord[0]))
                 
#                 model = recommender.buildModel(similarityMeasure, nNeighbors)
                model = (similarityMeasure, nNeighbors)
                evaluation = evaluateRecommender(recommender, model, testSet, topN)
                
                precision += evaluation["precision"]
                recall += evaluation["recall"]
                hitrate += evaluation["hitrate"]
                nTrials += 1
                
                del(trainSet)
                del(testSet)
                del(model)
        
        # Find final results
        result = {}
        result["precision"] = precision / nTrials
        result["recall"] = recall / nTrials
        result["hitrate"] = hitrate / nTrials
        return result
