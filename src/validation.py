from copy import deepcopy

import similarity


def evaluateRecommender(testSet, recommender, model, topN = 10, binaryMode = False, useOnlyPositives = True):
    # Evaluation metrics
    totalPrecision = 0
    totalRecall = 0
    totalF1score = 0
    totalHit = 0
    
    nTrials = 0
    for user in testSet:
        recommendation = recommender.Recommendation(model, user, topN, binaryMode, useOnlyPositives)
        hit = sum([1 for item in testSet[user] if item in recommendation])
        precision = hit / topN
        recall = hit / len(testSet[user])
        f1score = 2 * precision * recall / (precision + recall) 
        
        totalPrecision += precision
        totalRecall += recall
        totalF1score += f1score
        totalHit += hit
        nTrials += 1
    
    # Find final results
    result = {}
    result["Precision"] = totalPrecision / nTrials
    result["Recall"] = totalRecall / nTrials
    result["F1-score"] = totalF1score / nTrials
    result["Hit-rate"] = totalHit / nTrials
    return result

class CrossValidation(object):
    def LeaveOneOut(self, recommender, similarityMeasure = similarity.cosine_intersection, topN = 10, nNeighbors = 20):
        # Evaluation metrics
        totalPrecision = 0
        totalRecall = 0
        totalF1score = 0
        totalHitrate = 0
        
        nTrials = 0
        for user in recommender.prefs.keys():
            # The number of subjects for each object is more than 1 at least in order to leave one out.
            if len(recommender.prefs[user]) <= 1:
                continue
            
            for heldOutRecord in recommender.prefs[user].items():
                # Make training set and test set
                trainSet = deepcopy(recommender.prefs)
                testSet = {}
                testSet.setdefault(user, {})
                testSet[user][heldOutRecord[0]] = float(trainSet[user].pop(heldOutRecord[0]))
                
                model = (similarityMeasure, nNeighbors)
                evaluation = evaluateRecommender(testSet, recommender, model, topN)
                
                totalPrecision += evaluation["Precision"]
                totalRecall += evaluation["Recall"]
                totalF1score += evaluation["F1-score"]
                totalHitrate += evaluation["Hit-rate"]
                nTrials += 1
                
                del(trainSet)
                del(testSet)
                del(model)
        
        # Find final results
        result = {}
        result["Precision"] = totalPrecision / nTrials
        result["Recall"] = totalRecall / nTrials
        result["F1-score"] = totalF1score / nTrials
        result["Hit-rate"] = totalHitrate / nTrials
        return result
