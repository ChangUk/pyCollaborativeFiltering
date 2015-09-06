import os

from recommender import ItemBased, UserBased
import tool


def LeaveOneOutValidation(pathData, recommender, topN = 10, nNeighbors = 100):
    # Data split
    trainSet, testSet = tool.LeaveNOutSplit(pathData, N=1)
    
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
    TRIALS = 0
    HIT = 0
    for user in testSet.keys():
        recommendation = recommender.Recommendation(model, user, topN)
        cnt = 0
        for item in testSet[user].keys():
            if item in recommendation:
                cnt += 1
                HIT += 1
        precision += cnt / topN
        recall += cnt / len(testSet[user].keys())
        TRIALS += 1
    precision /= TRIALS
    recall /= TRIALS
    hitrate = HIT / TRIALS
    print("The result:")
    print("\t* Precision: " + str(precision))
    print("\t* Recall: " + str(recall))
    print("\t* HitRate: " + str(hitrate))
    return precision, recall, hitrate
    
if __name__ == "__main__":
    pathData = "/home2/test/shopping.dat"
    
    ubcf = UserBased()
    LeaveOneOutValidation(pathData, ubcf)
    
    ibcf = ItemBased()
    LeaveOneOutValidation(pathData, ibcf)
