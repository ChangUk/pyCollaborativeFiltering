import os

from recommender import ItemBased, UserBased
import tool


def LeaveOneOutValidation(pathData, recommender, topN = 10, nNeighbors = 20):
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
        hit = 0
        for item in testSet[user]:
            if item in recommendation:
                hit += 1
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
    
if __name__ == "__main__":
    pathData = "/home2/test/movielens.dat"
    
    print("u1_30")
    ubcf1 = UserBased()
    trainSet1u = tool.loadData("/home2/test/u1.base")
    testSet1u = tool.loadData("/home2/test/u1.test")
    evaluation(ubcf1, trainSet1u, testSet1u, nNeighbors=30)
    ibcf1 = ItemBased()
    trainSet1i = tool.loadData("/home2/test/u1.base")
    testSet1i = tool.loadData("/home2/test/u1.test")
    evaluation(ibcf1, trainSet1i , testSet1i )
    
    print("u1_50")
    ubcf1_ = UserBased()
    trainSet1u_ = tool.loadData("/home2/test/u1.base")
    testSet1u_ = tool.loadData("/home2/test/u1.test")
    evaluation(ubcf1_, trainSet1u_, testSet1u_, nNeighbors=50)
    ibcf1_ = ItemBased()
    trainSet1i_ = tool.loadData("/home2/test/u1.base")
    testSet1i_ = tool.loadData("/home2/test/u1.test")
    evaluation(ibcf1_, trainSet1i_, testSet1i_)
    
    print("u2")
    ubcf2 = UserBased()
    trainSet2u = tool.loadData("/home2/test/u2.base")
    testSet2u = tool.loadData("/home2/test/u2.test")
    evaluation(ubcf2, trainSet2u, testSet2u, nNeighbors=30)
    ibcf2 = ItemBased()
    trainSet2i = tool.loadData("/home2/test/u2.base")
    testSet2i = tool.loadData("/home2/test/u2.test")
    evaluation(ibcf2, trainSet2i, testSet2i)
    
    print("u3")
    ubcf3 = UserBased()
    trainSet3u = tool.loadData("/home2/test/u3.base")
    testSet3u = tool.loadData("/home2/test/u3.test")
    evaluation(ubcf3, trainSet3u, testSet3u, nNeighbors=30)
    ibcf3 = ItemBased()
    trainSet3i = tool.loadData("/home2/test/u3.base")
    testSet3i = tool.loadData("/home2/test/u3.test")
    evaluation(ibcf3, trainSet3i, testSet3i)
    
    print("u4")
    ubcf4 = UserBased()
    trainSet4u = tool.loadData("/home2/test/u4.base")
    testSet4u = tool.loadData("/home2/test/u4.test")
    evaluation(ubcf4, trainSet4u, testSet4u, nNeighbors=30)
    ibcf4 = ItemBased()
    trainSet4i = tool.loadData("/home2/test/u4.base")
    testSet4i = tool.loadData("/home2/test/u4.test")
    evaluation(ibcf4, trainSet4i, testSet4i)
    
    print("u5")
    ubcf5 = UserBased()
    trainSet5u = tool.loadData("/home2/test/u5.base")
    testSet5u = tool.loadData("/home2/test/u5.test")
    evaluation(ubcf5, trainSet5u, testSet5u, nNeighbors=30)
    ibcf5 = ItemBased()
    trainSet5i = tool.loadData("/home2/test/u5.base")
    testSet5i = tool.loadData("/home2/test/u5.test")
    evaluation(ibcf5, trainSet5i, testSet5i)
    
    print("ua")
    ubcfa = UserBased()
    trainSetau = tool.loadData("/home2/test/ua.base")
    testSetau = tool.loadData("/home2/test/ua.test")
    evaluation(ubcfa, trainSetau, testSetau, nNeighbors=30)
    ibcfa = ItemBased()
    trainSetai = tool.loadData("/home2/test/ua.base")
    testSetai = tool.loadData("/home2/test/ua.test")
    evaluation(ibcfa, trainSetai, testSetai)
    
    print("ub")
    ubcfb = UserBased()
    trainSetbu = tool.loadData("/home2/test/ub.base")
    testSetbu = tool.loadData("/home2/test/ub.test")
    evaluation(ubcfb, trainSetbu, testSetbu, nNeighbors=30)
    ibcfb = ItemBased()
    trainSetbi = tool.loadData("/home2/test/ub.base")
    testSetbi = tool.loadData("/home2/test/ub.test")
    evaluation(ibcfb, trainSetbi, testSetbi)
