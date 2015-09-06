import os
import random

from recommender import ItemBased, UserBased
import tool


def LeaveNOutSplit(pathData, N = 1):
    '''
    Split dataset into training data and testset data for Leave-one-out or Leave-N-out validation
    This task performs selecting randomly one of the non-zero entries of each row to be part of testset
    * Input file format: objectID \t subjectID \t rating \n
    * Output files format: objectID \t subjectID \t rating \n (same as input file)
    '''
    pathTrain = pathData + ".train"
    pathTest = pathData + ".test"
    if os.path.exists(pathTrain) == True and os.path.exists(pathTest) == True:
        trainData = tool.loadData(pathTrain)
        testData = tool.loadData(pathTest)
        return trainData, testData
    else:
        try:
            fpTrain = open(pathTrain, "w")
            fpTest = open(pathTest, "w")
        except IOError as e:
            print(e)
            return None, None
    
    data = tool.loadData(pathData)
    testset = {}
    for user in data.keys():
        # The number of subjects for each object is more than 2 at least in order to leave one out.
        if len(data[user]) > N:
            # Select N records randomly
            heldOutItems = random.sample(data[user].keys(), N)
            
            # Find rating score for each held-out subject
            heldOutRecords = {}
            for heldOut in heldOutItems:
                heldOutRecords[heldOut] = int(data[user].pop(heldOut))
            testset[user] = heldOutRecords
    
    # Write training data
    for user in data.keys():
        for item in data[user].keys():
            fpTrain.write(user + "\t" + item + "\t" + str(data[user][item]) + "\n")
    # Write test data
    for user in testset.keys():
        for item in testset[user].keys():
            fpTest.write(user + "\t" + item + "\t" + str(testset[user][item]) + "\n")
    
    fpTrain.close()
    fpTest.close()
    print("# of users: " + str(len(data)))
    print("# of held-out records: " + str(len(testset)))
    return data, testset

def LeaveOneOutValidation(pathData, recommender, topN=10, nNeighbors=100):
    # Data split
    trainSet, testSet = LeaveNOutSplit(pathData, N = 1)
    
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
