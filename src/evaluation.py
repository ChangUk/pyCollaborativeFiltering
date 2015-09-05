import os
import random

from recommender import ItemBased, UserBased
import tool


def LeaveNOutSplit(pathData, inv = False, N = 1):
    '''
    Split dataset into training data and testset data for Leave-one-out or Leave-N-out validation
    This task performs selecting randomly one of the non-zero entries of each row to be part of testset
    * Input file format: objectID \t subjectID \t rating \n
    * Output files format: objectID \t subjectID \t rating \n (same as input file)
    '''
    try:
        fpTrain = open(pathData + ".train", "w")
        fpTest = open(pathData + ".test", "w")
    except IOError as e:
        print(e)
        return
    
    data, objectList, subjectList = tool.loadData(pathData, inv)
    
    testset = {}
    for obj in data.keys():
        # The number of subjects for each object is more than 2 at least in order to leave one out.
        if len(data[obj]) > N:
            # Select N records randomly
            heldOutSubjects = random.sample(data[obj].keys(), N)
            
            # Find rating score for each held-out subject
            heldOutRecords = {}
            for heldOut in heldOutSubjects:
                heldOutRecords[heldOut] = int(data[obj].pop(heldOut))
            testset[obj] = heldOutRecords
    
    # Write training data
    for obj in data.keys():
        for subject in data[obj].keys():
            fpTrain.write(obj + "\t" + subject + "\t" + str(data[obj][subject]) + "\n")
    # Write test data
    for obj in testset.keys():
        for subject in testset[obj].keys():
            fpTest.write(obj + "\t" + subject + "\t" + str(testset[obj][subject]) + "\n")
    
    fpTrain.close()
    fpTest.close()
    print("# of the given objects: " + str(len(data)))
    print("# of the held-out objects: " + str(len(testset)))
    return data, testset

def LeaveOneOutValidation(recommender, pathData):
    # Data split
    trainSet, testSet = LeaveNOutSplit(pathData, N = 1)     # 'trainSet' is the preferences data on users
    
    # File path
    curDir = os.path.dirname(os.path.abspath(pathData)) + "/"
    basename = os.path.basename(pathData)
    pathModel = curDir + os.path.splitext(basename)[0] + "_model.pickle"
    
    # Build or load recommendation model
    recommender.loadData(trainSet)
    if recommender.loadExtModel(pathModel) == False:
        recommender.buildModel()
        recommender.dumpModel(pathModel)
    
    print("Leave-one-out validation...")
    precision = 0
    recall = 0
    TRIALS = 0
    HIT = 0
    for user in testSet.keys():
        recommendation = recommender.Recommendation(trainSet[user])
        cnt = 0
        for item in testSet[user].keys():
            if item in recommendation:
                cnt += 1
                HIT += 1
        precision += cnt / recommender.TOPN
        recall += cnt / len(testSet[user].keys())
        TRIALS += 1
    precision /= TRIALS
    recall /= TRIALS
    hitrate = HIT / TRIALS
    print("\t* Precision: " + str(precision))
    print("\t* Recall: " + str(recall))
    print("\t* HitRate: " + str(hitrate))
    return precision, recall, hitrate
    
if __name__ == "__main__":
    pathData = "/home2/test/shopping.dat"
    ubcf = UserBased()
    trainSet, testSet = LeaveNOutSplit(pathData, N = 1)     # 'trainSet' is the preferences data on users
    
    curDir = os.path.dirname(os.path.abspath(pathData)) + "/"
    basename = os.path.basename(pathData)
    pathModel = curDir + os.path.splitext(basename)[0] + "_ubcf_model.pickle"
    
    ubcf.loadData(trainSet)
    model = ubcf.loadExtModel(pathModel)
    if model == None:
        model = ubcf.buildModel(100, pathModel)
    
    print("Leave-one-out validation...")
    precision = 0
    recall = 0
    TRIALS = 0
    HIT = 0
    for user in testSet.keys():
        recommendation = ubcf.Recommendation(model, user, 10)
        cnt = 0
        for item in testSet[user].keys():
            if item in recommendation:
                cnt += 1
                HIT += 1
        precision += cnt / 10
        recall += cnt / len(testSet[user].keys())
        TRIALS += 1
    precision /= TRIALS
    recall /= TRIALS
    hitrate = HIT / TRIALS
    print("\t* Precision: " + str(precision))
    print("\t* Recall: " + str(recall))
    print("\t* HitRate: " + str(hitrate))
