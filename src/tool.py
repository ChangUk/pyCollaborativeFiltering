import os
import random


def loadData(filePath, inv = False):
    '''
    Load data from a input file into memory with dictionary format.
    * Input file format: userID \t itemID \t rating \n
    * Output data format: {userID: {itemID: rating, ...}, ...}
    '''
    data = {}
    try:
        with open(filePath) as file:
            for line in file:
                tokens = line.split('\t')
                user = tokens[0]
                item = tokens[1]
                rating = 1
                if len(tokens) == 3:
                    rating = tokens[2]
                
                if inv == False:
                    data.setdefault(user, {})
                    data[user][item] = int(rating)
                else:
                    data.setdefault(item, {})
                    data[item][user] = int(rating)
            file.close()
    except IOError as e:
        print(e)
        return None
    return data

def LeaveOneOutSplit(pathData):
    return LeaveNOutSplit(pathData, N=1)

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
        trainData = loadData(pathTrain)
        testData = loadData(pathTest)
        return trainData, testData
    else:
        try:
            fpTrain = open(pathTrain, "w")
            fpTest = open(pathTest, "w")
        except IOError as e:
            print(e)
            return None, None
    
    data = loadData(pathData)
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

def transposePrefs(prefs):
    '''
    Transpose the preference data by switching object and subject.
    For example, the preference data on users can be transformed into the preferences data on items.
    '''
    transposed = {}
    for obj in prefs:
        for subj in prefs[obj]:
            transposed.setdefault(subj, {})
            transposed[subj][obj] = prefs[obj][subj]
    return transposed
