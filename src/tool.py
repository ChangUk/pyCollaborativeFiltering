
def loadData(filePath, inv = False):
    '''
    Load data from a input file into memory with dictionary format.
    * Input file format: objectID \t subjectID \t rating \n
    * Output data format: {objectID1: {subjectID1: rating1, subjectID2: rating2, ...}, objectID2: {...}, ...}
    '''
    data = {}
    objectList = []
    subjectList = []
    try:
        with open(filePath) as file:
            for line in file:
                if inv == True:
                    subjectID, objectID, rating = line.split('\t')
                else:
                    objectID, subjectID, rating = line.split('\t')
                data.setdefault(objectID, {})
                data[objectID][subjectID] = int(rating)
                if objectID not in objectList:
                    objectList.append(objectID)
                if subjectID not in subjectList:
                    subjectList.append(subjectID)
            file.close()
    except IOError as e:
        print(e)
        return
    return data, objectList, subjectList

def loadRelationsOnly(filePath, inv = False):
    '''
    From a input file, load data which has no rating score into memory with dictionary format.
    * Input file format: objectID \t subjectID \n
    * Output data format: {objectID1: [subjectID2, subjectID2, ...], objectID2: [...], ...}
    '''
    data = {}
    objectList = []
    subjectList = []
    try:
        with open(filePath) as file:
            for line in file:
                if inv == True:
                    subjectID, objectID = line.split('\t')
                else:
                    objectID, subjectID = line.split('\t')
                data.setdefault(objectID, [])
                data[objectID].append(subjectID)
                if objectID not in objectList:
                    objectList.append(objectID)
                if subjectID not in subjectList:
                    subjectList.append(subjectID)
    except IOError as e:
        print(e)
        return
    return data, objectList, subjectList

def transposePrefs(prefs):
    '''
    Transpose the preference data by switching object and subject.
    For example, the preference data on users can be transformed into the preferences data on items.
    '''
    result = {}
    for obj in prefs:
        for subj in prefs[obj]:
            result.setdefault(subj, {})
            result[subj][obj] = prefs[obj][subj]
    return result
