
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
