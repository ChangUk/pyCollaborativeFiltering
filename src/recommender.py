import pickle

import numpy as np
import similarity
import tool


class CollaborativeFiltering:
    def __init__(self, similarityMeasure):
        self.prefs = {}     # Training data
        self.similarityMeasure = similarityMeasure
        
    def viewStatistics(self):
        objList = []
        subjList = []
        for obj in self.prefs.keys():
            if obj not in objList:
                objList.append(obj)
            for subj in self.prefs[obj].keys():
                if subj not in subjList:
                    subjList.append(subj)
        print("Data statistics")
        print("\t* Object: " + str(len(objList)))
        print("\t* Subject: " + str(len(subjList)))
        
    def loadExtModel(self, pathDump):
        print("Loading external model...")
        try:
            file = open(pathDump, "rb")
            model = pickle.load(file)
            file.close()
            print("\tDone!")
            return model
        except:
            print("\tFailed!")
            return None
        
    def dumpModel(self, model, pathDump):
        file = open(pathDump, "wb")
        pickle.dump(model, file)
        file.close()

class UserBased(CollaborativeFiltering):
    def __init__(self):
        CollaborativeFiltering.__init__(self, similarityMeasure=similarity.cosine)
        print("User-based collaborative filtering")
        
    def loadData(self, data):
        print("Loading training data...")
        if type(data) is dict:      # If 'data' is preferences on users for training
            self.prefs = data
            self.userList = []
            self.itemList = []
            for user in self.prefs:
                if user not in self.userList:
                    self.userList.append(user)
                for item in self.prefs[user]:
                    if item not in self.itemList:
                        self.itemList.append(item)
        elif type(data) is str:     # If 'data' is a file path of training data
            self.prefs, self.userList, self.itemList = tool.loadData(data)
            
    def buildModel(self, nSimilarUsers, pathDump):
        print("Model builder is running...")
        # Model contains Top-N similar users for each user and their similarities.
        # Model format: {user: [(neighbor, similarity), ...], ...}
        model = {}
        for user in self.prefs.keys():
            similarities = {}
            for other in self.prefs.keys():
                if user == other:
                    continue
                similarities[other] = self.similarityMeasure(self.prefs[user], self.prefs[other])
            sortedList = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            mostSimilarUsers = sortedList[0: nSimilarUsers]
            model[user] = mostSimilarUsers
        self.dumpModel(model, pathDump)
        print("\tComplete!")
        return model
        
    def getPredictedRating(self, model, user, item):
        similarNeighbors = model[user]
        
        meanRating = np.mean([r for r in self.prefs[user].values()])
        weightedSum = 0
        normalizingFactor = 0
        for neighborInfo in similarNeighbors:
            neighbor = neighborInfo[0]
            similarity = neighborInfo[1]
            if item not in self.prefs[user] and item in self.prefs[neighbor]:
                meanRatingOfNeighbor = np.mean([r for r in self.prefs[neighbor].values()])
                weightedSum += similarity * (self.prefs[neighbor][item] - meanRatingOfNeighbor)
                normalizingFactor += np.abs(similarity)
                
        if normalizingFactor == 0:
            return 0
        return meanRating + (weightedSum / normalizingFactor)
    
    def Recommendation(self, model, user, topN):
        predictedScores = {}
        for item in self.itemList:
            if item not in self.prefs[user]:
                predictedScores[item] = self.getPredictedRating(model, user, item)
        sortedCandidates = sorted(predictedScores.items(), key=lambda x: x[1], reverse=True)
        recommendation = [sortedCandidates[i][0] for i in range(topN)]
        return recommendation

class ItemBased(object):
    '''
    For more details, reference the following paper:
    Item-based Top-N Recommendation Algorithms - Deshpande, Karypis (TOIS 2004)
    '''
    def __init__(self, topN = 10, simItems = 20):
        print("Item-based collaborative filtering")
        self.TOPN = topN
        self.SIM_ITEMS = simItems
        self.model = {}
        
    def loadData(self, data):
        print("Loading training data...")
        if type(data) is dict:      # If 'data' is preferences on items
            self.dataTrain = tool.transposePrefs(data)
        elif type(data) is str:     # If 'data' is a file path of training data
            self.dataTrain, itemList, userList = tool.loadData(data, inv = True)
        print("\tDone!")
        
    def buildModel(self):                                   # Build model which contains item-item similarities
        print("Model builder is running...")
        for itemA in self.dataTrain.keys():
            similarities = {}                               # Contains item-item similarities between 'itemA' and other items
            for itemB in self.dataTrain.keys():
                if itemA == itemB:
                    continue
                similarities[itemB] = similarity.cosine(self.dataTrain[itemA], self.dataTrain[itemB])
            sortedItems = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            sortedIids = [sortedItems[i][0] for i in range(self.SIM_ITEMS)]
            self.model[itemA] = similarities                # FINAL ROW VECTOR FEEDS INTO MODEL
        
        for c in self.model.keys():                         # FOR EACH COLUMN,
            COLSUM = 0
            targetRows = []                                 # CONTAINS ITEM IDs WHICH HAVE SIMILARITY VALUE WITH ITEM 'c'
            for r in self.model.keys():
                if c in self.model[r]:
                    targetRows.append(r)
                    COLSUM += self.model[r][c]
            if COLSUM > 0:
                for r in targetRows:
                    self.model[r][c] /= COLSUM              # ROW NORMALIZATION
        print("\tComplete!")
    
    def Recommendation(self, prefsOnTargetUser):            # TYPE(prefsOnTargetUser) = DICTIONARY
        scores = {}
        for itemA in self.model.keys():                     # 'itemA' IS CANDIDATE ITEM FOR TARGET USER
            score = 0
            if itemA in prefsOnTargetUser.keys():           # IF THE CANDIDATE ITEM IS IN TARGET USER'S RATING HISTORY,
                score = 0                                   # THE 'itemA' GETS EXCEPTED FROM CANDIDATE SET BY SCORING ZERO
            else:                                           # IF 'itemA' IS NOT EVALUATED BY TARGET USER,
                for itemB in self.model.keys():
                    if itemB in prefsOnTargetUser.keys():
                        score += self.model[itemA][itemB]   # INNER PRODUCT (SIM MATRIX AND TARGET USER'S BINARY VECTOR)
            scores[itemA] = score                           # SCORE FOR EACH CANDIDATE ITEM
        sortedItems = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommend = [sortedItems[i][0] for i in range(self.TOPN)]
        return recommend
        
    def loadExtModel(self, pathDump):
        print("Loading external ibcf model...")
        try:
            file = open(pathDump, "rb")
            self.model = pickle.load(file)
            file.close()
            print("\tDone!")
            return True
        except:
            print("\tFailed!")
            return False
        
    def dumpModel(self, pathDump):
        file = open(pathDump, "wb")
        pickle.dump(self.model, file)
        file.close()
        