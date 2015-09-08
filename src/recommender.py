import pickle

import numpy as np
import similarity
import tool


class CollaborativeFiltering:
    def __init__(self, title, similarityMeasure):
        self.title = title
        self.prefs = {}     # Training data
        self.similarityMeasure = similarityMeasure
        
    def viewStatistics(self):
        objList = []        # Users in user-based, items in item-based
        subjList = []       # Items in user-based, users in item-based
        for obj in self.prefs.keys():
            if obj not in objList:
                objList.append(obj)
            for subj in self.prefs[obj].keys():
                if subj not in subjList:
                    subjList.append(subj)
        print("Data statistics")
        print("\t* Object: " + str(len(objList)))
        print("\t* Subject: " + str(len(subjList)))
        
    def getSubjectList(self, prefs):
        subjList = []
        for obj in prefs:
            for subj in prefs[obj]:
                if subj not in subjList:
                    subjList.append(subj)
        return subjList
        
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
    '''
    For more details, reference the following paper:
    An Algorithmic Framework for Performing Collaborative Filtering - Herlocker, Konstan, Borchers, Riedl (SIGIR 1999)
    '''
    def __init__(self):
        CollaborativeFiltering.__init__(self, title="ubcf", similarityMeasure=similarity.pearson)
        self.itemList = []
        print("User-based Collaborative Filtering")
        
    def loadData(self, data):
        print("Loading training data...")
        if type(data) is dict:      # If 'data' is preferences on users for training
            self.prefs = data
        elif type(data) is str:     # If 'data' is a file path of training data
            self.prefs = tool.loadData(data)
        self.itemList = self.getSubjectList(self.prefs)
        print("\tDone!")
        
    def buildModel(self, nNearestNeighbors = 50, pathDump = None):
        print("Model builder is running...")
        # Model contains top-K similar users for each user and their similarities.
        # Model format: {user: {neighbor: similarity, ...}, ...}
        model = {}
        for user in self.prefs.keys():
            similarities = {}
            for other in self.prefs.keys():
                if user == other: continue
                similarities[other] = self.similarityMeasure(self.prefs[user], self.prefs[other])
            sortedList = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            mostSimilarUsers = sortedList[0: nNearestNeighbors]
            model[user] = {}
            for neighbor, similarity in mostSimilarUsers:
                # Ignore similarity of zero or lower
                if similarity <= 0: continue
                model[user][neighbor] = similarity
        
        if pathDump != None:
            self.dumpModel(model, pathDump)
        print("\tComplete!")
        return model
    
    def getPredictedRating(self, model, user, item):
        meanRating = np.mean([r for r in self.prefs[user].values()])
        weightedSum = 0
        normalizingFactor = 0
        for similarUser in model[user].keys():
            if item not in self.prefs[user] and item in self.prefs[similarUser]:
                similarity = model[user][similarUser]
                meanRatingOfNeighbor = np.mean([r for r in self.prefs[similarUser].values()])
                weightedSum += similarity * (self.prefs[similarUser][item] - meanRatingOfNeighbor)
                normalizingFactor += np.abs(similarity)
        if normalizingFactor == 0:
            return 0
        return meanRating + (weightedSum / normalizingFactor)
    
    def Recommendation(self, model, user, topN = 10):
        predictedScores = {}
        for item in self.itemList:
            if item not in self.prefs[user]:
                predictedScores[item] = self.getPredictedRating(model, user, item)
        sortedCandidates = sorted(predictedScores.items(), key=lambda x: x[1], reverse=True)
        recommendation = [sortedCandidates[i][0] for i in range(len(sortedCandidates))]
        return recommendation[0:topN]
    
class ItemBased(CollaborativeFiltering):
    '''
    For more details, reference the following paper:
    Item-based Top-N Recommendation Algorithms - Deshpande, Karypis (TOIS 2004)
    '''
    def __init__(self):
        CollaborativeFiltering.__init__(self, title="ibcf", similarityMeasure=similarity.cosine)
        self.prefsOnUser = {}
        self.itemList = []
        print("Item-based Collaborative Filtering")
        
    def loadData(self, data):
        print("Loading training data...")
        if type(data) is dict:      # If 'data' is preferences on users for training
            self.prefsOnUser = data
            self.prefs = tool.transposePrefs(self.prefsOnUser)
        elif type(data) is str:     # If 'data' is a file path of training data
            self.prefsOnUser = tool.loadData(data)
            self.prefs = tool.transposePrefs(self.prefsOnUser)
        self.itemList = self.prefs.keys()
        print("\tDone!")
        
    def buildModel(self, nNeighbors = 20, pathDump = None):
        '''
        The j-th column of the model(matrix) stores the k most similar items to item j.
        But, in this project, the model is not matrix but dictionary type.
        '''
        print("Model builder is running...")
        # Model contains top-K similar items for each item and their similarities.
        # Model format: {item: {neighbor: similarity, ...}, ...}
        model = {}
        for item in self.prefs.keys():
            model.setdefault(item, {})
            similarities = {}
            for other in self.prefs.keys():
                if item == other:
                    continue
                similarities[other] = self.similarityMeasure(self.prefs[item], self.prefs[other])
            sortedList = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            mostSimilarItems = sortedList[0: nNeighbors]
            for entry in mostSimilarItems:
                model[item][entry[0]] = entry[1]
        
        # Row normalization
        for c in model.keys():
            COLSUM = 0
            for r in model[c]:
                COLSUM += model[c][r]
            if COLSUM > 0:
                for r in model[c]:
                    model[c][r] /= COLSUM
        
        if pathDump != None:
            self.dumpModel(model, pathDump)
        print("\tComplete!")
        return model
    
    def Recommendation(self, model, user, topN = 10):
        '''
        Pseudo code:
        ApplyModel(M, U, N):
            x <- MU            # i-th row, j-th column
            for j <- 1 to m:
                if U_i != 0:
                    x_i <- 0
            for j <- 1 to m:
                if x_i != among the N largest values in x:
                    x_i <- 0
        '''
        predictedScores = {}
        for candidate in self.itemList:
            if candidate in self.prefsOnUser[user]:
                continue
            score = 0
            for item in self.prefsOnUser[user]:
                if candidate in model[item]:
                    score += model[item][candidate] * self.prefsOnUser[user][item]
            predictedScores[candidate] = score
        sortedCandidates = sorted(predictedScores.items(), key=lambda x: x[1], reverse=True)
        recommendation = [sortedCandidates[i][0] for i in range(len(sortedCandidates))]
        return recommendation[0:topN]
    