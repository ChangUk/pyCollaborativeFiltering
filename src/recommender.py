import os
import pickle

import evaluation
import numpy as np
import similarity
import tool


class CollaborativeFiltering:
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
        try:
            file = open(pathDump, "wb")
            pickle.dump(model, file)
            file.close()
        except IOError as e:
            print(e)

class UserBased(CollaborativeFiltering):
    '''
    For more details, reference the following paper:
    An Algorithmic Framework for Performing Collaborative Filtering - Herlocker, Konstan, Borchers, Riedl (SIGIR 1999)
    '''
    def __init__(self):
        print("User-based Collaborative Filtering")
        self.title = "ubcf"
        self.prefs = {}             # Training data format: {user: {item: rating, ...}, ...}
        self.itemList = []
        
    def loadData(self, data):
        print("Loading training data...")
        if type(data) is dict:      # If 'data' is preferences on users for training
            self.prefs = data
        elif type(data) is str:     # If 'data' is a file path of training data
            self.prefs = tool.loadData(data)
        self.itemList = self.getSubjectList(self.prefs)
        print("\tDone!")
        
    def buildModel(self, data, similarityMeasure = similarity.cosineForInterSet, nNeighbors = 50, pathDump = None):
        self.loadData(data)
        
        # Model contains top-K similar users for each user and their similarities.
        # Model format: {user: {neighbor: similarity, ...}, ...}
        model = self.loadExtModel(pathDump)
        if model != None:
            return model
        
        print("Model builder is running...")
        model = {}
        for user in self.prefs:
            model.setdefault(user, {})
            similarities = [(similarityMeasure(self.prefs[user], self.prefs[other]), other) for other in self.prefs if user != other]
            similarities.sort(reverse = True)
            for similarity, neighbor in similarities[0:nNeighbors]:
                model[user][neighbor] = similarity
        if pathDump != None and type(pathDump) is str:
            self.dumpModel(model, pathDump)
        print("\tComplete!")
        return model
    
    def getPredictedRating(self, model, user, item):
        if item in self.prefs[user]:
            return self.prefs[user][item]
        meanRating = np.mean([r for r in self.prefs[user].values()])
        weightedSum = 0
        normalizingFactor = 0
        for neighbor in model[user]:
            similarity = model[user][neighbor]
            if similarity <= 0:
                continue
            if item in self.prefs[neighbor]:
                meanRatingOfNeighbor = np.mean([r for r in self.prefs[neighbor].values()])
                weightedSum += similarity * (self.prefs[neighbor][item] - meanRatingOfNeighbor)
                weightedSum += similarity * self.prefs[neighbor][item]
                normalizingFactor += np.abs(similarity)
        if normalizingFactor == 0:
            return 0
        return meanRating + (weightedSum / normalizingFactor)
        return weightedSum / normalizingFactor
    
    def Recommendation(self, model, user, topN = 10):
        predictedScores = [(self.getPredictedRating(model, user, item), item) for item in self.itemList if item not in self.prefs[user]]
        predictedScores.sort(reverse = True)
        recommendation = [item for similarity, item in predictedScores[0:topN]]
        return recommendation
    
class ItemBased(CollaborativeFiltering):
    '''
    For more details, reference the following paper:
    Item-based Top-N Recommendation Algorithms - Deshpande, Karypis (TOIS 2004)
    '''
    def __init__(self):
        print("Item-based Collaborative Filtering")
        self.title = "ibcf"
        self.prefs = {}             # Training data format: {item: {user: rating, ...}, ...}
        self.prefsOnUser = {}       # Transposed data format: {user: {item: rating, ...}, ...}
        self.itemList = []
        
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
        
    def buildModel(self, data, similarityMeasure = similarity.cosine, nNeighbors = 20, pathDump = None):
        '''
        The j-th column of the model(matrix) stores the k most similar items to item j.
        But, in this project, the model is not matrix but dictionary type.
        '''
        self.loadData(data)
        
        # Model contains top-K similar items for each item and their similarities.
        # Model format: {item: {neighbor: similarity, ...}, ...}
        model = self.loadExtModel(pathDump)
        if model != None:
            return model
        
        print("Model builder is running...")
        model = {}
        for item in self.prefs:
            model.setdefault(item, {})
            similarities = [(similarityMeasure(self.prefs[item], self.prefs[other]), other) for other in self.prefs if item != other]
            similarities.sort(reverse = True)
            for similarity, neighbor in similarities[0:nNeighbors]:
                model[item][neighbor] = similarity
        
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
        predictedScores = []
        for candidate in self.itemList:
            if candidate in self.prefsOnUser[user]: continue
            score = sum([model[item][candidate] * self.prefsOnUser[user][item] for item in self.prefsOnUser[user] if candidate in model[item]])
            predictedScores.append((score, candidate))
        predictedScores.sort(reverse = True)
        recommendation = [item for similarity, item in predictedScores[0:topN]]
        return recommendation
