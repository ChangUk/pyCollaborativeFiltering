import abc
from builtins import isinstance
import pickle

import numpy as np
import similarity
import tool


class CollaborativeFiltering(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self):
        self.prefs = {}
    
    @classmethod
    @abc.abstractmethod
    def buildModel(cls):
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def Recommendation(cls):
        raise NotImplementedError
    
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
        super().__init__()
        print("User-based Collaborative Filtering")
        
    def loadData(self, data):
        if isinstance(data, dict):          # If 'data' is preferences on users for training
            self.prefs = data
        elif isinstance(data, str):         # If 'data' is a file path of training data
            self.prefs = tool.loadData(data)
            
    def getNearestNeighbors(self, targetUser, simMeasure = similarity.cosine_intersection, nNeighbors = 50, useOnlyPositives = True):
        nearestNeighbors = {}
        similarities = [(simMeasure(self.prefs[targetUser], self.prefs[user]), user) for user in self.prefs if targetUser != user]
        similarities.sort(reverse = True)
        for similarity, neighbor in similarities[0:nNeighbors]:
            if useOnlyPositives == True and similarity <= 0:
                continue
            nearestNeighbors[neighbor] = similarity
        return nearestNeighbors
    
    def buildModel(self, simMeasure = similarity.cosine_intersection, nNeighbors = 50, pathDump = None):
        # Model contains top-K similar users for each user and their similarities.
        # Model format: {user: {neighbor: similarity, ...}, ...}
        model = self.loadExtModel(pathDump)
        if model != None:
            return model
        
        print("Model builder is running...")
        model = {}
        for user in self.prefs:
            model[user] = self.getNearestNeighbors(user, simMeasure, nNeighbors)
            
        if isinstance(pathDump, str):
            self.dumpModel(model, pathDump)
        print("\tComplete!")
        return model
    
    def getPredictedRating(self, user, item, nearestNeighbors, binaryMode = False):
        if binaryMode == True:
            if item in self.prefs[user]:
                return 1.0
            similarities = [similarity for neighbor, similarity in nearestNeighbors.items() if item in self.prefs[neighbor]]
            if len(similarities) == 0:
                return 0.0
            return np.mean(similarities)
        else:
            if item in self.prefs[user]:
                return self.prefs[user][item]
            meanRating = np.mean([r for r in self.prefs[user].values()])
            weightedSum = 0
            normalizingFactor = 0
            for neighbor, similarity in nearestNeighbors.items():
                if item not in self.prefs[neighbor]:
                    continue
                meanRatingOfNeighbor = np.mean([r for r in self.prefs[neighbor].values()])
                weightedSum += similarity * (self.prefs[neighbor][item] - meanRatingOfNeighbor)
                normalizingFactor += np.abs(similarity)
            if normalizingFactor == 0:
                return 0
            return meanRating + (weightedSum / normalizingFactor)
    
    def Recommendation(self, user, model, topN = 10, binaryMode = False, useOnlyPositives = True):
        if isinstance(model, dict):         # model = {user: {neighbor_user: similarity, ...}, ...}
            nearestNeighbors = model[user]
        elif isinstance(model, tuple):      # model = (similarityMeasure, nNearestNeighbors)
            nearestNeighbors = self.getNearestNeighbors(user, model[0], model[1], useOnlyPositives)
        else:
            return []
        
        # Get list of candidate items to be recommended
        candidateItems = {}
        for neighbor in nearestNeighbors:
            for item in self.prefs[neighbor]:
                candidateItems[item] = None
        
        predictedScores = [(self.getPredictedRating(user, item, nearestNeighbors, binaryMode), item)
                           for item in candidateItems if item not in self.prefs[user]]
        predictedScores.sort(reverse = True)
        recommendation = [item for similarity, item in predictedScores[0:topN]]
        return recommendation
    
class ItemBased(CollaborativeFiltering):
    '''
    For more details, reference the following paper:
    Item-based Top-N Recommendation Algorithms - Deshpande, Karypis (TOIS 2004)
    '''
    def __init__(self):
        super().__init__()
        print("Item-based Collaborative Filtering")
        
    def loadData(self, data):
        if isinstance(data, dict):          # If 'data' is preferences on users for training
            self.prefsOnUser = data
            self.prefs = tool.transposePrefs(self.prefsOnUser)
        elif isinstance(data, str):         # If 'data' is a file path of training data
            self.prefsOnUser = tool.loadData(data)
            self.prefs = tool.transposePrefs(self.prefsOnUser)
        self.itemList = self.prefs.keys()
    
    def getNearestNeighbors(self, targetItem, simMeasure = similarity.cosine, nNeighbors = 20):
        nearestNeighbors = {}
        similarities = [(simMeasure(self.prefs[targetItem], self.prefs[item]), item) for item in self.prefs if targetItem != item]
        similarities.sort(reverse = True)
        for similarity, neighbor in similarities[0:nNeighbors]:
            nearestNeighbors[neighbor] = similarity
        return nearestNeighbors
    
    def buildModel(self, simMeasure = similarity.cosine, nNeighbors = 20, pathDump = None):
        '''
        The j-th column of the model(matrix) stores the k most similar items to item j.
        But, in this project, the model is not matrix but dictionary type.
        '''
        # Model contains top-K similar items for each item and their similarities.
        # Model format: {item: {neighbor: similarity, ...}, ...}
        model = self.loadExtModel(pathDump)
        if model != None:
            return model
        
        print("Model builder is running...")
        model = {}
        for item in self.prefs:
            model[item] = self.getNearestNeighbors(item, simMeasure, nNeighbors)
        
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
    
    def Recommendation(self, user, model, topN = 10, binaryMode = None, useOnlyPositives = None):
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
            
            if isinstance(model, dict):         # model = {item: {neighbor_item: similarity, ...}, ...}
                nearestNeighbors = model[candidate]
            elif isinstance(model, tuple):      # model = (similarityMeasure, nNearestNeighbors)
                nearestNeighbors = self.getNearestNeighbors(candidate, model[0], model[1])
                
            score = sum([nearestNeighbors[candidate] * self.prefsOnUser[user][item] for item in self.prefsOnUser[user] if candidate in nearestNeighbors])
            predictedScores.append((score, candidate))
        predictedScores.sort(reverse = True)
        recommendation = [item for similarity, item in predictedScores[0:topN]]
        return recommendation
