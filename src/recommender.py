import abc
from builtins import isinstance
import pickle

import numpy as np
import similarity
import tool


class DataType:
    Unary = 1       # Like, purchase, etc
    Binary = 2      # Like/dislike, thumb-up/thumb-down, true/false, etc
    Explicit = 3    # User-Item-Score, etc

class CollaborativeFiltering(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, dataType = DataType.Explicit):
        self.dataType = dataType
        self.prefs = None
        self.itemList = None
    
    @classmethod
    @abc.abstractmethod
    def buildModel(cls):
        raise NotImplementedError
    
    @classmethod
    @abc.abstractmethod
    def Recommendation(cls):
        raise NotImplementedError
    
    def getNearestNeighbors(self, target, simMeasure, nNeighbors = None):
        similarities = [(simMeasure(self.prefs[target], self.prefs[other]), other) for other in self.prefs if target != other]
        similarities.sort(reverse = True)
        if nNeighbors != None:
            similarities = similarities[0:nNeighbors]
        return similarities     # similarities = [(similarity, neighbor), ...]
    
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
    def __init__(self, dataType = DataType.Explicit):
        super().__init__(dataType)
        print("User-based Collaborative Filtering")
        
    def loadData(self, data):
        if isinstance(data, dict):          # If 'data' is preferences on users for training
            self.prefs = data
        elif isinstance(data, str):         # If 'data' is a file path of training data
            self.prefs = tool.loadData(data)
        self.itemList = {}
        for user in self.prefs:
            for item in self.prefs[user]:
                self.itemList[item] = None
    
    def buildModel(self, simMeasure = similarity.cosine_intersection, nNeighbors = None, pathDump = None):
        # Model contains top-K similar users for each user and their similarities.
        # Model format: {user: [(similarity, neighbor), ...], ...}
        model = self.loadExtModel(pathDump)
        if model != None:
            return model
        
        print("Model builder is running...")
        model = {}
        for user in self.prefs:
            model[user] = self.getNearestNeighbors(user, simMeasure, nNeighbors)
            
        if pathDump != None:
            self.dumpModel(model, pathDump)
        print("\tComplete!")
        return model
    
    def getPredictedRating(self, user, item, nearestNeighbors):
        if self.dataType == DataType.Unary:
            if item in self.prefs[user]:
                return 1.0
            similarities = [similarity for neighbor, similarity in nearestNeighbors.items() if item in self.prefs[neighbor]]
            if len(similarities) == 0:
                return 0.0
            return np.mean(similarities)
        elif self.dataType == DataType.Binary:
            # Not supported yet
            return 0.0
        elif self.dataType == DataType.Explicit:
            if item in self.prefs[user]:
                return self.prefs[user][item]
            meanRating = np.mean([score for score in self.prefs[user].values()])
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
    
    def Recommendation(self, user, simMeasure = similarity.cosine_intersection, nNeighbors = 50, model = None, topN = None):
        if model != None:
            '''
            If a user-user similarity model is given,
            other parameters such as similarity measure and the number of nearest neighbors are ignored.
            It is because that the similarity measure and # of neighbors are determined during the model building.
            '''
            candidateItems = {}         # List of candidate items to be recommended
            nearestNeighbors = {}       # List of nearest neighbors
            for similarity, neighbor in model[user]:
                if similarity <= 0:
                    break
                nearestNeighbors[neighbor] = similarity
                for item in self.prefs[neighbor]:
                    candidateItems[item] = None
            predictedScores = [(self.getPredictedRating(user, item, nearestNeighbors), item)
                               for item in candidateItems if item not in self.prefs[user]]
        else:
            '''
            If a model is not given, the recommendation task follows the original CF method.
            After finding K-nearest neighbors who have rating history on the item,
            the recommendation is made by using their similarities.
            '''
            predictedScores = []        # predictedScores = [(predicted_score, item), ...]
            similarities = self.getNearestNeighbors(user, simMeasure)   # similarities = [(similarity, neighbor), ...]
            for item in self.itemList:
                if item in self.prefs[user]:
                    continue
                itemRaters = {}         # Nearest neighbors who rated on the item
                for similarity, neighbor in similarities:
                    if similarity <= 0 or len(itemRaters) == nNeighbors:
                        break
                    if item in self.prefs[neighbor]:
                        itemRaters[neighbor] = similarity
                predictedScores.append((self.getPredictedRating(user, item, itemRaters), item))
        
        predictedScores.sort(reverse = True)
        recommendation = [item for similarity, item in predictedScores]
        if topN != None:
            recommendation = recommendation[0:topN]
        return recommendation
    
class ItemBased(CollaborativeFiltering):
    '''
    For more details, reference the following paper:
    Item-based Top-N Recommendation Algorithms - Deshpande, Karypis (TOIS 2004)
    '''
    def __init__(self, dataType = DataType.Explicit):
        super().__init__(dataType)
        print("Item-based Collaborative Filtering")
        
    def loadData(self, data):
        if isinstance(data, dict):          # If 'data' is preferences on users for training
            self.prefsOnUser = data
            self.prefs = tool.transposePrefs(self.prefsOnUser)
        elif isinstance(data, str):         # If 'data' is a file path of training data
            self.prefsOnUser = tool.loadData(data)
            self.prefs = tool.transposePrefs(self.prefsOnUser)
        self.itemList = self.prefs.keys()
    
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
            model.setdefault(item, {})
            correlations = self.getNearestNeighbors(item, simMeasure, nNeighbors)
            for correlation, neighbor in correlations:
                model[item][neighbor] = correlation
        
        # Row normalization
        for c in model:
            COLSUM = sum([model[c][r] for r in model[c]])
            if COLSUM > 0:
                for r in model[c]:
                    model[c][r] /= COLSUM
        
        if pathDump != None:
            self.dumpModel(model, pathDump)
        print("\tComplete!")
        return model
    
    def Recommendation(self, user, simMeasure = similarity.cosine, nNeighbors = 20, model = None, topN = None):
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
            if candidate in self.prefsOnUser[user]:
                continue
            
            if model != None:
                correlations = model[candidate]
            else:
                correlations = self.getNearestNeighbors(candidate, simMeasure, nNeighbors)
            
            score = sum([correlations[candidate] * self.prefsOnUser[user][item]
                         for item in self.prefsOnUser[user] if candidate in correlations])
            predictedScores.append((score, candidate))
        predictedScores.sort(reverse = True)
        recommendation = [item for similarity, item in predictedScores]
        if topN != None:
            recommendation = recommendation[0:topN]
        return recommendation
