# pyCollaborativeFiltering
User-based and Item-based Collaborative Filtering algorithms written in Python

## Specification of user-based method
* User-user similarity does not include neighbors whose similarity is zero or lower value.
* The similarity between users is multiplied by a weight according to the number of co-rated items. (Significance Weighting)

## Develop enviroment
* Language: Python3
* Prerequisite libraries: [Numpy](http://numpy.org)

## Input data format
UserID \t ItemID \t Count \n

## Usage example
### Recommendation
```python
>>> import tool
>>> trainSet, testSet = tool.LeaveOneOutSplit("/home2/movielens/movielens.dat")
>>> from recommender import UserBased
>>> ubcf = UserBased()
>>> ubcf.loadData(trainSet)
>>> model = ubcf.loadExtModel("/home2/movielens/movielens_ubcf30model.pickle")
>>> if model == None:
...     model = ubcf.buildModel(nNeighbors=30)
>>> for user in testSet.keys():
...     recommendation = ubcf.Recommendation(model, user, topN=10)
```
### Evaluation
```python
>>> import tool
>>> trainSet, testSet = tool.LeaveOneOutSplit("/home2/movielens/movielens.dat")
>>> from recommender import ItemBased
>>> ibcf = ItemBased()
>>> ibcf.loadData(trainSet)
>>> model = ibcf.loadExtModel("/home2/movielens/movielens_ibcf20model.pickle")
>>> if model == None:
...     model = ibcf.buildModel(nNeighbors=20)
>>> import evaluation
>>> precision, recall, hitrate = evaluation.evaluation(ibcf, model, testSet, topN=10)
>>> precision
0.026511134676564248
>>> recall
0.2651113467656416
>>> hitrate
0.2651113467656416
```

## References
* An Algorithmic Framework for Performing Collaborative Filtering - Herlocker, Konstan, Borchers, Riedl (SIGIR 1999)
* Item-based Top-N Recommendation Algorithms - Deshpande, Karypis (TOIS 2004)
