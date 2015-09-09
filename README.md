# pyCollaborativeFiltering
User-based and Item-based Collaborative Filtering algorithms written in Python

## Develop enviroment
* Language: Python3
* Prerequisite libraries: [Numpy](http://numpy.org)

## Specification of user-based method
* User-user similarity does not include neighbors whose similarity is zero or lower value.
* The similarity between users is multiplied by a weight according to the number of co-rated items. (Significance Weighting)
* The algorithm basically uses the cosine similarity considering only co-rated items. (Another measure such as Pearson is also applicable by setting parameter.)

## Input data format
UserID \t ItemID \t Count \n

## Usage example
### Recommendation
```python
>>> import tool
>>> trainSet, testSet = tool.LeaveOneOutSplit("/home2/movielens/movielens.dat")
>>> from recommender import ItemBased
>>> ibcf = ItemBased()
>>> model = ibcf.buildModel(trainSet, nNeighbors=20)
>>> for user in testSet.keys():
...     recommendation = ibcf.Recommendation(model, user, topN=10)
```
### Evaluation
```python
>>> import tool
>>> trainSet = tool.loadData("/home2/movielens/u1.base")
>>> testSet = tool.loadData("/home2/movielens/u1.test")
>>> from recommender import UserBased
>>> ubcf = UserBased()
>>> model = ubcf.buildModel(trainSet, nNeighbors=30)
>>> import evaluation
>>> precision, recall, hitrate = evaluation.evaluation(ubcf, model, testSet, topN=10)
>>> print((precision, recall, hitrate))
(0.05163398692810463, 0.010009830619733972, 0.5163398692810458)
```

## References
* An Algorithmic Framework for Performing Collaborative Filtering - Herlocker, Konstan, Borchers, Riedl (SIGIR 1999)
* Item-based Top-N Recommendation Algorithms - Deshpande, Karypis (TOIS 2004)
