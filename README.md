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
>>> ibcf = ItemBased(trainSet)
>>> model = ibcf.buildModel(nNeighbors=20)
>>> for user in testSet.keys():
...     recommendation = ibcf.Recommendation(model, user, topN=10)
```
### Evaluation
```python
>>> import tool
>>> trainSet = tool.loadData("/home2/movielens/u1.base")
>>> testSet = tool.loadData("/home2/movielens/u1.test")
>>> from recommender import UserBased
>>> ubcf = UserBased(trainSet)
>>> model = ubcf.buildModel(nNeighbors=30)
>>> import validation
>>> result = validation.evaluateRecommender(ubcf, model, testSet, topN=10)
>>> print(result)
{'precision': 0.050980392156862, 'recall': 0.009698538130460, 'hitrate': 0.5098039215686}
```

## References
* An Algorithmic Framework for Performing Collaborative Filtering - Herlocker, Konstan, Borchers, Riedl (SIGIR 1999)
* Item-based Top-N Recommendation Algorithms - Deshpande, Karypis (TOIS 2004)
