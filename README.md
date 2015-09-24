# pyCollaborativeFiltering
User-based and Item-based Collaborative Filtering algorithms written in Python

## Develop enviroment
* Language: Python3
* Prerequisite libraries: [Numpy](http://numpy.org)

## Specification of user-based method
* Main difference from the original CF method is that when calculating the predicted score on the item the task finding nearest neighbors of target user precedes the work looking for users who rated the item.
* User-user similarity does not include those of neighbors whose similarity is zero or lower value.
* The similarity between users is multiplied by a weight according to the number of co-rated items. (Significance Weighting)
* The algorithm basically uses the cosine similarity considering only co-rated items. (Another measure such as Pearson is also applicable.)
* Support binary type(rated or not) of rating data.  In this case, the similarity average of the nearest neighbors who rated the item is used as a predicted score of the item.

## Input data format
`UserID \t ItemID \t Count \n`

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
* [An Algorithmic Framework for Performing Collaborative Filtering - Herlocker, Konstan, Borchers, Riedl (SIGIR 1999)](http://files.grouplens.org/papers/algs.pdf)
* [Item-based Top-N Recommendation Algorithms - Deshpande, Karypis (TOIS 2004)](http://glaros.dtc.umn.edu/gkhome/fetch/papers/itemrsTOIS04.pdf)
* [https://en.wikipedia.org/wiki/Collaborative_filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
