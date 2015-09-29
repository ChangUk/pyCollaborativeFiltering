# pyCollaborativeFiltering
User-based and Item-based Collaborative Filtering algorithms written in Python

## Develop enviroment
* Language: Python3
* IDE: Eclipse PyDev
* Prerequisite libraries: [Numpy](http://numpy.org)

## Specification of user-based method
* If you use a built-up model, the recommender system considers only the nearest neighbors existing in the model. Otherwise, the recommender looks for K-similar neighbors for each target user by using the given similarity measure and the number(K) of nearest neighbors.
* In unary data, the predicted score of the item is the average similarity of the nearest neighbors who rated on the item.
* User similarity does not include those of neighbors whose similarity is zero or lower value.
* The cosine similarity basically considers only co-rated items. (Another measures such as the basic cosine similarity and Pearson correlation coefficient are also applicable.)

## Input data format
`UserID \t ItemID \t Rating \n`

## Usage example
### User-based Recommendation
```python
>>> import tool
>>> data = tool.loadData("/home/changuk/data/MovieLens/movielens.dat")
>>> from recommender import UserBased
>>> ubcf = UserBased()
>>> ubcf.loadData(data)
>>> for user in data.keys():
...     recommendation = ubcf.Recommendation(user, similarity=cosine, nNeighbors=30, topN=10)
```
### Item-based Recommendation
```python
>>> import tool
>>> data = tool.loadData("/home/changuk/data/MovieLens/movielens.dat")
>>> from recommender import ItemBased
>>> ibcf = ItemBased()
>>> ibcf.loadData(data)
>>> model = ibcf.buildModel(nNeighbors=20)
>>> for user in data.keys():
...     recommendation = ibcf.Recommendation(user, model=model, topN=10)
```
### Validation
```python
>>> import tool
>>> trainSet = tool.loadData("/home/changuk/data/MovieLens/u1.base")
>>> testSet = tool.loadData("/home/changuk/data/MovieLens/u1.test")
>>> from recommender import UserBased
>>> ubcf = UserBased()
>>> ubcf.loadData(trainSet)
>>> model = ubcf.buildModel(nNeighbors=30)
>>> import validation
>>> result = validation.evaluateRecommender(testSet, ubcf, model=model, topN=10)
>>> print(result)
{'precision': 0.050980392156862, 'recall': 0.009698538130460, 'hitrate': 0.5098039215686}
```

## TODO list
* Support binary data
* Implement similarity normalization in Item-based CF

## References
* [An Algorithmic Framework for Performing Collaborative Filtering - Herlocker, Konstan, Borchers, Riedl (SIGIR 1999)](http://files.grouplens.org/papers/algs.pdf)
* [Item-based Top-N Recommendation Algorithms - Deshpande, Karypis (TOIS 2004)](http://glaros.dtc.umn.edu/gkhome/fetch/papers/itemrsTOIS04.pdf)
* [https://en.wikipedia.org/wiki/Collaborative_filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
