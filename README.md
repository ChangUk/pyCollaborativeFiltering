# pyCollaborativeFiltering
User-based and Item-based Collaborative Filtering algorithms written in Python

## Develop enviroment
* Language: Python3
* Prerequisite libraries: [Scipy](http://scipy.org), [Numpy](http://numpy.org)

## Input data format
UserID \t ItemID \t Count \n

## Recommendation example
```python
>>> import tool
>>> trainSet, testSet = tool.LeaveNOutSplit("/home2/test/shopping.dat", N=1)
>>> from recommender import ItemBased, UserBased
>>> ubcf = UserBased()
>>> ubcf.loadData(trainSet)
>>> model = ubcf.loadExtModel("/home2/test/shopping_ubcf_model.pickle")
>>> if model == None:
...     model = ubcf.buildModel(nNeighbors=100)
>>> for user in testSet.keys():
...     recommendation = ubcf.Recommendation(model, user, topN=10)
```

## References
* Item-based Top-N Recommendation Algorithms - Deshpande, Karypis (TOIS 2004)
