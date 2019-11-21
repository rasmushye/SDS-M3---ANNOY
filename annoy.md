



# Approximate Nearest Neighbors Oh Yeah (ANNOY)


## Course:
### Social Data Science 2019 - M3: Deep Learning - Portfolio

![alt_text](https://warehouse-camo.cmh1.psfhosted.org/378b31074a83939a9cf068b92be7ed28b5a02448/68747470733a2f2f7261772e6769746875622e636f6d2f73706f746966792f616e6e6f792f6d61737465722f616e6e2e706e67 "image_tooltip")

## What is ANNOY

ANNOY (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings to search for points in space that are close to a given query point eg. a specific point of interest. 

It also creates large data structures that are mapped into memory so that many processes may share the same data. The nearest neighbors search is a similarity problem to find the closest points to a given instance.

### Definition

Out of a set of _n_ instances _P = {p1, ..., pn}_ in some metric space _X_ the nearest neighbors search computes the closest instance _q_ to an instance _p_ under some measurement function.

![alt_text]( "image_tooltip")
The idea behind the approximate nearest neighbor search is to speed up the computation of the nearest neighbors, the exact algorithm can not be improved. The only way to speed up the computation is to allow errors[^1]. 

ANNOY (Approximate Nearest Neighbors Oh Yeah) is an algorithm based on random projections and trees. It was developed by Erik Bernhardsson in 2015 working at that time at Spotify. ANNOY is designed to search in date sets up to 100 to 1000 dense dimensions. To compute the nearest neighbors it is splitting the set of points into half and is doing this recursively until each set is having _k_ items. Usually _k_ should be around 100 _(see figure below)_. 




![alt_text]( "image_tooltip")
An approximate nearest neighbors search algorithm is allowed to return points, whose distance from the query is at most _c_ times the distance from the query to its nearest points.

The appeal of this approach is that, in many cases, an approximate nearest neighbors is almost as good as the exact one. In particular, if the distance measure accurately captures the notion of user quality, then small differences in the distance should not matter.

ANN classification output represents a class membership. An object is classified by the majority votes of its neighbors. The object is assigned to a particular class that is most common among its _k_ nearest neighbors. _k_ is a positive integer, typically small[^2].  \



# Why use ANNOY?

Annoy is almost as fast as the fastest libraries, but there is actually another feature that really sets Annoy apart: it has the ability to use static files as indexes. In particular, this means you can share index across processes. Annoy also decouples creating indexes from loading them, so you can pass around indexes as files and map them into memory quickly. Another nice thing of Annoy is that it tries to minimize memory footprint so the indexes are quite small[^3]. 

An exact algorithm to solve the similarity problem of the k-nearest neighbors is given with a runtime of _O(n2 ∗ m)_ with _n_ instances and _m_ dimensions. Given the curse of dimensionality the runtime of this algorithm is critical, approximate algorithms can be helpful to find a solution, while using less time and computational power. 

Current approximate nearest neighbors search algorithms are often dealing with a lack of support of very high dimensional but very sparse datasets. The benefits of a sparse dataset is that only the non-zero features of every instance need to be stored.

Algorithms like Annoy accept sparse datasets as an input but needs a lot of memory. The higher the dimensions are the more is the runtime of the algorithm depended on the number of dimensions and not as usually assumed by the number of input instances[^4]. 

This is where approximate nearest neighbors shines: returning approximate results but blazingly quickly. Many times you don’t need exact optimal results[^5]. 


# What’s the difference between nearest neighbors algorithms?

There are numerous variants of the Nearest Neighbors Search (NNS) problem and the two most well-known are the k-nearest neighbors search and the approximate nearest neighbors search.

**k-nearest neighbors **

k-nearest neighbors search identifies the top k nearest neighbors to the query. This technique is commonly used in predictive analytics to estimate or classify a point based on the consensus of its neighbors. k-nearest neighbors graphs are graphs in which every point is connected to its k-nearest neighbors.

**Approximate nearest neighbors (ANN)**

In some applications it may be acceptable to retrieve a "good guess" of the nearest neighbors. In those cases, we can use an algorithm which doesn't guarantee to return the actual nearest neighbors in every case, in return for improved speed or memory savings. Often such an algorithm will find the nearest neighbors in a majority of cases, but this depends strongly on the dataset being queried. ANNOY is an example of ANN[^6]. 

The difference between KNN and ANN is that in the prediction phase, all training points are involved in searching k-nearest neighbors in the KNN algorithm, but in ANN this search starts only on a small subset of candidates points[^7]. 


# What problems can occur when working with ANNOY 

Working with ANNOY has great potential in optimizing runtime, when executing Nearest Neighbors algorithm, but also the risk of decreasing runtime, if not handled properly. This is where ‘the curse of dimensionality’ comes in. If ANNOY is working with a large display of dimensionality, it may affect runtime in a bad way, opposite from the number of instances, that does not affect/influence runtime in the same way.

**Curse of dimensionality**

The curse of dimensionality was first mentioned by Bellman in 1957. The well known notion of big O to specify the theoretical runtime of an algorithm focuses on the number of input elements. Bellman showed that for a computation of _n_ instances with _m_ dimensions that the runtime is more depended on the number of dimensions if _m >> n_. For example if _n = 1000_ and _m = 100000_ then a quadratic factor for the runtime of the prediction for the k-nearest neighbors with _O(n2 ∗ m)_ is not influencing the runtime that drastically in comparison to the influence of the dimensions[^8]. 


# Get your hands dirty

To install, simply do `!pip install -qq annoy` to pull down the latest version from PyPI.

```python
from annoy import AnnoyIndex
import random

f = 40
t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
for i in range(1000):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_item(i, v)

t.build(10) # 10 trees
t.save('test.ann')

# ...

u = AnnoyIndex(f, 'angular')
u.load('test.ann') # super fast, will just mmap the file
print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors
```
Approximate Nearest neighbors search (ANNS) is a fundamental and essential operation in applications from many domains, such as databases, machine learning, multimedia, and computer vision.

Despite much research on this problem, it is commonly believed that it is very costly to find the exact nearest neighbors in high dimensional Euclidean space, due to the curse of dimensionality. Experiments showed that exact methods can rarely outperform the brute-force linear scan method when dimensionality is high (e.g., more than 20). Nevertheless, returning sufficiently nearby objects, referred to as approximate nearest neighbors search (ANNS), can be performed efficiently and are sufficiently useful for many practical problems[^9]. 


## Practical use of ANNOY: 

Approximate Nearest Neighbors search is used at Spotify for music recommendations. After running matrix factorization algorithms, every user/song can be represented as a vector in f-dimensional space. This ANN library helps with searching for similar users/song. Spotify has a database of millions of tracks in a high-dimensional space, with millions of users, so memory usage and decreasing the runtime, for every query is a prime concern for a system like Spotify’ recommender system[^10] [^11]. 


# Conclusion

ANNOY solves a fundamental problem and has both significant theoretical value and empowers a diverse range of applications. It is widely believed that there is no practically competitive algorithm to answer exact ANN queries in sublinear time with linear sized index. 


<!-- Footnotes themselves at the bottom. -->
## Sources

^1: http://www.bioinf.uni-freiburg.de/Lehre/Theses/MA_Joachim_Wolff.pdf

^2: https://apacheignite.readme.io/docs/ann-approximate-nearest-neighbor

^3: https://github.com/spotify/annoy

^4: http://www.bioinf.uni-freiburg.de/Lehre/Theses/MA_Joachim_Wolff.pdf 

^5: https://medium.com/@kevin_yang/simple-approximate-nearest-neighbors-in-python-with-annoy-and-lmdb-e8a701baf905

^6: https://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor

^7: https://apacheignite.readme.io/docs/ann-approximate-nearest-neighbor

^8: http://www.bioinf.uni-freiburg.de/Lehre/Theses/MA_Joachim_Wolff.pdf

^9: https://arxiv.org/pdf/1610.02455.pdf

^10: https://github.com/spotify/annoy 

^11: https://www.youtube.com/watch?v=QkCCyLW0ehU


<!-- Docs to Markdown version 1.0β17 -->
