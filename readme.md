---
title: "Dimensionality reduction"
author: "Nikola Sokolova"
date: "14 prosince 2020"
output: pdf_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Overview
The key dimensionality reduction techniques: ISOMAP, PCA (Principal Component Analysis), and t-SNE (t-Distributed Stochastic Neighbor Embedding) are presented and compared. The focus is on experimenting with ISOMAP's parameters and contrasting its performance with PCA and t-SNE, providing a concise yet comprehensive insight into dimensionality reduction strategies.

## Background

The data are vector representations of words in a latent (unknown) high-dimensional space i.e. embedding.  (Coming from the most popular algorithm for word embedding known as word2vec[^1] by Tomas Mikolov (VUT-Brno alumni)). 

[^1]: Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In *Advances in neural information processing systems*, pages 3111-3119, 2013.

## Data
300-dimensional word2vec vector embeddings in the file *data.csv* with corresponding word labels in *labels.txt* for each line. Each of these words comes from one of 10 selected classes of synonyms, which can be recognized (and depicted) w.r.t. labels denoted in the file *colors.csv*

```{r include=FALSE, message=FALSE}
PlotPoints <- function(X, labels, colors){
  library(deldir, quietly = TRUE)
  voronoi <- deldir(X[,1], X[,2])
  plot(X[,1], X[,2], type="n", asp=1, xlab = "", ylab = "")
  points(X[,1], X[,2], pch=20, col=colors[,1], cex=1.3)
  text(X[,1], X[,2], labels = labels[,1], pos = 1, cex = 0.6)
  plot(voronoi, wlines="tess", wpoints="none", number=FALSE, add=TRUE, lty=1)
  legend("topleft", legend = sort(unique(colors[,1])), col=sort(unique(colors[,1])), pch=20, cex = 0.8)
}

Classify <- function(X, colors, kfolds = 50){
  #install.packages('tree')
  library(tree)
  #library(caret)
  set.seed(17)
  #add class
  if(!any(names(X) == "class")){X <- cbind(X, class=as.factor(colors))}
  #randomly shuffle the data
  data <-X[sample(nrow(X)),]
  #Create 10 equally size folds
  folds <- cut(seq(1,nrow(data)),breaks=kfolds,labels=FALSE)
  acc <- rep(0, times = kfolds)
  #10 fold cross validation
  for(i in 1:kfolds){
    #Segment your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- data[testIndexes, ]
    trainData <- data[-testIndexes, ]
    model <- tree(formula = class ~., data = trainData)
    pre <- predict(model, newdata=testData, y = TRUE, type="class")
    acc[i] <- sum(pre == testData$class)/length(testData$class)
  }
  return(acc)
}
```

## Tasks
1. **Load the dataset of 165 words**, each represented as a 300-dimensional vector. Each word is assigned to one of 10 clusters.
```{r}
#Load the dataset of 165 words
mydata <- read.csv('data.csv', header = FALSE)
mylabels <- read.csv('labels.txt', header = FALSE)
mycolors <- read.csv('colors.csv', header = FALSE)
```

```{r}
PlotPoints(mydata[,c(1,2)], mylabels, mycolors)
```

2. **Implement ISO-MAP dimensionality reduction procedure**.
  * Use *k*-NN method to construct the neighborhood graph (sparse matrix). # Sparse matrix is a matrix which contains very few non-zero elements. 
    - For simplicity, you can use `get.knn` method available in `FNN` package.
  * Compute shortest-paths (geodesic) matrix using your favourite algorithm.
    - Tip: Floyd-Warshall algorithm can be implemented easily here.
  * Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
  * Challenge: you may simply use PCA to do the same, but be careful to account for a proper normalization (centering) of the geodesic (kernel) matrix (see Kernel PCA for details).
  
An expected result (for *k* = 5) should look similar (not necessarily exactly the same) to following
![Example output](graph_iso.pdf)

```{r}
#mydata
# k = number of neighbours considered 	
#neighbors = FNN::get.knn(mydata, 5, algorithm = "kd_tree")
#neighbors
#$nn.index
#$nn.dist

```


```{r}
isomap <- function(data, k) {
  #1.Determine the neighbors of each point (k-NN)
    # get.knn: k-nearest neighbour classification for test set from training set. For each row of the test set, the k nearest (in Euclidean distance) training set vectors are found, and the                classification is decided by majority vote, with ties broken at random. If there are ties for the kth nearest vector, all candidates are included in the vote. 
      neighbors = FNN::get.knn(mydata, k, algorithm = "kd_tree")  # initialize the neighbors list
   
  #2.Construct a neighborhood graph.
    #Each point is connected to other if it is a K nearest neighbor.
    #Edge length equal to Euclidean distance.
    
    size = nrow(data)
  
    # Initialize matrix
    # Pozn. na nenastavenych poziciach, t.j tam kde nebude dist je Inf
    matrix <- matrix(Inf, nrow = size, ncol = size)
    # https://en.wikipedia.org/wiki/Nearest_neighbor_graph
    for ( i in 1:size ){ 
      for ( j in 1:k ){ # pre kazdy bod reprezentovany na nejakej pozicii i, i je z rozsahu 0 - nrows == size, # nastav vzdialenost medzi bodmi
        index = neighbors$nn.index[i,j] # index: one of nearest neighbours
        dist = neighbors$nn.dist[i,j]
        matrix[i,index] = dist 
      }
    }
    
    # 3.Compute shortest path between two nodes. (Floyd-Warshall algorithm)
    # https://shybovycha.github.io/2017/09/04/floyd-warshall-algorithm.html
        # Set elements on diagonal to zero
    # straight routes
    for(r in 1:size) {
      matrix[r,r] = 0
    }
    
    for (k in 1:size) {
        for (i in 1:size) {
            for (j in 1:size) {
                if (matrix[j, i] >  matrix[i, k] + matrix[k, j]) {
                    matrix[j, i] = matrix[i,k] + matrix[k,j]
                }
            }
        }
    }
    
    
  return(matrix)
}

# 4.(classical) multidimensional scaling
matrix <- isomap(mydata, 5)
fit <- cmdscale(matrix,eig=TRUE, k=2)
# Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
PlotPoints(fit$points, mylabels, mycolors)
```

*Comment:*
For each point, we found the k nearest neighbors, selecting points based on a fixed number. We then connected each point with its nearest neighbors, where the distance between them was equal to the Euclidean distance between the points. Using the Floyd-Warshall algorithm, implemented according to the algorithm in the reference above, we calculated the shortest paths between all pairs of points, thus approximating the geodesic distance between points. This was followed by dimensionality reduction using MDS (Multidimensional Scaling), which preserves distances between points (i.e., points that were close to each other before reduction should remain close afterwards).

3. **Visually compare PCA, ISOMAP and t-SNE** by plotting the word2vec data, embedded into 2D using the `Plotpoints` function. 
```{r eval=FALSE}
#Principal component analysis
fitPCA <- prcomp(mydata, center = TRUE, scale. = TRUE)
PlotPoints(fitPCA$x[,c(1,2)], mylabels, mycolors)
```

```{r}
#t-SNE (T-Distributed Stochastic Neighbor Embedding)
#install.packages('tsne')
library(tsne)
fittsne <- tsne(mydata, k = 2)
PlotPoints(fittsne, mylabels, mycolors)
```

Try finding the optimal *k* value for ISOMAP's nearest neighbour.
```{r}
# k = 5
matrix <- isomap(mydata, 5)
fit <- cmdscale(matrix,eig=TRUE, k=2)
# Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
PlotPoints(fit$points, mylabels, mycolors)

# k = 2
matrix <- isomap(mydata, 3)
fit <- cmdscale(matrix,eig=TRUE, k=2)
# Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
PlotPoints(fit$points, mylabels, mycolors)

# k = 3
matrix <- isomap(mydata, 3)
fit <- cmdscale(matrix,eig=TRUE, k=2)
# Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
PlotPoints(fit$points, mylabels, mycolors)

# k = 4
matrix <- isomap(mydata, 4)
fit <- cmdscale(matrix,eig=TRUE, k=2)
# Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
PlotPoints(fit$points, mylabels, mycolors)

# k = 6
matrix <- isomap(mydata, 6)
fit <- cmdscale(matrix,eig=TRUE, k=2)
# Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
PlotPoints(fit$points, mylabels, mycolors)

# k = 7
matrix <- isomap(mydata, 7)
fit <- cmdscale(matrix,eig=TRUE, k=2)
# Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
PlotPoints(fit$points, mylabels, mycolors)

# k = 8
matrix <- isomap(mydata, 8)
fit <- cmdscale(matrix,eig=TRUE, k=2)
# Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
PlotPoints(fit$points, mylabels, mycolors)

# k = 9
matrix <- isomap(mydata, 9)
fit <- cmdscale(matrix,eig=TRUE, k=2)
# Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
PlotPoints(fit$points, mylabels, mycolors)

# k = 10
matrix <- isomap(mydata, 10)
fit <- cmdscale(matrix,eig=TRUE, k=2)
# Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
PlotPoints(fit$points, mylabels, mycolors)

# k = 11
matrix <- isomap(mydata, 11)
fit <- cmdscale(matrix,eig=TRUE, k=2)
# Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
PlotPoints(fit$points, mylabels, mycolors)

# k = 12
matrix <- isomap(mydata, 12)
fit <- cmdscale(matrix,eig=TRUE, k=2)
# Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
PlotPoints(fit$points, mylabels, mycolors)

# k = 13
matrix <- isomap(mydata, 13)
fit <- cmdscale(matrix,eig=TRUE, k=2)
# Project the geodesic distance matrix into 2D space with (Classical) Multidimensional Scaling (`cmdscale` functions in R).
PlotPoints(fit$points, mylabels, mycolors)
```

*Answer:*
Nájdenie vhodného počtu susedov k nie je jednoduché. Príliš malé k viedlo k nerozpoznateľným zhlukom a vzdialenosť medzi bodmi v grafe bola primalá, zatiaľčo priveľké hodnoty viedli k prekryvu tried. Ako optimálne hodnoty k sa zdajú byť k = 5, k = 4,k = 6 a pri k = 7 už dochádzalo k viditeľnému prekryvu.
Na základe vizuálneho pozorovania sa zdá byť metóda ISOMAP ako najvhodnejšia pre daný súbor dát, pre k = 5.
V porovnaní s ISOMAP metódou sa PCA metóda zdá byť menej presná. Body klasifikované do rovnakej triedy sa prekrývajú s bodmi z iných tried a sú priveľmi blízko seba a prekrývajú sa. To platí aj pre t-SNE metódu, ktorá sa zdá byť taktiež menej presná ako ISOMAP. 

4. **Observe the effect of dimensionality reduction on a classiffication algorithm**. 
The supporting code in a function `Classify` performs training and testing of classification trees and gives the classification accuracy (percentage of correctly classified samples) as its result. Compare the accuracy of prediction on plain data, PCA, ISOMAP and t-SNE.
```{r eval=FALSE}
#classify ISOMAP
matrix <- isomap(mydata, 5)
fit <- cmdscale(matrix,eig=TRUE, k=2)
accISOMAP <- Classify(as.data.frame(fit$points), mycolors$V1, 50)

#classify PCA
accPCA <- Classify(as.data.frame(fitPCA$x), mycolors$V1, 50)

#classify t-SNE
accTSNE <- Classify(as.data.frame(fittsne), mycolors$V1, 50)

#PLOT results
print(paste("ISOMAP ACC:", mean(accISOMAP)))
print(paste("PCA ACC:", mean(accPCA)))
print(paste("t-SNE ACC:", mean(accTSNE)))
```

# Result
**ISOMAP ACC**: 0.761666666666667" with k = 5
**PCA ACC**: 0.743333333333333"
**t-SNE ACC**: 0.606666666666667"

The resulting classification accuracy corresponds to the interpretation from the visual comparison of the methods (for k = 5). However, upon repeated application of the methods to the same dataset, these values changed (in the case of changing k for ISOMAP), and for t-SNE, it varied from 0.6 to the accuracy values of the PCA method. The ISOMAP method proves to be the most accurate for appropriately chosen values of k, followed by PCA, and finally t-SNE.