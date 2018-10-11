# Assignment 1 : Face Recognition Using Eigenfaces 
1st Assignment, Multivariate Statistics. Face recognition using Eigenfaces.

# Introduction

This report is devoted to a specific technique in face image recognition: the ``eigenface approach''. It was introduced in [1], by the year 1991. The approach consists on calculating the eigenvectors of the variance-covariance matrix of the empirical distribution of the vectors of images. Then a set of ``representative'' eigenvectors is attained through Principal Components Analysis (PCA) in order to reduce the high-dimensional representation of an image. The smaller set, or basis, representing the main features of the images is then formed by the eigenfaces. Moreover, recognition can be done by projecting every new image into the subspace spanned by the eigenfaces (called the ``face-space'') and comparing the distances of the projection from a predefined set of face classes.

# The Data

The study is motivated by a data base of images (all frontal views) from 100 individuals (half men, half women) taken in 6 different conditions. It is possible to identify, for example, neutral expressions, smiles, anger faces and screams. The images are colored and seems to be (somehow) centered. There is no appreciable change in light orientation and although some subjects wear glasses, there are no other accessories on them.  The author presumes this is a sample from the ``AR Face Database''.

The data will not be preprocessed, although this particular application starts by transforming the images to a standardized gray-scale using, successively, the MATLAB functions \emph{rgb2gray} $\rightarrow$ \emph{mat2gray}. This transform the dataset into a 3D matrix of $N = 19800$ rows (representing $165\times 120$ pixels) $\times$ 6 columns of face-types $\times$ 100 individuals. We could also see it as a $19800 \times 600$ 2D matrix. 

# References

[1] Matthew Turk and Alex Pentland. Eigenfaces for recognition. Journal of Cognitive Neuroscience, 3(1):71–86, 1991. PMID: 23964806.
