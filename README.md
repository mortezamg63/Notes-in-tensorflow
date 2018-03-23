# Notes-in-tensorflow
Before going to notes, it is necessary to bring out describtions about understanding the matrix in python language. For this purpose, I want to describe  definition of vect and matrix.

In order to define a vector, we use only []. This is a matrix with one row; but the number of columns is depend on number of numbers in the [], ie. [1, 2] or [1, 2, 3, 4, 5].

![image](https://user-images.githubusercontent.com/15813546/37853020-853a116e-2f02-11e8-800a-6375dee0b60f.png)


If we want to define a matrix with two rows, we must define it as [[first row],][second row]]. Each row has the same number of columns for example [[1, 2, 3],[4, 5, 6]]. It is a matrix with two rows and three columns. Actually, it is a matrix in shape of   2x3.


If we define the previous example as [[[1],[2],[3]],[[4],[5].[6]]], we are adding depth to the matrix. So the matrix is in shape of 2x3x1. Accordingly, [[[1,1,1],[2,2,2],[3,3,3]],[[4,4,4],[5,5,5],[6,6,6]]] is  a matrix in shape of 2x2x3.
In summary, [] is a vector (matrix with one row). For adding more rows we define as [[],[], ... ,[]]. 


[] is one row.

[[],[]] is two rows.

[[],[],[]] is three rows.


while we are adding numbers to the above brackets, we are defining the columns in each row. For example [[1, 2, 3],[4, 5, 6]] is 2x3 (two rows and three columns). If we replace numbers with brackets, we are defining depth in matrix like [[[1],[2],[3]],[[4],[5],[6]]]. This matrix is in shape of 2x3x1. By replacing numbers with brackets, we are actually adding more dimension to matrix.

## tf.unstack
This function is used to chop a matrix to slices. In order to show how this function can be used for separating a matrix to different parts/slices, I want to illustrate this by means of picture. 

At first, it should be mentioned that it is necessary to specify dimension or axis in which separation will be done. For example, assume that there is a 3D matrix in shape of (3,3,2).
