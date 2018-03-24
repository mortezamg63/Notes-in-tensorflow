# Notes-in-tensorflow
Before going to notes in this page, it is necessary to bring out some describtions about understanding the matrix in python language. For this purpose, I want to describe  definition of vect and matrix.

In order to define a vector, we use only []. This is a matrix with one row; but the number of columns is depend on number of numbers in the [], ie. [1, 2] or [1, 2, 3, 4, 5].

<p align="center">
<img src="https://user-images.githubusercontent.com/15813546/37853255-69454360-2f03-11e8-8777-fa67017dd2f6.png">
</p>


If we want to define a matrix with two rows, we must define it as [[first row],][second row]]. Each row has the same number of columns for example [[1, 2, 3],[4, 5, 6]]. It is a matrix with two rows and three columns. Actually, it is a matrix in shape of   2x3.

<p align="center">
<img src="https://user-images.githubusercontent.com/15813546/37853254-672a1d80-2f03-11e8-9abc-3aec0f6f8ba3.png">
</img>

If we define the previous example as [[[1],[2],[3]],[[4],[5].[6]]], we are adding depth to the matrix. So the matrix is in shape of 2x3x1. Accordingly, [[[1,1,1],[2,2,2],[3,3,3]],[[4,4,4],[5,5,5],[6,6,6]]] is  a matrix in shape of 2x2x3.
In summary, [] is a vector (matrix with one row). For adding more rows we define as [[],[], ... ,[]]. 

<p align="center">
<img src="https://user-images.githubusercontent.com/15813546/37853770-c98d6034-2f05-11e8-8ac6-476f7e1a817d.png">
</p>

[] is one row.

[[],[]] is two rows.

[[],[],[]] is three rows.


In summary, while we are adding numbers to the above brackets, we are defining the columns in each row. For example [[1, 2, 3],[4, 5, 6]] is 2x3 (two rows and three columns). If we replace numbers with brackets, we are defining depth in matrix like [[[1],[2],[3]],[[4],[5],[6]]]. This matrix is in shape of 2x3x1. By replacing numbers with brackets, we are actually adding more dimension to the matrix. 

## tf.unstack
All matrices are described in previous section are known as tensor in tensorflow and other deep learning libraries. TensorFlow provides several operations to slice or extract parts of a tensor, or join multiple tensors together (for seeing the list click [here](https://www.tensorflow.org/api_guides/python/array_ops#Slicing_and_Joining). Among all operations, our goal is look at unstack operation closely.

This operation is used to chop a matrix to slices. In order to show how this operation can be used for separating a matrix to different parts/slices, I want to illustrate by means of pictures. This operation take matrix or tensor and axis number as input parameters. Then chop the input matrix to different parts in axis direction.

At first, it should be mentioned that it is necessary to specify dimension or axis in which separation will be done. For example, assume that there is a 3D matrix in shape of (3,3,2) and this matrix's name is X.

<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37860397-a22ab88e-2f41-11e8-8b0a-357ff9e5bc7c.png">
</p>

Two important parameters in tf.unstack operation are input matrix and axis. I want to extract rows in X (above matrix). In other words, linear patches can be extracted in axis zero by means of tf.unstack(X, axis=0). The following pictures show how the above matrix is sliced by unstack operation.

<p align="center">
   <img src="https://user-images.githubusercontent.com/15813546/37860474-7c4ed580-2f43-11e8-9740-f44b77c42a67.png"> 
  colors shows direction of chopping the matrix by unstack operation
</p>
