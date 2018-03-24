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

## tf.unstack operation
All matrices are described in previous section are known as tensor in tensorflow and other deep learning libraries. TensorFlow provides several operations to slice or extract parts of a tensor, or join multiple tensors together (for seeing the list click [here](https://www.tensorflow.org/api_guides/python/array_ops#Slicing_and_Joining). Among all operations, our goal is look at unstack operation closely.

This operation is used to chop a matrix to slices. In order to show how this operation can be used for separating a matrix to different parts/slices, I want to illustrate by means of pictures. This operation take matrix or tensor and axis number as input parameters. Then chop the input matrix to different parts in axis direction.

At first, it should be mentioned that it is necessary to specify dimension or axis in which separation will be done. For example, assume that there is a 3D matrix in shape of (3,3,2) and this matrix's name is X.

<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37860397-a22ab88e-2f41-11e8-8b0a-357ff9e5bc7c.png">
</p>

Two important parameters in tf.unstack operation are input matrix and axis. I want to extract rows in X (above matrix). In other words, linear patches can be extracted in axis zero by means of tf.unstack(X, axis=0). The following pictures show how the above matrix is sliced by unstack operation. As it is shown, colors show direction of chopping the matrix by unstack operation

<p align="center">
   <img src="https://user-images.githubusercontent.com/15813546/37860474-7c4ed580-2f43-11e8-9740-f44b77c42a67.png">   
</p>

In case of extracting columnar patches, unstack(X, axis=1) must be used. The following pictures show slicing matrix in columnar direction.

<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37860516-9c4a4972-2f44-11e8-8960-cd1934297916.png">  
</p>
Finally, we can extract depth patches from matrix using tf.unstack(X, axis=2) as following pictures.

<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37860550-04a4bfca-2f45-11e8-8485-bd9219c0d07d.png">
</p>

## tf.stack operation
This operation goes against tf.unstack. It gets patches and join them together based on axis. For instance we can join the patches in linear and columnar direction. To illustrate this operation, assume that we have the following patches are specified by colors.

<p align="center">
   <img src="https://user-images.githubusercontent.com/15813546/37860648-6d3b671c-2f47-11e8-8ca5-35ed8960e230.png">
</p>

Now stacking or packing the patches together in axis zero is as following picture. This axis joins patches in row direction.

<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37860656-bc296a7c-2f47-11e8-8cc3-b1951989d251.png">
</p>
<p align="center">
  tf.stack(X, axis=0)
</p>

Moreover, the following picture shows the stack operation in axis one. This axis joins patches in column direction

<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37860671-fbd886e4-2f47-11e8-870f-2211bcbfc48a.png">
</p>
<p align="center">
  tf.stack(X, axis=0)
</p>
