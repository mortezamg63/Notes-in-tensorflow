# Notes-in-tensorflow
Before going to notes in this page, it is necessary to bring out some describtions about understanding the matrix in python language. For this purpose, I want to describe  definition of vector and matrix.

In order to define a vector, we use only []. This is a matrix with one row; but the number of columns is depend on the number of values in the [], ie. [1, 2] or [1, 2, 3, 4, 5].

<p align="center">
<img src="https://user-images.githubusercontent.com/15813546/37853255-69454360-2f03-11e8-8777-fa67017dd2f6.png">
</p>


If we want to define a matrix with two rows, we must define it as [[first row],][second row]]. Each row has the same number of columns for example [[1, 2, 3],[4, 5, 6]]. It is a matrix with two rows and three columns. In fact, it is a matrix in shape of   2x3.

<p align="center">
<img src="https://user-images.githubusercontent.com/15813546/37853254-672a1d80-2f03-11e8-9abc-3aec0f6f8ba3.png">
</img>

If we define the previous example as [[[1],[2],[3]],[[4],[5].[6]]], we are adding depth to the matrix. So the matrix is in shape of 2x3x1. Accordingly, [[[1,1,1],[2,2,2],[3,3,3]],[[4,4,4],[5,5,5],[6,6,6]]] is  a matrix in shape of 2x2x3.
 [] is a vector (matrix with one row). For adding more raws (extra dimension) we define as [[],[], ... ,[]]. 

<p align="center">
<img src="https://user-images.githubusercontent.com/15813546/37853770-c98d6034-2f05-11e8-8ac6-476f7e1a817d.png">
</p>

[] is one row.

[[],[]] is two rows.

[[],[],[]] is three rows.


In other words, while we are adding numbers to the above brackets, we are defining the columns in each row. For example [[1, 2, 3],[4, 5, 6]] is 2x3 (two rows and three columns). If we replace numbers with brackets, we are defining depth in matrix like [[[1],[2],[3]],[[4],[5],[6]]]. This matrix is in shape of 2x3x1. By replacing numbers with brackets, we are actually adding more dimension to the matrix. 

## tf.unstack operation
All matrices are described in previous section are known as tensor in tensorflow and other deep learning libraries. TensorFlow provides several operations to slice or extract parts of a tensor, or join multiple tensors together (for seeing the list click [here](https://www.tensorflow.org/api_guides/python/array_ops#Slicing_and_Joining). Among all operations, our goal is look at unstack operation closely.

This operation is used to chop a matrix to slices. In order to show how this operation can be used for separating a matrix to different parts/slices, I want to illustrate by means of pictures. This operation take matrix or tensor and axis number as input parameters. Then chop the input matrix to different parts in axis direction.

At first, it should be mentioned that it is necessary to specify dimension or axis in which separation will be done. For example, assume that there is a 3D matrix in shape of (3,3,2) and the matrix's name is X.

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
This operation goes against tf.unstack. It gets patches and joins them together based on axis. For instance we can join the patches in linear and columnar direction. To illustrate this operation, assume that we have the following patches are specified by colors.

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
  tf.stack(X, axis=1)
</p>

## tf.reduce_mean, tf.reduce_max and ...
In this section I am going to introduce a series of functions that are along with doing one operation and dimension reduction in tensorflow library. These functions apply some basic operations on elements in matrices or tensors, then they turn out matrices with fewer dimensions. Although the functions can do more tasks far from the current description, the dimension reduction is focused here, and other taks are ignored. I describe basic tasks in the functinos that return outputs with smaller matrices. The list of common functions are here:

tf.reduce_mean()

tf.reduce_max()

tf.reduce_min()

tf.reduce_prod()

tf.reduce_sum()
and other functions that are not mentioned here.

Anyway, the mentioned functions use the same method for doing different operations. In other words, they do operations in different axises in matrix/tensor. Two important parameters of these functions are input matrix and axis. If we just send input parameter, the operation will be done on all elements and all axises of matrix. For instance, take the following picture that shows a matrix as base matrix for using the functions. Assume the name of the matrix is 'X'.

<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37861894-3e313c98-2f61-11e8-9bd3-b54604d88e2e.png">
</p>

In case of using tf.reduce_max(X). The result is 8.

```ruby

X = tf.constant([[[1., 2.],[3., 4.]],[[5., 6.],[7., 8.]]])
sess = tf.Session()
print(sess.run(tf.reduce_max(X)))  # output: 8.

```

From now I am going to talk over tf.reduce_mean() to show operation in different axises. At first, the following picture is case in point to show mean operation in different axises.

<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37861949-b6d415ac-2f62-11e8-9f75-b8f688fc01ba.png">
</p>

```ruby

X = tf.constant([[1, 1],[2, 2]])
sess = tf.Session()
print(sess.run(tf.reduce_mean(X)))  # output: 1.5
print(sess.run(tf.reduce_mean(X,axis=0)))  # output: [1.5, 1.5]
print(sess.run(tf.reduce_mean(X,axis=1)))  # output: [1., 2.]

```

```ruby

X=tf.constant([[1, 1],[2, 2],[3, 3]])
sess = tf.Session()
print(sess.run(tf.reduce_mean(X)))  # output: 2.
print(sess.run(tf.reduce_mean(X,axis=0)))  # output: [2., 2.]
print(sess.run(tf.reduce_mean(X,axis=1)))  # output: [1., 2., 3.]

```

Now, I want to show a 3D matrix and the output of tf.reduce_mean in different axises again. 


<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37861894-3e313c98-2f61-11e8-9bd3-b54604d88e2e.png">
</p>

```ruby

X=tf.constant([[[1, 2], [3, 4]],[[5, 6], [7, 8]]])
sess = tf.Session()
print(sess.run(tf.reduce_mean(X)))  # output: 4.5
print(sess.run(tf.reduce_mean(X,axis=0)))  
# output: [[3., 4.],
#           [5., 6.]]
print(sess.run(tf.reduce_mean(X,axis=1)))  
# output: [[2., 3.],
#           [6., 7.]]
print(sess.run(tf.reduce_mean(X, axis=2)))
#output: [[1.5, 3.5]
          [5.5, 7.5]]

```

<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37862310-76a94a50-2f68-11e8-90cf-2f3887b5f4bc.png">
</p>
<p align="center">
  tf.reduce_mean(X, axis=0)
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37862415-13b6218c-2f6a-11e8-917a-7c134f2e835c.png">
</p>
<p align="center">
  tf.reduce_mean(X, axis=1)
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37862550-4f164782-2f6c-11e8-8a09-b8b4da6022bc.png">
</p>
<p align="center">
  tf.reduce_mean(X, axis=2)
</p>

#   parallel and rotation invariant layers in tensorflow

# Parallel Layer

 All things is same as defining one layer. But we must define a function for declaring layers as array to roll up them in parallel form using tf.stack function in tensorflow module. the layer is a convolution 2D layer. Therefore, we define our convolution layer in a function to call it repetitively to put every convolution in one element of array. For this purpose we define our layer as follow:
 
 ```ruby
 def conv_layer(input_):
   input_layer = tf.reshape(input_, [-1,28,28,1])
   with tf.variable_scope('conv1') as scope:
      conv1 = tf.layers.conv2d(inputs=input_layer,
						filters = 1,
						kernel_size=[5, 5],
						strides=[5,5],
						activation=tf.nn.relu)
      return conv1

 ```
 
  Here I want to define 5 layers in parallel. So I must use the conv_layer function 5 times and stack them as following function:

 ```ruby
 
 def parallel_layer(input_, angles=0):
	branches = []
	with tf.variable_scope('branches') as scope:
		for index, angle in enumerate(angles):
		branches.append(conv_layer(input_)) 
		concatenated = tf.stack(branches, axis=1)
		return  concatenated
  ```
  
  As you see in the parallel_layer function above, I use a for statement and create 5 parallel convolution layers. The number 5 is the number of angles that are sent in function throught the angles parameter in the first line of above code.
To use the layer and show the result, let me define a session and use the functions. For this purpose look at the following codes:

```ruby

N_INPUT = 28*28
BATCH_SIZE=1
NUM_PARALLEL_CONVS = 5
tf.set_random_seed(1)
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist_data/", one_hot = True)
x = tf.placeholder(tf.float32, [None,N_INPUT])
logit,concat = parallel_layer(x,NUM_PARALLEL_CONVS)
with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   image, label = mnist.train.next_batch(BATCH_SIZE)
   out = sess.run(logit, feed_dict={x:image})
   print(out.shape) →
	# output : (1, 5, 24, 24, 1) = (batch_size, output of parallel layers, height, width,channel)

```

 In case of setting NUM_PARALLEL_CONVS to 8, output of print(out.shape) is (1,8,24,24,1). Also if I set BATCH_SIZE to 10, the output of print(out.shape) is (10,8, 24, 24, 1). It is clear that everything is true. It seems that based on number of parallel layers and batch_size nothing is wrong and everything comes true.

 In most papers this method is used to extract invariant transformations. But, element-wise max operation from output feature maps in the parallel layer is used to send final feature map and feed to next layers.
For this purpose tf.reduce_max function with axis same as tf.stack must be used. So the parallel_layer function must be changed as following:

```ruby
def parallel_layer(input_, angles=0):
	branches = []
	with tf.variable_scope('branches') as scope:
		for index, angle in enumerate(angles):
		branches.append(conv_layer(input_)) 		
		concatenated = tf.stack(branches, axis=1)
		max_pool = tf.reduce_max(concatenated, reduction_indices=[1])
		return max_pool, concatenated

```

An instance output of the above code is shown in the following picture.

<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37915672-52a753d6-312f-11e8-93c7-805c5db9b355.png">
</p>

It is necessary to mention one problem of the parallel layer. The problem is that every layer has its own weights, biases values and filters. Therefore another salient point that must be mentioned here is about using the same variables and parameters in parallel models because of benefits such as easy training and occupying less space on memory. In this way back-propagation is applicable and learning is possible in parallel layer. In practice one layer is used as parallel in a way that in all branches of the parallel layer same values and parameters (bias, weight and filter) are used. Fortunately, It's an easy job. We just need to add two lines in the parallel_layer function as following:

```ruby
def parallel_layer(input_, angles=0):
	branches = []
	with tf.variable_scope('branches') as scope:
		for index, angle in enumerate(angles):
			branches.append(conv_layer(input_)) 		
			if (index==0):
				scope.reuse_variables() 
		concatenated = tf.stack(branches, axis=1)
		max_pool = tf.reduce_max(concatenated, reduction_indices=[1])
		return max_pool

```

Based on new definition of parallel convolution (new change in the parallel_layer function), all outputs from 5 branches (parallel convolution layers) are same as the first branch as it is shown in above picture. In other words, all columns have values same as the first column  because we are feeding the same picture to all five branches which has the same parameters, values and filters.


Moreover , I use 4 filters in the conv_layer function, which is shown in the following code. The result of reduce_max is shown in the following picture. 

```ruby
def conv_layer(input_):
   input_layer = tf.reshape(input_, [-1,28,28,1])
   with tf.variable_scope('conv1') as scope:
      conv1 = tf.layers.conv2d(inputs=input_layer,
						filters = 1,
						kernel_size=[5, 5],
						strides=[5,5],
						activation=tf.nn.relu)
      return conv1
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/15813546/37915768-820dcc86-312f-11e8-9464-54c248e83ca1.png">
</p>


# Rotation Invariant Layer

In this section, the goal is describing how to change conv_layer in a way that making the convolutional layer invariant to rotation. We just need to rotate kernel in convolutional layer. It ,however, is not pre-defined in tensorflow. Hence, we must implement and define the layer from scratch. The implication is that we define the following function rather thanconv_layer function, which is described already.

```ruby
def rotInvar_conv_layer(input,index, num_input_channels, filter_size, num_filters, stride=1,    
                                                                                      angle=0, reusevars=tf.AUTO_REUSE):
   input_layer = tf.reshape(input, [-1,28,28,1]) 
   with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE) as scope:
      # Shape of the filter-weights for the convolution
      shape = [filter_size, filter_size, num_input_channels, num_filters]

      # Create new weights (filters) with the given shape
      if index==0:
         weights = tf.get_variable("weights", initializer=tf.truncated_normal(shape,
                                                                                                                    stddev=0.05))
      else:
         weights = tf.get_variable("weights")

      # Rotate filter
      if angle != 0:
         radian = angle * math.pi / 180
      weights = tf.contrib.image.rotate(weights, radian)
      # Create new biases, one for each filter
      biases = tf.Variable(tf.constant(0.05, shape=[num_filters])) 
 
      layer1 = tf.nn.conv2d(input=input_layer, filter=weights, 
                                         strides=[1, stride, stride, 1],     padding='SAME')

      # Add the biases to the results of the convolution.
      layer1 += biases

      return layer1
```

Here tf.get_variable is used to create and reuse it in our implementation.  Finally the  rotInvar_conv_layer function is used in the parallel_layer function as follow:

```ruby
def parallel_layer(input_, angles=0):
    branches = []
    with tf.variable_scope('branches') as scope:
       for index, angle in enumerate(angles):
           #branches.append(conv_layer(input_)) 
           branches.append(rotInvar_conv_layer(input_,index,num_input_channels=1,
                                                              _size=5, num_filters=1, stride=5,angle=angle))
           if (index==0):
                scope.reuse_variables() 
        concatenated = tf.stack(branches, axis=1)
        max_pool = tf.reduce_max(concatenated, reduction_indices=[1])
        return max_pool, concatenated

```

All defined functions are ready to use. Therefore, the rest of the code must be as following.

```ruby
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import os

N_INPUT = 28*28
BATCH_SIZE=1
NUM_PARALLEL_CONVS = 5
tf.set_random_seed(1)
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist_data/", one_hot = True)

x = tf.placeholder(tf.float32, [None,N_INPUT])

logit,concat = parallel_layer(x,[0,90, -90])

with tf.Session() as sess:
sess.run(tf.global_variables_initializer())
image, label = mnist.train.next_batch(BATCH_SIZE)
image = np.ones(image.shape)
max_out, con = sess.run([logit,concat], feed_dict={x:image})
print(max_out.shape)

```

In order to verify that all things are working fine, I took one image with values one in all pixels as input image. Then I apply my defined convolutional layer to four angles 0, 90,-90 and 180 degrees. For this purpose I create an image using np.ones function in numpy module and feed it to the convolution layer as following:

```ruby
logit,concat = parallel_layer(x,[0,90, -90, 180])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    image, label = mnist.train.next_batch(BATCH_SIZE)
    image = np.ones(image.shape)
    max_out, con = sess.run([logit,concat], feed_dict={x:image})
    print(max_out.shape)

```

The outputs from every parallel layer is shown in the following picture. 

<p align="center">
     <img src="https://user-images.githubusercontent.com/15813546/37915979-eebd10bc-312f-11e8-9bfe-63e570a1c2a0.png">
</p>
<p align="center">
 The colors show that outputs of every layers are symmetric. Also, output of layers for angles 90 and -90 are symmetric. Likewise, output of layers for angles 0 and 180 are symmetric too.
</p>

