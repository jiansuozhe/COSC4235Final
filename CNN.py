import tensorflow as tf
 
 
def inference(images, batch_size, n_classes):
    #a simple convolutional neural network, (conv layer + pooling layer) * 2, fully connected layer * 2, softmax layer * 1
    
    #conv layer1
    #128 3*3 neurons, activation function relu()
    with tf.variable_scope('conv1') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 128], stddev=1.0, dtype=tf.float32),
                              name='weights', dtype=tf.float32)
 
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                             name='biases', dtype=tf.float32)
 
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME') #the padding operation does not change the image size
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
 
    #pooling layer1
    #3 * 3 max pooling, strides = 2
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
 
    #conv layer2
    #16 3*3 neurons, activation function relu()
    with tf.variable_scope('conv2') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 16], stddev=0.1, dtype=tf.float32),
                              name='weights', dtype=tf.float32)
 
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[16]),
                             name='biases', dtype=tf.float32)
 
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
 
   	#pooling layer2
   	#3 * 3 max pooling, strides = 1
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')
 
    #fully connected layer1
    #256 neurons, reshape the output from the previous pooling layer to one line
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal(shape=[dim, 256], stddev=0.005, dtype=tf.float32),
                              name='weights', dtype=tf.float32)
 
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[256]),
                             name='biases', dtype=tf.float32)
 
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
 
    #fully connected layer2
    #256 neurons
    with tf.variable_scope('local4') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[256, 256], stddev=0.005, dtype=tf.float32),
                              name='weights', dtype=tf.float32)
 
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[256]),
                             name='biases', dtype=tf.float32)
 
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
 
    #softmax layer
    #linear regression on the previous output
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[256, n_classes], stddev=0.005, dtype=tf.float32),
                              name='softmax_linear', dtype=tf.float32)
 
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[n_classes]),
                             name='biases', dtype=tf.float32)
 
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
 
    return softmax_linear
 
 
#loss function, returns the data loss rate
#logits is the predicted value by CNN, labels is the real value
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy =tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss
 
 
#training function using Adam Optimizer
def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
 
 
#calculate the training accuracy
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
