import os
import numpy as np
import tensorflow as tf
import CNN
import GetCnnData

#define the number of classes, image size, batch size, capacity of layer, number of epochs and learning rate in advance for convenient use 
N_CLASSES = 0  
IMG_W = 128  
IMG_H = 128
BATCH_SIZE =20
CAPACITY = 200
MAX_STEP = 15000
learning_rate = 0.0001 
 
train_dir = r'C:\important\python\dataset\train_data'  
logs_train_dir = r'C:\important\python\logs'              
 

for str in os.listdir(train_dir):
    N_CLASSES = N_CLASSES+1

#get data
train,trian_label,val,val_label = GetCnnData.get_files(train_dir,0.3)
#training data and label
train_batch,train_label_batch = GetCnnData.get_batch(train,trian_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)
#validation data and label
val_batch,val_label_batch = GetCnnData.get_batch(val,val_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)
 
#define training in CNN
train_logits = CNN.inference(train_batch,BATCH_SIZE,N_CLASSES)
train_loss = CNN.losses(train_logits, train_label_batch)
train_op = CNN.trainning(train_loss, learning_rate)
train_acc = CNN.evaluation(train_logits, train_label_batch)

#define testing in CNN
test_logits = CNN.inference(val_batch, BATCH_SIZE, N_CLASSES)
test_loss = CNN.losses(test_logits, val_label_batch)
test_acc = CNN.evaluation(test_logits, val_label_batch)

#LOGS
summary_op = tf.summary.merge_all()
 

sess = tf.Session()
#write down logs
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)

#create a saver for trained model
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
 
#train batch
try:
    
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
       
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
 
        #print the training loss rate and accuracy for every 10 steps
        if step % 10 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
        
        #save the trained model
        if (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
 
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
 
finally:
    coord.request_stop()
