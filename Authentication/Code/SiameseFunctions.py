import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l
from tensorflow.keras import backend as K

import math

from sklearn.metrics import roc_curve

import numpy as np
import os

def euclidean_distance(vects):
    # This is a simple euclidean distance
    # We know (x, y) = vects because when defining the network this layer has inputs [embeddings_a, embeddings_b]
    x, y = vects
    # axis=1 here because the first dim is corresponds with the number of values per batch
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    out = K.sqrt(K.maximum(sum_square, K.epsilon()))
    return out

def eucl_dist_output_shape(shapes):
    #shape1[0] is the number of trajectories per batch
    shape1, shape2 = shapes
    return (shape1[0], 1)

def ContrastiveLoss(y_true, y_pred):
    margin = 1.0
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0.0))
    mean = K.mean(y_true * square_pred + (1 - y_true) * margin_square, axis=-1)
    return mean

def CreateSiameseModel(input_shape, backbone_fn, n_feature_map, pseudo=False):
    input_a = l.Input(shape=input_shape)
    input_b = l.Input(shape=input_shape)

    if (pseudo):
        left_limb = backbone_fn(input_shape, n_feature_map, "left")
        right_limb = backbone_fn(input_shape, n_feature_map, "right")

        proc_a = left_limb(input_a)
        proc_b = right_limb(input_b)
    else:
        shared_limb = backbone_fn(input_shape, n_feature_map, "shared")
        proc_a = shared_limb(input_a)
        proc_b = shared_limb(input_b)
       
    distance = l.Lambda(euclidean_distance,
                        output_shape=eucl_dist_output_shape)([proc_a, proc_b]) 
    return keras.Model(inputs=[input_a, input_b], outputs=distance)


### A Siamese Model subclassing Keras Models
### If this is used it will result in testing metrics being different from the reported ones
### To reproduce paper results use <CreateSiameseModel> function
### I think the difference is due to the <train_step> method updating weights differently
### It seems like this model can be trained longer (on 5Fold/10Fold)
### This model usually results in higher accuracies but lower EERs
### This model also tracks the learning rate, this was used to verify the cyclic implementation worked
class SiameseModel(keras.Model):
    def __init__(self, input_shape, backbone_fn, n_feature_map, pseudo=False):
        super(SiameseModel, self).__init__()
        input_a = l.Input(shape=input_shape)
        input_b = l.Input(shape=input_shape)

        if (pseudo):
            left_limb = backbone_fn(input_shape, n_feature_map, "left")
            right_limb = backbone_fn(input_shape, n_feature_map, "right")

            proc_a = left_limb(input_a)
            proc_b = right_limb(input_b)
        else:
            shared_limb = backbone_fn(input_shape, n_feature_map, "shared")
            proc_a = shared_limb(input_a)
            proc_b = shared_limb(input_b)
           
        distance = l.Lambda(euclidean_distance,
                            output_shape=eucl_dist_output_shape)([proc_a, proc_b]) 
        self.model = keras.Model(inputs=[input_a, input_b], outputs=distance)

        self.train_loss_tracker = keras.metrics.Mean(name='loss')
        self.learn_rate_tracker = keras.metrics.Mean(name='lr')

    def call(self, inputs, training=None):
        if training is not None:
            return self.model(inputs, training=training)
        else:
            return self.model(inputs, training=False) 

    @property
    def metrics(self):
        return [self.train_loss_tracker, self.learn_rate_tracker]

    def compile(self, optimizer, loss):
        super(SiameseModel, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y, y_pred)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights)
        )
        self.train_loss_tracker.update_state(loss)
        self.learn_rate_tracker.reset_state()
        self.learn_rate_tracker.update_state(self.optimizer.lr.cur_lr)
        return {
            'loss': self.train_loss_tracker.result(),
            'lr': self.learn_rate_tracker.result()
        }

## Callback and Schedule are provided because TF complained <on_batch_end> was taking too long with Callback 
## Schedule has the same performance per batch but I don't think it complains
class CyclicLearnRateCallback(keras.callbacks.Callback):
    def __init__(self, min_lr, max_lr, step_size):
        super(CyclicLearnRateScheduler, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.step = 0

    def compute_learn_rate(self):
        cycle = tf.math.floor(1 + (self.step / (2*self.step_size)))
        x = tf.math.abs((self.step/self.step_size) - (2*cycle) + 1)
        return self.min_lr + (self.max_lr - self.min_lr)*tf.math.maximum(0.0, 1.0-x)

    def on_batch_end(self, batch, logs=None):
        self.step = tf.math.floormod(self.step, (self.step_size * 2)) + 1
        K.set_value(self.model.optimizer.lr, self.compute_learn_rate())



class CyclicLearnRateScheduler(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, min_lr, max_lr, step_size):
        super(CyclicLearnRateScheduler, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.cur_lr = min_lr
        self.learn_rate_tracker = keras.metrics.Mean(name='lr')

    def __call__(self, step):
        cycle_step = tf.math.floormod(step, self.step_size*2)+1
        self.cur_lr = self.compute_learn_rate(cycle_step)
        self.learn_rate_tracker.update_state(self.cur_lr)
        return self.cur_lr

    def compute_learn_rate(self, step):
        cycle = tf.math.floor(1 + (step / (2*self.step_size)))
        x = tf.math.abs((step/self.step_size) - (2*cycle) + 1)
        return self.min_lr + (self.max_lr - self.min_lr)*tf.math.maximum(0.0, 1.0-x)


class TestData():
    def __init__(self, test_x, test_y, TestUsers, MODEL_DIR, epoch, prepend_logs=''):
        self.test_x = test_x
        self.test_y = test_y
        self.TestUsers = TestUsers
        self.MODEL_DIR = MODEL_DIR
        self.epoch = epoch
        with open(os.path.join(self.MODEL_DIR, "models.csv"), "a") as f:
            f.write(prepend_logs)
            f.close()

    
    def compute_acc(self, throws_to_test, filename, model=None):
        num_throws = len(throws_to_test)
        num_users = len(self.TestUsers)

        # TEST ACCURACY
        assert(model is not None), 'No model passed to <compute_acc> fn'
        #model.trainable = False

        file_txt = ""
        correct = 0
        total = 0
        all_predictions = model.predict(self.test_x, batch_size=256)
        
        ### Verify Batch Normalization layers don't change during inference ###
        #all_predictions2 = model.predict(self.test_x, batch_size=128)
        #diff = all_predictions-all_predictions2
        #small_enough = tf.math.less_equal(diff, 1e-3)
        
        #assert(tf.math.reduce_all(small_enough)), \
        #    'Running Test Data through <model.predict> twice gave different results'
        
        ##Reshape so that the second dimension is 
        ##all the pairs a single library user has for a particular query
        eer_pred = np.reshape(all_predictions, (-1, num_throws))
        
        ##Grab closest match per library user
        eer_pred = np.min(eer_pred, axis=-1)
        eer_pred = 1.0 - eer_pred ## Invert so scipy works
        
        eer_ys = (self.test_y[:,0]==self.test_y[:,2]).astype(float)
        eer_ys = np.reshape(eer_ys, (-1, num_throws))[:, 0] 
        fpr, tpr, _ = roc_curve(np.squeeze(eer_ys), np.squeeze(eer_pred), pos_label=1)
        fnr = 1 - tpr
        idx = np.nanargmin(np.absolute((fnr - fpr)))
        EER = (fpr[idx] + fnr[idx]) / 2
        
        all_predictions = np.reshape(all_predictions, (num_users*num_throws, -1))
        y_s = np.reshape(self.test_y, (num_users*num_throws, -1, 4))
        
        for i in range(0, y_s.shape[0]):
            minpred = np.argmin(all_predictions[i, :])
        
            if y_s[i, minpred, 0] == y_s[i, minpred, 2]:
                correct += 1
            total += 1
            for j in range(0, y_s.shape[1]):
                file_txt += "{}, {}, {}, {}, {},,\n".format(y_s[i,j,0], y_s[i,j,1],
                                                            y_s[i,j,2], y_s[i,j,3],
                                                            all_predictions[i, j])
            file_txt += ",,,,,{},{}\n".format(y_s[i, minpred, 2], y_s[i, minpred, 3])

        f2 = open(filename, "at")
        f2.write(file_txt)
        f2.close()
        # Return accuracy as value 0-100
        del all_predictions, eer_pred, eer_ys, fpr, tpr, fnr, idx, y_s
        acc = correct / total * 100
        EER = EER

        #model.trainable = True
        return acc, EER

    def run(self, model=None):
        acc, eer = self.compute_acc([x for x in range(10)],
                                    os.path.join(self.MODEL_DIR, "FoldAcc{}.csv".format(self.epoch)),
                                    model)
        # Record the accuracy and eer
        with open(os.path.join(self.MODEL_DIR, "models.csv"), "a") as f:
            f.write(",,,{},{},\n".format(
                    acc, eer))
            f.close()
        return acc,eer

