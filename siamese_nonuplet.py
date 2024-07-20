#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.metrics import roc_auc_score


def graph_embed(X, msg_mask, N_x, N_embed, N_o, iter_level, Wnode, Wembed, W_output, b_output):
    #X -- affine(W1) -- ReLU -- (Message -- affine(W2) -- add (with aff W1)
    # -- ReLU -- )* MessageAll  --  output
    node_val = tf.reshape(tf.matmul( tf.reshape(X, [-1, N_x]) , Wnode),
            [tf.shape(X)[0], -1, N_embed])
    
    cur_msg = tf.nn.relu(node_val)   #[batch, node_num, embed_dim]
    for t in range(iter_level):
        #Message convey
        Li_t = tf.matmul(msg_mask, cur_msg)  #[batch, node_num, embed_dim]
        #Complex Function
        cur_info = tf.reshape(Li_t, [-1, N_embed])
        for Wi in Wembed:
            if (Wi == Wembed[-1]):
                cur_info = tf.matmul(cur_info, Wi)
            else:
                cur_info = tf.nn.relu(tf.matmul(cur_info, Wi))
        neigh_val_t = tf.reshape(cur_info, tf.shape(Li_t))
        #Adding
        tot_val_t = node_val + neigh_val_t
        #Nonlinearity
        tot_msg_t = tf.nn.tanh(tot_val_t)
        cur_msg = tot_msg_t   #[batch, node_num, embed_dim]

    g_embed = tf.reduce_sum(cur_msg, 1)   #[batch, embed_dim]
    output = tf.matmul(g_embed, W_output) + b_output
    
    return output


# Siamese Nonuplet Network Construction 
class graphnn(object):
    def __init__(self,
                    N_x1,
                    N_x2,
                    N_x3,
                    Dtype, 
                    N_embed,
                    depth_embed,
                    N_o,
                    ITER_LEVEL,
                    lr,
                    device = '/cpu:0'
                ):

        self.NODE_LABEL_DIM = N_x1

        tf.compat.v1.reset_default_graph()
        with tf.device(device):
            Wnode1 = tf.Variable(tf.random.truncated_normal(
                shape = [N_x1, N_embed], stddev = 0.1, dtype = Dtype))
            Wnode2 = tf.Variable(tf.random.truncated_normal(
                shape = [N_x2, N_embed], stddev = 0.1, dtype = Dtype))
            Wnode3 = tf.Variable(tf.random.truncated_normal(
                shape = [N_x3, N_embed], stddev = 0.1, dtype = Dtype))
            Wembed = []
            for i in range(depth_embed):
                Wembed.append(tf.Variable(tf.random.truncated_normal(
                    shape = [N_embed, N_embed], stddev = 0.1, dtype = Dtype)))

            W_output = tf.Variable(tf.random.truncated_normal(
                shape = [N_embed, N_o], stddev = 0.1, dtype = Dtype))
            b_output = tf.Variable(tf.constant(0, shape = [N_o], dtype = Dtype))

            # model 1 -------------
            X1 = tf.compat.v1.placeholder(Dtype, [None, None, N_x1])
            msg1_mask = tf.compat.v1.placeholder(Dtype, [None, None, None])
                                            #[B, N_node, N_node]
            self.X1 = X1
            self.msg1_mask = msg1_mask
            embed1 = graph_embed(X1, msg1_mask, N_x1, N_embed, N_o, ITER_LEVEL,
                    Wnode1, Wembed, W_output, b_output)

            X2 = tf.compat.v1.placeholder(Dtype, [None, None, N_x1])
            msg2_mask = tf.compat.v1.placeholder(Dtype, [None, None, None])
            self.X2 = X2
            self.msg2_mask = msg2_mask
            embed2 = graph_embed(X2, msg2_mask, N_x1, N_embed, N_o, ITER_LEVEL,
                    Wnode1, Wembed, W_output, b_output)

            X3 = tf.compat.v1.placeholder(Dtype, [None, None, N_x1])
            msg3_mask = tf.compat.v1.placeholder(Dtype, [None, None, None])
                                            #[B, N_node, N_node]
            self.X3 = X3
            self.msg3_mask = msg3_mask
            embed3 = graph_embed(X3, msg3_mask, N_x1, N_embed, N_o, ITER_LEVEL,
                    Wnode1, Wembed, W_output, b_output)

            label1 = tf.compat.v1.placeholder(Dtype, [None, ]) #same: 1; different:-1
            self.label1 = label1
            # ----------------
            
            # model 2 ------------
            X4 = tf.compat.v1.placeholder(Dtype, [None, None, N_x2])
            msg4_mask = tf.compat.v1.placeholder(Dtype, [None, None, None])
                                            #[B, N_node, N_node]
            self.X4 = X4
            self.msg4_mask = msg4_mask
            embed4 = graph_embed(X4, msg4_mask, N_x2, N_embed, N_o, ITER_LEVEL,
                    Wnode2, Wembed, W_output, b_output)

            X5 = tf.compat.v1.placeholder(Dtype, [None, None, N_x2])
            msg5_mask = tf.compat.v1.placeholder(Dtype, [None, None, None])
            self.X5 = X5
            self.msg5_mask = msg5_mask
            embed5 = graph_embed(X5, msg5_mask, N_x2, N_embed, N_o, ITER_LEVEL,
                    Wnode2, Wembed, W_output, b_output)

            X6 = tf.compat.v1.placeholder(Dtype, [None, None, N_x2])
            msg6_mask = tf.compat.v1.placeholder(Dtype, [None, None, None])
                                            #[B, N_node, N_node]
            self.X6 = X6
            self.msg6_mask = msg6_mask
            embed6 = graph_embed(X6, msg6_mask, N_x2, N_embed, N_o, ITER_LEVEL,
                    Wnode2, Wembed, W_output, b_output) 

            label2 = tf.compat.v1.placeholder(Dtype, [None, ]) #same: 1; different:-1
            self.label2 = label2
            # -------------

            # model 3 ------------
            X7 = tf.compat.v1.placeholder(Dtype, [None, None, N_x3])
            msg7_mask = tf.compat.v1.placeholder(Dtype, [None, None, None])
                                            #[B, N_node, N_node]
            self.X7 = X7
            self.msg7_mask = msg7_mask
            embed7 = graph_embed(X7, msg7_mask, N_x3, N_embed, N_o, ITER_LEVEL,
                    Wnode3, Wembed, W_output, b_output)

            X8 = tf.compat.v1.placeholder(Dtype, [None, None, N_x3])
            msg8_mask = tf.compat.v1.placeholder(Dtype, [None, None, None])
            self.X8 = X8
            self.msg8_mask = msg8_mask
            embed8 = graph_embed(X8, msg8_mask, N_x3, N_embed, N_o, ITER_LEVEL,
                    Wnode3, Wembed, W_output, b_output)

            X9 = tf.compat.v1.placeholder(Dtype, [None, None, N_x3])
            msg9_mask = tf.compat.v1.placeholder(Dtype, [None, None, None])
                                            #[B, N_node, N_node]
            self.X9 = X9
            self.msg9_mask = msg9_mask
            embed9 = graph_embed(X9, msg9_mask, N_x3, N_embed, N_o, ITER_LEVEL,
                    Wnode3, Wembed, W_output, b_output) 

            label3 = tf.compat.v1.placeholder(Dtype, [None, ]) #same: 1; different:-1
            self.label3 = label3
            # -------------

            
            self.embed1 = embed1
            
            # -------- model 1
            cos1 = tf.reduce_sum(embed1*embed2, 1) / tf.sqrt(tf.reduce_sum(
                embed1**2, 1) * tf.reduce_sum(embed2**2, 1) + 1e-10)

            cos2 = tf.reduce_sum(embed2*embed3, 1) / tf.sqrt(tf.reduce_sum(
                embed2**2, 1) * tf.reduce_sum(embed3**2, 1) + 1e-10)

            cos3 = tf.reduce_sum(embed1*embed3, 1) / tf.sqrt(tf.reduce_sum(
                embed1**2, 1) * tf.reduce_sum(embed3**2, 1) + 1e-10)

            diff1 = -cos1
            self.diff1 = diff1

            diff2 = -cos3
            self.diff2 = diff2


            # -------- model 2
            cos4 = tf.reduce_sum(embed4*embed5, 1) / tf.sqrt(tf.reduce_sum(
                embed4**2, 1) * tf.reduce_sum(embed5**2, 1) + 1e-10)

            cos5 = tf.reduce_sum(embed5*embed6, 1) / tf.sqrt(tf.reduce_sum(
                embed5**2, 1) * tf.reduce_sum(embed6**2, 1) + 1e-10)

            cos6 = tf.reduce_sum(embed4*embed6, 1) / tf.sqrt(tf.reduce_sum(
                embed4**2, 1) * tf.reduce_sum(embed6**2, 1) + 1e-10)

            diff3 = -cos4
            self.diff3 = diff3

            diff4 = -cos6
            self.diff4 = diff4


            # -------- model 3
            cos7 = tf.reduce_sum(embed7*embed8, 1) / tf.sqrt(tf.reduce_sum(
                embed7**2, 1) * tf.reduce_sum(embed8**2, 1) + 1e-10)

            cos8 = tf.reduce_sum(embed8*embed9, 1) / tf.sqrt(tf.reduce_sum(
                embed8**2, 1) * tf.reduce_sum(embed9**2, 1) + 1e-10)

            cos9 = tf.reduce_sum(embed7*embed9, 1) / tf.sqrt(tf.reduce_sum(
                embed7**2, 1) * tf.reduce_sum(embed9**2, 1) + 1e-10)

            diff5 = -cos7
            self.diff5 = diff5

            diff6 = -cos9
            self.diff6 = diff6

            #--------
            
            
            #pos = label1[0] * ((diff1[0] + diff3[0]) - (diff2[0] + diff4[0])) + 0.2            
            #neg = label1[0] * ((diff2[1] + diff4[1]) - (diff1[1] + diff3[1])) + 0.2
            
            pos1 = label1[0] * (diff1[0] - diff2[0]) + 0.2            
            neg1 = label1[0] * (diff2[1] - diff1[1]) + 0.2

            pos2 = label2[0] * (diff3[0] - diff4[0]) + 0.2            
            neg2 = label2[0] * (diff4[1] - diff3[1]) + 0.2
            
            pos3 = label3[0] * (diff5[0] - diff6[0]) + 0.2            
            neg3 = label3[0] * (diff6[1] - diff5[1]) + 0.2
            
            loss = tf.maximum(pos1, 0) + tf.maximum(neg1, 0) + tf.maximum(pos2, 0) + tf.maximum(neg2, 0) + tf.maximum(pos3, 0) + tf.maximum(neg3, 0)
            self.loss = loss
            
            optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
            self.optimizer = optimizer
    
    def say(self, string):
        print (string)
        if self.log_file != None:
            self.log_file.write(string+'\n')
    
    def init(self, LOAD_PATH, LOG_PATH):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        saver = tf.train.Saver()
        self.sess = sess
        self.saver = saver
        self.log_file = None
        if (LOAD_PATH is not None):
            if LOAD_PATH == '#LATEST#':
                checkpoint_path = tf.train.latest_checkpoint('./')
            else:
                checkpoint_path = LOAD_PATH
            saver.restore(sess, checkpoint_path)
            if LOG_PATH != None:
                self.log_file = open(LOG_PATH, 'a+')
            self.say('{}, model loaded from file: {}'.format(
                datetime.datetime.now(), checkpoint_path))
        else:
            sess.run(tf.global_variables_initializer())
            if LOG_PATH != None:
                self.log_file = open(LOG_PATH, 'w')
            self.say('Training start @ {}'.format(datetime.datetime.now()))
    
    def get_embed(self, X1, mask1):
        vec, = self.sess.run(fetches=[self.embed1],
                feed_dict={self.X1:X1, self.msg1_mask:mask1})
        return vec

    def calc_loss(self, X1, X2, X3, mask1, mask2, mask3, y1, X4, X5, X6, mask4, mask5, mask6, y2, X7, X8, X9, mask7, mask8, mask9, y3):
        cur_loss, = self.sess.run(fetches=[self.loss], feed_dict={
            self.X1:X1,self.X2:X2,self.X3:X3,self.msg1_mask:mask1,self.msg2_mask:mask2,self.msg3_mask:mask3,self.label1:y1,
            self.X4:X4,self.X5:X5,self.X6:X6,self.msg4_mask:mask4,self.msg5_mask:mask5,self.msg6_mask:mask6,self.label2:y2,
            self.X7:X7,self.X8:X8,self.X9:X9,self.msg7_mask:mask7,self.msg8_mask:mask8,self.msg9_mask:mask9,self.label3:y3})
        return cur_loss
        
    def calc_diff1(self, X1, X2, X3, mask1, mask2, mask3):
        diff1, = self.sess.run(fetches=[self.diff1], feed_dict={self.X1:X1,
            self.X2:X2, self.X3:X3, self.msg1_mask:mask1, self.msg2_mask:mask2, self.msg3_mask:mask3})
        return diff1

    def calc_diff2(self, X1, X2, X3, mask1, mask2, mask3):
        diff2, = self.sess.run(fetches=[self.diff2], feed_dict={self.X1:X1,
            self.X2:X2, self.X3:X3, self.msg1_mask:mask1, self.msg2_mask:mask2, self.msg3_mask:mask3})
        return diff2

    def calc_diff3(self, X4, X5, X6, mask4, mask5, mask6):
        diff3, = self.sess.run(fetches=[self.diff3], feed_dict={self.X4:X4,
            self.X5:X5, self.X6:X6, self.msg4_mask:mask4, self.msg5_mask:mask5, self.msg6_mask:mask6})
        return diff3

    def calc_diff4(self, X4, X5, X6, mask4, mask5, mask6):
        diff4, = self.sess.run(fetches=[self.diff4], feed_dict={self.X4:X4,
            self.X5:X5, self.X6:X6, self.msg4_mask:mask4, self.msg5_mask:mask5, self.msg6_mask:mask6})
        return diff4

    def calc_diff5(self, X7, X8, X9, mask7, mask8, mask9):
        diff5, = self.sess.run(fetches=[self.diff5], feed_dict={self.X7:X7,
            self.X8:X8, self.X9:X9, self.msg7_mask:mask7, self.msg8_mask:mask8, self.msg9_mask:mask9})
        return diff5

    def calc_diff6(self, X7, X8, X9, mask7, mask8, mask9):
        diff6, = self.sess.run(fetches=[self.diff6], feed_dict={self.X7:X7,
            self.X8:X8, self.X9:X9, self.msg7_mask:mask7, self.msg8_mask:mask8, self.msg9_mask:mask9})
        return diff6
    
    
    def train(self, X1, X2, X3, mask1, mask2, mask3, y1, X4, X5, X6, mask4, mask5, mask6, y2, X7, X8, X9, mask7, mask8, mask9, y3):
        loss,_ = self.sess.run([self.loss,self.optimizer],feed_dict={
            self.X1:X1,self.X2:X2,self.X3:X3,self.msg1_mask:mask1,self.msg2_mask:mask2,self.msg3_mask:mask3,self.label1:y1,
            self.X4:X4,self.X5:X5,self.X6:X6,self.msg4_mask:mask4,self.msg5_mask:mask5,self.msg6_mask:mask6,self.label2:y2,
            self.X7:X7,self.X8:X8,self.X9:X9,self.msg7_mask:mask7,self.msg8_mask:mask8,self.msg9_mask:mask9,self.label3:y3})
        return loss
    
    def save(self, path, epoch=None):
        checkpoint_path = self.saver.save(self.sess, path, global_step=epoch)
        return checkpoint_path
