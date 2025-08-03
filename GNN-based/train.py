import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


import tensorflow as tf
print (tf.__version__)

import numpy as np
import sys
import argparse
import json
from datetime import datetime
from siamese_triplet import graphnn
from utils import *
#from utils_cflg import *
from settings import *

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
        help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=7, #200, #32, #7, #2560
        help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64,
        help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2, 
        help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64,
        help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5,
        help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4,
        help='learning rate')
parser.add_argument('--epoch', type=int, default=100,
        help='epoch number')
parser.add_argument('--batch_size', type=int, default=1,
        help='batch size')
parser.add_argument('--load_path', type=str, default=None,
        help='path for model loading')
parser.add_argument('--save_path', type=str,
        default='./saved_model/graphnn-model')
parser.add_argument('--log_path', type=str, default=None,
        help='path for training log')




if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print(f"Paramters: {args}")
    print()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    Dtype = args.dtype
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    MAX_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LOAD_PATH = args.load_path
    SAVE_PATH = args.save_path
    LOG_PATH = args.log_path

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 25

    FUNC_NAME_DICT = {}

    # Process the input graphs
    F_NAME = ["./data/train_data.json"]
            
    print ("Training functions: {}".format(F_NAME))        
    FUNC_NAME_DICT = get_f_dict(F_NAME)
     
    Gs, classes = read_graph(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM)
    print ("{} graphs, {} functions".format(len(Gs), len(classes)))

    if os.path.isfile('./data/class_perm.npy'):
        perm = np.load('./data/class_perm.npy')
    else:
        perm = np.random.permutation(len(classes))
        np.save('./data/class_perm.npy', perm)
    if len(perm) < len(classes):
        perm = np.random.permutation(len(classes))
        np.save('./data/class_perm.npy', perm)

  
    Gs_train, classes_train =\
            partition_data(Gs,classes,[1],perm)
    # Model
    gnn = graphnn(
            N_x = NODE_FEATURE_DIM,
            Dtype = Dtype, 
            N_embed = EMBED_DIM,
            depth_embed = EMBED_DEPTH,
            N_o = OUTPUT_DIM,
            ITER_LEVEL = ITERATION_LEVEL,
            lr = LEARNING_RATE
        )
    gnn.init(LOAD_PATH, LOG_PATH)

    valid_epoch = None
    auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_train, classes_train,
            BATCH_SIZE, load_data=valid_epoch)
    gnn.say("Initial Training AUC = {0} @ {1}".format(auc, datetime.now()))

    best_auc = 0
    for i in range(1, MAX_EPOCH+1):
        l = train_epoch(gnn, Gs_train, classes_train, BATCH_SIZE)
        gnn.say("Epoch {3}/{0}, loss = {1} @ {2}".format(
            MAX_EPOCH, l, datetime.now(), i))

        if (i % TEST_FREQ == 0):
            auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_train, classes_train,
                    BATCH_SIZE, load_data=valid_epoch)
            gnn.say("Testing model: training auc = {0} @ {1}".format(
                auc, datetime.now()))

            if auc > best_auc:
                path = gnn.save(SAVE_PATH+'_best')
                best_auc = auc
                gnn.say("Model saved in {}".format(path))

        if (i % SAVE_FREQ == 0):
            path = gnn.save(SAVE_PATH, i)
            gnn.say("Model saved in {}".format(path))
    
