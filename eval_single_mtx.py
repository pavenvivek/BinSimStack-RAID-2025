import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
print (tf.__version__)
import numpy as np
import argparse, subprocess
import json
from datetime import datetime
from siamese_nonuplet import graphnn
#from utils_top_mtx import *
from utils_top_mtx_cflg import *
from settings import *
from insert_db import *
#from inc_fns_lst_viz_patch import *


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
        help='visible gpu device')
parser.add_argument('--fea_dim1', type=int, default=32, #32, #7, #200, #2560, #32,
        help='feature dimension')
parser.add_argument('--fea_dim2', type=int, default=7,
        help='feature dimension')
parser.add_argument('--fea_dim3', type=int, default=200,
        help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64, #64,
        help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2, #2,
        help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64, #64,
        help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5, #5,
        help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4,
        help='learning rate')
parser.add_argument('--epoch', type=int, default=100,
        help='epoch number')
parser.add_argument('--batch_size', type=int, default=1,
        help='batch size')
parser.add_argument('--load_path', type=str,
        default='./saved_model/graphnn-model_best')
parser.add_argument('--log_path', type=str, default=None,
        help='path for training log')



if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print(f"Parameters: {args}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    Dtype = args.dtype

    NODE_FEATURE_DIM1 = args.fea_dim1
    NODE_FEATURE_DIM2 = args.fea_dim2
    NODE_FEATURE_DIM3 = args.fea_dim3
    

    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    MAX_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LOAD_PATH = args.load_path
    LOG_PATH = args.log_path

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 25

    FUNC_NAME_DICT = {}

    # Process the input graphs
    F_NAME = ["./data/test_data.json"]  

    print ("Functions: {}".format(F_NAME))
    FUNC_NAME_DICT = get_f_dict(F_NAME)
    print (f"Func name: {FUNC_NAME_DICT}\n\n")

    Gs, classes = read_graph(F_NAME, FUNC_NAME_DICT) #, NODE_FEATURE_DIM)

    avg_acu = 0
    total_auc = 0
    avg_acc2 = 0
    total_auc2 = 0
    total_auc2_1 = 0
    total_auc2_2 = 0

    #for i in range(0, 1):

    perm = [i for i in range(0, len(classes))]

    Gs_test, classes_test =\
          partition_data(Gs,classes,[1],perm)

    gnn = graphnn(
          N_x1 = NODE_FEATURE_DIM1,
          N_x2 = NODE_FEATURE_DIM2,
          N_x3 = NODE_FEATURE_DIM3,
          Dtype = Dtype, 
          N_embed = EMBED_DIM,
          depth_embed = EMBED_DEPTH,
          N_o = OUTPUT_DIM,
          ITER_LEVEL = ITERATION_LEVEL,
          lr = LEARNING_RATE
      )
    gnn.init(LOAD_PATH, LOG_PATH)

    print (f"len(classes) -> {len(classes_test)}")

    uuid = subprocess.getoutput("uuidgen --random") # "2585f7ec-4cf8-4174-add3-1494eb5312fa"
    st_ind = 0

    # openssl
    # 1096 - flt
    # 1068 - viz
    # O0 <-> O1 - 1051
    # O0 <-> O2 - 991
    # O0 <-> O3 - 297
    end_ind = 297

    k = 0
    for j in range(st_ind, end_ind):
        info_lst = []
        fn_lst = list(FUNC_NAME_DICT.keys())
        st = j #st_ind
        end = st + 1
        cls_st = 0
        BATCH_SIZE = 1500
        diff_lst_1_x = []
        diff_lst_2_x = []
        y1_lst_x = []
        diff_lst_3_x = []
        diff_lst_4_x = []
        y2_lst_x = []
        diff_lst_5_x = []
        diff_lst_6_x = []
        y3_lst_x = []

        while cls_st < end_ind:

            test_epoch, test_ids, ids = generate_epoch_pair(
                Gs_test, classes_test, BATCH_SIZE, cls_st, st, end, output_id=True, mode = "Testing")

            #print (f"ids len -> {len(ids)}")
            #print (f"outside here 1 ! cls_st -> {cls_st}")

            if len(ids) != 0:
                diff_lst_1, diff_lst_2, y1_lst, diff_lst_3, diff_lst_4, y2_lst, diff_lst_5, diff_lst_6, y3_lst = get_auc_epoch(
                      gnn, Gs_test, classes_test, BATCH_SIZE, cls_st, load_data=test_epoch, ids=ids, mode="Testing")
                #print (f"outside here 2 ! cls_st -> {cls_st}")
                #print (f"diff_lst_1 keys len -> {len(diff_lst_1.keys())}")

                diff_lst_1_x = diff_lst_1_x + diff_lst_1[0]
                diff_lst_2_x = diff_lst_2_x + diff_lst_2[0]
                y1_lst_x = y1_lst_x + y1_lst[0]

                diff_lst_3_x = diff_lst_3_x + diff_lst_3[0]
                diff_lst_4_x = diff_lst_4_x + diff_lst_4[0]
                y2_lst_x = y2_lst_x + y2_lst[0]

                diff_lst_5_x = diff_lst_5_x + diff_lst_5[0]
                diff_lst_6_x = diff_lst_6_x + diff_lst_6[0]
                y3_lst_x = y3_lst_x + y3_lst[0]
                
            cls_st = cls_st + BATCH_SIZE

        #for i in range(0, len(diff_lst_1.keys())):
        #info_lst.append((st, fn_lst[st].split("_openssl")[0], model, transformation, inc_libraries, diff_lst_1_x, diff_lst_2_x, y1_lst_x, uuid))
        info_lst.append((st, fn_lst[st], model, transformation, inc_libraries, diff_lst_1_x, diff_lst_2_x, y1_lst_x, diff_lst_3_x, diff_lst_4_x, y2_lst_x, diff_lst_5_x, diff_lst_6_x, y3_lst_x, uuid))
        #print (f"cls_st -> {cls_st}, diff_lst_1_x len -> {len(diff_lst_1_x)}")

        #print (f"diff_lst 1 -> {diff_lst_1_x[:20]}")
        #print (f"cls_st -> {cls_st}, diff_lst_1_x len -> {len(diff_lst_1_x)}")
        #print (f"diff_lst 2 -> {diff_lst_3_x[:20]}")
        #print (f"cls_st -> {cls_st}, diff_lst_3_x len -> {len(diff_lst_3_x)}")
        #break

        insert_db_mongo(info_lst)

        #k = k + 1
        #if k > 2:
        #    break
        #print("\nj : {}, info_lst len -> {}\n".format(j, len(info_lst)))


