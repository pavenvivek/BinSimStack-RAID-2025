import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
print (tf.__version__)
import numpy as np
import argparse
import json
from datetime import datetime
from siamese_nonuplet import graphnn
#from utils import *
from utils_top_stat import *
from settings import *
from insert_db import *




parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
        help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=32, #2560, 7, #32,
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
        default='./saved_model_angr_flt_top_ge4_cirrina/graphnn-model_best')
parser.add_argument('--log_path', type=str, default=None,
        help='path for training log')


def get_data_from_db():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[DIFF_DB]
    col = db[DIFF_COLLECTION]
    data_lst = {}


    #print ("Inside get data from db")
    for data in col.find({}, {"_id": 0, "index" : 1, "function": 1, "model" : 1, "transformation" : 1, "data_type" : 1, "diff_lst_1" : 1, "diff_lst_2" : 1, "y_lst_1" : 1, "diff_lst_3" : 1, "diff_lst_4" : 1, "y_lst_2" : 1,
                              "diff_lst_5" : 1, "diff_lst_6" : 1, "y_lst_3" : 1, "uuid" : 1}):

        if data["index"] not in data_lst.keys():
            data_lst[data["index"]] = {}
            data_lst[data["index"]]["function"] = data["function"]
            #data_lst[data["index"]]["model"] = data["model"]
        if data["model"] == "Nonuplet": #"Cirrina":
            data_lst[data["index"]]["diff1"] = data["diff_lst_1"]
            data_lst[data["index"]]["diff2"] = data["diff_lst_2"]
            data_lst[data["index"]]["y1"]   = data["y_lst_1"]
        #elif data["model"] == "Gemini_triplet":
            data_lst[data["index"]]["diff3"] = data["diff_lst_3"]
            data_lst[data["index"]]["diff4"] = data["diff_lst_4"]
            data_lst[data["index"]]["y2"]   = data["y_lst_2"]
        #elif data["model"] == "Palmtree":
            data_lst[data["index"]]["diff5"] = data["diff_lst_5"]
            data_lst[data["index"]]["diff6"] = data["diff_lst_6"]
            data_lst[data["index"]]["y3"]   = data["y_lst_3"]

    info_lst = []
    for k in data_lst.keys():
        info_lst.append((k, data_lst[k]["function"], data_lst[k]["diff1"], data_lst[k]["diff2"], data_lst[k]["diff3"], data_lst[k]["diff4"], data_lst[k]["diff5"], data_lst[k]["diff6"], data_lst[k]["y1"], data_lst[k]["y2"], data_lst[k]["y3"]))
            
    return info_lst


if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("=================================")
    print(args)
    print("=================================")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    

    total_auc2_1 = 0
    total_auc2_2 = 0
    total_auc2_3 = 0
    total_auc_cmb = 0


    data_lst = get_data_from_db()
    test_auc, fpr, tpr, thres, auc2_1, auc2_2, auc2_3, auc_cmb2 = get_auc_epoch_combined_mtx(data_lst)

    total_auc2_1 = total_auc2_1 + auc2_1
    total_auc2_2 = total_auc2_2 + auc2_2
    total_auc2_3 = total_auc2_3 + auc2_3
    total_auc_cmb = total_auc_cmb + auc_cmb2 
      
    print ("\n\n-------------------------------------")
    print ("Cirrina  average accuracy: {}".format(total_auc2_1))
    print ("Gemini   average accuracy: {}".format(total_auc2_2))
    print ("Palmtree average accuracy: {}".format(total_auc2_3))
    print ("Average accuracy combined (Aggregate): {}".format(total_auc_cmb))
    print ("-------------------------------------")
    
