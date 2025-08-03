import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
print (tf.__version__)
import numpy as np
import argparse, subprocess
import json
from datetime import datetime
from siamese_triplet import graphnn
from utils_top_mtx import *
#from utils_top_mtx_cflg import *
from settings import *
from insert_db import *
#from inc_fns_lst_viz_patch import *


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0',
        help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=200, #32, #7, #200, #2560, #32,
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


def get_data_from_db():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[DIFF_DB]
    col = db[DIFF_COLLECTION]
    data_lst = []


    #print ("Inside get data from db")
    for data in col.find({}, {"_id": 0, "index" : 1, "function": 1, "model" : 1, "transformation" : 1, "diff_lst_1" : 1, "diff_lst_2" : 1, "y_lst" : 1, "uuid" : 1}):

        if data["model"] == "Palmtree_trp": # Cirrina # Gemini_trp
            data_lst.append((data["index"], data["function"], data["model"], data["transformation"], data["diff_lst_1"], data["diff_lst_2"], data["y_lst"]))

    return data_lst


if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print(f"Parameters: {args}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
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
    LOG_PATH = args.log_path

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 25

    FUNC_NAME_DICT = {}

    # Process the input graphs
    F_NAME = ["./data/test_data.json"]  

    print ("Functions: {}".format(F_NAME))

    #-- new logic to remove mongo inserted order (also patch for viz)
    func_lst = get_func_from_db()
    func_info = get_func_info(F_NAME)

    #func_lst_n = []
    
    func_dict = {}
    for i in range(0, len(func_info)):
        fname, func, lib = func_info[i]

        #func_lst_n.append(func)
        func_dict[fname] = func_lst.index(func)

    FUNC_NAME_DICT = dict(sorted(func_dict.items(), key=lambda item: item[1])) #func_dict #get_f_dict(F_NAME)

    #print (f"\n\nnew func list -> {FUNC_NAME_DICT.keys()}\n\n")
    print (f"new Func name: {FUNC_NAME_DICT}\n\n")
    #--
    
    #FUNC_NAME_DICT = get_f_dict(F_NAME)
    #print (f"Func name: {FUNC_NAME_DICT}\n\n")

    Gs, classes = read_graph(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM)

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

    #ids = []
    #if os.path.isfile('./data/test_ids1.json'):
     #with open('./data/test.json') as inf:
     #    test_ids = json.load(inf)
     #with open('./data/test_ids.json') as inf:
     #   ids = json.load(inf)
     #test_epoch = generate_epoch_pair_3(
     #   Gs_test, classes_test, BATCH_SIZE, load_id=test_ids, mode = "Testing")
     #print ("loading existing data !")
    #else:
    #    test_epoch, test_ids, ids = generate_epoch_pair_3(
    #          Gs_test, classes_test, BATCH_SIZE, output_id=True, mode = "Testing")


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

    print (f"len(classes) -> {len(classes_test)}")

    uuid = subprocess.getoutput("uuidgen --random") # "2585f7ec-4cf8-4174-add3-1494eb5312fa"
    st_ind = 0

    # openssl
    # 1096 - flt
    # 1068 - viz
    # O0 <-> O1 - 1051
    # O0 <-> O2 - 991
    # O0 <-> O3 - 297
    # redis
    # 698 - flt
    # 602 - viz
    # O0 <-> O1 - 709
    # O0 <-> O2 - 648
    # O0 <-> O3 - 190
    # binutils
    # 968 - flt
    # 778 - viz
    # O0 <-> O1 - 1350
    # O0 <-> O2 - 1196
    # O0 <-> O3 - 529
    # imagemagick
    # 506 - flt
    # 422 - viz
    # O0 <-> O1 - 474
    # O0 <-> O2 - 450
    # O0 <-> O3 - 193
    # libxml2
    # 669 - flt
    # 428 - viz
    # O0 <-> O1 - 452
    # O0 <-> O2 - 370
    # O0 <-> O3 - 164
    end_ind = 1096  #2997 #3173

    for j in range(st_ind, end_ind):
        info_lst = []
        fn_lst = list(FUNC_NAME_DICT.keys())
        st = j #st_ind
        end = st + 1
        cls_st = 0
        BATCH_SIZE = 50
        diff_lst_1_x = []
        diff_lst_2_x = []
        y_lst_x = []

        while cls_st < end_ind:

            test_epoch, test_ids, ids = generate_epoch_pair_3(
                Gs_test, classes_test, BATCH_SIZE, cls_st, st, end, output_id=True, mode = "Testing")

            #print (f"ids len -> {len(ids)}")
            #print (f"outside here 1 ! cls_st -> {cls_st}")

            if len(ids) != 0:
                diff_lst_1, diff_lst_2, y_lst = get_auc_epoch(
                      gnn, Gs_test, classes_test, BATCH_SIZE, cls_st, load_data=test_epoch, ids=ids, mode="Testing")
                #print (f"outside here 2 ! cls_st -> {cls_st}")

                #print (f"diff_lst_1 len -> {len(diff_lst_1.keys())}")
                diff_lst_1_x = diff_lst_1_x + diff_lst_1[0]
                diff_lst_2_x = diff_lst_2_x + diff_lst_2[0]
                y_lst_x = y_lst_x + y_lst[0]
            
            cls_st = cls_st + BATCH_SIZE

        #for i in range(0, len(diff_lst_1.keys())):
        #info_lst.append((st, fn_lst[st].split("_openssl")[0], model, transformation, inc_libraries, diff_lst_1_x, diff_lst_2_x, y_lst_x, uuid))
        info_lst.append((st, fn_lst[st], model, transformation, inc_libraries, diff_lst_1_x, diff_lst_2_x, y_lst_x, uuid))
        #print (f"cls_st -> {cls_st}, diff_lst_1_x len -> {len(diff_lst_1_x)}")

        #print (f"diff_lst -> {diff_lst_2_x}")
        #print (f"cls_st -> {cls_st}, diff_lst_1_x len -> {len(diff_lst_1_x)}")
        #break
    
        insert_db_mongo(info_lst)
        #if j > 1:
        #    break
        #print("\nj : {}, info_lst len -> {}\n".format(j, len(info_lst)))


