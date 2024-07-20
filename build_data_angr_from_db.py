import numpy as np
from utils import *
import os, subprocess
import argparse
import json
import pymongo
import ast
import re
import random, math

from settings import *
from insert_db import *
#from instr_repo import *
#from gemini_encoding_angr import *
#from cirrina_encoding_angr import *
#from palmtree_encoding_angr import *

fnc_inst_cnt = {}
#trs_cf = "Flatten" #"Virtualize" #"EncodeArithmetic, Virtualize" #"EncodeArithmetic, Flatten" #
    

def generate_training_and_testing_files_from_db():

    testing_fns = []
    validation_fns = []
    training_fns = []
    
    nd_cnt_dct = {}
    skip_functions = []

    test_data = []
    validation_data = []
    train_data = []

    #skipped_funcs = []

    

    #inc_libraries = ["igraph", "dbus", "allegro", "libxml2", "libmicrohttpd", "gsl", "alsa", "libmongoc", "binutils", "libtomcrypt", "imagemagick", "coreutils", "redis", "sqlite", "curl", "musl", "openssl"]
    #inc_libraries = ["igraph", "libmicrohttpd", "gsl", "binutils", "libtomcrypt", "imagemagick", "sqlite", "curl", "musl", "openssl"] #"redis", 

    inc_libraries = ["openssl"]

    client1 = pymongo.MongoClient(MONGO_CLIENT)
    db1 = client1[FUNC_INFO_DB]
    col1 = db1[FUNC_INFO_COLLECTION]

    for data in col1.find({"experiment" : expr_num}, {"_id": 0, "testing_fns" : 1, "validation_fns": 1, "training_fns" : 1, "library" : 1, "experiment" : 1, "trs" : 1}):
        testing_fns = data["testing_fns"]
        validation_fns = data["validation_fns"]
        training_fns = data["training_fns"]
        #exp_num = data["experiment"]
        inc_libraries = data["library"]

    print(f"\n** testing_fns -> {testing_fns}\n")
    print(f"\ntesting_cnt -> {len(testing_fns)}")
    print(f"training_cnt -> {len(training_fns)}\n")
            
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[CODE_ENC_DB]
    col = db[CODE_ENC_COLLECTION]

    model_ins_count = {}
    model_ins_count["Cirrina"] = {"training" : 0, "testing" : 0}
    model_ins_count["Gemini_trp"] = {"training" : 0, "testing" : 0}
    model_ins_count["Palmtree_trp"] = {"training" : 0, "testing" : 0}
    
    inc_models = ["Cirrina", "Gemini_trp", "Palmtree_trp"]

    for model in inc_models:
        print (f"\n\nRetrieving encodings for {model} and {inc_libraries} ...")
        for el in col.find({"model" : model},
                           {"_id": 0, "src" : 1, "n_num": 1, "succs" : 1, "features" : 1, "fname" : 1, "func" : 1, "compiler_flag" : 1, "arch" : 1, "trs" : 1, "library" : 1, "model" : 1}):

            if el["library"] not in ["openssl"]:
                continue
            
            if (el["func"] in testing_fns):
                test_data.append(el)
                model_ins_count[el["model"]]["testing"] = model_ins_count[el["model"]]["testing"] + 1

                if el["func"] not in nd_cnt_dct.keys():
                    nd_cnt_dct[el["func"]] = {}
                    nd_cnt_dct[el["func"]]["inst"] = 1
                    nd_cnt_dct[el["func"]]["testing"] = 1
                else:
                    nd_cnt_dct[el["func"]]["inst"] = nd_cnt_dct[el["func"]]["inst"] + 1
                    nd_cnt_dct[el["func"]]["testing"] = nd_cnt_dct[el["func"]]["testing"] + 1

            elif (el["func"] in training_fns):
                train_data.append(el)
                model_ins_count[el["model"]]["training"] = model_ins_count[el["model"]]["training"] + 1

                if el["func"] not in nd_cnt_dct.keys():
                    nd_cnt_dct[el["func"]] = {}
                    nd_cnt_dct[el["func"]]["inst"] = 1
                    nd_cnt_dct[el["func"]]["training"] = 1
                else:
                    nd_cnt_dct[el["func"]]["inst"] = nd_cnt_dct[el["func"]]["inst"] + 1
                    nd_cnt_dct[el["func"]]["training"] = nd_cnt_dct[el["func"]]["training"] + 1
            #else:
            #    pass
       

    for k,v in nd_cnt_dct.items():
       if nd_cnt_dct[k]["inst"] != 6:
         print (f"inst_cnt != 6 =====> {k} -> {v}")
         if 'testing' in nd_cnt_dct[k].keys():
             print (f"testing cnt -> {nd_cnt_dct[k]['testing']}")
         if 'training' in nd_cnt_dct[k].keys():
             print (f"training cnt -> {nd_cnt_dct[k]['training']}")
         skip_functions.append(k)


    #print ("\n----------------------------\n")
    with open("./data/train_data.json", 'w') as f:
       for el in train_data:
         if el["func"] not in skip_functions:
           #print("including fn:{}, arch: {}, flg: {}, trs: {}, lib: {}, model: {}, n_num: {} for training !".format(el["fname"], el["arch"], el["compiler_flag"], el["trs"], el["library"], el["model"], el["n_num"]))
           json.dump(el, f)
           f.write("\n")
    
    print ("\n----------------------------\n")
    with open("./data/test_data.json", 'w') as f:
       for el in test_data:
         if el["func"] not in skip_functions:
           print("including fn:{}, arch: {}, flg: {}, trs: {}, lib: {}, model: {}, n_num: {} for testing !".format(el["fname"], el["arch"], el["compiler_flag"], el["trs"], el["library"], el["model"], el["n_num"]))
           json.dump(el, f)
           f.write("\n")

    print ("\n----------------------------\n")

    # print ("\n----------------------------\n")
    # with open("./data/validation_data.json", 'w') as f:
    #    for el in validation_data:
    #      if el["func"] not in skip_functions:
    #        print("including fn:{}, arch: {}, flg: {}, trs: {}, lib: {}, model: {}, n_num: {} for validation !".format(el["fname"], el["arch"], el["compiler_flag"], el["trs"], el["library"], el["model"], el["n_num"]))
    #        json.dump(el, f)
    #        f.write("\n")
    # print ("\n----------------------------\n")
    
    print(f"\ntesting_cnt -> {len(testing_fns)}")
    #print(f"validation_cnt -> {len(validation_fns)}")
    print(f"training_cnt -> {len(training_fns)}\n")
    #print(f"skipped functions -> {skipped_funcs}")

    for k,v in model_ins_count.items():
        print (f"model: {k}, training cnt: {model_ins_count[k]['training']}, testing_cnt: {model_ins_count[k]['testing']}")
    
    
    print ("Exiting Script...")   


if __name__ == '__main__':

    #insert_enc_data_to_db()
    generate_training_and_testing_files_from_db() # pass experiment number as arg
