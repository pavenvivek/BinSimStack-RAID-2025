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
from instr_repo import *
from insert_db import *
from gemini_encoding_angr import *
#from cirrina_encoding_angr import *
#from palmtree_encoding_angr import *
#from inc_fns_lst_flt import *
#from inc_fns_lst_viz import *

fnc_inst_cnt = {}
trs_cf = "N/A" #"Virtualize" #"Flatten" #"EncodeArithmetic, Virtualize" #"EncodeArithmetic, Flatten" #


def get_data_from_db():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[TREES_DB]
    col = db[CODE_GRAPH_COLLECTION]
    data_lst = []
    data_lst_dc = {}
    unique_adj_dict = {}
    unique_adj_dict_dc = {}
    unique_adj_dict_all = {}
    unique_asm = {}
    rev_inst_cnt = {}
    max_len = 0
    skip_fn_name_lst = []
    func_add_lst = []
    clash_cnt = 0
    le4_cnt = 0
    ins1_cnt = 0
    insge2_cnt = 0
    skip_fn_mul_inst = []
    dup_ins_skip = 0

    #--- use this block only for O2 and O3 (to avoid O1 duplicates)
    client1 = pymongo.MongoClient(MONGO_CLIENT)
    db1 = client1[FUNC_INFO_DB]
    col1 = db1[FUNC_INFO_COLLECTION]

    for data in col1.find({"experiment" : "header_O3_lib2"}, {"_id": 0, "testing_fns" : 1, "validation_fns": 1, "training_fns" : 1, "library" : 1, "experiment" : 1, "trs" : 1}):
        inc_fns = data["training_fns"]
    #---
    
    #inc_libraries1 = ["igraph", "dbus", "allegro", "libxml2", "libmicrohttpd", "gsl", "alsa", "libmongoc", "binutils", "libtomcrypt", "imagemagick", "coreutils", "sqlite", "curl", "musl"] #, "openssl"] "redis", 
    inc_libraries1 = ["igraph", "dbus", "allegro", "libmicrohttpd", "gsl", "alsa", "libmongoc", "libtomcrypt", "coreutils", "sqlite", "curl", "musl"] #, "openssl"]
    #inc_libraries1 = ["openssl", "redis", "binutils", "imagemagick", "libxml2"]
    #inc_libraries1 = ["openssl"]
    #{"library" : {'$in' : inc_libraries1}}

    
    for data in col.find({"library" : {'$in' : inc_libraries1}}, {"_id": 0, "graph" : 1, "ref_dict": 1, "adj_dict" : 1, "vex_ir" : 1, "calls" : 1, "consts" : 1, "str_consts_cnt" : 1, "instr_str" : 1, "instr_byte" : 1, "function" : 1, "filename" : 1, "transformation" : 1, "compiler" : 1, "compiler_flags" : 1, "architecture" : 1, "library" : 1, "uuid" : 1}):
       
       if data["architecture"] not in ["x86_64"]:
           print ("skipping in arch !")
           continue
       
       if data["compiler"] != "gcc":
           print ("skipping in compiler !")
           continue

       if data["compiler_flags"] not in ["O3", "O0"]:
           continue
       
       #--- use this block only for O2 and O3 (to avoid O1 duplicates)
       if data["function"] not in inc_fns:
           continue
       #---
       
       if data["transformation"] not in ["N/A"]:
           print ("skipping in trs !")
           continue
       
       # filtering duplicate insertion of same funcs
       if data["function"] not in unique_asm.keys():
           unique_asm[data["function"]] = {}
           unique_asm[data["function"]]["adj_dict"] = data["adj_dict"]
           unique_asm[data["function"]]["instr_str"] = ast.literal_eval(data["instr_str"])
           unique_asm[data["function"]]["cflg"] = [data["compiler_flags"]]
       elif data["compiler_flags"] in unique_asm[data["function"]]["cflg"]:
           clash_cnt = clash_cnt + 1
           skip_fn_mul_inst.append(data["function"])
           continue # skipping repeating function with same compilation flag (duplicates)
       elif data["adj_dict"] == unique_asm[data["function"]]["adj_dict"] and data["compiler_flags"] not in unique_asm[data["function"]]["cflg"]:

           unique_asm[data["function"]]["cflg"].append(data["compiler_flags"])

           asm1 = unique_asm[data["function"]]["instr_str"].values()
           asm2 = ast.literal_eval(data["instr_str"]).values()

           asm1 = [op for inst in asm1 for op in inst]
           asm2 = [op for inst in asm2 for op in inst]

           
           if len(asm1) == len(asm2):
               skip_func = 1
               for i in range(0, len(asm1)):
                   ins1 = asm1[i]
                   ins2 = asm2[i]

                   ins1 = re.sub("0x[0-9a-f]*", "addr", ins1)
                   ins1 = re.sub("(?<![a-z])r[0-9a-z]*", "reg", ins1)
                   ins2 = re.sub("0x[0-9a-f]*", "addr", ins2)
                   ins2 = re.sub("(?<![a-z])r[0-9a-z]*", "reg", ins2)

                   if ins1 != ins2:
                       clash_cnt = clash_cnt + 1
                       skip_func = 0
                       break
                       
               if skip_func == 1:
                   dup_ins_skip = dup_ins_skip + 1
                   continue
           else:
               pass
       else:
           pass

       
       unique_adj_dict_all[data["adj_dict"]] = [data["function"]]
       if (data["function"] + "___" + data["library"]) not in rev_inst_cnt.keys():
               rev_inst_cnt[data["function"] + "___" + data["library"]] = 1
       else:
               rev_inst_cnt[data["function"] + "___" + data["library"]] = rev_inst_cnt[data["function"] + "___" + data["library"]] + 1 
       
       if len(data["ref_dict"]) > max_len:
           max_len = len(data["ref_dict"])
       
       data_lst.append((ast.literal_eval(data["graph"]), ast.literal_eval(data["ref_dict"]), ast.literal_eval(data["adj_dict"]), ast.literal_eval(data["vex_ir"]), ast.literal_eval(data["calls"]), ast.literal_eval(data["consts"]), ast.literal_eval(data["str_consts_cnt"]), ast.literal_eval(data["instr_str"]), ast.literal_eval(data["instr_byte"]), data["function"], data["filename"], data["transformation"], data["compiler"], data["compiler_flags"], data["architecture"], data["library"], data["uuid"]))

       data_lst_dc[data["uuid"]] = (ast.literal_eval(data["graph"]), ast.literal_eval(data["ref_dict"]), ast.literal_eval(data["adj_dict"]), ast.literal_eval(data["vex_ir"]), ast.literal_eval(data["calls"]), ast.literal_eval(data["consts"]), ast.literal_eval(data["str_consts_cnt"]), ast.literal_eval(data["instr_str"]), ast.literal_eval(data["instr_byte"]), data["function"], data["filename"], data["transformation"], data["compiler"], data["compiler_flags"], data["architecture"], data["library"], data["uuid"])

    rev_root_fns_dict = {}
    skip_fn = {}
    c = 0
    print ("revised unique instance count ...")    
    for k,v in rev_inst_cnt.items():
      if v == 2:
          print ("*** including {} => {}, class : {}".format(k, v, c))
          rev_root_fns_dict[k] = c
          c = c + 1

          if "musl" in k:
              func_add_lst.append(k.split("___musl")[0])
          elif "openssl" in k:
              func_add_lst.append(k.split("___openssl")[0])
          elif "curl" in k:
              func_add_lst.append(k.split("___curl")[0])
          elif "sqlite" in k:
              func_add_lst.append(k.split("___sqlite")[0])
          elif "redis" in k:
              func_add_lst.append(k.split("___redis")[0])
          elif "coreutils" in k:
              func_add_lst.append(k.split("___coreutils")[0])
          elif "imagemagick" in k:
              func_add_lst.append(k.split("___imagemagick")[0])
          elif "libtomcrypt" in k:
              func_add_lst.append(k.split("___libtomcrypt")[0])
          elif "binutils" in k:
              func_add_lst.append(k.split("___binutils")[0])
          elif "libmongoc" in k:
              func_add_lst.append(k.split("___libmongoc")[0])
          elif "alsa" in k:
              func_add_lst.append(k.split("___alsa")[0])
          elif "gsl" in k:
              func_add_lst.append(k.split("___gsl")[0])
          elif "libmicrohttpd" in k:
              func_add_lst.append(k.split("___libmicrohttpd")[0])
          elif "libxml2" in k:
              func_add_lst.append(k.split("___libxml2")[0])
          elif "allegro" in k:
              func_add_lst.append(k.split("___allegro")[0])
          elif "dbus" in k:
              func_add_lst.append(k.split("___dbus")[0])
          elif "igraph" in k:
              func_add_lst.append(k.split("___igraph")[0])
          else:
              insge2_cnt = insge2_cnt + 1
              print ("*** skipping 1 invalid inclusion {} => {}, class : {}".format(k, v, c))
      else:
          if v == 1:
              ins1_cnt = ins1_cnt + 1
          
          print ("skipping {} => {}, insufficient instance".format(k, v))
          skip_fn[k] = 1
          
          if "musl" in k:
              skip_fn_name_lst.append(k.split("___musl")[0])
          elif "openssl" in k:
              skip_fn_name_lst.append(k.split("___openssl")[0])
          elif "curl" in k:
              skip_fn_name_lst.append(k.split("___curl")[0])
          elif "sqlite" in k:
              skip_fn_name_lst.append(k.split("___sqlite")[0])
          elif "redis" in k:
              skip_fn_name_lst.append(k.split("___redis")[0])
          elif "coreutils" in k:
              skip_fn_name_lst.append(k.split("___coreutils")[0])
          elif "imagemagick" in k:
              skip_fn_name_lst.append(k.split("___imagemagick")[0])
          elif "libtomcrypt" in k:
              skip_fn_name_lst.append(k.split("___libtomcrypt")[0])
          elif "binutils" in k:
              skip_fn_name_lst.append(k.split("___binutils")[0])
          elif "libmongoc" in k:
              skip_fn_name_lst.append(k.split("___libmongoc")[0])
          elif "alsa" in k:
              skip_fn_name_lst.append(k.split("___alsa")[0])
          elif "gsl" in k:
              skip_fn_name_lst.append(k.split("___gsl")[0])
          elif "libmicrohttpd" in k:
              skip_fn_name_lst.append(k.split("___libmicrohttpd")[0])
          elif "libxml2" in k:
              skip_fn_name_lst.append(k.split("___libxml2")[0])
          elif "allegro" in k:
              skip_fn_name_lst.append(k.split("___allegro")[0])
          elif "dbus" in k:
              skip_fn_name_lst.append(k.split("___dbus")[0])
          elif "igraph" in k:
              skip_fn_name_lst.append(k.split("___igraph")[0])
          else:
              #insge2_cnt = insge2_cnt + 1
              print ("*** skipping 2 invalid inclusion {} => {}, class : {}".format(k, v, c))

    print (f"clash count : {clash_cnt}")
    print (f"le4_cnt : {le4_cnt}")
    print (f"ins1_cnt : {ins1_cnt}")
    print (f"insge2_cnt : {insge2_cnt}")
    print (f"dup_ins_skip : {dup_ins_skip}")

    return (data_lst, max_len, rev_root_fns_dict, skip_fn, skip_fn_name_lst, skip_fn_mul_inst, func_add_lst)





def insert_enc_data_to_db():    

    print ("Generating training and testing files for ML models...")

    data_lst, max_len, root_fns_dict, skip_fn, skip_fn_name_lst, skip_fn_mul_inst, func_add_lst = get_data_from_db()
    

    #print ("skip_fn : {}".format(skip_fn.keys()))
    #print ("skip_fn_name_lst : {}".format(skip_fn_name_lst))
    print ("skip_fn_name_lst len : {}".format(len(skip_fn_name_lst)))
    print ("skip_fn_mul_inst: {}".format(list(set(skip_fn_mul_inst))))
    print ("skip_fn_mul_inst len : {}".format(len(list(set(skip_fn_mul_inst)))))
    print ("func_add_lst len : {}".format(len(func_add_lst)))
    print ("func_add_lst set len : {}".format(len(set(func_add_lst))))
    
    fn_lst = func_add_lst #get_functions_from_db()
    #fn_lst = list(set(fn_lst) - set(skip_fn_name_lst))

    random.shuffle(fn_lst)
    testing_cnt  = math.ceil(len(fn_lst) * .20)
    validation_cnt = 0 #testing_cnt
    
    testing_fns  = fn_lst[:testing_cnt] #testing_fns_O0_O2
    training_fns = list(set(fn_lst) - set(testing_fns)) #training_fns_cflg_O0_O2

    print(f"\ntesting_cnt -> {len(testing_fns)}")
    print(f"training_cnt -> {len(training_fns)}\n")

    print(f"\ntesting_fns -> {testing_fns}\n")
    #print(f"\ntraining_fns -> {training_fns}\n")

    #--- use this block only for O2 and O3 (to avoid O1 duplicates) # comment the next block (till function end) when running this
    #info_lst = []
    ##inc_libs = ["openssl", "redis", "binutils", "imagemagick", "libxml2"]
    #inc_libs = ["igraph", "dbus", "allegro", "libmicrohttpd", "gsl", "alsa", "libmongoc", "libtomcrypt", "coreutils", "sqlite", "curl", "musl"]
    #info_lst.append(([], [], list(set(fn_lst)), inc_libs, "header_O3_lib2", trs_cf))
    #insert_db_function_info_mongo(info_lst)
    #print(f"\nfns_cnt -> {len(list(set(fn_lst)))}")
    #---
    
    print ("Generating Layer Encoding Data ...")
    data = encode_data(data_lst)

    data_enc_lst = []
    for el in data:
       if el["fname"] not in skip_fn.keys():
            if el["func"] in skip_fn_mul_inst:
                print(f"skipping duplicate multi instance function {el['func']}")
                continue

            data_enc_lst.append((el["src"], el["n_num"], el["succs"], el["features"], el["fname"], el["func"], el["compiler_flag"], el["arch"], el["trs"], el["library"], el["model"]))

    insert_db_encodings_mongo(data_enc_lst)
    print (f"Inserted encoding data to db ...")


def generate_training_and_testing_files_from_db(model, exp_num="0", pull_test_data_from_db=False):

    testing_fns = []
    validation_fns = []
    training_fns = []
    
    nd_cnt_dct = {}
    skip_functions = []

    test_data = []
    validation_data = []
    train_data = []

    #skipped_funcs = []

    
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[CODE_ENC_DB]
    col = db[CODE_ENC_COLLECTION]

    #inc_libraries = ["igraph", "dbus", "allegro", "libxml2", "libmicrohttpd", "gsl", "alsa", "libmongoc", "binutils", "libtomcrypt", "imagemagick", "coreutils", "redis", "sqlite", "curl", "musl", "openssl"]
    #inc_libraries = ["igraph", "libmicrohttpd", "gsl", "binutils", "libtomcrypt", "imagemagick", "sqlite", "curl", "musl", "openssl"] #"redis", 

    inc_libraries = ["openssl", "redis", "binutils", "imagemagick", "libxml2"] #["libxml2"]

    if pull_test_data_from_db:

        client = pymongo.MongoClient(MONGO_CLIENT)
        db1 = client[FUNC_INFO_DB]
        col1 = db1[FUNC_INFO_COLLECTION]

        for data in col1.find({"experiment" : exp_num}, {"_id": 0, "testing_fns" : 1, "validation_fns": 1, "training_fns" : 1, "library" : 1, "experiment" : 1, "trs" : 1}):
            testing_fns = data["testing_fns"]
            validation_fns = data["validation_fns"]
            training_fns = data["training_fns"]
            exp_num = data["experiment"]
            inc_libraries = data["library"]

        print(f"\n** testing_fns -> {testing_fns}\n")
            
    else:

        # fn_lst1 = []
        # for data in col.find({"library" : "redis", "model" : "Gemini_trp"}, {"_id": 0, "func" : 1}):
        #     fn_lst1.append(data["func"])

        # fn_lst2 = []
        # for data in col.find({"library" : "redis", "model" : "Palmtree_trp"}, {"_id": 0, "func" : 1}):
        #     fn_lst2.append(data["func"])

        # missing_fns = list(set(fn_lst1) - set(fn_lst2))
        # print (f"missing_fns -> {missing_fns}")
            
        # return
    
        fn_lst = []
        for data in col.find({"library" : {'$in' : inc_libraries}}, {"_id": 0, "func" : 1}):
            fn_lst.append(data["func"])

        fn_lst = list(set(fn_lst))    
        random.shuffle(fn_lst)
        testing_cnt  = math.ceil(len(fn_lst) * .20)
        #validation_cnt = 0 #testing_cnt

        testing_fns  = fn_lst[:testing_cnt] #testing_fns_ge0 #
        validation_fns  = [] #fn_lst[testing_cnt:(testing_cnt + validation_cnt)] #[] #validation_fns_ge0 #
        training_fns = list(set(fn_lst) - set(testing_fns + validation_fns + vuln_fns))

        info_lst = []
        info_lst.append((testing_fns, validation_fns, training_fns, inc_libraries, exp_num, trs_cf))
        insert_db_function_info_mongo(info_lst)

        print (f"Inserted function info to db ...")
        print(f"\ntesting_fns -> {testing_fns}\n")
        
    n_num_test = 0
    n_num_vald = 0
    n_num_train = 0
    n_num_test_1 = 0
    n_num_vald_1 = 0
    n_num_train_1 = 0

    print (f"\n\nGenerating Training and Testing JSON files for {model} and {inc_libraries} ...")
    for el in col.find({"library" : {'$in' : inc_libraries}, "model" : model},
                       {"_id": 0, "src" : 1, "n_num": 1, "succs" : 1, "features" : 1, "fname" : 1, "func" : 1, "compiler_flag" : 1, "arch" : 1, "trs" : 1, "library" : 1, "model" : 1}):

        if (el["func"] in testing_fns):
            test_data.append(el)

            if el["func"] not in nd_cnt_dct.keys():
                nd_cnt_dct[el["func"]] = {}
                nd_cnt_dct[el["func"]]["inst"] = 1
                nd_cnt_dct[el["func"]]["n_num"] = el["n_num"]
            else:
                nd_cnt_dct[el["func"]]["inst"] = nd_cnt_dct[el["func"]]["inst"] + 1

                if nd_cnt_dct[el["func"]]["n_num"] != el["n_num"]:
                    n_num_test = n_num_test + 1
                else:
                    n_num_test_1 = n_num_test_1 + 1

        elif (el["func"] in validation_fns):
            validation_data.append(el)

            if el["func"] not in nd_cnt_dct.keys():
                nd_cnt_dct[el["func"]] = {}
                nd_cnt_dct[el["func"]]["inst"] = 1
                nd_cnt_dct[el["func"]]["n_num"] = el["n_num"]
            else:
                nd_cnt_dct[el["func"]]["inst"] = nd_cnt_dct[el["func"]]["inst"] + 1

                if nd_cnt_dct[el["func"]]["n_num"] != el["n_num"]:
                    n_num_vald = n_num_vald + 1
                else:
                    n_num_vald_1 = n_num_vald_1 + 1

        elif (el["func"] in training_fns):
            train_data.append(el)


            if el["func"] not in nd_cnt_dct.keys():
                nd_cnt_dct[el["func"]] = {}
                nd_cnt_dct[el["func"]]["inst"] = 1
                nd_cnt_dct[el["func"]]["n_num"] = el["n_num"]
            else:
                nd_cnt_dct[el["func"]]["inst"] = nd_cnt_dct[el["func"]]["inst"] + 1

                if nd_cnt_dct[el["func"]]["n_num"] != el["n_num"]:
                    n_num_train = n_num_train + 1
                else:
                    n_num_train_1 = n_num_train_1 + 1
       

    for k,v in nd_cnt_dct.items():
       if nd_cnt_dct[k]["inst"] != 2:
         print (f"inst_cnt != 2 =====> {k} -> {v}")
         skip_functions.append(k)
       
    print ("\n----------------------------\n")
    with open("./data/train_data.json", 'w') as f:
       for el in train_data:
         if el["func"] not in skip_functions:
           #print("including fn:{}, arch: {}, flg: {}, trs: {}, lib: {}, n_num: {} for training !".format(el["fname"], el["arch"], el["compiler_flag"], el["trs"], el["library"], el["n_num"]))
           json.dump(el, f)
           f.write("\n")
           
    print ("\n----------------------------\n")
    with open("./data/test_data.json", 'w') as f:
       for el in test_data:
         if el["func"] not in skip_functions:
           print("including fn:{}, arch: {}, flg: {}, trs: {}, lib: {}, n_num: {}, model: {} for testing !".format(el["fname"], el["arch"], el["compiler_flag"], el["trs"], el["library"], el["n_num"], el["model"]))
           json.dump(el, f)
           f.write("\n")

    print ("\n----------------------------\n")


    print(f"\ntesting_cnt -> {len(testing_fns)}")
    print(f"validation_cnt -> {len(validation_fns)}")
    print(f"training_cnt -> {len(training_fns)}\n")

    print(f"\nn_num_test -> {n_num_test}")
    print(f"n_num_test_1 -> {n_num_test_1}")
    print(f"n_num_vald -> {n_num_vald}")
    print(f"n_num_vald_1 -> {n_num_vald_1}")
    print(f"n_num_train -> {n_num_train}")
    print(f"n_num_train_1 -> {n_num_train_1}\n")
    
    print ("Exiting Script...")   


if __name__ == '__main__':

    insert_enc_data_to_db()
    #generate_training_and_testing_files_from_db(model, exp_num="35", pull_test_data_from_db=True) # pass model, experiment number as arg
