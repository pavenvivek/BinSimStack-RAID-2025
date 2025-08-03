import numpy as np
#from utils import *
import os, subprocess
import argparse
import json
import pymongo
import ast
import re
import random, math

from settings import *
#from instr_repo import *
from insert_db import *
from jtrans_encoding_angr import *
#from gemini_encoding_angr import *
#from cirrina_encoding_angr import *
#from palmtree_encoding_angr import *
#from inc_fns_lst_flt import *
#from inc_fns_lst_viz import *

fnc_inst_cnt = {}
trs_cf = "Flatten" #"EncodeArithmetic, Flatten" #"Virtualize" #"EncodeArithmetic, Virtualize" #"EncodeArithmetic, Flatten" #


def get_data_from_db():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[TREES_DB]
    col = db[CODE_GRAPH_COLLECTION]
    data_lst = []
    data_lst_dc = {}
    unique_adj_dict = {}
    unique_adj_dict_dc = {}
    rev_inst_cnt = {}
    max_len = 0
    skip_fn_name_lst = []
    func_add_lst = []
    clash_cnt = 0
    le4_cnt = 0
    ins1_cnt = 0
    insge2_cnt = 0

    #--------------
    #scalability training and testing
    # client1 = pymongo.MongoClient(MONGO_CLIENT)
    # db1 = client1[FUNC_INFO_DB]
    # col1 = db1[FUNC_INFO_COLLECTION]

    # training_fns = []
    # testing_fns = []
    # #exp_num_ls = ["1", "8", "15", "22", "29"]
    # #exp_num_ls = ["2", "9", "16", "23", "30"]
    # #exp_num_ls = ["5", "12", "19", "26", "33"]
    # #exp_num_ls = ["6", "13", "20", "27", "34"]
    # #exp_num_ls = ["7", "14", "21", "28", "35"]

    # exp_num_ls = ["39-scale-viz-7500"]
    
    # for data in col1.find({"experiment" : { '$in' : exp_num_ls}}, {"_id": 0, "testing_fns" : 1, "validation_fns": 1, "training_fns" : 1, "library" : 1, "experiment" : 1, "trs" : 1}):
    #     testing_fns = testing_fns + data["testing_fns"]
    #     validation_fns = [] #data["validation_fns"]
    #     training_fns = [] #training_fns + data["training_fns"]
    #     exp_num = "" #data["experiment"]
    #     #inc_libraries = "" #data["library"]
    # inc_fns = list(set(testing_fns)) # - set(vuln_fns))
    # #inc_libraries1 = ["openssl", "redis", "binutils", "imagemagick", "libxml2"]
    # inc_libraries1 = ["igraph", "dbus", "allegro", "libmicrohttpd", "gsl", "alsa", "libmongoc", "libtomcrypt", "coreutils", "sqlite", "curl", "musl"]

    # print (f"len inc_fns -> {len(inc_fns)}")
    #-------------
    
    # invalid functions - too big to insert into mongo for Flt
    #inc_funcs = ["setup_curl", "testPutExternal", "testMultithreadedPostCancelPart", "testWithoutTimeout", "testMultithreadedDelete", "testInternalDelete", "aarch64_opcode_lookup_1", "dohprobe", "testExternalDelete", "testPutThreadPool", "testMultithreadedPoolDelete", "testPutThreadPerConn", "setup_easy_handler_params", "testPutInternalThread", "testWithTimeout", "je_realloc", "je_memalign", "malloc_default", "je_aligned_alloc", "je_valloc", "je_mallocx", "je_posix_memalign", "je_calloc"]
    
    #inc_libraries1 = ["igraph", "dbus", "allegro", "libxml2", "libmicrohttpd", "gsl", "alsa", "libmongoc", "binutils", "libtomcrypt", "imagemagick", "coreutils", "redis", "sqlite", "curl", "musl"] #, "openssl"]
    #inc_libraries1 = ["igraph", "dbus", "allegro", "libmicrohttpd", "gsl", "alsa", "libmongoc", "libtomcrypt", "coreutils", "sqlite", "curl", "musl"] #, "openssl"]
    inc_libraries1 = ["openssl"]
    #{"library" : {'$in' : inc_libraries1}}
    
    for data in col.find({"library" : {'$in' : inc_libraries1}}, {"_id": 0, "graph" : 1, "ref_dict": 1, "adj_dict" : 1, "vex_ir" : 1, "calls" : 1, "consts" : 1, "instr_str" : 1, "instr_byte" : 1, "function" : 1, "filename" : 1, "transformation" : 1, "compiler" : 1, "compiler_flags" : 1, "architecture" : 1, "library" : 1, "uuid" : 1}):

       #if data["function"] not in ["ossl_a2ulabel", "rsa_ossl_private_decrypt", "ossl_policy_level_add_node", "X509_issuer_and_serial_hash", "DH_check", "dsa_sign_setup", "EC_GROUP_set_generator", "use_certificate_chain_file", "AES_ige_encrypt", "ossl_i2c_ASN1_BIT_STRING"]: #inc_funcs:
       #    continue
       
        
       if data["architecture"] not in ["x86_64"]:
           continue
       
       if data["compiler"] != "gcc":
           print ("skipping in compiler !")
           continue

       #--- use this block only for O2 and O3 (to avoid O1 duplicates)
       #if data["function"] not in inc_fns:
       #    continue
       #---

       if data["transformation"] not in ["N/A", trs_cf]: #, trs_cf,  []]: #["Flatten", "Flatten"]]: # # # 
           #print ("skipping in trs !")
           continue
       
       # filtering duplicate insertion of same funcs
       if data["function"] not in unique_adj_dict.keys():
           unique_adj_dict[data["function"]] = [data["adj_dict"]]
       elif data["adj_dict"] not in unique_adj_dict[data["function"]]:
           unique_adj_dict[data["function"]].append(data["adj_dict"])
       else:
           clash_cnt = clash_cnt + 1
           continue
      
       if (data["function"] + "___" + data["library"]) not in rev_inst_cnt.keys():
               rev_inst_cnt[data["function"] + "___" + data["library"]] = 1
       else:
               rev_inst_cnt[data["function"] + "___" + data["library"]] = rev_inst_cnt[data["function"] + "___" + data["library"]] + 1 
       
       if len(data["ref_dict"]) > max_len:
           max_len = len(data["ref_dict"])
       
       data_lst.append((ast.literal_eval(data["graph"]), ast.literal_eval(data["ref_dict"]), ast.literal_eval(data["adj_dict"]), ast.literal_eval(data["vex_ir"]), ast.literal_eval(data["calls"]), ast.literal_eval(data["consts"]), ast.literal_eval(data["instr_str"]), ast.literal_eval(data["instr_byte"]), data["function"], data["filename"], data["transformation"], data["compiler"], data["compiler_flags"], data["architecture"], data["library"], data["uuid"]))

       data_lst_dc[data["uuid"]] = (ast.literal_eval(data["graph"]), ast.literal_eval(data["ref_dict"]), ast.literal_eval(data["adj_dict"]), ast.literal_eval(data["vex_ir"]), ast.literal_eval(data["calls"]), ast.literal_eval(data["consts"]), ast.literal_eval(data["instr_str"]), ast.literal_eval(data["instr_byte"]), data["function"], data["filename"], data["transformation"], data["compiler"], data["compiler_flags"], data["architecture"], data["library"], data["uuid"])

    rev_root_fns_dict = {}
    skip_fn = {}
    c = 0
    print ("revised unique instance count ...")    
    for k,v in rev_inst_cnt.items():
      if v == 2:
          print ("*** including {} => {}, class : {}".format(k, v, c))
          rev_root_fns_dict[k] = c
          c = c + 1

          if "___musl" in k:
              func_add_lst.append(k.split("___musl")[0])
          elif "___libmongoc" in k:
              func_add_lst.append(k.split("___libmongoc")[0])
          elif "___sqlite" in k:
              func_add_lst.append(k.split("___sqlite")[0])
          elif "___redis" in k:
              func_add_lst.append(k.split("___redis")[0])
          elif "___coreutils" in k:
              func_add_lst.append(k.split("___coreutils")[0])
          elif "___imagemagick" in k:
              func_add_lst.append(k.split("___imagemagick")[0])
          elif "___libtomcrypt" in k:
              func_add_lst.append(k.split("___libtomcrypt")[0])
          elif "___binutils" in k:
              func_add_lst.append(k.split("___binutils")[0])
          elif "___alsa" in k:
              func_add_lst.append(k.split("___alsa")[0])
          elif "___gsl" in k:
              func_add_lst.append(k.split("___gsl")[0])
          elif "___libmicrohttpd" in k:
              func_add_lst.append(k.split("___libmicrohttpd")[0])
          elif "___libxml2" in k:
              func_add_lst.append(k.split("___libxml2")[0])
          elif "___allegro" in k:
              func_add_lst.append(k.split("___allegro")[0])
          elif "___curl" in k:
              func_add_lst.append(k.split("___curl")[0])
          elif "___dbus" in k:
              func_add_lst.append(k.split("___dbus")[0])
          elif "___igraph" in k:
              func_add_lst.append(k.split("___igraph")[0])
          elif "___openssl" in k:
              func_add_lst.append(k.split("___openssl")[0])
          elif "malware" in k:
              func_add_lst.append(k.split("___malware")[0] + k.split("___malware")[1])
          else:
              insge2_cnt = insge2_cnt + 1
              print ("*** skipping 1 invalid inclusion {} => {}, class : {}".format(k, v, c))
      else:
          if v == 1:
              ins1_cnt = ins1_cnt + 1
          else:
              insge2_cnt = insge2_cnt + 1
              
          print ("skipping {} => {}, insufficient instance".format(k, v))
          skip_fn[k] = 1
          
          if "___musl" in k:
              skip_fn_name_lst.append(k.split("___musl")[0])
          elif "___openssl" in k:
              skip_fn_name_lst.append(k.split("___openssl")[0])
          elif "___curl" in k:
              skip_fn_name_lst.append(k.split("___curl")[0])
          elif "___sqlite" in k:
              skip_fn_name_lst.append(k.split("___sqlite")[0])
          elif "___redis" in k:
              skip_fn_name_lst.append(k.split("___redis")[0])
          elif "___coreutils" in k:
              skip_fn_name_lst.append(k.split("___coreutils")[0])
          elif "___imagemagick" in k:
              skip_fn_name_lst.append(k.split("___imagemagick")[0])
          elif "___libtomcrypt" in k:
              skip_fn_name_lst.append(k.split("___libtomcrypt")[0])
          elif "___binutils" in k:
              skip_fn_name_lst.append(k.split("___binutils")[0])
          elif "___libmongoc" in k:
              skip_fn_name_lst.append(k.split("___libmongoc")[0])
          elif "___alsa" in k:
              skip_fn_name_lst.append(k.split("___alsa")[0])
          elif "___gsl" in k:
              skip_fn_name_lst.append(k.split("___gsl")[0])
          elif "___libmicrohttpd" in k:
              skip_fn_name_lst.append(k.split("___libmicrohttpd")[0])
          elif "___libxml2" in k:
              skip_fn_name_lst.append(k.split("___libxml2")[0])
          elif "___allegro" in k:
              skip_fn_name_lst.append(k.split("___allegro")[0])
          elif "___dbus" in k:
              skip_fn_name_lst.append(k.split("___dbus")[0])
          elif "___igraph" in k:
              skip_fn_name_lst.append(k.split("___igraph")[0])
          elif "malware" in k:
              skip_fn_name_lst.append(k.split("___malware")[0] + k.split("___malware")[1])
          else:
              #insge2_cnt = insge2_cnt + 1
              print ("*** skipping 2 invalid inclusion {} => {}, class : {}".format(k, v, c))

    print (f"clash count : {clash_cnt}")
    print (f"le4_cnt : {le4_cnt}")
    print (f"ins1_cnt : {ins1_cnt}")
    print (f"insge2_cnt : {insge2_cnt}")

    return (data_lst, max_len, rev_root_fns_dict, skip_fn, skip_fn_name_lst, func_add_lst)


def insert_enc_data_to_db():    
    print ("Generating training and testing files for ML models ...")

    data_lst, max_len, root_fns_dict, skip_fn, skip_fn_name_lst, func_add_lst = get_data_from_db()

    print ("skip_fn : {}".format(skip_fn.keys()))
    print ("skip_fn_name_lst : {}".format(skip_fn_name_lst))
    print ("skip_fn_name_lst len : {}".format(len(skip_fn_name_lst)))
    print ("func_add_lst len : {}".format(len(func_add_lst)))
    print ("func_add_lst set len : {}".format(len(set(func_add_lst))))
    
    fn_lst = func_add_lst
    #fn_lst = list(set(fn_lst) - set(skip_fn_name_lst))
    print ("func_lst len : {}".format(len(fn_lst)))

    print ("func_lst : {}".format(set(fn_lst)))
    
    #dup = {x for x in func_add_lst if func_add_lst.count(x) > 1}
    #print(f"Duplicates in func list -> {dup}")

    random.shuffle(fn_lst)
    testing_cnt  = math.ceil(len(fn_lst) * .20)
    validation_cnt = testing_cnt
    
    testing_fns  = fn_lst[:testing_cnt] #testing_fns_ge0 #
    validation_fns  = [] #fn_lst[testing_cnt:(testing_cnt + validation_cnt)] #[] #validation_fns_ge0 #
    training_fns = list(set(fn_lst) - set(testing_fns + validation_fns + vuln_fns))
    

    print(f"\ntesting_cnt -> {len(testing_fns)}")
    print(f"validation_cnt -> {len(validation_fns)}")
    print(f"training_cnt -> {len(training_fns)}\n")

    #print(f"\ntesting_fns -> {testing_fns}\n")
    #print(f"\nvalidation_fns -> {validation_fns}\n")
    #print(f"\ntraining_fns -> {training_fns}\n")

    
    print ("Generating Layer Encoding Data ...")
    data = encode_data(data_lst)

    data_enc_lst = []
    for el in data:
       if el["fname"] not in skip_fn.keys():
           if "malware" in el["library"]:
               data_enc_lst.append((el["src"], el["n_num"], el["succs"], el["features"], el["fname"], el["library"] + "_" + el["func"], el["compiler_flag"], el["arch"], el["trs"], el["library"], el["model"]))
           else:
               data_enc_lst.append((el["src"], el["n_num"], el["succs"], el["features"], el["fname"], el["func"], el["compiler_flag"], el["arch"], el["trs"], el["library"], el["model"]))

    insert_db_encodings_mongo(data_enc_lst)
    print (f"Inserted encoding data to db ...")
    

def generate_training_and_testing_files_from_db(model, exp_num="1", pull_test_data_from_db=False):

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

    inc_libraries = ["openssl"]

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
        validation_cnt = testing_cnt

        testing_fns  = fn_lst[:testing_cnt] #testing_fns_ge0 #
        validation_fns  = [] #fn_lst[testing_cnt:(testing_cnt + validation_cnt)] #[] #validation_fns_ge0 #
        training_fns = list(set(fn_lst) - set(testing_fns + validation_fns + vuln_fns))

        info_lst = []
        info_lst.append((testing_fns, validation_fns, training_fns, inc_libraries, exp_num, trs_cf))
        insert_db_function_info_mongo(info_lst)

        print (f"Inserted function info to db ...")
        print(f"\ntesting_fns -> {testing_fns}\n")
        
    print (f"\n\nGenerating Training and Testing JSON files for {model} and {inc_libraries} ...")
    for el in col.find({"library" : {'$in' : inc_libraries}, "model" : model},
                       {"_id": 0, "src" : 1, "n_num": 1, "succs" : 1, "features" : 1, "fname" : 1, "func" : 1, "compiler_flag" : 1, "arch" : 1, "trs" : 1, "library" : 1, "model" : 1}):
    #for el in data:

        #if el["fname"] in skip_fn.keys() and el["func"] in testing_fns:
        #    skipped_funcs.append(el["func"])

        #if el["fname"] not in skip_fn.keys():

        #print ("enters here ...")
        
        if (el["func"] in testing_fns):
            test_data.append(el)

            if el["func"] not in nd_cnt_dct.keys():
                nd_cnt_dct[el["func"]] = {}
                nd_cnt_dct[el["func"]]["inst"] = 1
            else:
                nd_cnt_dct[el["func"]]["inst"] = nd_cnt_dct[el["func"]]["inst"] + 1

        elif (el["func"] in validation_fns):
            validation_data.append(el)

            if el["func"] not in nd_cnt_dct.keys():
                nd_cnt_dct[el["func"]] = {}
                nd_cnt_dct[el["func"]]["inst"] = 1
            else:
                nd_cnt_dct[el["func"]]["inst"] = nd_cnt_dct[el["func"]]["inst"] + 1


        elif (el["func"] in training_fns):
            train_data.append(el)

            if el["func"] not in nd_cnt_dct.keys():
                nd_cnt_dct[el["func"]] = {}
                nd_cnt_dct[el["func"]]["inst"] = 1
            else:
                nd_cnt_dct[el["func"]]["inst"] = nd_cnt_dct[el["func"]]["inst"] + 1
        #else:
        #    pass
       

    for k,v in nd_cnt_dct.items():
       if nd_cnt_dct[k]["inst"] != 2:
         print (f"inst_cnt != 2 =====> {k} -> {v}")
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
    
    print ("Exiting Script...")   


def generate_data(exp_num="1", pull_test_data_from_db=False):    
    print ("Generating training and testing files for ML models ...")

    data_lst, max_len, root_fns_dict, skip_fn, skip_fn_name_lst, func_add_lst = get_data_from_db()

    print ("skip_fn : {}".format(skip_fn.keys()))
    print ("skip_fn_name_lst : {}".format(skip_fn_name_lst))
    print ("skip_fn_name_lst len : {}".format(len(skip_fn_name_lst)))
    print ("func_add_lst len : {}".format(len(func_add_lst)))
    print ("func_add_lst set len : {}".format(len(set(func_add_lst))))
    
    fn_lst = func_add_lst
    #fn_lst = list(set(fn_lst) - set(skip_fn_name_lst))
    print ("func_lst : {}".format(set(fn_lst)))
    print ("\nfunc_lst len : {}\n".format(len(set(fn_lst))))
    print ("len data_lst : {}".format(len(data_lst)))
    
    #dup = {x for x in func_add_lst if func_add_lst.count(x) > 1}
    #print(f"Duplicates in func list -> {dup}")

    random.shuffle(fn_lst)
    #testing_cnt  = math.ceil(len(fn_lst) * .20)
    #validation_cnt = 0 #testing_cnt


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
    
        testing_fns  = fn_lst #[] #fn_lst[:testing_cnt] #testing_fns_ge0 #
        validation_fns  = [] #fn_lst[testing_cnt:(testing_cnt + validation_cnt)] #[] #validation_fns_ge0 #
        training_fns = [] #fn_lst #list(set(fn_lst) - set(testing_fns + validation_fns + vuln_fns))
    

    print(f"\ntesting_cnt -> {len(testing_fns)}")
    print(f"validation_cnt -> {len(validation_fns)}")
    print(f"training_cnt -> {len(training_fns)}\n")

    test_data = []
    validation_data = []
    train_data = []
    nd_cnt_dct = {}
    skip_functions = []

    #print(f"\ntesting_fns -> {testing_fns}\n")
    #print(f"\nvalidation_fns -> {validation_fns}\n")
    #print(f"\ntraining_fns -> {training_fns}\n")

    
    print ("Generating Layer Encoding Data ...")
    data = encode_data(data_lst)

    for el in data:

        #if el["fname"] in skip_fn.keys() and el["func"] in testing_fns:
        #    skipped_funcs.append(el["func"])

        #if el["fname"] not in skip_fn.keys():

        #print ("enters here ...")

        if (el["func"] in testing_fns):
            test_data.append(el)

            #if el["func"] == "_mongoc_stream_tls_openssl_setsockopt":
            #    print ("\n_mongoc_stream_tls_openssl_setsockopt is present !\n")
        
            if el["func"] not in nd_cnt_dct.keys():
                nd_cnt_dct[el["func"]] = {}
                nd_cnt_dct[el["func"]]["inst"] = 1
            else:
                nd_cnt_dct[el["func"]]["inst"] = nd_cnt_dct[el["func"]]["inst"] + 1

        elif (el["func"] in validation_fns):
            validation_data.append(el)

            if el["func"] not in nd_cnt_dct.keys():
                nd_cnt_dct[el["func"]] = {}
                nd_cnt_dct[el["func"]]["inst"] = 1
            else:
                nd_cnt_dct[el["func"]]["inst"] = nd_cnt_dct[el["func"]]["inst"] + 1


        elif (el["func"] in training_fns):
            train_data.append(el)

            if el["func"] not in nd_cnt_dct.keys():
                nd_cnt_dct[el["func"]] = {}
                nd_cnt_dct[el["func"]]["inst"] = 1
            else:
                nd_cnt_dct[el["func"]]["inst"] = nd_cnt_dct[el["func"]]["inst"] + 1
        #else:
        #    pass
       

    for k,v in nd_cnt_dct.items():
       if nd_cnt_dct[k]["inst"] != 2:
         print (f"inst_cnt != 2 =====> {k} -> {v}")
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
           #print("including fn:{}, arch: {}, flg: {}, trs: {}, lib: {}, model: {}, n_num: {} for testing !".format(el["fname"], el["arch"], el["compiler_flag"], el["trs"], el["library"], el["model"], el["n_num"]))
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
    print(f"skipped functions -> {skip_functions}")
    
    print ("Exiting Script...")   

               
    
if __name__ == '__main__':

    #insert_enc_data_to_db()
    #generate_training_and_testing_files_from_db(model, exp_num="2", pull_test_data_from_db=True) # pass model, experiment number as arg
    generate_data(exp_num="1", pull_test_data_from_db=True)
