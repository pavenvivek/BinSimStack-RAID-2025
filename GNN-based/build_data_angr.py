import numpy as np
from utils import *
import os
import argparse
import json
import pymongo
import ast
import re
import random, math

from settings import *
from instr_repo import *
#from gemini_encoding_angr import *
#from cirrina_encoding_angr import *
from palmtree_encoding_angr import *
from inc_fns_lst_flt import *
#from inc_fns_lst_viz import *

fnc_inst_cnt = {}
trs_cf = "Flatten" #"Virtualize" #"EncodeArithmetic, Virtualize" #"EncodeArithmetic, Flatten" #


def get_data_from_db():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[TREES_DB]
    col = db[CODE_GRAPH_COLLECTION]
    data_lst = []
    data_lst_dc = {}
    unique_adj_dict = {}
    unique_adj_dict_dc = {}
    #unique_adj_dict_all = {}
    #unique_asm = {}
    rev_inst_cnt = {}
    max_len = 0
    skip_fn_name_lst = []
    func_add_lst = []
    clash_cnt = 0
    le4_cnt = 0
    ins1_cnt = 0
    insge2_cnt = 0

    for data in col.find({"library" : "openssl"}, {"_id": 0, "graph" : 1, "ref_dict": 1, "adj_dict" : 1, "vex_ir" : 1, "calls" : 1, "consts" : 1, "str_consts_cnt" : 1, "instr_str" : 1, "instr_byte" : 1, "function" : 1, "filename" : 1, "transformation" : 1, "compiler" : 1, "compiler_flags" : 1, "architecture" : 1, "library" : 1, "uuid" : 1}):

       #if data["library"] not in ["openssl"]:
       #    continue

       if data["architecture"] not in ["x86_64"]:
           continue
       
       if data["compiler"] != "gcc":
           print ("skipping in compiler !")
           continue

       if data["transformation"] not in ["N/A", trs_cf]: #, trs_cf,  []]: #["Flatten", "Flatten"]]: # # # 
           print ("skipping in trs !")
           continue
       
       # filtering duplicate insertion of same funcs
       if data["function"] not in unique_adj_dict.keys():
           unique_adj_dict[data["function"]] = [data["adj_dict"]]
           #unique_asm[data["function"]] = [ast.literal_eval(data["instr_str"])]
       elif data["adj_dict"] not in unique_adj_dict[data["function"]]:
           unique_adj_dict[data["function"]].append(data["adj_dict"])
       else:
           clash_cnt = clash_cnt + 1
           continue

      
       #unique_adj_dict_all[data["adj_dict"]] = [data["function"]]
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
          else:
              insge2_cnt = insge2_cnt + 1
              
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

    return (data_lst, max_len, rev_root_fns_dict, skip_fn, skip_fn_name_lst, func_add_lst)


if __name__ == '__main__':
    
    print ("Generating training and testing files for ML models ...")
    #root_fns_dict = get_root_fns()

    data_lst, max_len, root_fns_dict, skip_fn, skip_fn_name_lst, func_add_lst = get_data_from_db()
    
    
    #print (f"data_lst -> {data_lst}")
    print ("skip_fn : {}".format(skip_fn.keys()))
    print ("skip_fn_name_lst : {}".format(skip_fn_name_lst))
    print ("skip_fn_name_lst len : {}".format(len(skip_fn_name_lst)))
    print ("func_add_lst len : {}".format(len(func_add_lst)))
    print ("func_add_lst set len : {}".format(len(set(func_add_lst))))
    
    fn_lst = func_add_lst
    #fn_lst = list(set(fn_lst) - set(skip_fn_name_lst))
    print ("func_lst len : {}".format(len(fn_lst)))

    dup = {x for x in func_add_lst if func_add_lst.count(x) > 1}
    print(f"Duplicates in func list -> {dup}")

    random.shuffle(fn_lst)
    testing_cnt  = math.ceil(len(fn_lst) * .10)
    validation_cnt = testing_cnt
    
    testing_fns  = testing_fns_ge0 #fn_lst[:testing_cnt] #
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
    
    nd_cnt_dct = {}
    skip_functions = []

    test_data = []
    validation_data = []
    train_data = []

    skipped_funcs = []

    for el in data:
        if el["fname"] in skip_fn.keys() and el["func"] in testing_fns:
            skipped_funcs.append(el["func"])

        if el["fname"] not in skip_fn.keys():
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
        else:
            pass
       

    for k,v in nd_cnt_dct.items():
       if nd_cnt_dct[k]["inst"] != 2:
         print (f"inst_cnt != 2 =====> {k} -> {v}")
         skip_functions.append(k)


    print ("\n----------------------------\n")
    with open("./data/train_data.json", 'w') as f:
       for el in train_data:
         if el["func"] not in skip_functions:
           print("including fn:{}, arch: {}, flg: {}, trs: {}, lib: {}, model: {}, n_num: {} for training !".format(el["fname"], el["arch"], el["compiler_flag"], el["trs"], el["library"], el["n_num"], el["model"]))
           json.dump(el, f)
           f.write("\n")
    
    print ("\n----------------------------\n")
    with open("./data/test_data.json", 'w') as f:
       for el in test_data:
         if el["func"] not in skip_functions:
           print("including fn:{}, arch: {}, flg: {}, trs: {}, lib: {}, model: {}, n_num: {} for testing !".format(el["fname"], el["arch"], el["compiler_flag"], el["trs"], el["library"], el["n_num"], el["model"]))
           json.dump(el, f)
           f.write("\n")

    print ("\n----------------------------\n")

    print ("\n----------------------------\n")
    with open("./data/validation_data.json", 'w') as f:
       for el in validation_data:
         if el["func"] not in skip_functions:
           print("including fn:{}, arch: {}, flg: {}, trs: {}, lib: {}, model: {}, n_num: {} for validation !".format(el["fname"], el["arch"], el["compiler_flag"], el["trs"], el["library"], el["n_num"], el["model"]))
           json.dump(el, f)
           f.write("\n")
    print ("\n----------------------------\n")
    
    print(f"\ntesting_cnt -> {len(testing_fns)}")
    print(f"validation_cnt -> {len(validation_fns)}")
    print(f"training_cnt -> {len(training_fns)}\n")
    print(f"skipped functions -> {skipped_funcs}")
    
    print ("Exiting Script...")   

