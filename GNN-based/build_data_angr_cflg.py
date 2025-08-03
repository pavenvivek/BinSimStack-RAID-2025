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
from gemini_encoding_angr import *
#from cirrina_encoding_angr import *
#from palmtree_encoding_angr import *
#from inc_fnc_cflg import *
#from inc_fns_lst_cflg_O0_O2 import *

fnc_inst_cnt = {}
trs_cf = "N/A"

def get_root_fns():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[TREES_DB]
    col = db[CODE_GRAPH_COLLECTION]
    root_fns_dict = {}
    c = 0
    
    for f in col.distinct("function"):
        if True: #f in inc_fns:
            root_fns_dict[f] = c
            fnc_inst_cnt[f] = col.distinct("adj_dict", {"function" : f, "transformation" : [trs_cf]})
            c = c + 1
        
    return root_fns_dict

    
def get_functions_from_db():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[TREES_DB]
    col = db[CODE_GRAPH_COLLECTION]

    return col.distinct("function")
    

def get_data_from_db(root_fns_dict):
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

    for data in col.find({}, {"_id": 0, "graph" : 1, "ref_dict": 1, "adj_dict" : 1, "vex_ir" : 1, "calls" : 1, "consts" : 1, "str_consts_cnt" : 1, "instr_str" : 1, "instr_byte" : 1, "function" : 1, "filename" : 1, "transformation" : 1, "compiler" : 1, "compiler_flags" : 1, "architecture" : 1, "library" : 1, "uuid" : 1}):
       
       if data["architecture"] not in ["x86_64"]:
           continue
       
       if data["compiler"] != "gcc":
           continue

       if data["compiler_flags"] not in ["O0", "O1"]:
           continue

       
       if data["transformation"] not in ["N/A"]:
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
                   continue
           else:
               pass
       else:
           pass

       
       unique_adj_dict_all[data["adj_dict"]] = [data["function"]]
       if (data["function"] + "_" + data["library"]) not in rev_inst_cnt.keys():
               rev_inst_cnt[data["function"] + "_" + data["library"]] = 1
       else:
               rev_inst_cnt[data["function"] + "_" + data["library"]] = rev_inst_cnt[data["function"] + "_" + data["library"]] + 1 
       
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
            func_add_lst.append(k.split("_musl")[0])
        elif "openssl" in k:
            func_add_lst.append(k.split("_openssl")[0])
        elif "curl" in k:
            skip_fn_name_lst.append(k.split("_curl")[0])
      else:
          if v == 1:
              ins1_cnt = ins1_cnt + 1
          
          print ("skipping {} => {}, insufficient instance".format(k, v))
          skip_fn[k] = 1
          
          if "musl" in k:
              skip_fn_name_lst.append(k.split("_musl")[0])
          elif "openssl" in k:
              skip_fn_name_lst.append(k.split("_openssl")[0])
          elif "curl" in k:
              skip_fn_name_lst.append(k.split("_curl")[0])

    print (f"clash count : {clash_cnt}")
    print (f"le4_cnt : {le4_cnt}")
    print (f"ins1_cnt : {ins1_cnt}")
    print (f"insge2_cnt : {insge2_cnt}")

    return (data_lst, max_len, rev_root_fns_dict, skip_fn, skip_fn_name_lst, skip_fn_mul_inst, func_add_lst)


if __name__ == '__main__':
    
    print ("Generating training and testing files for ML models...")
    root_fns_dict = get_root_fns()

    data_lst, max_len, root_fns_dict, skip_fn, skip_fn_name_lst, skip_fn_mul_inst, func_add_lst = get_data_from_db(root_fns_dict)
    
    data = encode_data(data_lst)
    
    fn_lst = func_add_lst #get_functions_from_db()
    fn_lst = list(set(fn_lst) - set(skip_fn_name_lst))

    random.shuffle(fn_lst)
    testing_cnt  = math.ceil(len(fn_lst) * .10)
    validation_cnt = testing_cnt
    
    testing_fns  = fn_lst[:testing_cnt] #testing_fns_O0_O2
    validation_fns  = fn_lst[testing_cnt:(testing_cnt + validation_cnt)] #validation_fns_O0_O2
    training_fns = list(set(fn_lst) - set(testing_fns + validation_fns)) #training_fns_cflg_O0_O2

    print(f"\ntesting_cnt -> {len(testing_fns)}")
    print(f"validation_cnt -> {len(validation_fns)}")
    print(f"training_cnt -> {len(training_fns)}\n")

    print(f"\ntesting_fns -> {testing_fns}\n")
    print(f"\nvalidation_fns -> {validation_fns}\n")
    print(f"\ntraining_fns -> {training_fns}\n")

    nd_cnt_dct = {}
    skip_functions = []

    test_data = []
    validation_data = []
    train_data = []

    n_num_test = 0
    n_num_vald = 0
    n_num_train = 0
    n_num_test_1 = 0
    n_num_vald_1 = 0
    n_num_train_1 = 0

    for el in data:
        if el["fname"] not in skip_fn.keys():
            if el["func"] in skip_fn_mul_inst:
                print(f"skipping duplicate multi instance function {el['func']}")
                continue

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
           print("including fn:{}, arch: {}, flg: {}, trs: {}, lib: {}, n_num: {} for training !".format(el["fname"], el["arch"], el["compiler_flag"], el["trs"], el["library"], el["n_num"]))
           json.dump(el, f)
           f.write("\n")
           
    print ("\n----------------------------\n")
    with open("./data/test_data.json", 'w') as f:
       for el in test_data:
         if el["func"] not in skip_functions:
           print("including fn:{}, arch: {}, flg: {}, trs: {}, lib: {}, n_num: {} for testing !".format(el["fname"], el["arch"], el["compiler_flag"], el["trs"], el["library"], el["n_num"]))
           json.dump(el, f)
           f.write("\n")

    print ("\n----------------------------\n")

    print ("\n----------------------------\n")
    with open("./data/validation_data.json", 'w') as f:
       for el in validation_data:
         if el["func"] not in skip_functions:
           print("including fn:{}, arch: {}, flg: {}, trs: {}, lib: {}, n_num: {} for validation !".format(el["fname"], el["arch"], el["compiler_flag"], el["trs"], el["library"], el["n_num"]))
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

