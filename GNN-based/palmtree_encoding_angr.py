##################################
#
# This file implements the Palmtree-trp encoding
#
##################

from settings import *
from instr_repo import *
import json
import re

import os
from config import *
from torch import nn
from scipy.ndimage.filters import gaussian_filter1d
from torch.autograd import Variable
import torch
import numpy as np
import eval_utils as utils



def encode_data(data_lst):
        
    data = []
    func_dt = {}

    max_block_sz = 0
    total_blks = 0
    tot_bl_twty = 0
    tot_bl_10 = 0
    tot_bl_15 = 0

    palmtree = utils.UsableTransformer(model_path="./palmtree/transformer.ep19",
                                       vocab_path="./palmtree/vocab")

    
    for g, refd, adj_dct, vex_ir, calls, consts, n_str_consts, inst_str, inst_byt, func, flname, trs, cmpl, cflg, arch, lib, uid in data_lst:

       #if trs == "Virtualize":
           #print ("skipping Viz !")
       #    continue
       
       n_num = len(refd.keys()) # node count
       features = []
       succs = []
       ft_dct = {}
       out_deg = 0
       in_deg = 0
       out_deg_dct = {}
       in_deg_dct = {}
       
       all_vals = []
       for k,v in adj_dct.items():
           all_vals = all_vals + v       
       
       # store node count and edge count for function
       if func not in func_dt.keys():
           func_dt[func] = {}
       
       if trs[0] not in list(func_dt[func].keys()):
           func_dt[func][trs[0]] = [n_num, len(all_vals)] 
       
       for k,v in refd.items():
           # node level features of Gemini
           str_consts  = 0
           num_consts  = 0
           offspr      = 0
           parent_ft   = []
           

           if k not in inst_str.keys():
               print ("fl: {}, lib: {}".format(flname, lib))

           total_blks = total_blks + 1
           
           if len(inst_str[k]) <= 20:
               tot_bl_twty = tot_bl_twty + 1           

           if len(inst_str[k]) <= 15:
               tot_bl_15 = tot_bl_15 + 1           

           if len(inst_str[k]) <= 10:
               tot_bl_10 = tot_bl_10 + 1           
           
           
           if v in adj_dct.keys():
               succs.append(adj_dct[v])    
           else:
               succs.append([])


           ins_ls = []
           for ins in inst_str[k]:
               ins = ins.replace(",", " ")
               ins = ins.replace("[", "[ ")
               ins = ins.replace("]", " ]")
               ins_ls.append(ins)

           embeddings = []    
           try:    
               embeddings = palmtree.encode(ins_ls)
           except Exception as exp:
               print(f"Func -> {func}\nIns_lst -> {ins_ls}\nException in Palmtree Encoding -> {exp}\n")
               
           embeddings_trn = []

           i = 0
           for emb in embeddings:
               embeddings_trn = embeddings_trn + list(emb[:10])
               i = i + 1

               if i == 20:
                   break

           while len(embeddings_trn) < 200:
               embeddings_trn.append(0)
               
           embeddings = [str(f) for f in embeddings_trn]

           features.append(embeddings)

       # length of encoding should be 200 for Palmtree-trp
       for ft in features:
          if len(ft) != 200:
              print ("\n!!! feature length is different -> {}\n".format(len(ft)))
       
       data_i = {"src": flname, "n_num" : n_num, "succs" : succs, "features" : features, "fname" : func + "___" + lib, "func" : func, "compiler_flag" : cflg, "arch" : arch,
                 "trs" : [trs], "library" : lib, "model" : "Palmtree_trp"}     
       data.append(data_i)
     
    print ("max_block_sz -> {}".format(max_block_sz))
    print ("total_blks -> {}".format(total_blks))
    print ("tot_bl_twty -> {}".format(tot_bl_twty))
    print ("tot_bl_15 -> {}".format(tot_bl_15))
    print ("tot_bl_10 -> {}".format(tot_bl_10))

    return data
