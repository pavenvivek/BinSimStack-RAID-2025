##################################
#
# This file implements the Cirrina encoding
#
##################

from settings import *
from instr_repo import *
import json
import re


def encode_data(data_lst):
        
    data = []
    func_dt = {}

    max_block_sz = 0
    total_blks = 0
    tot_bl_twty = 0
    tot_bl_10 = 0
    tot_bl_15 = 0
    total_blks_pl = 0
    total_blks_trs = 0

    
    for g, refd, adj_dct, vex_ir, calls, consts, n_str_consts, inst_str, inst_byt, func, flname, trs, cmpl, cflg, arch, lib, uid in data_lst:
          
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
           str_consts  = 0
           num_consts  = 0
           trs_instrs  = 0
           n_calls     = 0
           instrs      = 0
           arth_instrs = 0
           offspr      = 0

           log_instrs  = 0
           parent_ft   = []
           

           total_blks = total_blks + 1
           
           if len(inst_str[k]) <= 20:
               tot_bl_twty = tot_bl_twty + 1           

           if len(inst_str[k]) <= 15:
               tot_bl_15 = tot_bl_15 + 1           

           if len(inst_str[k]) <= 10:
               tot_bl_10 = tot_bl_10 + 1           


           if cflg == "O0": #trs == "N/A":
               total_blks_pl = total_blks_pl + 1
           else:
               total_blks_trs = total_blks_trs + 1
               

           if k not in inst_str.keys():
               print ("fl: {}, lib: {}".format(flname, lib))

           for ins in inst_str[k]:
           
               op, _, args = ins.partition(' ')
               instrs = instrs + 1
               
               op = op.lower()
               if op in ARCH_MNEM[arch]["transfer"]:
                   trs_instrs = trs_instrs + 1
               elif op in ARCH_MNEM[arch]["arithmetic"]:
                   arth_instrs = arth_instrs + 1
               elif op in ARCH_MNEM[arch]["logic"]:
                   log_instrs = log_instrs + 1
               
           if v in adj_dct.keys():
               out_deg = len(adj_dct[v])
           
           in_deg = 0
           for k1,v1 in adj_dct.items():
               if v in adj_dct[k1] and v != k1:
                   in_deg = in_deg + 1
                   
           
           if v in adj_dct.keys():
               v_dsc = adj_dct[v]
               
               for k1,v1 in adj_dct.items():
                   if k1 != v and k1 in v_dsc:
                       v_dsc = v_dsc + adj_dct[k1]
                       
               offspr = len(list(set(v_dsc)))
  

           if k in calls.keys():
               n_calls = len(calls[k])
           
           if v in adj_dct.keys():
               succs.append(adj_dct[v])    
           else:
               succs.append([])
           
           # node label encoding for Cirrina
           features.append([out_deg, in_deg, n_calls, offspr])
           
           out_deg_dct[v] = out_deg
           in_deg_dct[v] = in_deg
           ft_dct[v] = [out_deg, in_deg, n_calls, offspr]
       
       # Tracelet encoding for Cirrina
       i = 0
       for ft in features:
           j = 0
           ft_p = []
           max_in_deg = -1
           max_in_deg_el = ""

           # find max in-degree parent (busy parent)
           for k,v in adj_dct.items():
               if i in adj_dct[k] and i != k:
                   
                   if in_deg_dct[k] > max_in_deg:
                       max_in_deg = in_deg_dct[k]
                       max_in_deg_el = k


           # skip for root node
           if max_in_deg != -1:
               features[i] = ft_dct[max_in_deg_el] + ft
           else:    
           #if i != 0 and max_in_deg == -1:
               features[i] = [0, 0, 0, 0] + features[i]
           i = i + 1
       
       # encode node label of childrens (keeping the limit to 6 childrens for our dataset)
       i = 0
       for ft in features:
           ft_c = []
           m = 0
           for k,v in adj_dct.items():
               if k == i:
                   for j in range(0, len(v)):
                       
                       ft_c = ft_c + ft_dct[v[j]]
                       
                       m = m + 1
                       if m > 5:
                           break
                   
           while m <= 5:
               ft_c = ft_c + [0, 0, 0, 0]
               m = m + 1
           features[i] = ft + ft_c
           i = i + 1

       # length of encoding should be 32 for Cirrina
       for ft in features:
          if len(ft) != 32:
              print ("\n!!! feature length is different -> {}\n".format(len(ft)))
       
       data_i = {"src": flname, "n_num" : n_num, "succs" : succs, "features" : features, "fname" : func + "___" + lib, "func" : func, "compiler_flag" : cflg, "arch" : arch,
                 "trs" : [trs], "library" : lib, "model" : "Cirrina"}     
       data.append(data_i)
     
    print ("max_block_sz -> {}".format(max_block_sz))
    print ("total_blks -> {}".format(total_blks))
    print ("tot_bl_twty -> {}".format(tot_bl_twty))
    print ("tot_bl_15 -> {}".format(tot_bl_15))
    print ("tot_bl_10 -> {}".format(tot_bl_10))

    print ("\ntotal_blks_pl -> {}".format(total_blks_pl))
    print ("total_blks_trs -> {}".format(total_blks_trs))

    
    return data
