##################################
#
# This file implements the Gemini encoding
#
##################

from settings import *
from instr_repo import *
import json
import re


def encode_data(data_lst):
        
    data = []
    func_dt = {}
    n_str_cnst_blks = 0
    log_inst_blk = False
    n_log_inst_blks = 0
    arth_inst_blk = False
    n_arth_inst_blks = 0

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
           # node level features of Gemini
           str_consts  = 0
           num_consts  = 0
           trs_instrs  = 0
           n_calls     = 0
           instrs      = 0
           arth_instrs = 0
           offspr      = 0
           str_consts  = 0
           log_instrs  = 0
           #childs      = 0
           parent_ft   = []
           

           if k not in inst_str.keys():
               print ("fl: {}, lib: {}".format(flname, lib))

           if k in n_str_consts.keys():
               str_consts = n_str_consts[k]
               n_str_cnst_blks = n_str_cnst_blks + 1

           log_inst_blk = False
           arth_inst_blk = False

           for ins in inst_str[k]:
           
               op, _, args = ins.partition(' ')
               instrs = instrs + 1
               
               op = op.lower()
               if op in ARCH_MNEM[arch]["transfer"]:
                   trs_instrs = trs_instrs + 1
               elif op in ARCH_MNEM[arch]["arithmetic"]:
                   arth_instrs = arth_instrs + 1
                   arth_inst_blk = True
               elif op in ARCH_MNEM[arch]["logic"]:
                   log_instrs = log_instrs + 1
                   log_inst_blk = True

           if arth_inst_blk:
               n_arth_inst_blks = n_arth_inst_blks + 1

           if log_inst_blk:
               n_log_inst_blks = n_log_inst_blks + 1
                   
           if v in adj_dct.keys():
               out_deg = len(adj_dct[v])
           
           in_deg = 0
           for k1,v1 in adj_dct.items():
               if v in adj_dct[k1] and v != k1:
                   in_deg = in_deg + 1
                   
           
           for c in consts[k]:
               c_val = str(c)
               # filtering consts introduced by vex
               if len(c_val) < 10:
                   num_consts = num_consts + 1
           
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
           
           # node label encoding for Gemini - use log_instrs instead of str_consts for Gemini-trp
           features.append([log_instrs, num_consts, trs_instrs, n_calls, instrs, arth_instrs, offspr])
           #features.append([str_consts, num_consts, trs_instrs, n_calls, instrs, arth_instrs, offspr])
           

       # length of encoding should 7 for Gemini
       for ft in features:
          if len(ft) != 7:
              print ("\n!!! feature length is different -> {}\n".format(len(ft)))
       
       data_i = {"src": flname, "n_num" : n_num, "succs" : succs, "features" : features, "fname" : func + "___" + lib, "func" : func, "compiler_flag" : cflg, "arch" : arch,
                 "trs" : [trs], "library" : lib, "model" : "Gemini_trp"}     
       data.append(data_i)
     
    print (f"Total string constant blocks: {n_str_cnst_blks}")
    print (f"Total logical instr. blocks: {n_log_inst_blks}")
    print (f"Total arithmetic instr. blocks: {n_arth_inst_blks}")
    
    return data
