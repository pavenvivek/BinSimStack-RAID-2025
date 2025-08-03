import sys
from datautils.playdata import DatasetBase as DatasetBase
import networkx
import os
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
import pickle
import argparse
import re
import readidadata
import torch
import random
import time, json
from utils_top_mtx import *
#from sample_funcs_angr import *
from inc_fns_lst_flt_patch import *
#from inc_fns_lst_viz_patch import *
#from inc_fns_lst_flt_patch_scale import *
#from inc_fns_lst_viz_patch_scale import *
#from inc_fns_lst_viz_patch_scale_malw import *
#from inc_fns_lst_O0_O1_scale import *
#from inc_fns_lst_O0_O2_scale import *
#from inc_fns_lst_O0_O3_scale import *
#from inc_fns_lst_flt_encArth_patch import *
MAXLEN=512

vocab_data = open("./jtrans_tokenizer/vocab.txt").read().strip().split("\n") + ["[SEP]", "[PAD]", "[CLS]", "[MASK]"]
my_vocab = defaultdict(lambda: 512, {vocab_data[i] : i for i in range(len(vocab_data))})


def help_tokenize(line):
    global my_vocab
    ret = {}
    split_line = line.strip().split(' ')
    split_line_len = len(split_line)
    if split_line_len <= 509:
        split_line = ['[CLS]']+split_line+['[SEP]']
        attention_mask = [1] * len(split_line) + [0] * (512 - len(split_line))
        split_line = split_line + (512-len(split_line))*['[PAD]']
    else:
        split_line = ['[CLS]'] + split_line[:510] + ['[SEP]']
        attention_mask = [1]*512
    input_ids = [my_vocab[e] for e in split_line]
    ret['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
    ret['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
    return ret

def gen_funcstr(f,convert_jump):
    cfg=f[3]
    #print(hex(f[0]))
    #print(f"cfg -> {cfg.nodes}")
    bb_ls,code_lst,map_id=[],[],{}
    for bb in cfg.nodes:
        bb_ls.append(bb)
    bb_ls.sort()
    #print(f"bb_ls -> {bb_ls}")
    for bx in range(len(bb_ls)):
        bb=bb_ls[bx]
        asm=cfg.nodes[bb]['asm']
        map_id[bb]=len(code_lst)
        for code in asm:
            operator,operand1,operand2,operand3,annotation=readidadata.parse_asm(code)
            code_lst.append(operator)
            if operand1!=None:
                code_lst.append(operand1)
            if operand2!=None:
                code_lst.append(operand2)
            if operand3!=None:
                code_lst.append(operand3)
    #print(f"map_id -> {map_id}")
    for c in range(len(code_lst)):
        op=code_lst[c]
        #print(f"op -> {op}")
        if op.startswith('hex_'):
            jumpaddr=int(op[4:],base=16)
            #print(f"jumpaddr -> {jumpaddr}")
            if map_id.get(jumpaddr):
                jumpid=map_id[jumpaddr]
                if jumpid < MAXLEN:
                    code_lst[c]='JUMP_ADDR_{}'.format(jumpid)
                else:
                    code_lst[c]='JUMP_ADDR_EXCEEDED'
            else:
                code_lst[c]='UNK_JUMP_ADDR'
            if not convert_jump:
                code_lst[c]='CONST'
    func_str=' '.join(code_lst)
    #print (f"func_str -> {func_str}")
    return func_str

def load_unpair_data(datapath,filt=None,alldata=True,convert_jump=True,opt=None, fp=None):
    dataset = DatasetBase(datapath,filt, alldata)
    dataset.load_unpair_data()
    functions=[]
    for i in dataset.get_unpaird_data():  #proj, func_name, func_addr, asm_list, rawbytes_list, cfg, bai_featrue
        f = (i[2], i[3], i[4], i[5], i[6])
        func_str=gen_funcstr(f,convert_jump)
        if len(func_str) > 0:
            fp.write(func_str+"\n")

def load_paired_data(datapath,filt=None,alldata=True,convert_jump=True,opt=None,add_ebd=False):
   
    dataset = DatasetBase(datapath,filt,alldata, opt=opt)
    functions=[]
    func_emb_data=[]
    SUM=0
    for i in dataset.get_paired_data_iter():  #proj, func_name, {func_addr, asm_list, rawbytes_list, cfg, bai_featrue}
        functions.append([])
        if add_ebd:
            func_emb_data.append({'proj':i[0],'funcname':i[1]})
        for o in opt:
            if i[2].get(o):                   
                f=i[2][o]
                func_str=gen_funcstr(f,convert_jump)
                if len(func_str)>0:
                    if add_ebd:
                        func_emb_data[-1][o]=len(functions[-1])
                    functions[-1].append(func_str)
                    SUM+=1

    #print('TOTAL ',SUM)
    print(f"TOTAL -> {SUM}")
    #print(f"funs -> {functions}, \n\n\nemb_data -> {func_emb_data}")
    return functions,func_emb_data


def load_paired_data_angr_train(datapath,filt=None,alldata=True,convert_jump=True,opt=None,add_ebd=False):
   
    #dataset = DatasetBase(datapath,filt,alldata, opt=opt)
    data_path = [datapath] #"./data/train_data.json"]
    functions=[]
    func_emb_data=[]
    SUM=0
    func_info = {}

    for f_name in data_path:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                fname = g_info['fname']
                func  = g_info['func']
                func_str  = g_info['func_str']
                trs  = g_info['trs'][0]
                cflg = g_info['compiler_flag']
                arch = g_info['arch']

                if func not in func_info.keys():
                    func_info[func] = {}
                func_info[func][trs] = func_str
                #func_info[func][cflg] = func_str

    for k,v in func_info.items():
        #functions.append([v["EncodeArithmetic, Flatten"],v["N/A"]])
        functions.append([v["Flatten"],v["N/A"]])
        #functions.append([v["Virtualize"],v["N/A"]])
        #functions.append([v["O3"],v["O0"]])
        func_emb_data.append(k)
        
    #print(f"funs -> {functions}, \n\n\nemb_data -> {func_emb_data}")
    #print (f"\n\nemb_data -> {func_emb_data}\n\n")
    #print (f"func_info_idx -> {func_info_idx}")
    return functions,func_emb_data


def load_paired_data_angr(datapath,filt=None,alldata=True,convert_jump=True,opt=None,add_ebd=False):
   
    #dataset = DatasetBase(datapath,filt,alldata, opt=opt)
    data_path = [datapath] #"./data/train_data.json"]
    functions=[]
    func_emb_data=[]
    SUM=0
    func_info = {}

    for f_name in data_path:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                fname = g_info['fname']
                func  = g_info['func']
                func_str  = g_info['func_str']
                trs  = g_info['trs'][0]
                cflg = g_info['compiler_flag']
                arch = g_info['arch']
                lib = g_info['library']

                #----
                # normal routine (not scalability)
                if func not in func_info.keys():
                    func_info[func] = {}

                func_info[func][trs] = func_str                
                #func_info[func][cflg] = func_str
                #----

                #----
                # for scalability only
                #if fname not in func_info.keys():
                #    func_info[fname] = {} 

                # for scalability only
                #func_info[fname][trs] = func_str
                #func_info[fname][cflg] = func_str
                #----
                
    #print (f"func_info keys -> {func_info.keys()}")

    func_lst = gem_openssl #get_func_from_db() #scale_5000 #get_func_from_db() #gem_libxml2 #
    for fn in func_lst:
        #print (f"fn -> {fn}")
        
        #functions.append([func_info[fn]["EncodeArithmetic, Flatten"],func_info[fn]["N/A"]])
        functions.append([func_info[fn]["Flatten"],func_info[fn]["N/A"]])
        #functions.append([func_info[fn]["Virtualize"],func_info[fn]["N/A"]])
        #functions.append([func_info[fn]["O3"],func_info[fn]["O0"]])
        func_emb_data.append(fn) # + "___" + func_info[fn]["lib"])
                
        
    #print(f"funs -> {functions}, \n\n\nemb_data -> {func_emb_data}")
    print (f"\n\nemb_data -> {func_emb_data}\n\n")
    #print (f"func_info_idx -> {func_info_idx}")
    return functions,func_emb_data


class FunctionDataset_CL_top(torch.utils.data.Dataset): #binary version dataset
    def __init__(self,tokenizer,path='./data/test_data.json',filt=None,alldata=True,convert_jump_addr=True,opt=None,add_ebd=True):  #random visit
        functions,ebds=load_paired_data_angr(datapath=path,filt=filt,alldata=alldata,convert_jump=convert_jump_addr,opt=opt,add_ebd=add_ebd)
        print (f"\n\nEnters Top Eval: Path -> {path}\n\n")
        self.datas=functions
        self.ebds=ebds
        self.tokenizer=tokenizer
        self.opt=opt
        self.convert_jump_addr=True

    def __getitem__(self, idx):             #also return bad pair

        #print (f"idx -> {idx}, len -> {len(self.datas[idx])}")
        pairs=self.datas[idx]
        embd =self.ebds[idx] 
        if False: #self.opt==None:
            pos=random.randint(0,len(pairs)-1)
            pos2=random.randint(0,len(pairs)-1)
            while pos2==pos:
                pos2=random.randint(0,len(pairs)-1)
            f1=pairs[pos]   #give three pairs
            f2=pairs[pos2]
        else:
            pos=0
            pos2=1
            f1=pairs[pos]
            f2=pairs[pos2]
            emb_pos = embd
            #print (f"emb_f1 -> {emb_f1}\n\n")
            #print (f"f2 -> {f2}\n\n")
            
        ftype=random.randint(0,len(self.datas)-1)
        while ftype==idx:
            ftype=random.randint(0,len(self.datas)-1)
        pair_opp=self.datas[ftype]
        pos3=1 #random.randint(0,len(pair_opp)-1)
        f3=pair_opp[pos3]
        emb_neg = self.ebds[ftype]

        #print (f"f3 -> {f3}\n\n")
        
        ret1 = help_tokenize(f1)
        token_seq1=ret1['input_ids']
        mask1=ret1['attention_mask']

        ret2 = help_tokenize(f2)
        token_seq2=ret2['input_ids']
        mask2=ret2['attention_mask']

        ret3 = help_tokenize(f3)
        token_seq3=ret3['input_ids']
        mask3=ret3['attention_mask']

        return token_seq1,token_seq2,token_seq3,mask1,mask2,mask3,emb_pos,emb_neg

    def __len__(self):
        return len(self.datas)


class FunctionDataset_CL(torch.utils.data.Dataset): #binary version dataset
    def __init__(self,tokenizer,path='./data/test_data.json',filt=None,alldata=True,convert_jump_addr=True,opt=None,add_ebd=True):  #random visit
        functions,ebds=load_paired_data_angr(datapath=path,filt=filt,alldata=alldata,convert_jump=convert_jump_addr,opt=opt,add_ebd=add_ebd)
        #print ("\n\nEnters here -------------->\n\n")
        self.datas=functions
        self.ebds=ebds
        self.tokenizer=tokenizer
        self.opt=opt
        self.convert_jump_addr=True
    def __getitem__(self, idx):             #also return bad pair

        #print (f"idx -> {idx}, len -> {len(self.datas[idx])}")
        pairs=self.datas[idx]
        if False: #self.opt==None:
            pos=random.randint(0,len(pairs)-1)
            pos2=random.randint(0,len(pairs)-1)
            while pos2==pos:
                pos2=random.randint(0,len(pairs)-1)
            f1=pairs[pos]   #give three pairs
            f2=pairs[pos2]
        else:
            pos=0
            pos2=1
            f1=pairs[pos]
            f2=pairs[pos2]
            #print (f"f1 -> {f1}\n\n")
            #print (f"f2 -> {f2}\n\n")
        ftype=random.randint(0,len(self.datas)-1)
        while ftype==idx:
            ftype=random.randint(0,len(self.datas)-1)
        pair_opp=self.datas[ftype]
        pos3=1 #random.randint(0,len(pair_opp)-1)
        f3=pair_opp[pos3]

        #print (f"f3 -> {f3}\n\n")
        
        ret1 = help_tokenize(f1)
        token_seq1=ret1['input_ids']
        mask1=ret1['attention_mask']

        ret2 = help_tokenize(f2)
        token_seq2=ret2['input_ids']
        mask2=ret2['attention_mask']

        ret3 = help_tokenize(f3)
        token_seq3=ret3['input_ids']
        mask3=ret3['attention_mask']

        return token_seq1,token_seq2,token_seq3,mask1,mask2,mask3
    def __len__(self):
        return len(self.datas)
    
class FunctionDataset_CL_Load(torch.utils.data.Dataset): #binary version dataset
    def __init__(self,tokenizer,path='./data/train_data.json',filt=None,alldata=True,convert_jump_addr=True,opt=None,add_ebd=True, load=None):  #random visit
        if load:
            start = time.time()
            self.datas = pickle.load(open(load, 'rb'))
            print('load time:', time.time() - start)
            self.tokenizer=tokenizer
            self.opt=opt
            self.convert_jump_addr=True
        else:
            functions,ebds=load_paired_data_angr_train(datapath=path,filt=filt,alldata=alldata,convert_jump=convert_jump_addr,opt=opt,add_ebd=add_ebd)
            #func_info = load_paired_data_angr_train(datapath=path,filt=filt,alldata=alldata,convert_jump=convert_jump_addr,opt=opt,add_ebd=add_ebd)
            print (f"\n\nEnters Train: Path -> {path}\n\n")
            self.datas=[]
            for func_list in functions: #train_funcs: #
                tmp = []
                for f in func_list:
                    tmp.append(help_tokenize(f))
                self.datas.append(tmp)
                #print (f"datas -> {self.datas}")
            self.ebds=None #ebds
            self.tokenizer=tokenizer
            self.opt=opt
            self.convert_jump_addr=True
    def __getitem__(self, idx):             #also return bad pair

        #print (f"idx -> {idx}, len -> {len(self.datas[idx])}")
        #print (f"datas[idx] -> {self.datas[idx]}")
        pairs=self.datas[idx]
        if False: #self.opt!=None:
            pos=random.randint(0,len(pairs)-1)
            pos2=random.randint(0,len(pairs)-1)
            while pos2==pos:
                pos2=random.randint(0,len(pairs)-1)
            f1=pairs[pos]   #give three pairs
            f2=pairs[pos2]
        else:
            pos=0
            pos2=1
            f1=pairs[pos]
            f2=pairs[pos2]
        ftype=random.randint(0,len(self.datas)-1)
        while ftype==idx:
            ftype=random.randint(0,len(self.datas)-1)
        pair_opp=self.datas[ftype]
        pos3=1 #random.randint(0,len(pair_opp)-1)
        f3=pair_opp[pos3]

        token_seq1=f1['input_ids']
        mask1=f1['attention_mask']

        token_seq2=f2['input_ids']
        mask2=f2['attention_mask']

        token_seq3=f3['input_ids']
        mask3=f3['attention_mask']

        return token_seq1,token_seq2,token_seq3,mask1,mask2,mask3
    def __len__(self):
        return len(self.datas)

def load_filter_list(name):
    import csv
    f=csv.reader(open(name,'r'))
    S=set()
    for i in f:
        S.add(i[1])
    return list(S)
