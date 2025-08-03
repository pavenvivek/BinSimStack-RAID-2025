##################################
#
# This file implements the Cirrina encoding
#
##################

from settings import *
#from instr_repo import *
import json
import re
MAXLEN=512

def parse_operand(operator,location,operand1):
    operand1=operand1.strip(' ')
    operand1=operand1.replace('ptr ','')
    operand1=operand1.replace('offset ','')
    operand1=operand1.replace('xmmword ','')
    operand1=operand1.replace('dword ','')
    operand1=operand1.replace('qword ','')
    operand1=operand1.replace('word ','')
    operand1=operand1.replace('byte ','')
    operand1=operand1.replace('short ','')
    operand1=operand1.replace('-','+')
    #operand1=operand1.replace('-','*')

    #print (f"pcode -> {operator, location, operand1}")
    
    if operand1[0:3]=='cs:' :
        operand1='cs:xxx'
        return operand1
    if operand1[0:3]=='ss:' :
        operand1='ss:xxx'
        return operand1
    if operand1[0:3]=='fs:' :
        operand1='fs:xxx'
        return operand1
    if operand1[0:3]=='ds:' :
        operand1='ds:xxx'
        return operand1
    if operand1[0:3]=='es:' :
        operand1='es:xxx'
        return operand1
    if operand1[0:3]=='gs:' :
        operand1='gs:xxx'
        return operand1
    if operator[0]=='j' and not isregister(operand1):
        if operand1[0:4]=='loc_' or operand1[0:7]=='locret_' or operand1[0:4]=='sub_' or operand1[0:2]=='0x':
            operand1='hex_'+operand1 #[operand1.find('_')+1:]
            return operand1
        else:
            #print("JUMP ",operand1)
            operand1='UNK_ADDR'
            return operand1

    if operand1[0:4]=='loc_' :
        operand1='loc_xxx'
        return operand1
    if operand1[0:4]=='off_' :
        operand1='off_xxx'
        return operand1
    if operand1[0:4]=='unk_' :
        operand1='unk_xxx'
        return operand1
    if operand1[0:6]=='locret' :
        operand1='locretxxx'
        return operand1
    if operand1[0:4]=='sub_' :
        operand1='sub_xxx'
        return operand1
    if operand1[0:4]=='arg_' :
        operand1='arg_xxx'
        return operand1
    if operand1[0:4]=='def_' :
        operand1='def_xxx'
        return operand1
    if operand1[0:4]=='var_' :
        operand1='var_xxx'
        return operand1
    if operand1[0]=='(' and operand1[-1]==')':
        operand1='CONST'
        return operand1
    if operator=='lea' and location==2:
        if not ishexnumber(operand1) and not isaddr(operand1):  #handle some address constants
            operand1='GLOBAL_VAR'
            return operand1

    if operator=='call' and location==1:
        if len(operand1)>3:
            operand1='callfunc_xxx'
            return operand1

    if operator=='extrn':
        operand1='extrn_xxx'
        return operand1
    if ishexnumber(operand1):
        operand1='CONST'
        return operand1
    elif ispurenumber(operand1):
        operand1='CONST'
        return operand1
    if isaddr(operand1):
        params=operand1[1:-1].split('+')
        #print (f"params -> {params}")
        for i in range(len(params)):
            params[i] = params[i].strip()
            if ishexnumber(params[i]):
                params[i]='var_xxx' #'CONST'
            elif ispurenumber(params[i]):
                params[i]='CONST'
            elif params[i][0:4]=='var_':
                params[i]='var_xxx'
            elif params[i][0:4]=='arg_':
                params[i]='arg_xxx'
            elif not isregister(params[i]):
                if params[i].find('*')==-1:
                    params[i]='var_xxx' #'CONST_VAR'
        s1='+'
        operand1='['+s1.join(params)+']'
        return operand1

    if not isregister(operand1) and len(operand1)>4:
        operand1='CONST'
        return operand1
    return operand1


def parse_asm(code):   #handle ida code to better quality code for NLP model
    annotation=None
    operator,operand=None,None
    operand1,operand2,operand3=None,None,None
    if code.find(';')!=-1:
        id=code.find(';')
        annotation=code[id+1:]
        code=code[0:id]
    if code.find(' ')!=-1:
        id=code.find(' ')
        operand=code[id+1:]
        operator=code[0:id]
    else:
        operator=code
    if operand!=None:
        if operand.find(',')!=-1:
            strs=operand.split(',')
            if len(strs)==2:
                operand1,operand2=strs[0],strs[1]
            else:
                operand1,operand2,operand3=strs[0],strs[1],strs[2]
        else:
            operand1=operand
            operand2=None

    if operand1 == "":
        operand1 = None
    if operand2 == "":
        operand2 = None
    if operand3 == "":
        operand3 = None

    #print (f"code -> {operator, operand1, operand2, operand3}")

            
    if operand1!=None:
        operand1=parse_operand(operator,1,operand1)
    if operand2!=None:
        operand2=parse_operand(operator,2,operand2)
    if operand3!=None:
        operand3=parse_operand(operator,3,operand3)
    return operator,operand1,operand2,operand3,annotation
def isregister(x):
    registers=['rax','rbx','rcx','rdx','esi','edi','rbp','rsp','rip','r8','r9','r10','r11','r12','r13','r14','r15']
    return x in registers
def ispurenumber(number):
    if len(number)==1 and str.isdigit(number):
        return True
    return False
def isaddr(number):
    return number[0]=='[' and number[-1]==']'
def ishexnumber(number):

    if number[0:2] == "0x":
        return True
    
    if number[-1]=='h':
        for i in range(len(number)-1):
            if str.isdigit(number[i]) or (number[i] >='A' and number[i]<='F'):
                continue
            else:
                return False
    else:
        return False
    return True




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
    convert_jump = True
    func_code = {}
    
    for g, refd, adj_dct, vex_ir, calls, consts, inst_str, inst_byt, func, flname, trs, cmpl, cflg, arch, lib, uid in data_lst:
        
       n_num = len(refd.keys()) # node count
       features = []
       succs = []
       ft_dct = {}
       out_deg = 0
       in_deg = 0
       out_deg_dct = {}
       in_deg_dct = {}
       code_lst = []
       map_id = {}
       
       all_vals = []
       for k,v in adj_dct.items():
           all_vals = all_vals + v       
       
       if func not in func_code.keys():
           func_code[func] = []

           
       #if trs[0] not in list(func_dt[func].keys()):
       #    func_dt[func][trs[0]] = [n_num, len(all_vals)] 

       bbs = list(refd.keys())
       #print (f"bbs -> {bbs}\n")

       bbs.sort()
       #print (f"sorted bbs -> {bbs}")
       
       for k in bbs: #,v in refd.items():
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


           if trs == "N/A": #cflg == "O0": #
               total_blks_pl = total_blks_pl + 1
           else:
               total_blks_trs = total_blks_trs + 1
               

           if k not in inst_str.keys():
               print ("fl: {}, lib: {}".format(flname, lib))


           map_id[k] = len(code_lst)
           
           for ins in inst_str[k]:
               operator,operand1,operand2,operand3,annotation = parse_asm(ins)

               if operator == "ret":
                   operator = "retn"
                   
               if operator == "endbr64":
                   operator = ""
                   continue

               code_lst.append(operator)
               if operand1!=None:
                   code_lst.append(operand1)
               if operand2!=None:
                   code_lst.append(operand2)
               if operand3!=None:
                   code_lst.append(operand3)


               #print (f"parsed code -> {operator,operand1,operand2,operand3}")

       for c in range(len(code_lst)):
           op=code_lst[c]
           #print(f"op -> {op}")
           if op.startswith('hex_'):
               jumpaddr=int(op[4:],base=16)
               #print(f"jumpaddr -> {jumpaddr}, {map_id.get(jumpaddr)}")
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
                   

       # length of encoding should be 32 for Cirrina
       #for ft in features:
       #   if len(ft) != 32:
       #       print ("\n!!! feature length is different -> {}\n".format(len(ft)))

       func_str=' '.join(code_lst)
       
       #print ("\n\n\n")
       #print (f"map_id -> {map_id}")
       #print ("\n\n\n")
       #print (f"code_lst -> {code_lst}")
       #print (f"func_str -> {func_str}")
       #print ("\n\n\n")

       func_code[func].append(func_str)

       data_i = {"src": flname, "n_num" : n_num, "succs" : succs, "func_str" : func_str, "fname" : func + "___" + lib, "func" : func, "compiler_flag" : cflg, "arch" : arch,
                 "trs" : [trs], "library" : lib, "model" : "jTrans"}     
       data.append(data_i)
     
    print ("max_block_sz -> {}".format(max_block_sz))
    print ("total_blks -> {}".format(total_blks))
    print ("tot_bl_twty -> {}".format(tot_bl_twty))
    print ("tot_bl_15 -> {}".format(tot_bl_15))
    print ("tot_bl_10 -> {}".format(tot_bl_10))

    print ("\ntotal_blks_pl -> {}".format(total_blks_pl))
    print ("total_blks_trs -> {}".format(total_blks_trs))

    #print (f"func_code -> {func_code}")

    func_code_lst = []
    for k,v in func_code.items():
        func_code_lst.append(v)

    #print (f"\n\nfunc_code_lst -> {func_code_lst}")

        
    return data
