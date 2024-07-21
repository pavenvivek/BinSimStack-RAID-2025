import sys, os, subprocess
import glob, json
import pyhidra
from params import *
from insert_db import *

pyhidra.start()

import ghidra
from ghidra.program.model.block import SimpleBlockModel
from ghidra.program.util.string import StringSearcher
from ghidra.program.flatapi import FlatProgramAPI 
from ghidra.program.model.address import *
#from utils import *

from ghidra.program.util import DefinedDataIterator
from ghidra.app.util import XReferenceUtil



def build_function_graph(code_block, to_visit_list, monitor, model, prg, str_ref, fn_ref):

    graph = {}
    visited_list = []
    ref_dict = {}
    adj_dict = {}
    instr = {}
    instr_str = {}
    instr_byte = {}
    calls = {}
    flow_type_dict = {}
    #node_type = {code_block.getName(): "Entry"}
    #func_call = {code_block.getName(): None}
    ref = 0
    block_strs = {}
    indirect = {}

    
    try:
    
      while len(to_visit_list) > 0:
        visit_block = to_visit_list.pop()
        
        insts = prg.getListing().getInstructions(AddressSet(visit_block.getMinAddress(), visit_block.getMaxAddress()), True)
        #source_block = visit_block.getName()
        # for virtualize to handle double switch
        source_block = visit_block.getName() + "_" + visit_block.getMinAddress().toString()
        calls[source_block] = []

        strs = []
        flow_type = {}
        ins_ls = []
        ins_byte_ls = []
        for inst in insts:
            addr = inst.getAddressString(False, False)

            if addr in str_ref.keys():
                strs.append(str_ref[addr])

            ibytes = inst.getBytes()
            ins_ls.append(inst.toString()) #, ibytes.tolist()))
            ins_byte_ls.append(' '.join(str(e) for e in list(ibytes))) #.tolist()))

            # checking if 2nd operand is indirect call. Direct "Call" has only 1 operand. not working for cross-archs aarch64, powerpc64
            opType = inst.getOperandType(2)
            if opType is not None:
                for fn_addr in fn_ref.keys():
                    if fn_addr in str(inst):
                        if source_block not in indirect.keys():
                            indirect[source_block] = []
                        indirect[source_block].append(fn_ref[fn_addr])


        
        if source_block in ref_dict.keys():
            k = ref_dict[source_block]
        else:
            ref_dict[source_block] = ref
            k = ref
            ref = ref + 1
            

        instr[k] = ins_ls
        instr_str[source_block] = ins_ls
        instr_byte[source_block] = ins_byte_ls
        block_strs[source_block] = strs

        # mark as visited
        visited_list.append(visit_block)

        dest_it = visit_block.getDestinations(monitor)
        while dest_it.hasNext():
            dest_ref = dest_it.next()
            
            if dest_ref.getFlowType().isCall():
                if source_block not in calls.keys():
                    calls[source_block] = []
                calls[source_block].append(dest_ref.getDestinationBlock().getName())
                continue
        
            dest_block = dest_ref.getDestinationBlock()
            # for virtualize to handle double switch
            destination_block = dest_block.getName() + "_" + dest_block.getMinAddress().toString()

            if source_block not in graph.keys():
                graph[source_block] = []
            
            graph[source_block].append(destination_block)
            
            if destination_block in ref_dict.keys():
                v = ref_dict[destination_block]
            else:
                ref_dict[destination_block] = ref
                v = ref
                ref = ref + 1

            if k in adj_dict.keys(): 
                adj_dict[k].append(v)
            else:
                adj_dict[k] = [v]


            if dest_block not in visited_list and dest_block not in to_visit_list:
                to_visit_list.append(dest_block)

            
            #dest_ref = dest_it.next()
    
    except Exception as Exc:
        print ("Exception in build_function_graph: {}".format(Exc))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("\nType: {}, fname: {}, line: {}\n".format(exc_type, fname, exc_tb.tb_lineno))
    
    
    
    return (graph, instr_str, instr_byte, ref_dict, adj_dict, calls, block_strs, indirect)


if __name__ == "__main__":

    #src_path = f"{binary_path}" #/home/paven/Studies/Spring_2024/sample_obj/"
    files = glob.glob(f"{binary_path}" + "*.o")
    files_2 = glob.glob(f"{trs_binary_path}" + "*.o")
    files = files + files_2 # files + 
    file_names = [os.path.basename(fl) for fl in files]

    print(f"\nfiles -> {files}\n")

    adj_mat_lst = []
    with open("func_trs_lst.json") as fn_trs_lst_file:

        fn_trs_lst = json.load(fn_trs_lst_file)
        print (f"\nfn_trs_lst - > {fn_trs_lst}\n")
        for fl in files:
            if os.path.basename(fl) in fn_trs_lst.keys():
                with pyhidra.open_program(fl) as flat_api:


                    print (f"\n\n\n<---------- New program : ## {fl} ## ------------->\n\n")
                    program = flat_api.currentProgram #getCurrentProgram()


                    model = SimpleBlockModel(program, True)
                    fns = program.getFunctionManager().getFunctions(True)
                    flst = list(fns)

                    #print(f"func list -> {flst}")

                    str_ref = {}
                    for string in DefinedDataIterator.definedStrings(program):
                        for refr in XReferenceUtil.getXRefList(string):
                            str_ref[str(refr)] = string


                    func_ref = {}
                    for fn in flst:
                        if "<EXTERNAL>" not in str(fn):
                            #print (f"name -> {fn}, addr -> {fn.getEntryPoint()}")
                            ref = str(fn.getEntryPoint())
                            func_ref["0x" + ref[2:]] = str(fn)


                    for fn in flst:
                        if "<EXTERNAL>" not in str(fn) and str(fn) in fn_trs_lst[os.path.basename(fl)]:

                            print(f"\n<- function ## {str(fn)} ## ----->\n")

                            #print(f"\n=================> {str(fn)} addr space -------------> {fn.getBody()}\n")
                            #codeBlkitr = model.getCodeBlocksContaining(fn.getBody(), None)

                            #blk_cnt = 0
                            #while codeBlkitr.hasNext():
                            #    blk_cnt = blk_cnt + 1
                            #    codeBlkitr.next()

                            # skip functions with basic block count < 5
                            #if blk_cnt < 5:
                                #print (f"block count -> {blk_cnt}")
                            #    continue

                            entry_points = []

                            cblk = model.getCodeBlockAt(fn.getEntryPoint(), ghidra.util.task.TaskMonitor.DUMMY)
                            entry_points.insert(0, cblk)
                            #print("=========> entry_points : {}".format(cblk.getName()))
                            graph, instr_str, instr_byte, ref_dict, adj_dict, calls, block_strs, indirect = build_function_graph(cblk, entry_points, ghidra.util.task.TaskMonitor.DUMMY, model, program, str_ref, func_ref)

                            addr_space = list(fn.getBody())

                            trs = "N/A"
                            if "trs_obj" in fl:
                                trs = transformation

                            print(f"graph -> {graph}, instr_str -> {instr_str}, ref_dict -> {ref_dict}, adj_dict -> {adj_dict}, calls -> {calls}, block_strs -> {block_strs}, indirect -> {indirect}, addr_space -> {addr_space}, len addr_space -> {len(addr_space)}, trs -> {trs}")

                            uuid = subprocess.getoutput("uuidgen --random")

                            adj_mat_lst.append((graph, instr_str, instr_byte, ref_dict, adj_dict, calls, str(block_strs), indirect, addr_space, str(fn), os.path.basename(fl), \
                                                   architecture, compiler, compiler_version, cpu, library, os_type, os_version, trs, compiler_flags, uuid))


    #print("\nadj_mat_lst -> {}\n".format(adj_mat_lst))
    insert_db_mongo(adj_mat_lst)
    print("\nadj_mat_lst len -> {}\n".format(len(adj_mat_lst)))
