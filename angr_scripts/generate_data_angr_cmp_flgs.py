import sys, os, subprocess
import glob, json
from params import *
from insert_db import *
import angr


def generate_function_data(proj, cfg, func, trs, cflg, filename):

    print ("Current Function: {}".format(func.name))
     
    ref_dict   = {}
    instr_byte = {}
    instr_str  = {}
    vex_ir     = {}
    consts     = {}
    calls      = {}
    graph      = {}
    n_str_cnst = {}
    
    blk_addrs = list(func.block_addrs)
    entry_block = func.addr
    call_sites = list(func.get_call_sites())
    nodes = list(func.nodes)

    ref = 0
    for i in blk_addrs:
        
        ref_dict[ref] = i
        ref = ref + 1
        
        irsb = proj.factory.block(i)

        instr_byte[i] = irsb.bytes

        instr_str[i] = []
        for ins in irsb.capstone.insns:
            instr_str[i].append(ins.mnemonic + " " + ins.op_str)
            
        
        vex_ir[i] = []
        for vex in irsb.vex.statements:
            vex_ir[i].append(str(vex))
            
        consts[i] = []
        
        for c in irsb.vex.constants:
            consts[i].append(int(str(c), 16))

        call_target = ""
        if i in call_sites:
            call_target = func.get_call_target(i)

            if cfg.kb.functions[call_target] is not None:
                call_target = cfg.kb.functions[call_target].name
            
            calls[i] = call_target
        
        graph[i] = []

    for n in nodes:
        xrefs = list(func._function_manager._kb.xrefs.get_xrefs_by_ins_addr_region(n.addr, n.addr + n.size))
        for xref in xrefs:
            if xref.memory_data is not None and xref.memory_data.sort == "string":
                #print(f"{i} -> {xrefs}")
                n_str_cnst[n.addr] = len(xrefs)

        
    ref_inv = {v: k for k, v in ref_dict.items()}
    #print(f"bloc_addr -> {blk_addrs}")
    #print(f"ref_inv -> {ref_inv}")
    #print(f"graph -> {graph}")

    adj_dict = {}
    graph = list(func.graph.edges)
    graph_new = []
    for edg in graph:
        (e1, e2) = edg
        graph_new.append((e1.addr, e2.addr))
        e1 = ref_inv[e1.addr]
        e2 = ref_inv[e2.addr]

        if e1 in adj_dict.keys():
            adj_dict[e1].append(e2)
        else:
            adj_dict[e1] = [e2]
            
    #print("graph: {}\n, adj_dict: {}\n, calls: {}\n, consts: {}\n, ref_dict: {}\n, inst_str: {}\n, inst_byte: {}\n, vex_ir: {}\n".format(graph, adj_dict, calls, consts, ref_inv, instr_str, instr_byte, vex_ir))
            
    uuid = subprocess.getoutput("uuidgen --random")
            
    return (graph_new, instr_str, instr_byte, vex_ir, calls, consts, n_str_cnst, ref_inv, adj_dict, func.name, filename, \
            architecture, compiler, compiler_version, cpu, library, os_type, os_version, trs, cflg, uuid)



if __name__ == "__main__":

    files = glob.glob(f"{binary_path}O0/" + "*.o")
    files_2 = glob.glob(f"{binary_path}O1/" + "*.o")
    files_3 = glob.glob(f"{binary_path}O2/" + "*.o")
    files_4 = glob.glob(f"{binary_path}O3/" + "*.o")
    files = files + files_2 + files_3 + files_4 
    file_names = [os.path.basename(fl) for fl in files]

    print(f"\nfiles -> {files}\n")

    adj_mat_lst = []


    with open("func_trs_lst.json") as fn_trs_lst_file:

        fn_trs_lst = json.load(fn_trs_lst_file)
        print (f"\nfn_trs_lst - > {fn_trs_lst}\n")
        for fl in files:
            if os.path.basename(fl) in fn_trs_lst.keys():

                try:
                    print (f"\n\n\n<---------- New program : ## {fl} ## ------------->\n\n")
                    proj = angr.Project(f"{fl}", load_options={'auto_load_libs': False})
                    cfg = proj.analyses.CFGFast()

                    trs = "N/A"
                    cflg = "O0"
                    if "O1" in fl:
                        cflg = "O1"
                    elif "O2" in fl:
                        cflg = "O2"
                    elif "O3" in fl:
                        cflg = "O3"
                    else:
                        cflg = "O0"

                    for j in cfg.kb.functions:
                        if str(cfg.kb.functions[j].name) in fn_trs_lst[os.path.basename(fl)]:
                            f_info = generate_function_data(proj, cfg, cfg.kb.functions[j], trs, cflg, os.path.basename(fl)) # N/A -> no transformation applied
                            adj_mat_lst.append(f_info)
                except Exception as exc:
                    print ("Angr Exception Raised: {exc}")
                    continue


    #print("\nadj_mat_lst -> {}\n".format(adj_mat_lst))
    insert_db_mongo(adj_mat_lst)
    print("\nadj_mat_lst len -> {}\n".format(len(adj_mat_lst)))
