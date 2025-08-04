import angr, json
import sys, os, subprocess, glob
#import pymongo

from params import *
#from insert_db import *

include_lib = igraph_inc

def generate_obj_files(files, adj_mat_lst, bin_path="./obj", trs="N/A"):

    file_names = [os.path.basename(fl) for fl in files]
    
    print ("\nfiles -> {}\nfile_names -> {}\n".format(files, file_names))

    db_files = [] #get_functions_and_trs_from_db()
    #print (f"db_files -> {db_files}")
    
    file_dct = {}
    obj_dct  = {}
    for i in range(0, len(files)):
        fl_name = file_names[i].split(".c")[0]
        
        #if (fl_name+".c", trs) in db_files:
        #    print (f"fl_name -> {fl_name}, trs -> {trs}")
        #    continue
        
        command = (f"gcc -c -m64 -O0 "
                   #f"mips64-linux-gnuabi64-gcc -c " 
                   #f"aarch64-linux-gnu-gcc -c -O0 " #"
                   #f"powerpc-linux-gnu-gcc -c -O0 "
                   + include_lib +
                   #f"--param case-values-threshold=5 " # additional flag for aarch64
                   f"{files[i]} "
                   f"-o {bin_path}O0/{fl_name}.o")

        compile_output = subprocess.getoutput(command)

        print ("compile command -> {}\n".format(command))
        if "error:" in compile_output.lower():
            print(compile_output)
            #exit(1)
            continue

        command = (f"gcc -c -m64 -O1 "
                   #f"mips64-linux-gnuabi64-gcc -c " 
                   #f"aarch64-linux-gnu-gcc -c -O1 " #"
                   #f"powerpc-linux-gnu-gcc -c -O1 "
                   + include_lib +
                   #f"--param case-values-threshold=5 " # additional flag for aarch64
                   f"{files[i]} "
                   f"-o {bin_path}O1/{fl_name}.o")

        compile_output = subprocess.getoutput(command)

        print ("compile command -> {}\n".format(command))
        if "error:" in compile_output.lower():
            print(compile_output)
            #exit(1)
            continue

        command = (f"gcc -c -m64 -O2 "
                   #f"mips64-linux-gnuabi64-gcc -c " 
                   #f"aarch64-linux-gnu-gcc -c -O2 " #"
                   #f"powerpc-linux-gnu-gcc -c -O2 "
                   + include_lib +
                   #f"--param case-values-threshold=5 " # additional flag for aarch64
                   f"{files[i]} "
                   f"-o {bin_path}O2/{fl_name}.o")

        compile_output = subprocess.getoutput(command)

        print ("compile command -> {}\n".format(command))
        if "error:" in compile_output.lower():
            print(compile_output)
            #exit(1)
            continue
        

        command = (f"gcc -c -m64 -O3 "
                   #f"mips64-linux-gnuabi64-gcc -c " 
                   #f"aarch64-linux-gnu-gcc -c -O3 " #"
                   #f"powerpc-linux-gnu-gcc -c -O3 "
                   + include_lib +
                   #f"--param case-values-threshold=5 " # additional flag for aarch64
                   f"{files[i]} "
                   f"-o {bin_path}O3/{fl_name}.o")

        compile_output = subprocess.getoutput(command)

        print ("compile command -> {}\n".format(command))
        if "error:" in compile_output.lower():
            print(compile_output)
            #exit(1)
            continue

        
        fn_lst = []
        if trs == "N/A":

            try:
                proj = angr.Project(f"{bin_path}O0/{fl_name}.o", load_options={'auto_load_libs': False})
                cfg = proj.analyses.CFGFast()

                #existing_fns = [] #get_functions_from_db()

                for j in cfg.kb.functions:
                    # drop functions with basic block cnt < 5
                    if len(cfg.kb.functions[j].block_addrs) >= 5: # and (cfg.kb.functions[j].name not in existing_fns):
                        fn_lst.append(cfg.kb.functions[j].name)
            except Exception as exp:
                print (f"Angr Exception Raised: {exp}")
                continue
        
        file_dct[files[i]] = fn_lst
        obj_dct[f"{fl_name}.o"] = fn_lst
        
    return file_dct, obj_dct #, adj_mat_lst

    
if __name__ == "__main__":

    #print("existing functions -> {}".format(get_functions_and_trs_from_db()))

    adj_mat_lst = []
    files = glob.glob(src_path + "*.c")

    file_dct, obj_dct = generate_obj_files(files, adj_mat_lst, bin_path=binary_path, trs="N/A")    
    print("file_dct -> {}\n".format(file_dct))
    
    with open(f"{code_path}func_trs_lst.json", "w") as outfile: 
        json.dump(obj_dct, outfile)
    



    
