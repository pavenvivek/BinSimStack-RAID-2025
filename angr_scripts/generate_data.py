import angr, json
import sys, os, subprocess, glob
#import pymongo

from params import *
#from insert_db import *


def tigress_trs(file_dct):

    trs_lst = []    # to store individual func trs
    files = list(file_dct.keys())
    file_names = [os.path.basename(fl) for fl in files]
    print ("files -> {}\nfile_names -> {}\n".format(files, file_names))

    for i in range(0, len(files)):
        fns = ""
        for j in file_dct[files[i]]:
            # skipping list of wrongly linked external functions (ticker_geom_fixup -> redis) and multiple function names by angr
            #if j not in ['ticker_geom_fixup', 'sub_40251c', 'sub_400ca3','sub_400c2a','sub_400ba7','sub_400b2e', 'main', 'usage', 'do_scrub_chars'
            #              'SetCacheAlphaChannel._omp_fn.0', 'ClonePixelCacheRepository._omp_fn.1', 'ClonePixelCacheRepository._omp_fn.0', 'GetMagickNumberThreads'] and ("sub_" not in j):
            if j not in ['has_in_name', 'testStopRace']: #libmicrohttpd - angr/tigress introducing non-existing fns
                fns = j + "," + fns


        fl_name = file_names[i].split(".c")[0]

        command = (f"tigress --Environment=x86_64:Linux:Gcc:4.6 "
               f"--gcc=gcc "
               #f"{transformation_cmd_2} "
               #f"--Functions={fns} "
               f"{transformation_cmd} "
               f"--Functions={fns} "
               + igraph_inc +
               f"--out={trs_src_path}{fl_name}.c "
               f"-c {files[i]}")

        try:
            print("tigress command -> {}\n".format(command))
            exe_output = subprocess.getoutput(command)

            if "usage" in exe_output.lower(): # or "error" in exe_output.lower():
                print(exe_output)
                #exit(1)
            else:
                trs_lst.append(f"{trs_src_path}{fl_name}.c")
        except Exception as exp:
            print (f"Tigress Exception Raised: {exp}")
            continue
            
            
    # clearing tigress generated object files
    compile_output = subprocess.getoutput(f"rm {code_path}*.o")

    return trs_lst

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
        
        command = (f"gcc -c -m64 "
                   #f"mips64-linux-gnuabi64-gcc -c " 
                   #f"aarch64-linux-gnu-gcc -c " #"
                   #f"powerpc-linux-gnu-gcc -c "
                   + igraph_inc +
                   #f"--param case-values-threshold=5 " # additional flag for aarch64
                   f"{files[i]} "
                   f"-o {bin_path}{fl_name}.o")

        compile_output = subprocess.getoutput(command)

        print ("compile command -> {}\n".format(command))
        if "error:" in compile_output.lower():
            print(compile_output)
            #exit(1)
            continue

        fn_lst = []
        if trs == "N/A":

            try:
                proj = angr.Project(f"{bin_path}{fl_name}.o", load_options={'auto_load_libs': False})
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
    
    trs_lst = tigress_trs(file_dct)
    print("trs_lst -> {}\n".format(trs_lst))
    
    file_dct2, obj_dct2 = generate_obj_files(trs_lst, adj_mat_lst, bin_path=trs_binary_path, trs=transformation)    
    #print("file_dct -> {}\n".format(file_dct))

    with open(f"{code_path}func_trs_lst.json", "w") as outfile: 
        json.dump(obj_dct, outfile)
    



    
