import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import tensorflow as tf
#print (tf.__version__)
import numpy as np
import argparse
import json, pymongo
from datetime import datetime
#from siamese_triplet import graphnn
from utils_top_stat import *
#from utils_top_heat import *
from settings import *
#from insert_db import *


def get_data_from_db():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[DIFF_DB]
    col = db[DIFF_COLLECTION]
    data_lst = {}
    model_flst = {}
    fn_lst = {}

    inc_models = ["Cirrina", "Gemini_trp", "Palmtree_trp", "jTrans"]
    for data in col.find({"model" : {'$in' : inc_models}}, {"_id": 0, "index" : 1, "function": 1, "model" : 1, "transformation" : 1, "library" : 1, "diff_lst_1" : 1, "diff_lst_2" : 1, "y_lst" : 1, "uuid" : 1}):

      #print (f"lib -> {data['library']}")
      if data["library"] == inc_libraries:
      #if data["uuid"] in ["4a5e516c-cee5-4e56-b1c3-eaba3176efea", "a953cf53-ad9d-432c-890c-85bd9e01f634", "d49c4f27-a593-4f27-820f-e1970c46d3b4"]:    
        
        #print (f"len diff_lst -> {len(data['diff_lst_2'])}")  

        if data["index"] not in data_lst.keys():
            data_lst[data["index"]] = {}
            data_lst[data["index"]]["function"] = data["function"]
            #data_lst[data["index"]]["model"] = data["model"]

        if data["model"] not in model_flst.keys():
            model_flst[data["model"]] = {}
            
        if data["model"] == "Cirrina":
            fn_lst[data["index"]] = data["function"]
            model_flst[data["model"]][data["function"]] = data["index"]
            data_lst[data["index"]]["diff1"] = data["diff_lst_1"]
            data_lst[data["index"]]["diff2"] = data["diff_lst_2"]
            data_lst[data["index"]]["y1"]   = data["y_lst"]
        elif data["model"] == "Gemini_trp":
            model_flst[data["model"]][data["function"]] = data["index"]
            data_lst[data["index"]]["diff3"] = data["diff_lst_1"]
            data_lst[data["index"]]["diff4"] = data["diff_lst_2"]
            data_lst[data["index"]]["y2"]   = data["y_lst"]
        elif data["model"] == "Palmtree_trp":
            model_flst[data["model"]][data["function"]] = data["index"]
            data_lst[data["index"]]["diff5"] = data["diff_lst_1"]
            data_lst[data["index"]]["diff6"] = data["diff_lst_2"]
            data_lst[data["index"]]["y3"]   = data["y_lst"]
        elif data["model"] == "jTrans":
            model_flst[data["model"]][data["function"]] = data["index"]
            data_lst[data["index"]]["diff7"] = data["diff_lst_1"]
            data_lst[data["index"]]["diff8"] = data["diff_lst_2"]
            data_lst[data["index"]]["y4"]   = data["y_lst"]

    cnt = 0        
    for i in model_flst["Cirrina"].keys():

        if (model_flst["Cirrina"][i] != model_flst["Palmtree_trp"][i]) or (model_flst["Cirrina"][i] != model_flst["Gemini_trp"][i]) or (model_flst["Cirrina"][i] != model_flst["jTrans"][i]):
            print (f"({model_flst['Cirrina'][i]}, {model_flst['jTrans_e50'][i]})\t -> {i} not same for Cir and jTrans !")
            cnt += 1

    print (f"\n\nTotal function count -> {len(model_flst['Cirrina'].keys())}")
    print (f"Total clash -> {cnt}\n\n")
            
    info_lst = []
    for k in data_lst.keys():
        #print (f"k -> {k}")
        info_lst.append((k, data_lst[k]["function"], data_lst[k]["diff1"], data_lst[k]["diff2"], data_lst[k]["diff3"], data_lst[k]["diff4"], data_lst[k]["diff5"], data_lst[k]["diff6"], data_lst[k]["diff7"], data_lst[k]["diff8"], data_lst[k]["y1"], data_lst[k]["y2"], data_lst[k]["y3"], data_lst[k]["y4"]))


    
    f_lst = []
    for ind in range(0, len(fn_lst)):
        f_lst.append(fn_lst[ind])
        
    #print (f"f_lst -> {f_lst}")    
        
    return info_lst


if __name__ == '__main__':

    avg_acu = 0
    total_auc = 0
    avg_acc2 = 0
    total_auc2 = 0
    total_auc2_1 = 0
    total_auc2_2 = 0
    total_auc2_3 = 0
    total_auc2_4 = 0
    total_auc_cmb_2 = 0
    i = 0
    
    for i in range(0, 1):


      data_lst = get_data_from_db()
      test_auc, fpr, tpr, thres, auc2_1, auc2_2, auc2_3, auc2_4, auc_cmb2 = get_auc_epoch_combined_mtx(data_lst)
      #break
      
      total_auc2_1 = total_auc2_1 + auc2_1
      total_auc2_2 = total_auc2_2 + auc2_2
      total_auc2_3 = total_auc2_3 + auc2_3
      total_auc2_4 = total_auc2_4 + auc2_4
      total_auc_cmb_2 = total_auc_cmb_2 + auc_cmb2 
    print ("\n\n-------------------------------------")
    #print ("Average accuracy (AUC) after {} runs: {}".format(i+1, (total_auc/(i+1)) * 100)) # Palmtree
    #print ("Cirrina Average accuracy after {} runs: {}".format(i+1, total_auc2/(i+1)))
    print ("Cirrina  average accuracy after {} runs: {}".format(i+1, total_auc2_1/(i+1)))
    print ("Gemini   average accuracy after {} runs: {}".format(i+1, total_auc2_2/(i+1)))
    print ("Palmtree average accuracy after {} runs: {}".format(i+1, total_auc2_3/(i+1)))
    print ("jTrans   average accuracy after {} runs: {}".format(i+1, total_auc2_4/(i+1)))
    print ("Average accuracy combined (Aggregate) after {} runs: {}".format(i+1, total_auc_cmb_2/(i+1)))
