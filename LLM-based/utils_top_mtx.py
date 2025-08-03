import numpy as np
#from sklearn.metrics import auc, roc_curve
#from siamese_triplet import graphnn
import json, sys
from settings import *
import pymongo

np.set_printoptions(threshold=sys.maxsize)

def get_func_from_db():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db1 = client[FUNC_INFO_DB]
    col1 = db1[FUNC_INFO_COLLECTION]

    for data in col1.find({"experiment" : expr_num}, {"_id": 0, "testing_fns" : 1, "validation_fns": 1, "training_fns" : 1, "library" : 1, "experiment" : 1, "trs" : 1}):
      testing_fns = data["testing_fns"]
      validation_fns = data["validation_fns"]
      training_fns = data["training_fns"]
      exp_num = data["experiment"]
      inc_libraries = data["library"]
 
    return testing_fns          


def get_func_info(F_NAME):
    name_num = 0
    name_dict = {}
    func_lst = []
    
    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                if (g_info['fname'] not in name_dict):
                    name_dict[g_info['fname']] = name_num
                    name_num += 1

                    func_lst.append((g_info['fname'], g_info['func'], g_info['library']))
    return func_lst


def get_f_dict(F_NAME):
    name_num = 0
    name_dict = {}
    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                if (g_info['fname'] not in name_dict):
                    name_dict[g_info['fname']] = name_num
                    name_num += 1
    return name_dict

