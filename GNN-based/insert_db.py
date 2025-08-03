import pymongo
from settings import *


def setup_db():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[TREES_DB]
    block_flow_col = db[CODE_GRAPH_COLLECTION]

    if block_flow_col.count_documents({}) == 0:
        block_flow_col.create_index([("uuid", 1)], unique=True)
        print ("created index for block flow collection !")
            


def insert_db_mongo(data):
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[DIFF_DB]
    col = db[DIFF_COLLECTION]
    
    print ("insert_db_mongo called !") #.format(data))
    
    for ind, func, model, trs, lib, diff_lst_1, diff_lst_2, y_lst, uuid in data:
    
        print ("Inserting {}, {}, {}, {} in db".format(ind, func+"_"+uuid, trs, model))

        record = {"index" : ind, "function" : func, "model": model, "library" : lib, "transformation" : trs, "diff_lst_1" : diff_lst_1, "diff_lst_2" : diff_lst_2, "y_lst" : y_lst, "uuid" : uuid}

        try:
            idt = col.insert_one(record)
            #print ("collection inserted with id {}".format(idt))
        except Exception as Exc:
            print ("Exception in inserting ({}, {}, {}): {}".format(func, model, trs, Exc))


def insert_db_function_info_mongo(data):
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[FUNC_INFO_DB]
    col = db[FUNC_INFO_COLLECTION]
    
    print ("insert_db_function_info_mongo called !") #.format(data))
    
    for testing_fns, validation_fns, training_fns, lib, exp, trs in data:
    
        record = {"testing_fns" : testing_fns, "validation_fns" : validation_fns, "training_fns" : training_fns, "library" : lib, "experiment" : exp, "trs" : trs}

        try:
            idt = col.insert_one(record)
            #print ("collection inserted with id {}".format(idt))
        except Exception as Exc:
            print ("Exception in inserting ({}, {}, {}): {}".format(model, trs, uudi, Exc))

def insert_db_encodings_mongo(data):
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[CODE_ENC_DB]
    col = db[CODE_ENC_COLLECTION]
    
    print ("insert_db_encodings_mongo called !") #.format(data))
    
    for src, n_num, succs, features, fname, func, compiler_flag, arch, trs, library, model in data:
    
        #print ("Inserting {}, {}, {} in db".format(ind, func+"_"+uuid, trs))

        record = {"src": src, "n_num" : n_num, "succs" : succs, "features" : features, "fname" : fname, "func" : func, "compiler_flag" : compiler_flag, "arch" : arch, "trs" : trs, "library" : library, "model" : model}

        try:
            idt = col.insert_one(record)
            #print ("collection inserted with id {}".format(idt))
        except Exception as Exc:
            print ("Exception in inserting ({}, {}, {}): {}".format(func, library, trs, Exc))

            
def get_functions_from_db():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[TREES_DB]
    col = db[CODE_GRAPH_COLLECTION]

    return col.distinct("function")

def get_functions_and_trs_from_db():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[TREES_DB]
    col = db[CODE_GRAPH_COLLECTION]

    cursor = col.aggregate([{ "$group": { "_id": { "filename": "$filename", "transformation": "$transformation" } } }])
    docs = []
    
    for doc in cursor:
        #print (f"document -> {doc.values()}")
        docs.append((doc["_id"]["filename"], doc["_id"]["transformation"]))

    #print (f"docs -> {docs}")

    return docs #cursor #col.aggregate([{ "$group": { "_id": { "function": "$function", "transformation": "$transformation" } } }])


