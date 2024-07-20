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
    
    for ind, func, model, trs, typ, diff_lst_1, diff_lst_2, y_lst_1, diff_lst_3, diff_lst_4, y_lst_2, diff_lst_5, diff_lst_6, y_lst_3, uuid in data:
    
        print ("Inserting {}, {}, {} in db".format(ind, func+"_"+uuid, trs))

        record = {"index" : ind, "function" : func, "model": model, "transformation" : trs, "data_type" : typ,
                  "diff_lst_1" : diff_lst_1, "diff_lst_2" : diff_lst_2, "y_lst_1" : y_lst_1,
                  "diff_lst_3" : diff_lst_3, "diff_lst_4" : diff_lst_4, "y_lst_2" : y_lst_2,
                  "diff_lst_5" : diff_lst_5, "diff_lst_6" : diff_lst_6, "y_lst_3" : y_lst_3, "uuid" : uuid}

        try:
            idt = col.insert_one(record)
            #print ("collection inserted with id {}".format(idt))
        except Exception as Exc:
            print ("Exception in inserting ({}, {}, {}): {}".format(func, model, trs, Exc))

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

    return docs #cursor #col.aggregate([{ "$group": { "_id": { "function": "$function", "transformation": "$transformation" } } }])

