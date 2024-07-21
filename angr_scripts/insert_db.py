import pymongo
from params import *


def setup_db():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[TREES_DB]
    block_flow_col = db[CODE_GRAPH_COLLECTION]

    if block_flow_col.count_documents({}) == 0:
        block_flow_col.create_index([("uuid", 1)], unique=True)
        print ("created index for block flow collection !")
            


def insert_db_mongo(data):
    client = pymongo.MongoClient(MONGO_CLIENT)
    db = client[TREES_DB]
    col = db[CODE_GRAPH_COLLECTION]
    
    print ("insert_db_mongo called !") #.format(data))
    
    for graph, instr_str, instr_byte, vex_ir, calls, consts, n_str_cnst, ref_dict, adj_dict, func, filename, arch, comp, comp_version, cpu, lib, os_type, os_ver, trs, cflgs, uuid in data:
    
        print ("Inserting {}, {} in db".format(func+"_"+uuid, trs))
    
        record = {"graph" : str(graph), "instr_str": str(instr_str), "instr_byte": str(instr_byte), "vex_ir" : str(vex_ir), "calls" : str(calls), "consts" : str(consts), "str_consts_cnt" : str(n_str_cnst), \
                  "ref_dict" : str(ref_dict), "adj_dict" : str(adj_dict), "function" : func, "filename": filename, "architecture" : arch, "compiler" : comp, \
                  "compiler_version" : comp_version, "cpu" : cpu, "library" : lib, "os_type" : os_type, "os_version" : os_ver, "transformation" : trs, "compiler_flags" : cflgs, "uuid" : uuid}

        try:
            idt = col.insert_one(record)
            #print ("collection inserted with id {}".format(idt))
        except Exception as Exc:
            print ("Exception in inserting ({}, {}, {}, {}): {}".format(func, filename, arch, lib, Exc))

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

