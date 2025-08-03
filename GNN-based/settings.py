
MONGO_CLIENT = "mongodb://localhost:27017/"

CODE_GRAPH_COLLECTION = "code_graph"
FUNC_INFO_COLLECTION = "func_info"
CODE_ENC_COLLECTION = "code_encodings"
DIFF_COLLECTION = "diff_graph"

FUNC_INFO_DB = "func_info_db"
TREES_DB = "angr_proj_db"
CODE_ENC_DB = "encodings_db"
DIFF_DB = "diff_proj_db"

transformation = "Flatten" #"Virtualize" #"O0<->O3" #"N/A" # "EncodeArithmetic, Virtualize" #"EncodeArithmetic, Flatten" # # # #
model = "Palmtree_trp" #"Cirrina" #"Gemini_trp" # ##"Gemini"
inc_libraries = ["openssl"]
expr_num = "1"


