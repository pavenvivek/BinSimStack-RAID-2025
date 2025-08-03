
MONGO_CLIENT = "mongodb://localhost:27017/"

CODE_GRAPH_COLLECTION = "code_graph"
FUNC_INFO_COLLECTION = "func_info"
CODE_ENC_COLLECTION = "code_encodings"
DIFF_COLLECTION = "diff_graph"

FUNC_INFO_DB = "func_info_db"
TREES_DB = "angr_proj_db"
CODE_ENC_DB = "encodings_db"
DIFF_DB = "diff_proj_db"



transformation = "Flatten" #"O0<->O1" #"EncodeArithmetic, Flatten" #"Virtualize" #"O0<->O3" #"O0<->O3" #"N/A" # "EncodeArithmetic, Virtualize" #"EncodeArithmetic, Flatten" # # # #
model = "jTrans" #"Gemini_trp" #"Palmtree_trp" #"Cirrina"
model_db = model
inc_libraries = ["openssl"]
lib2 = "___openssl"
expr_num = "1"

#libraries
#"igraph" #"dbus" #"allegro" #"libxml2" #"libmicrohttpd" #"gsl" #"alsa" #"libmongoc" #"binutils" #"libtomcrypt" #"imagemagick" #"coreutils" #"redis" #"sqlite" #"curl" #"musl" #"openssl" ##  # zlib - compiled as part of binutils

# Experiment number info:

# openssl
# 1 -> Flt
# 2 -> Viz
# 3 -> Flt, EncArth
# 4 -> Viz, EncArth
# 5 -> O0 <-> O1
# 6 -> O0 <-> O2
# 7 -> O0 <-> O3

# redis
# 8 -> Flt
# 9 -> Viz
# 10 -> Flt, EncArth
# 11 -> Viz, EncArth
# 12 -> O0 <-> O1
# 13 -> O0 <-> O2
# 14 -> O0 <-> O3

# binutils
# 15 -> Flt
# 16 -> Viz
# 17 -> Flt, EncArth
# 18 -> Viz, EncArth
# 19 -> O0 <-> O1
# 20 -> O0 <-> O2
# 21 -> O0 <-> O3

# imagemagick
# 22 -> Flt
# 23 -> Viz
# 24 -> Flt, EncArth
# 25 -> Viz, EncArth
# 26 -> O0 <-> O1
# 27 -> O0 <-> O2
# 28 -> O0 <-> O3

# libxml2
# 29 -> Flt
# 30 -> Viz
# 31 -> Flt, EncArth
# 32 -> Viz, EncArth
# 33 -> O0 <-> O1
# 34 -> O0 <-> O2
# 35 -> O0 <-> O3

# scalablity and generalizability
# 36 -> flt
# 37 -> O0 <-> O1
# 38 -> viz # malw and vuln
# 39 -> viz
# 40 -> O0 <-> O2
# 41 -> O0 <-> O3

