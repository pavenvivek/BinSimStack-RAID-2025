
MONGO_CLIENT = "mongodb://localhost:27017/"
NEO4J_CLIENT = "bolt://localhost:7687"
NEO4J_URI = "bolt://localhost:11003"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWD = "neo4j"

CODE_GRAPH_COLLECTION = "code_graph"
FUNC_INFO_COLLECTION = "func_info"
CODE_ENC_COLLECTION = "code_encodings"
DIFF_COLLECTION = "diff_graph"

FUNC_INFO_DB = "trees_db_func_info_flt"
#FUNC_INFO_DB = "trees_db_func_info_viz"
#FUNC_INFO_DB = "trees_db_func_info_cflgs"
#FUNC_INFO_DB = "trees_db_func_info_cflgs_O2"
#FUNC_INFO_DB = "trees_db_func_info_cflgs_O3"


TREES_DB = "angr_proj_db"
CODE_ENC_DB = "encodings_db"
DIFF_DB = "diff_proj_db"


#--
#TREES_DB = "trees_db_angr_flt_proj_final"
#TREES_DB = "trees_db_angr_viz_proj_final"
#TREES_DB = "trees_db_angr_malw_vuln_viz_proj_final"
#TREES_DB = "trees_db_angr_cflgs_proj_final"

#--

#--
#CODE_ENC_DB = "trees_db_encodings_flt"
#CODE_ENC_DB = "trees_db_encodings_viz"
#CODE_ENC_DB = "trees_db_encodings_malw_vuln_viz"

#CODE_ENC_DB = "trees_db_encodings_cflgs_O0_O1"
#CODE_ENC_DB = "trees_db_encodings_cflgs_O0_O2"
#CODE_ENC_DB = "trees_db_encodings_cflgs_O0_O3"

#--

#--
# test_data
#DIFF_DB = "trees_db_angr_diff_flt_proj_final"
#DIFF_DB = "trees_db_angr_diff_flt_proj_final_scale_500"
#DIFF_DB = "trees_db_angr_diff_flt_proj_final_scale_3500"
#DIFF_DB = "trees_db_angr_diff_flt_proj_final_scale_7500"

#DIFF_DB = "trees_db_angr_diff_viz_proj_final"
#DIFF_DB = "trees_db_angr_diff_viz_proj_final_scale_malw"
#DIFF_DB = "trees_db_angr_diff_viz_proj_final_scale_500"
#DIFF_DB = "trees_db_angr_diff_viz_proj_final_scale_3500"
#DIFF_DB = "trees_db_angr_diff_viz_proj_final_scale_7500"

#DIFF_DB = "trees_db_angr_diff_cflgs_proj_final_O0_O1"
#DIFF_DB = "trees_db_angr_diff_cflgs_proj_final_O0_O1_scale_500"
#DIFF_DB = "trees_db_angr_diff_cflgs_proj_final_O0_O1_scale_3500"
#DIFF_DB = "trees_db_angr_diff_cflgs_proj_final_O0_O1_scale_7500"

#DIFF_DB = "trees_db_angr_diff_cflgs_proj_final_O0_O2_scale_500"
#DIFF_DB = "trees_db_angr_diff_cflgs_proj_final_O0_O2_scale_3500"
#DIFF_DB = "trees_db_angr_diff_cflgs_proj_final_O0_O2_scale_7500"

#DIFF_DB = "trees_db_angr_diff_cflgs_proj_final_O0_O3_scale_500"
#DIFF_DB = "trees_db_angr_diff_cflgs_proj_final_O0_O3_scale_2500"
#DIFF_DB = "trees_db_angr_diff_cflgs_proj_final_O0_O3_scale_5000"
#DIFF_DB = ""

#DIFF_DB = "trees_db_angr_diff_cflgs_proj_final_O0_O2"
#DIFF_DB = "trees_db_angr_diff_cflgs_proj_final_O0_O3"

# validation_data
#DIFF_DB = "trees_db_angr_diff_flt_proj_final_vald"
#DIFF_DB = "trees_db_angr_diff_viz_proj_final_vald"
#--

transformation = "Flatten" #"Virtualize" #"O0<->O3" #"N/A" # "EncodeArithmetic, Virtualize" #"EncodeArithmetic, Flatten" # # # #
model = "Palmtree_trp" #"Cirrina" #"Gemini_trp" # ##"Gemini"
inc_libraries = ["openssl"]
#inc_libraries = ["igraph", "dbus", "allegro", "libmicrohttpd", "gsl", "alsa", "libmongoc", "libtomcrypt", "coreutils", "sqlite", "curl", "musl", "scale-O0-O3-5000"] #"scale-viz-7500"] #, "scale-flt-7500"] # #"scale-malw-viz-7500"
#inc_libraries = ["igraph", "dbus", "allegro", "libxml2", "libmicrohttpd", "gsl", "alsa", "libmongoc", "binutils", "libtomcrypt", "imagemagick", "coreutils", "redis", "sqlite", "curl", "musl", "openssl", "scale-malw-viz-7500"]
expr_num = "1" #"38-scale-malw-viz-7500" #"36-scale-500" # "41-scale-O0-O3-500" #"39-scale-viz-7500" #"37-scale-500" #"37" #

#libraries
#"igraph" #"dbus" #"allegro" #"libxml2" #"libmicrohttpd" #"gsl" #"alsa" #"libmongoc" #"binutils" #"libtomcrypt" #"imagemagick" #"coreutils" #"redis" #"sqlite" #"curl" #"musl" #"openssl" ##  # zlib - compiled as part of binutils

# Exclude the following functions from training to avoid clash during evaluation
vuln_fns = ["siv_cipher", "DH_check", "ossl_a2ulabel", "dtls1_process_heartbeat", "tls1_process_heartbeat", "BN_mod_sqrt", "dsa_sign_setup", "EC_GROUP_set_generator", "ssl3_get_message", "ssl_verify_cert_chain", "X509_issuer_and_serial_hash", "Curl_getworkingpath", "file_connect", "ftp_setup_connection", "Curl_http_method", "imap_done", "rtsp_do", "check_telnet_options", "Curl_follow", "Curl_resolv_timeout", "curl_share_setopt", "rsa_ossl_private_decrypt", "ossl_ffc_validate_public_key_partial", "BIO_new_NDEF", "ossl_policy_level_add_node", "PEM_read_bio_ex", "ossl_pkcs7_resolve_libctx", "pkcs7_bio_add_digest", "check_policy"]

malw_fns = ['malware_Jackshell_analyz', 'malware_Moses_DoPrivmsg', 'malware_Buhtrap_BcCreateSession', 'malware_Gozi_DllMain', 'malware_Minipig_Infects', 'malware_Xorusb_main', 'malware_Sdbot_WinMain', 'malware_Dexter_Infect', 'malware_MyDoom_msg_b64enc', 'malware_MooBot_bot_event', 'malware_RedMenshenBPFDoor_packet_loop', 'malware_TrojoDaemon_main', 'malware_LizardSquad_epollEventLoop', 'malware_Jynxkit_backconnect', 'malware_Beurk_drop_pty_connection', 'malware_Rrs_foreachline', 'malware_Op_main', 'malware_Lyceum_main', 'malware_Pop3d_fld_release', 'malware_Mushroom_main', 'malware_Hidepak_main', 'malware_Phantasmagoria_main', 'malware_Invader_main', 'malware_Dataseg_infect_me_baby', 'malware_ApacheBd_main', 'malware_Kaiten_tsunami', 'malware_Reptile_runshell', 'malware_Mirai_attack_start']

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

