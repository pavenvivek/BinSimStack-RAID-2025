# Malware and Vulnerability Analysis using Graph-Synchronized Language Model

This repository contains the implementation for the paper titled _Malware and Vulnerability Analysis using Graph-Synchronized Language Model_ that got accepted into the 28th International Symposium on Research in Attacks, Intrusions, and Defenses ([RAID 2025](https://raid2025.github.io/call.html)), Queensland, Australia (October 2025).

## Abstract
Malware and Vulnerability analysis are integral components of Cybersecurity. Malware authors employ code obfuscation techniques such as Control-flow flattening and Virtualization to escape detection from antivirus tools. Also, vulnerable code analysis gets complicated when the code is optimized using compiler flags. This paper proposes a binary function similarity detection (BFSD) framework that combines traditional Graph Neural Networks and relatively more recent Large Language Models using ensemble techniques to break code obfuscation and code optimization problems. The framework projects different facets of a binary program, such as control-flow graphs and assembly codes, into different Neural Network architectures, such as GNN and LLM, synchronizes their training process by maintaining the same testing and training data, and finally combines the predictions using ensemble techniques. The diverse features employed by the machine learning models expose unique subsets of functions, and the ensemble takes advantage of this. Experiments show state-of-the-art accuracy in breaking both code obfuscation and code optimization. We use two transformations for code obfuscation: _Control-flow Flattening_ and _Virtualization_. We use three compiler flags for code optimization: _O<sub>1</sub>_, _O<sub>2</sub>_, and _O<sub>3</sub>_. We also evaluate the robustness of the framework by cascading _Mixed-Boolean Arithmetic_ with _Flatten_.

## Dataset
Our dataset includes 17 libraries: 

  - OpenSSL 
  - ImageMagick 
  - Libxml2 
  - Binutils
  - Redis
  - SQLite
  - Curl 
  - Musl
  - Libmicrohttpd
  - LibTomCrypt
  - Coreutils
  - Alsa
  - Libmongoc
  - Dbus
  - Allegro
  - Igraph
  - Gsl 

We filtered the functions whose CFG had less than five basic blocks. This resulted in 43,048 unique C source code functions, which we used for our analysis.

## Experimental Setup and Details
For all the experiments, we used 80% of the dataset for training and 20% for testing. Also, the training and testing sets are kept disjoint for all experiments in the paper. We conducted 33 experiments in total spanning more than 250 models (including baselines) as follows.

  - 25 experiments to analyze the five transformations _Flatten_, _Virtualize_, _O<sub>0</sub> <-> O<sub>1</sub>_, _O<sub>0</sub> <-> O<sub>2</sub>_, and _O<sub>0</sub> <-> O<sub>3</sub>_ applied to five libraries: _OpenSSL_, _ImageMagick_, _Libxml2_, _Binutils_, and  _Redis_. 
  - 5 experiments for scalability and generalizability analysis.
  - 1 experiment for malware and vulnerability analysis.
  - 1 experiment to assesses the robustness of the ensemble framework. 
  - 1 experiment to analyze the relative performance of different ensemble methods.
  
For the ensemble approach BinSimStack, we selected jTrans, Palmtree, Gemini-trp, and Cirrina to operate at layers L1, L2, L3, and L4, respectively. For all the experiments, BinSimStack combines the predictions of individually trained models using the weighted aggregate ensemble method. We conducted all the experiments on Ubuntu 22.04 LTS with 64 GB RAM, 32-core Intel i9-13900HX x86_64 CPU architecture, and 8 GB NVIDIA GeForce RTX 4070 GPU. We used Angr as the disassembler for all the tools in the ensemble. Please refer the main paper for more details. 

## Run Commands
The repository is built on top of [Gemini](https://github.com/xiaojunxu/dnn-binary-code-similarity) and [jTrans](https://github.com/vul337/jTrans). The folder "GNN-based" contains the code for building the graph-based models Cirrina, Gemini-trp, and Palmtree. The folder "LLM-based" contains the code for BERT-based jTrans. To run the codes inside LLM-based, first download "models.tar.gz" from jTrans [git-repo](https://github.com/vul337/jTrans) and extract the two files optimizer.pt and pytorch_model.bin and place it inside "LLM-based/models/jTrans-pretrain/".

For building the dataset, use the code inside Preprocessing. The following commands will run Tigress/Compiler Optimization, build the necessary cfg and assembly contents, and push them into MongoDB.

For code obfuscation:
```
python3 generate_data.py
python3 generate_data_angr.py
```

For code optimization:

```
python3 generate_data_cmp_flags.py
python3 generate_data_angr_cmp_flags.py
```

Note: You might have to do make build on the necessary C libraries before running the above commands. Also, set the parameters in params.py file according to your local configurations.

Use the following commands for building the graph-based models.

For code obfuscation:

```
python3 build_data_angr.py
python3 train.py
python3 eval_single_mtx.py
```

For code optimization:

```
python3 build_data_angr_cflg.py
python3 train.py
python3 eval_single_mtx.py
```

Note: You have to set the correct parameters (such as transformation type - "Flatten" or "Virtualize") inside settings.py. Also, set the embedding size correctly in train.py (line 22). Also, utils.py (line 226 to 230) and utils_cflg.py (line 230 to 234) should be set correctly according to the transformation used. Repeat the above steps for all the three graph-based models: Cirrina, Gemini-trp, and Palmtree. 

"python3 eval_single_mtx.py" will insert the final cosine scores inside MongoDB for all the three graph-based models. 

Next, we should run the code inside LLM-based to build cosine scores for jTrans.

For code obfuscation:

```
python3 build_data_angr_from_db.py
python3 finetune_simple.py
python3 eval_save_simple.py
```

For code optimization:

```
python3 build_data_angr_from_db_cflg.py
python3 finetune_simple.py
python3 eval_save_simple.py
```

Note: Make sure to set the correct parameters in settings.py and data_simple.py (line 154 to 157 and 212 to 215) according to the transformation and library.

"python3 eval_save_simple.py" will insert the final cosine score for jTrans inside MongoDB. We used external files (like inc_fns_lst_flt_patch.py for OpenSSL and Flatten) to make sure the order of functions inside MongoDB is the same for all the 4 models. This is important for the ensemble to merge the scores correctly.

Once we have the cosine scores for all the models, we can run the following command to see the performance of the ensemble and contributions of the individual models.

```
python3 eval_cmb.py
```

Note: Set the weights for different models (refer to the paper) inside utils_top_stat.py (line 283 to 285).

## Sample output for OpenSSL and Flatten

```
BinSimStack:
top_1 -> 95.8029197080292
top_2 -> 98.35766423357664
top_3 -> 99.08759124087592
top_4 -> 99.81751824817519
top_5 -> 99.90875912408758
top_7 -> 99.90875912408758
top_10 -> 100.0
top_15 -> 100.0
top_20 -> 100.0
top_25 -> 100.0
top_50 -> 100.0
top_100 -> 100.0
MRR -> 97.53446877534472


Top-1
-------------------
Cirrina        : total -> 45.52919708029197, unique -> bef. aggregate: 0.18248175182481752, aft. aggregate: 0.0 
Gemini_triplet : total -> 71.44160583941606, unique -> bef. aggregate: 1.0036496350364963, aft. aggregate: 0.5474452554744526 
Palmtree       : total -> 73.35766423357664, unique -> bef. aggregate: 2.9197080291970803, aft. aggregate: 0.18248175182481752 
jTrans         : total -> 89.59854014598541, unique -> bef. aggregate: 6.021897810218978, aft. aggregate: 0.18248175182481752 
BinSimStack      : total -> 95.8029197080292, unique -> 0.8211678832116789 
Agg. missed    : total -> 0.9124087591240875 
Agg. retained  : total -> 94.98175182481752 
Model union    : total -> 95.8941605839416 

Top-1 info:
Model_wt_aggregate : 95.8029197080292 
Model_or_pred      : 97.62773722627736
Model_and_pred     : 30.83941605839416
Model_vote_pred    : 80.74817518248175
Model_best_pred    : 95.8029197080292
Model_wt_best_pred : 95.8029197080292
-------------------


Top-2
-------------------
Cirrina        : total -> 60.94890510948905, unique -> bef. aggregate: 0.18248175182481752, aft. aggregate: 0.09124087591240876 
Gemini_triplet : total -> 82.75547445255475, unique -> bef. aggregate: 0.36496350364963503, aft. aggregate: 0.18248175182481752 
Palmtree       : total -> 83.94160583941606, unique -> bef. aggregate: 1.0036496350364963, aft. aggregate: 0.0 
jTrans         : total -> 95.34671532846716, unique -> bef. aggregate: 3.2846715328467155, aft. aggregate: 0.2737226277372263 
BinSimStack      : total -> 98.35766423357664, unique -> 0.09124087591240876 
Agg. missed    : total -> 0.7299270072992701 
Agg. retained  : total -> 98.26642335766424 
Model union    : total -> 98.9963503649635 

Top-2 info:
Model_wt_aggregate : 98.35766423357664 
Model_or_pred      : 99.08759124087592
Model_and_pred     : 46.62408759124087
Model_vote_pred    : 89.87226277372264
Model_best_pred    : 98.35766423357664
Model_wt_best_pred : 98.35766423357664
-------------------


Top-3
-------------------
Cirrina        : total -> 69.52554744525547, unique -> bef. aggregate: 0.18248175182481752, aft. aggregate: 0.18248175182481752 
Gemini_triplet : total -> 87.5, unique -> bef. aggregate: 0.36496350364963503, aft. aggregate: 0.2737226277372263 
Palmtree       : total -> 87.86496350364963, unique -> bef. aggregate: 0.36496350364963503, aft. aggregate: 0.0 
jTrans         : total -> 97.99270072992701, unique -> bef. aggregate: 1.7335766423357664, aft. aggregate: 0.18248175182481752 
BinSimStack      : total -> 99.08759124087592, unique -> 0.0 
Agg. missed    : total -> 0.7299270072992701 
Agg. retained  : total -> 99.08759124087592 
Model union    : total -> 99.81751824817519 

Top-3 info:
Model_wt_aggregate : 99.08759124087592 
Model_or_pred      : 99.90875912408758
Model_and_pred     : 56.02189781021898
Model_vote_pred    : 92.7919708029197
Model_best_pred    : 99.17883211678831
Model_wt_best_pred : 99.17883211678831
-------------------


Top-4
-------------------
Cirrina        : total -> 75.72992700729927, unique -> bef. aggregate: 0.0, aft. aggregate: 0.0 
Gemini_triplet : total -> 91.42335766423358, unique -> bef. aggregate: 0.0, aft. aggregate: 0.0 
Palmtree       : total -> 90.32846715328468, unique -> bef. aggregate: 0.2737226277372263, aft. aggregate: 0.0 
jTrans         : total -> 98.8138686131387, unique -> bef. aggregate: 1.3686131386861315, aft. aggregate: 0.0 
BinSimStack      : total -> 99.81751824817519, unique -> 0.0 
Agg. missed    : total -> 0.0 
Agg. retained  : total -> 99.81751824817519 
Model union    : total -> 99.81751824817519 

Top-4 info:
Model_wt_aggregate : 99.81751824817519 
Model_or_pred      : 99.90875912408758
Model_and_pred     : 63.503649635036496
Model_vote_pred    : 95.62043795620438
Model_best_pred    : 99.72627737226277
Model_wt_best_pred : 99.72627737226277
-------------------


Top-5
-------------------
Cirrina        : total -> 80.47445255474453, unique -> bef. aggregate: 0.0, aft. aggregate: 0.0 
Gemini_triplet : total -> 93.33941605839416, unique -> bef. aggregate: 0.0, aft. aggregate: 0.0 
Palmtree       : total -> 92.06204379562044, unique -> bef. aggregate: 0.09124087591240876, aft. aggregate: 0.0 
jTrans         : total -> 99.81751824817519, unique -> bef. aggregate: 1.0036496350364963, aft. aggregate: 0.0 
BinSimStack      : total -> 99.90875912408758, unique -> 0.0 
Agg. missed    : total -> 0.0 
Agg. retained  : total -> 99.90875912408758 
Model union    : total -> 99.90875912408758 

Top-5 info:
Model_wt_aggregate : 99.90875912408758 
Model_or_pred      : 100.0
Model_and_pred     : 69.06934306569343
Model_vote_pred    : 97.53649635036497
Model_best_pred    : 99.90875912408758
Model_wt_best_pred : 99.90875912408758
-------------------


AUC Readings:
-------------------------------------
Cirrina  average accuracy after 1 runs: 99.64578542145786
Gemini   average accuracy after 1 runs: 99.84176582341766
Palmtree average accuracy after 1 runs: 99.75769089757691
jTrans   average accuracy after 1 runs: 99.98275172482751
Average accuracy combined (BinSimStack) after 1 runs: 99.99333399993334


```
