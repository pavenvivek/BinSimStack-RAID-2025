# Malware and Vulnerability Analysis using Graph-Synchronized Language Model

This repository contains the implementation for the paper titled _Malware and Vulnerability Analysis using Graph-Synchronized Language Model_ that got accepted into the 28th International Symposium on Research in Attacks, Intrusions, and Defenses ([RAID 2025](https://raid2025.github.io/call.html)), Queensland, Australia (October 2025).

## Abstract
Malware and Vulnerability analysis are integral components of Cybersecurity. Malware authors employ code obfuscation techniques such as Control-flow flattening and Virtualization to escape detection from antivirus tools. Also, vulnerable code analysis gets complicated when the code is optimized using compiler flags. This paper proposes a binary function similarity detection (BFSD) framework that combines traditional Graph Neural Networks and relatively more recent Large Language Models using ensemble techniques to break code obfuscation and code optimization problems. The framework projects different facets of a binary program, such as control-flow graphs and assembly codes, into different Neural Network architectures, such as GNN and LLM, synchronizes their training process by maintaining the same testing and training data, and finally combines the predictions using ensemble techniques. The diverse features employed by the machine learning models expose unique subsets of functions, and the ensemble takes advantage of this. Experiments show state-of-the-art accuracy in breaking both code obfuscation and code optimization. We use two transformations for code obfuscation: _Control-flow Flattening_ and _Virtualization_. We use three compiler flags for code optimization: _O1_, _O2_, and _O3_. We also evaluate the robustness of the framework by cascading _Mixed-Boolean Arithmetic_ with _Flatten_.

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
For all the experiments, we used 80% of the dataset for training and 20% for testing. Also, the training and testing sets are kept disjoint for all experiments in the paper. We conducted 33 experiments spanning more than 250 models (including baselines). We conducted 25 experiments to analyze the five transformations _Flatten_, _Virtualize_, _O0 <-> O1_, _O0 <-> O2_, and _O0 <-> O3_ applied to five libraries: _OpenSSL_, _ImageMagick_, _Libxml2_, _Binutils_, and  _Redis_. We conducted five more experiments for scalability and generalizability analysis, and one experiment for malware and vulnerability analysis. We also conducted one experiment that assesses the robustness of the ensemble framework and another experiment that analyzes the relative performance of different ensemble methods. For the ensemble approach BinSimStack, we selected jTrans, Palmtree, Gemini-trp, and Cirrina to operate at layers L1, L2, L3, and L4, respectively. For all the experiments, BinSimStack combines the predictions of individually trained models using the weighted aggregate ensemble method. We conducted all the experiments on Ubuntu 22.04 LTS with 64 GB RAM, 32-core Intel i9-13900HX x86_64 CPU architecture, and 8 GB NVIDIA GeForce RTX 4070 GPU. We used Angr as the disassembler for all the tools in the ensemble. Please refer the main paper for more details. 

## Run Commands
The repository is built on top of [Gemini](https://github.com/xiaojunxu/dnn-binary-code-similarity) and [jTrans](https://github.com/vul337/jTrans). The folder "GNN-based" contains the code for building the graph-based models Cirrina, Gemini-trp, and Palmtree. The folder "LLM-based" contains the code for BERT-based jTrans. To run the codes inside LLM-based, first download "models.tar.gz" from jTrans [git-repo](https://github.com/vul337/jTrans) and extract the two files optimizer.pt and pytorch_model.bin and place it inside "LLM-based/models/jTrans-pretrain/".

For building the dataset, use the code inside angr_scripts. The following commands will run Tigress and build the necessary cfg and assembly contents and will also push it into mongodb.

'''
python3 generate_data.py
python3 generate_data_angr.py
'''

For code optimization flags, use the following commands.

'''
python3 generate_data_cmp_flags.py
python3 generate_data_angr_cmp_flags.py
'''

Note: You might have to do make build on the necessary C libraries before running the above commands. Also, set the parameters in params.py file according to your local configurations.


