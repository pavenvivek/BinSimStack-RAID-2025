from transformers import BertTokenizer, BertForMaskedLM, BertModel
from tokenizer import *
import pickle
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from data_simple import help_tokenize, load_paired_data,FunctionDataset_CL,FunctionDataset_CL_top
from transformers import AdamW
import torch.nn.functional as F
import argparse
import wandb
import logging
import sys
import time, subprocess
import data_simple
from insert_db import *
from settings import *
WANDB = False

def get_logger(name):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=name)
    logger = logging.getLogger(__name__)
    s_handle = logging.StreamHandler(sys.stdout)
    s_handle.setLevel(logging.INFO)
    s_handle.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
    logger.addHandler(s_handle)
    return logger

def eval(model, args, valid_set, logger):

    if WANDB:
        wandb.init(project=f'jTrans-finetune')
        wandb.config.update(args)

    print ("\nenters here ..........\n")
    
    logger.info("Initializing Model...")
    device = torch.device("cuda")
    model.to(device)
    logger.info("Finished Initialization...")
    valid_dataloader = DataLoader(valid_set, batch_size=50, num_workers=0, shuffle=False) #args.eval_batch_size
    global_steps = 0
    etc=0
    logger.info("Doing Evaluation ...")
    mrr = finetune_eval(model, valid_dataloader)
    logger.info("Evaluate: mrr={mrr}")
    if WANDB:
        wandb.log({
                    'mrr': mrr
                })

def finetune_eval(net, data_loader):
    net.eval()
    print(net)
    with torch.no_grad():
        avg=[]
        gt=[]
        cons=[]
        Recall_AT_1=[]
        Recall_AT_2=[]
        Recall_AT_3=[]
        Recall_AT_4=[]
        Recall_AT_5=[]
        Recall_AT_10=[]
        eval_iterator = tqdm(data_loader)

        anchor_all, pos_all, emb_all_pos, emb_all_neg = [], [], [], []
        for i, (seq1,seq2,seq3,mask1,mask2,mask3,emb_pos,emb_neg) in enumerate(eval_iterator):
                input_ids1, attention_mask1= seq1.cuda(),mask1.cuda()
                input_ids2, attention_mask2= seq2.cuda(),mask2.cuda()

                print(f"input_ids1.shape -> {input_ids1.shape}")
                print(attention_mask1.shape)
                print(f"input_ids2.shape -> {input_ids2.shape}")
                print(attention_mask2.shape)

                anchor,pos=0,0

                output=net(input_ids=input_ids1,attention_mask=attention_mask1)
                anchor=output.pooler_output

                output=net(input_ids=input_ids2,attention_mask=attention_mask2)
                pos=output.pooler_output

                print (f"anchor size -> {anchor.shape}")
                print (f"pos size -> {pos.shape}")

                anchor_all.append(anchor)
                pos_all.append(pos)
                emb_all_pos = emb_all_pos + list(emb_pos)
                #emb_all_neg.append(emb_neg)
                
        print (f"\nlen anchor_all -> {len(anchor_all)}\n")
        print (f"\nlen pos_all -> {len(pos_all)}\n")
        print (f"\nemb_pos -> {emb_all_pos}\n")

        anchor = torch.cat(anchor_all, 0)
        pos = torch.cat(pos_all, 0)
        
        print (f"cat anchor size -> {anchor.shape}")
        print (f"cat pos size -> {pos.shape}")

        #for anchor, pos in zip(anchor_all, pos_all):

        uuid = subprocess.getoutput("uuidgen --random")
        z = 0
        top_1 = 0
        top_2 = 0
        top_3 = 0
        top_4 = 0
        top_5 = 0
        top_7 = 0
        top_10 = 0
        top_15 = 0
        top_all = 0
        ans=0
        for k in range(len(anchor)):    # check every vector of (vA,vB)
            info_lst = []
            vA=anchor[k:k+1].cpu()
            sim=[]
            diff_lst=[]
            for j in range(len(pos)):
                vB=pos[j:j+1].cpu()
                #vB=vB[0]
                AB_sim=F.cosine_similarity(vA, vB).item()
                sim.append(AB_sim)
                diff_lst.append(-1*AB_sim) # calculating cosine distance
                
                if j!=k:
                    cons.append(AB_sim)

            sim=np.array(sim)
            y = np.argsort(-sim)

            #print (f"sim -> {sim}, y -> {y}, k -> {k}")

            posi=0
            for j in range(len(pos)):
                if y[j]==k:
                    posi=j+1
                    #break

            pos_dis = diff_lst[k]
            del diff_lst[k]
            pos_dis = [pos_dis] * len(diff_lst)
            y_lst = [1] * len(diff_lst)

            #print (f"* diff_lst -> {diff_lst}")
            #print (f"* pos_dis -> {pos_dis}")

            info_lst.append((k, emb_all_pos[k] + lib2, model_db, transformation, inc_libraries, pos_dis, diff_lst, y_lst, uuid))
            #info_lst.append((k, emb_all_pos[k], model_db, transformation, inc_libraries, pos_dis, diff_lst, y_lst, uuid)) # for scalability expr only
            print (f"k -> {info_lst[0][0]}, f -> {info_lst[0][1]}")
            insert_db_mongo(info_lst)

            cnt = 0
            for val in range(0, len(diff_lst)):
                if ((-1 * pos_dis[val]) <= (-1 * diff_lst[val])):
                    cnt = cnt + 1

            if cnt == 0:
                top_1 = top_1 + 1
                top_2 = top_2 + 1
                top_3 = top_3 + 1
                top_4 = top_4 + 1
                top_5 = top_5 + 1
                top_7 = top_7 + 1
                top_10 = top_10 + 1
                top_15 = top_15 + 1
                top_all = top_all + 1
            elif cnt == 1:
                top_2 = top_2 + 1
                top_3 = top_3 + 1
                top_4 = top_4 + 1
                top_5 = top_5 + 1
                top_7 = top_7 + 1
                top_10 = top_10 + 1
                top_15 = top_15 + 1
                top_all = top_all + 1
            elif cnt == 2:
                top_3 = top_3 + 1
                top_4 = top_4 + 1
                top_5 = top_5 + 1
                top_7 = top_7 + 1
                top_10 = top_10 + 1
                top_15 = top_15 + 1
                top_all = top_all + 1
            elif cnt == 3:
                top_4 = top_4 + 1
                top_5 = top_5 + 1
                top_7 = top_7 + 1
                top_10 = top_10 + 1
                top_15 = top_15 + 1
                top_all = top_all + 1
            elif cnt == 4:
                top_5 = top_5 + 1
                top_7 = top_7 + 1
                top_10 = top_10 + 1
                top_15 = top_15 + 1
                top_all = top_all + 1
            elif cnt >= 4 and cnt < 7:
                top_7 = top_7 + 1
                top_10 = top_10 + 1
                top_15 = top_15 + 1
                top_all = top_all + 1
            elif cnt >= 7 and cnt < 10:
                top_10 = top_10 + 1
                top_15 = top_15 + 1
                top_all = top_all + 1
            elif cnt >= 10 and cnt < 15:
                top_15 = top_15 + 1
                top_all = top_all + 1
            else:
                top_all = top_all + 1
                
            

            gt.append(sim[k])

            ans+=1/posi

            z = z + 1
        print (f"\n\nz -> {z}\n\n")        

        ans=ans/len(anchor)
        avg.append(ans)
        print("now mrr ",np.mean(np.array(avg)))

        fi=open("logft.txt","a")
        print("MRR ",np.mean(np.array(avg)),file=fi)
        print("FINAL MRR ",np.mean(np.array(avg)))
        fi.close()

        print ("\n\n<------- Top-k values -------->\n")
        print (f"top_1 -> {(top_1/z) * 100}")
        print (f"top_2 -> {(top_2/z) * 100}")
        print (f"top_3 -> {(top_3/z) * 100}")
        print (f"top_4 -> {(top_4/z) * 100}")
        print (f"top_5 -> {(top_5/z) * 100}")
        print (f"top_7 -> {(top_7/z) * 100}")
        print (f"top_10 -> {(top_10/z) * 100}")        
        print (f"top_15 -> {(top_15/z) * 100}")        
        print (f"top_all -> {(top_all/z) * 100}")        
        print ("\n--------------------------\n\n")
        
        return np.mean(np.array(avg))


class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings.position_embeddings=self.embeddings.word_embeddings
from datautils.playdata import DatasetBase as DatasetBase

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="jTrans-EvalSave")
    parser.add_argument("--model_path", type=str, default='./models/jTrans-finetune', help="Path to the model")
    parser.add_argument("--dataset_path", type=str, default='./data/test_data.json', help="Path to the dataset")
    parser.add_argument("--experiment_path", type=str, default='./experiments/BinaryCorp-3M/jTrans.pkl', help="Path to the experiment")
    parser.add_argument("--tokenizer", type=str, default='./jtrans_tokenizer/')

    args = parser.parse_args()

    from datetime import datetime
    now = datetime.now() # current date and time
    TIMESTAMP="%Y%m%d%H%M"
    tim = now.strftime(TIMESTAMP)
    logger = get_logger(f"logfile.txt") #jTrans-{args.model_path}-eval-{args.dataset_path}_savename_{args.experiment_path}_{tim}")
    logger.info("Loading Pretrained Model from {args.model_path} ...")
    model = BinBertModel.from_pretrained(args.model_path)

    model.eval()
    device = torch.device("cuda")
    model.to(device)

    logger.info("Done ...")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    logger.info("Tokenizer Done ...")
   
    logger.info("Preparing Datasets ...")
    ft_valid_dataset=FunctionDataset_CL_top(tokenizer,args.dataset_path,None,True,opt=None, add_ebd=True, convert_jump_addr=True) #, 'O2', 'O3', 'Os'
    # k = 0
    # for i in tqdm(range(len(ft_valid_dataset.datas))):
    #     pairs=ft_valid_dataset.datas[i]
    #     for j in ['O0','O1']: #,'O2','O3','Os']:
    #         if ft_valid_dataset.ebds[i].get(j) is not None:
    #             idx=ft_valid_dataset.ebds[i][j]
    #             #print (f"{k} ft_valid_dataset.ebds -> {ft_valid_dataset.ebds}")
    #             print (f"{idx} pairs[idx] -> {pairs[idx]}\npairs -> {pairs}\n\n")
    #             ret1=tokenizer([pairs[idx]], add_special_tokens=True,max_length=512,padding='max_length',truncation=True,return_tensors='pt') #tokenize them
    #             seq1=ret1['input_ids']
    #             mask1=ret1['attention_mask']
    #             input_ids1, attention_mask1= seq1.cuda(),mask1.cuda()
    #             output=model(input_ids=input_ids1,attention_mask=attention_mask1)
    #             anchor=output.pooler_output
    #             ft_valid_dataset.ebds[i][j]=anchor.detach().cpu()
    #             k = k + 1

    #sys.exit(-1)
                
    eval(model, args, ft_valid_dataset, logger)            
    #logger.info("ebds start writing")
    #fi=open(args.experiment_path,'wb')
    #pickle.dump(ft_valid_dataset.ebds,fi)
    #fi.close()

