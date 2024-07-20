import numpy as np
from sklearn.metrics import auc, roc_curve
from siamese_nonuplet import graphnn
import json
from settings import *


def get_auc_epoch_combined_mtx(data_lst, mode=None):
    tot_diff = []
    tot_truth = []

    acc = 0
    acc_cnt = 0
    acc_flip = 0


    acc_1 = 0
    acc_2 = 0
    acc_3 = 0
    acc_4 = 0
    acc_cnt_1 = 0
    acc_cnt_2 = 0
    acc_cnt_3 = 0

    acc_1_flip = 0
    acc_2_flip = 0

    acc_pm_1 = 0
    acc_pm_2 = 0
    acc_pm = 0
    
    acc_cmb_1 = 0
    acc_cmb_2 = 0
    acc_cmb_3 = 0
    acc_cmb_4 = 0
    acc_cmb = 0

    false_pred_cnt = 0
    top_1 = 0
    top_2 = 0
    top_3 = 0
    top_4 = 0
    top_5 = 0
    top_7 = 0
    top_10 = 0
    top_15 = 0
    top_20 = 0
    top_25 = 0
    top_50 = 0
    top_100 = 0

    mdl1_pred_top_1 = []
    mdl1_pred_top_2 = []
    mdl1_pred_top_3 = []
    mdl1_pred_top_4 = []
    mdl1_pred_top_5 = []

    mdl2_pred_top_1 = []
    mdl2_pred_top_2 = []
    mdl2_pred_top_3 = []
    mdl2_pred_top_4 = []
    mdl2_pred_top_5 = []

    mdl3_pred_top_1 = []
    mdl3_pred_top_2 = []
    mdl3_pred_top_3 = []
    mdl3_pred_top_4 = []
    mdl3_pred_top_5 = []

    agg_pred_top_1 = []
    agg_pred_top_2 = []
    agg_pred_top_3 = []
    agg_pred_top_4 = []
    agg_pred_top_5 = []

    wt_agg_pred_top_1 = []
    wt_agg_pred_top_2 = []
    wt_agg_pred_top_3 = []
    wt_agg_pred_top_4 = []
    wt_agg_pred_top_5 = []
    
    mdl_or_pred_avg_top_1 = 0
    mdl_or_pred_avg_top_2 = 0
    mdl_or_pred_avg_top_3 = 0
    mdl_or_pred_avg_top_4 = 0
    mdl_or_pred_avg_top_5 = 0

    mdl_and_pred_avg_top_1 = 0
    mdl_and_pred_avg_top_2 = 0
    mdl_and_pred_avg_top_3 = 0
    mdl_and_pred_avg_top_4 = 0
    mdl_and_pred_avg_top_5 = 0

    mdl_vote_pred_avg_top_1 = 0
    mdl_vote_pred_avg_top_2 = 0
    mdl_vote_pred_avg_top_3 = 0
    mdl_vote_pred_avg_top_4 = 0
    mdl_vote_pred_avg_top_5 = 0

    mdl_best_pred_avg_top_1 = 0
    mdl_best_pred_avg_top_2 = 0
    mdl_best_pred_avg_top_3 = 0
    mdl_best_pred_avg_top_4 = 0
    mdl_best_pred_avg_top_5 = 0

    mdl_wt_best_pred_avg_top_1 = 0
    mdl_wt_best_pred_avg_top_2 = 0
    mdl_wt_best_pred_avg_top_3 = 0
    mdl_wt_best_pred_avg_top_4 = 0
    mdl_wt_best_pred_avg_top_5 = 0
    
    fnc_cnt = 0
    for i, func, diff1, diff2, diff3, diff4, diff5, diff6, y1, y2, y3 in data_lst: 
        
        false_pred_cnt_lcl = 0
        

        y1 = np.array(y1)
        diff1 = np.array(diff1)
        diff2 = np.array(diff2)

        y2 = np.array(y2)
        diff3 = np.array(diff3)
        diff4 = np.array(diff4)

        y3 = np.array(y3)
        diff5 = np.array(diff5)
        diff6 = np.array(diff6)

        #print (f"diff1 -> {diff1}")
        #print (f"diff2 -> {diff2}")

        #break
    
        tot_diff += list(diff1)
        tot_truth += list(y1 > 0)


        mdl1_fls_pred_cnt = 0
        mdl2_fls_pred_cnt = 0
        mdl3_fls_pred_cnt = 0
        agg_fls_pred_cnt = 0
        wt_agg_fls_pred_cnt = 0

        mdl_or_pred = 0
        mdl_and_pred = 0        
        mdl_vote_pred = 0        
        mdl_best_pred = 0        
        mdl_wt_best_pred = 0        
        mdl_pred_cnt = 0        
        pred1 = False
        pred2 = False         
        pred3 = False
        
        hf = len(diff1) #int(len(diff1)/2)

        for j in range(0, hf):
            if (-1 * diff1[j]) > (-1 * diff2[j]):
                acc_1 = acc_1 + 1
                pred1 = True
            else:
                mdl1_fls_pred_cnt = mdl1_fls_pred_cnt + 1

            if (-1 * diff3[j]) > (-1 * diff4[j]):
                acc_2 = acc_2 + 1
                pred2 = True
            else:
                mdl2_fls_pred_cnt = mdl2_fls_pred_cnt + 1

            if (-1 * diff5[j]) > (-1 * diff6[j]):
                acc_3 = acc_3 + 1
                pred3 = True
            else:
                mdl3_fls_pred_cnt = mdl3_fls_pred_cnt + 1

                
            acc_cnt_1 = acc_cnt_1 + 1
            acc_cnt_2 = acc_cnt_2 + 1
            acc_cnt_3 = acc_cnt_3 + 1


            dif_val_1 = abs((-1 * diff1[j]) - (-1 * diff2[j]))
            dif_val_2 = abs((-1 * diff3[j]) - (-1 * diff4[j]))
            dif_val_3 = abs((-1 * diff5[j]) - (-1 * diff6[j]))
                
            if dif_val_1 == max(dif_val_1, dif_val_2, dif_val_3):
                if (-1 * diff1[j]) > (-1 * diff2[j]):
                    mdl_best_pred = mdl_best_pred + 1
            elif dif_val_2 == max(dif_val_1, dif_val_2, dif_val_3):
                if (-1 * diff3[j]) > (-1 * diff4[j]):
                    mdl_best_pred = mdl_best_pred + 1
            elif dif_val_3 == max(dif_val_1, dif_val_2, dif_val_3):
                if (-1 * diff5[j]) > (-1 * diff6[j]):
                    mdl_best_pred = mdl_best_pred + 1

            # model weights for viz
            #w1 = 0.32 #0.35 #0.32
            #w2 = 0.48 #0.47 #0.48
            #w3 = 0.70 #0.76 #0.70

            # model weights for flt
            w1 = 0.38 #0.54 #0.52
            w2 = 0.33 #0.69 #0.78
            w3 = 0.29 #0.77 #0.83
            
            if (w1 * dif_val_1) == max((w1 * dif_val_1), (w2 * dif_val_2), (w3 * dif_val_3)):
                if (-1 * diff1[j]) > (-1 * diff2[j]):
                    mdl_wt_best_pred = mdl_wt_best_pred + 1
            elif (w2 * dif_val_2) == max((w1 * dif_val_1), (w2 * dif_val_2), (w3 * dif_val_3)):
                if (-1 * diff3[j]) > (-1 * diff4[j]):
                    mdl_wt_best_pred = mdl_wt_best_pred + 1
            elif (w3 * dif_val_3) == max((w1 * dif_val_1), (w2 * dif_val_2), (w3 * dif_val_3)):
                if (-1 * diff5[j]) > (-1 * diff6[j]):
                    mdl_wt_best_pred = mdl_wt_best_pred + 1

                    
            if pred1 == True or pred2 == True or pred3 == True:
                mdl_or_pred = mdl_or_pred + 1

            if pred1 == True and pred2 == True and pred3 == True:
                mdl_and_pred = mdl_and_pred + 1

            lst = [pred1, pred2, pred3]
            if lst.count(True) >= 2:
                mdl_vote_pred = mdl_vote_pred + 1
                
            mdl_pred_cnt = mdl_pred_cnt + 1
                
            pred1 = False
            pred2 = False         
            pred3 = False

            # weighted aggregate
            if ((w3 * -1 * diff5[j]) + (w2 * -1 * diff3[j]) + (w1 * -1 * diff1[j])) > ((w3 * -1 * diff6[j]) + (w2 * -1 * diff4[j]) + (w1 * -1 * diff2[j])): #(top-1 weights)
                #acc_cmb_2 = acc_cmb_2 + 1
                pass
            else:
                wt_agg_fls_pred_cnt = wt_agg_fls_pred_cnt + 1


            # no weights    
            if ((1 * -1 * diff5[j]) + (1 * -1 * diff3[j]) + (1 * -1 * diff1[j])) > ((1 * -1 * diff6[j]) + (1 * -1 * diff4[j]) + (1 * -1 * diff2[j])):
                acc_cmb_2 = acc_cmb_2 + 1
            else:
                false_pred_cnt = false_pred_cnt + 1
                false_pred_cnt_lcl = false_pred_cnt_lcl + 1
                agg_fls_pred_cnt = agg_fls_pred_cnt + 1


                
            acc_cmb = acc_cmb + 1

        if mdl_or_pred == mdl_pred_cnt: 
            mdl_or_pred_avg_top_1 = mdl_or_pred_avg_top_1 + 1
            mdl_or_pred_avg_top_2 = mdl_or_pred_avg_top_2 + 1
            mdl_or_pred_avg_top_3 = mdl_or_pred_avg_top_3 + 1
            mdl_or_pred_avg_top_4 = mdl_or_pred_avg_top_4 + 1
            mdl_or_pred_avg_top_5 = mdl_or_pred_avg_top_5 + 1
        elif mdl_or_pred == mdl_pred_cnt-1: 
            mdl_or_pred_avg_top_2 = mdl_or_pred_avg_top_2 + 1
            mdl_or_pred_avg_top_3 = mdl_or_pred_avg_top_3 + 1
            mdl_or_pred_avg_top_4 = mdl_or_pred_avg_top_4 + 1
            mdl_or_pred_avg_top_5 = mdl_or_pred_avg_top_5 + 1
        elif mdl_or_pred == mdl_pred_cnt-2: 
            mdl_or_pred_avg_top_3 = mdl_or_pred_avg_top_3 + 1
            mdl_or_pred_avg_top_4 = mdl_or_pred_avg_top_4 + 1
            mdl_or_pred_avg_top_5 = mdl_or_pred_avg_top_5 + 1
        elif mdl_or_pred == mdl_pred_cnt-3: 
            mdl_or_pred_avg_top_4 = mdl_or_pred_avg_top_4 + 1
            mdl_or_pred_avg_top_5 = mdl_or_pred_avg_top_5 + 1
        elif mdl_or_pred == mdl_pred_cnt-4: 
            mdl_or_pred_avg_top_5 = mdl_or_pred_avg_top_5 + 1
        

        if mdl_and_pred == mdl_pred_cnt: 
            mdl_and_pred_avg_top_1 = mdl_and_pred_avg_top_1 + 1
            mdl_and_pred_avg_top_2 = mdl_and_pred_avg_top_2 + 1
            mdl_and_pred_avg_top_3 = mdl_and_pred_avg_top_3 + 1
            mdl_and_pred_avg_top_4 = mdl_and_pred_avg_top_4 + 1
            mdl_and_pred_avg_top_5 = mdl_and_pred_avg_top_5 + 1
        elif mdl_and_pred == mdl_pred_cnt-1: 
            mdl_and_pred_avg_top_2 = mdl_and_pred_avg_top_2 + 1
            mdl_and_pred_avg_top_3 = mdl_and_pred_avg_top_3 + 1
            mdl_and_pred_avg_top_4 = mdl_and_pred_avg_top_4 + 1
            mdl_and_pred_avg_top_5 = mdl_and_pred_avg_top_5 + 1
        elif mdl_and_pred == mdl_pred_cnt-2: 
            mdl_and_pred_avg_top_3 = mdl_and_pred_avg_top_3 + 1
            mdl_and_pred_avg_top_4 = mdl_and_pred_avg_top_4 + 1
            mdl_and_pred_avg_top_5 = mdl_and_pred_avg_top_5 + 1
        elif mdl_and_pred == mdl_pred_cnt-3: 
            mdl_and_pred_avg_top_4 = mdl_and_pred_avg_top_4 + 1
            mdl_and_pred_avg_top_5 = mdl_and_pred_avg_top_5 + 1
        elif mdl_and_pred == mdl_pred_cnt-4: 
            mdl_and_pred_avg_top_5 = mdl_and_pred_avg_top_5 + 1

        if mdl_vote_pred == mdl_pred_cnt: 
            mdl_vote_pred_avg_top_1 = mdl_vote_pred_avg_top_1 + 1
            mdl_vote_pred_avg_top_2 = mdl_vote_pred_avg_top_2 + 1
            mdl_vote_pred_avg_top_3 = mdl_vote_pred_avg_top_3 + 1
            mdl_vote_pred_avg_top_4 = mdl_vote_pred_avg_top_4 + 1
            mdl_vote_pred_avg_top_5 = mdl_vote_pred_avg_top_5 + 1
        elif mdl_vote_pred == mdl_pred_cnt-1: 
            mdl_vote_pred_avg_top_2 = mdl_vote_pred_avg_top_2 + 1
            mdl_vote_pred_avg_top_3 = mdl_vote_pred_avg_top_3 + 1
            mdl_vote_pred_avg_top_4 = mdl_vote_pred_avg_top_4 + 1
            mdl_vote_pred_avg_top_5 = mdl_vote_pred_avg_top_5 + 1
        elif mdl_vote_pred == mdl_pred_cnt-2: 
            mdl_vote_pred_avg_top_3 = mdl_vote_pred_avg_top_3 + 1
            mdl_vote_pred_avg_top_4 = mdl_vote_pred_avg_top_4 + 1
            mdl_vote_pred_avg_top_5 = mdl_vote_pred_avg_top_5 + 1
        elif mdl_vote_pred == mdl_pred_cnt-3: 
            mdl_vote_pred_avg_top_4 = mdl_vote_pred_avg_top_4 + 1
            mdl_vote_pred_avg_top_5 = mdl_vote_pred_avg_top_5 + 1
        elif mdl_vote_pred == mdl_pred_cnt-4: 
            mdl_vote_pred_avg_top_5 = mdl_vote_pred_avg_top_5 + 1

        if mdl_best_pred == mdl_pred_cnt: 
            mdl_best_pred_avg_top_1 = mdl_best_pred_avg_top_1 + 1
            mdl_best_pred_avg_top_2 = mdl_best_pred_avg_top_2 + 1
            mdl_best_pred_avg_top_3 = mdl_best_pred_avg_top_3 + 1
            mdl_best_pred_avg_top_4 = mdl_best_pred_avg_top_4 + 1
            mdl_best_pred_avg_top_5 = mdl_best_pred_avg_top_5 + 1
        elif mdl_best_pred == mdl_pred_cnt-1: 
            mdl_best_pred_avg_top_2 = mdl_best_pred_avg_top_2 + 1
            mdl_best_pred_avg_top_3 = mdl_best_pred_avg_top_3 + 1
            mdl_best_pred_avg_top_4 = mdl_best_pred_avg_top_4 + 1
            mdl_best_pred_avg_top_5 = mdl_best_pred_avg_top_5 + 1
        elif mdl_best_pred == mdl_pred_cnt-2: 
            mdl_best_pred_avg_top_3 = mdl_best_pred_avg_top_3 + 1
            mdl_best_pred_avg_top_4 = mdl_best_pred_avg_top_4 + 1
            mdl_best_pred_avg_top_5 = mdl_best_pred_avg_top_5 + 1
        elif mdl_best_pred == mdl_pred_cnt-3: 
            mdl_best_pred_avg_top_4 = mdl_best_pred_avg_top_4 + 1
            mdl_best_pred_avg_top_5 = mdl_best_pred_avg_top_5 + 1
        elif mdl_best_pred == mdl_pred_cnt-4: 
            mdl_best_pred_avg_top_5 = mdl_best_pred_avg_top_5 + 1
            
        if mdl_wt_best_pred == mdl_pred_cnt: 
            mdl_wt_best_pred_avg_top_1 = mdl_wt_best_pred_avg_top_1 + 1
            mdl_wt_best_pred_avg_top_2 = mdl_wt_best_pred_avg_top_2 + 1
            mdl_wt_best_pred_avg_top_3 = mdl_wt_best_pred_avg_top_3 + 1
            mdl_wt_best_pred_avg_top_4 = mdl_wt_best_pred_avg_top_4 + 1
            mdl_wt_best_pred_avg_top_5 = mdl_wt_best_pred_avg_top_5 + 1
        elif mdl_wt_best_pred == mdl_pred_cnt-1: 
            mdl_wt_best_pred_avg_top_2 = mdl_wt_best_pred_avg_top_2 + 1
            mdl_wt_best_pred_avg_top_3 = mdl_wt_best_pred_avg_top_3 + 1
            mdl_wt_best_pred_avg_top_4 = mdl_wt_best_pred_avg_top_4 + 1
            mdl_wt_best_pred_avg_top_5 = mdl_wt_best_pred_avg_top_5 + 1
        elif mdl_wt_best_pred == mdl_pred_cnt-2: 
            mdl_wt_best_pred_avg_top_3 = mdl_wt_best_pred_avg_top_3 + 1
            mdl_wt_best_pred_avg_top_4 = mdl_wt_best_pred_avg_top_4 + 1
            mdl_wt_best_pred_avg_top_5 = mdl_wt_best_pred_avg_top_5 + 1
        elif mdl_wt_best_pred == mdl_pred_cnt-3: 
            mdl_wt_best_pred_avg_top_4 = mdl_wt_best_pred_avg_top_4 + 1
            mdl_wt_best_pred_avg_top_5 = mdl_wt_best_pred_avg_top_5 + 1
        elif mdl_wt_best_pred == mdl_pred_cnt-4: 
            mdl_wt_best_pred_avg_top_5 = mdl_wt_best_pred_avg_top_5 + 1

        if false_pred_cnt_lcl == 0:
          top_1 = top_1 + 1
          top_2 = top_2 + 1
          top_3 = top_3 + 1
          top_4 = top_4 + 1
          top_5 = top_5 + 1
          top_7 = top_7 + 1
          top_10 = top_10 + 1
          top_15 = top_15 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl == 1:
          top_2 = top_2 + 1
          top_3 = top_3 + 1
          top_4 = top_4 + 1
          top_5 = top_5 + 1
          top_7 = top_7 + 1
          top_10 = top_10 + 1
          top_15 = top_15 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl == 2:
          top_3 = top_3 + 1
          top_4 = top_4 + 1
          top_5 = top_5 + 1
          top_7 = top_7 + 1
          top_10 = top_10 + 1
          top_15 = top_15 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl == 3:
          top_4 = top_4 + 1
          top_5 = top_5 + 1
          top_7 = top_7 + 1
          top_10 = top_10 + 1
          top_15 = top_15 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl == 4:
          top_5 = top_5 + 1
          top_7 = top_7 + 1
          top_10 = top_10 + 1
          top_15 = top_15 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl >= 4 and false_pred_cnt_lcl < 7:
          top_7 = top_7 + 1
          top_10 = top_10 + 1
          top_15 = top_15 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl >= 7 and false_pred_cnt_lcl < 10:
          top_10 = top_10 + 1
          top_15 = top_15 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl >= 10 and false_pred_cnt_lcl < 15:
          top_15 = top_15 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl >= 15 and false_pred_cnt_lcl < 20:
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl >= 20 and false_pred_cnt_lcl < 25:
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl >= 25 and false_pred_cnt_lcl < 50:
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl >= 50 and false_pred_cnt_lcl < 100:
          top_100 = top_100 + 1                


        fnc_cnt = fnc_cnt + 1

        if mdl1_fls_pred_cnt == 0:
            mdl1_pred_top_1.append(i)
            mdl1_pred_top_2.append(i)
            mdl1_pred_top_3.append(i)
            mdl1_pred_top_4.append(i)
            mdl1_pred_top_5.append(i)
        elif mdl1_fls_pred_cnt == 1:
            mdl1_pred_top_2.append(i)
            mdl1_pred_top_3.append(i)
            mdl1_pred_top_4.append(i)
            mdl1_pred_top_5.append(i)
        elif mdl1_fls_pred_cnt == 2:
            mdl1_pred_top_3.append(i)
            mdl1_pred_top_4.append(i)
            mdl1_pred_top_5.append(i)
        elif mdl1_fls_pred_cnt == 3:
            mdl1_pred_top_4.append(i)
            mdl1_pred_top_5.append(i)
        elif mdl1_fls_pred_cnt == 4:
            mdl1_pred_top_5.append(i)


        if mdl2_fls_pred_cnt == 0:
            mdl2_pred_top_1.append(i)
            mdl2_pred_top_2.append(i)
            mdl2_pred_top_3.append(i)
            mdl2_pred_top_4.append(i)
            mdl2_pred_top_5.append(i)
        elif mdl2_fls_pred_cnt == 1:
            mdl2_pred_top_2.append(i)
            mdl2_pred_top_3.append(i)
            mdl2_pred_top_4.append(i)
            mdl2_pred_top_5.append(i)
        elif mdl2_fls_pred_cnt == 2:
            mdl2_pred_top_3.append(i)
            mdl2_pred_top_4.append(i)
            mdl2_pred_top_5.append(i)
        elif mdl2_fls_pred_cnt == 3:
            mdl2_pred_top_4.append(i)
            mdl2_pred_top_5.append(i)
        elif mdl2_fls_pred_cnt == 4:
            mdl2_pred_top_5.append(i)


        if mdl3_fls_pred_cnt == 0:
            mdl3_pred_top_1.append(i)
            mdl3_pred_top_2.append(i)
            mdl3_pred_top_3.append(i)
            mdl3_pred_top_4.append(i)
            mdl3_pred_top_5.append(i)
        elif mdl3_fls_pred_cnt == 1:
            mdl3_pred_top_2.append(i)
            mdl3_pred_top_3.append(i)
            mdl3_pred_top_4.append(i)
            mdl3_pred_top_5.append(i)
        elif mdl3_fls_pred_cnt == 2:
            mdl3_pred_top_3.append(i)
            mdl3_pred_top_4.append(i)
            mdl3_pred_top_5.append(i)
        elif mdl3_fls_pred_cnt == 3:
            mdl3_pred_top_4.append(i)
            mdl3_pred_top_5.append(i)
        elif mdl3_fls_pred_cnt == 4:
            mdl3_pred_top_5.append(i)
            

        if agg_fls_pred_cnt == 0:
            agg_pred_top_1.append(i)
            agg_pred_top_2.append(i)
            agg_pred_top_3.append(i)
            agg_pred_top_4.append(i)
            agg_pred_top_5.append(i)
        elif agg_fls_pred_cnt == 1:
            agg_pred_top_2.append(i)
            agg_pred_top_3.append(i)
            agg_pred_top_4.append(i)
            agg_pred_top_5.append(i)
        elif agg_fls_pred_cnt == 2:
            agg_pred_top_3.append(i)
            agg_pred_top_4.append(i)
            agg_pred_top_5.append(i)
        elif agg_fls_pred_cnt == 3:
            agg_pred_top_4.append(i)
            agg_pred_top_5.append(i)
        elif agg_fls_pred_cnt == 4:
            agg_pred_top_5.append(i)


        if wt_agg_fls_pred_cnt == 0:
            wt_agg_pred_top_1.append(i)
            wt_agg_pred_top_2.append(i)
            wt_agg_pred_top_3.append(i)
            wt_agg_pred_top_4.append(i)
            wt_agg_pred_top_5.append(i)
        elif wt_agg_fls_pred_cnt == 1:
            wt_agg_pred_top_2.append(i)
            wt_agg_pred_top_3.append(i)
            wt_agg_pred_top_4.append(i)
            wt_agg_pred_top_5.append(i)
        elif wt_agg_fls_pred_cnt == 2:
            wt_agg_pred_top_3.append(i)
            wt_agg_pred_top_4.append(i)
            wt_agg_pred_top_5.append(i)
        elif wt_agg_fls_pred_cnt == 3:
            wt_agg_pred_top_4.append(i)
            wt_agg_pred_top_5.append(i)
        elif wt_agg_fls_pred_cnt == 4:
            wt_agg_pred_top_5.append(i)
            
            
        tot_diff += list(diff3)
        tot_truth += list(y2 > 0)


    diff = np.array(tot_diff)
    truth = np.array(tot_truth)
    
    fpr, tpr, thres = roc_curve(truth, (1-diff)/2)
    model_auc = auc(fpr, tpr)

    auc2_1 = (acc_1/acc_cnt_2) * 100
    auc2_2 = (acc_2/acc_cnt_2) * 100
    auc2_3 = (acc_3/acc_cnt_2) * 100

    auc_cmb2 = (acc_cmb_2/acc_cmb) * 100

    fls_pd = (false_pred_cnt/acc_cmb) * 100

    print ("Cirrina accuracy: {}".format((acc_1/acc_cnt_1) * 100))
    print ("Gemini  accuracy: {}".format((acc_2/acc_cnt_2) * 100))
    print ("Palmtree  accuracy: {}".format((acc_3/acc_cnt_2) * 100))
    print ("fls_pd_cnt: {}, per: {}".format(false_pred_cnt, fls_pd))


    print (f"top_1 -> {top_1}")
    print (f"top_2 -> {top_2}")
    print (f"top_3 -> {top_3}")
    print (f"top_4 -> {top_4}")
    print (f"top_5 -> {top_5}")
    print (f"top_7 -> {top_7}")
    print (f"top_10 -> {top_10}")
    print (f"top_15 -> {top_15}")
    print (f"top_20 -> {top_20}")
    print (f"top_25 -> {top_25}")
    print (f"top_50 -> {top_50}")
    print (f"top_100 -> {top_100}")

    print ("Accuracy Combined (Aggregate): {}".format(auc_cmb2))

    print (f"\ntop_1 -> {(top_1/fnc_cnt) * 100}")
    print (f"top_2 -> {(top_2/fnc_cnt) * 100}")
    print (f"top_3 -> {(top_3/fnc_cnt) * 100}")
    print (f"top_4 -> {(top_4/fnc_cnt) * 100}")
    print (f"top_5 -> {(top_5/fnc_cnt) * 100}")
    print (f"top_7 -> {(top_7/fnc_cnt) * 100}")
    print (f"top_10 -> {(top_10/fnc_cnt) * 100}")
    print (f"top_15 -> {(top_15/fnc_cnt) * 100}")
    print (f"top_20 -> {(top_20/fnc_cnt) * 100}")
    print (f"top_25 -> {(top_25/fnc_cnt) * 100}")
    print (f"top_50 -> {(top_50/fnc_cnt) * 100}")
    print (f"top_100 -> {(top_100/fnc_cnt) * 100}\n")


    print_stat(1, mdl1_pred_top_1, mdl2_pred_top_1, mdl3_pred_top_1, agg_pred_top_1, wt_agg_pred_top_1,
               mdl_or_pred_avg_top_1, mdl_and_pred_avg_top_1, mdl_vote_pred_avg_top_1, mdl_best_pred_avg_top_1, mdl_wt_best_pred_avg_top_1, fnc_cnt)
    print_stat(2, mdl1_pred_top_2, mdl2_pred_top_2, mdl3_pred_top_2, agg_pred_top_2, wt_agg_pred_top_2,
               mdl_or_pred_avg_top_2, mdl_and_pred_avg_top_2, mdl_vote_pred_avg_top_2, mdl_best_pred_avg_top_2, mdl_wt_best_pred_avg_top_2, fnc_cnt)
    print_stat(3, mdl1_pred_top_3, mdl2_pred_top_3, mdl3_pred_top_3, agg_pred_top_3, wt_agg_pred_top_3,
               mdl_or_pred_avg_top_3, mdl_and_pred_avg_top_3, mdl_vote_pred_avg_top_3, mdl_best_pred_avg_top_3, mdl_wt_best_pred_avg_top_3, fnc_cnt)
    print_stat(4, mdl1_pred_top_4, mdl2_pred_top_4, mdl3_pred_top_4, agg_pred_top_4, wt_agg_pred_top_4,
               mdl_or_pred_avg_top_4, mdl_and_pred_avg_top_4, mdl_vote_pred_avg_top_4, mdl_best_pred_avg_top_4, mdl_wt_best_pred_avg_top_4, fnc_cnt)
    print_stat(5, mdl1_pred_top_5, mdl2_pred_top_5, mdl3_pred_top_5, agg_pred_top_5, wt_agg_pred_top_5,
               mdl_or_pred_avg_top_5, mdl_and_pred_avg_top_5, mdl_vote_pred_avg_top_5, mdl_best_pred_avg_top_5, mdl_wt_best_pred_avg_top_5, fnc_cnt)
    
    
    
    return model_auc, fpr, tpr, thres, auc2_1, auc2_2, auc2_3, auc_cmb2



def print_stat(top, mdl1_pred, mdl2_pred, mdl3_pred, agg_pred, wt_agg_pred, mdl_or_pred_avg, mdl_and_pred_avg, mdl_vote_pred_avg, mdl_best_pred_avg, mdl_wt_best_pred_avg, fnc_cnt):

    total_m2_m3 = list(set(mdl2_pred).union(set(mdl3_pred)))
    total_m1_m3 = list(set(mdl1_pred).union(set(mdl3_pred)))
    total_m1_m2 = list(set(mdl1_pred).union(set(mdl2_pred)))
    total = list(set(mdl1_pred).union(set(mdl2_pred).union(set(mdl3_pred))))

    mdl1_uniq_bef_agg = list(set(mdl1_pred) - set(total_m2_m3))
    mdl1_uniq_aft_agg = list(set(mdl1_uniq_bef_agg) - set(agg_pred))

    mdl2_uniq_bef_agg = list(set(mdl2_pred) - set(total_m1_m3))
    mdl2_uniq_aft_agg = list(set(mdl2_uniq_bef_agg) - set(agg_pred))

    mdl3_uniq_bef_agg = list(set(mdl3_pred) - set(total_m1_m2))
    mdl3_uniq_aft_agg = list(set(mdl3_uniq_bef_agg) - set(agg_pred))

    agg_missed = list(set(total) - set(agg_pred))
    agg_uniq = list(set(agg_pred) - set(total))
    #wt_agg_uniq = list(set(wt_agg_pred) - set(total))
    total_cmb = list(set(total) - set(mdl1_uniq_aft_agg).union(set(mdl2_uniq_aft_agg).union(set(mdl3_uniq_aft_agg))))
    
    print(f"\nTop-{top}")
    print("-------------------")
    print(f"Cirrina        : total -> {(len(mdl1_pred)/fnc_cnt) * 100}, unique -> bef. aggregate: {(len(mdl1_uniq_bef_agg)/fnc_cnt) * 100}, aft. aggregate: {(len(mdl1_uniq_aft_agg)/fnc_cnt) * 100} ")
    print(f"Gemini_triplet : total -> {(len(mdl2_pred)/fnc_cnt) * 100}, unique -> bef. aggregate: {(len(mdl2_uniq_bef_agg)/fnc_cnt) * 100}, aft. aggregate: {(len(mdl2_uniq_aft_agg)/fnc_cnt) * 100} ")
    print(f"Palmtree       : total -> {(len(mdl3_pred)/fnc_cnt) * 100}, unique -> bef. aggregate: {(len(mdl3_uniq_bef_agg)/fnc_cnt) * 100}, aft. aggregate: {(len(mdl3_uniq_aft_agg)/fnc_cnt) * 100} ")
    print(f"Aggregate      : total -> {(len(agg_pred)/fnc_cnt) * 100}, unique -> {(len(agg_uniq)/fnc_cnt) * 100} ")
    print(f"Agg. Missed    : total -> {(len(agg_missed)/fnc_cnt) * 100} ")
    print(f"Model combined : total -> {(len(total_cmb)/fnc_cnt) * 100} ")
    print(f"Model total    : total -> {(len(total)/fnc_cnt) * 100} ")
    print (f"\nTop-{top} info:")
    print (f"Model_wt_aggregate : {(len(wt_agg_pred)/fnc_cnt) * 100} ")
    print (f"Model_or_pred      : {(mdl_or_pred_avg/fnc_cnt) * 100}")
    print (f"Model_and_pred     : {(mdl_and_pred_avg/fnc_cnt) * 100}")
    print (f"Model_vote_pred    : {(mdl_vote_pred_avg/fnc_cnt) * 100}")
    print (f"Model_best_pred    : {(mdl_best_pred_avg/fnc_cnt) * 100}")
    print (f"Model_wt_best_pred : {(mdl_wt_best_pred_avg/fnc_cnt) * 100}")
    print("-------------------\n")
