import numpy as np
from sklearn.metrics import auc, roc_curve
from siamese_triplet import graphnn
import json, sys
from settings import *
import pymongo

np.set_printoptions(threshold=sys.maxsize)

def get_func_from_db():
    client = pymongo.MongoClient(MONGO_CLIENT)
    db1 = client[FUNC_INFO_DB]
    col1 = db1[FUNC_INFO_COLLECTION]

    for data in col1.find({"experiment" : expr_num}, {"_id": 0, "testing_fns" : 1, "validation_fns": 1, "training_fns" : 1, "library" : 1, "experiment" : 1, "trs" : 1}):
      testing_fns = data["testing_fns"]
      validation_fns = data["validation_fns"]
      training_fns = data["training_fns"]
      exp_num = data["experiment"]
      inc_libraries = data["library"]
 
    return testing_fns          


def get_func_info(F_NAME):
    name_num = 0
    name_dict = {}
    func_lst = []
    
    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                if (g_info['fname'] not in name_dict):
                    name_dict[g_info['fname']] = name_num
                    name_num += 1

                    func_lst.append((g_info['fname'], g_info['func'], g_info['library']))
    return func_lst


def get_f_dict(F_NAME):
    name_num = 0
    name_dict = {}
    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                if (g_info['fname'] not in name_dict):
                    name_dict[g_info['fname']] = name_num
                    name_num += 1
    return name_dict

class graph(object):
    def __init__(self, node_num = 0, label = None, name = None):
        self.node_num = node_num
        self.label = label
        self.function = ""
        self.name = name
        self.compiler_flag = ""
        self.transformation = ""
        self.library = ""
        self.arch = ""
        self.features = []
        self.succs = []
        self.preds = []
        if (node_num > 0):
            for i in range(node_num):
                self.features.append([])
                self.succs.append([])
                self.preds.append([])
                
    def add_node(self, feature = []):
        self.node_num += 1
        self.features.append(feature)
        self.succs.append([])
        self.preds.append([])
        
    def add_edge(self, u, v):
        self.succs[u].append(v)
        self.preds[v].append(u)

    def toString(self):
        ret = '{} {}\n'.format(self.node_num, self.label)
        for u in range(self.node_num):
            for fea in self.features[u]:
                ret += '{} '.format(fea)
            ret += str(len(self.succs[u]))
            for succ in self.succs[u]:
                ret += ' {}'.format(succ)
            ret += '\n'
        return ret

        
def read_graph(F_NAME, FUNC_NAME_DICT, FEATURE_DIM):
    graphs = []
    classes = []
    if FUNC_NAME_DICT != None:
        for f in range(len(FUNC_NAME_DICT)):
            classes.append([])

    for f_name in F_NAME:
        with open(f_name) as inf:
            for line in inf:
                g_info = json.loads(line.strip())
                label = FUNC_NAME_DICT[g_info['fname']]
                classes[label].append(len(graphs))
                #print ("grp: {}, trs: {} -> {}".format(g_info['fname'], g_info['trs'], len(graphs)))
                cur_graph = graph(g_info['n_num'], label, g_info['src'])
                cur_graph.function = g_info['fname']
                cur_graph.compiler_flag = g_info['compiler_flag']
                cur_graph.transformation = g_info['trs']
                cur_graph.library = g_info['library']
                cur_graph.arch = g_info['arch']
                for u in range(g_info['n_num']):
                    cur_graph.features[u] = np.array(g_info['features'][u])
                    for v in g_info['succs'][u]:
                        cur_graph.add_edge(u, v)
                graphs.append(cur_graph)

    return graphs, classes


def partition_data(Gs, classes, partitions, perm):
    C = len(classes)
    st = 0.0
    ret = []
    for part in partitions:
        cur_g = []
        cur_c = []
        ed = st + part * C
        for cls in range(int(st), int(ed)):
            prev_class = classes[perm[cls]]
            cur_c.append([])
            for i in range(len(prev_class)):
                cur_g.append(Gs[prev_class[i]])
                cur_g[-1].label = len(cur_c)-1
                cur_c[-1].append(len(cur_g)-1)

        ret.append(cur_g)
        ret.append(cur_c)
        st = ed
    
    return ret


def generate_epoch_pair_3(Gs, classes, M, cls_st, st_ind, end_ind, output_id = False, load_id = None, mode = None):
    epoch_data = []
    id_data = []  
    ids = []

    if len(Gs) < end_ind:
        end_ind = len(Gs)
    
    #print (f"len(classes) -> {len(classes)}")
    #print (f"st -> {st_ind}, end -> {end_ind}")    

    if load_id is None:
        st = st_ind
        while st < len(classes) and st < end_ind:  #len(Gs):
            if output_id:
                X1, X2, X3, m1, m2, m3, y, pos_id, neg_id, gid = get_pair_3(Gs, classes, M, cls_st, target_cls_ind=st, output_id=True, mode=mode)
                id_data.append( (pos_id, neg_id) )
                ids = ids + gid # (pos_id, neg_id) )
            else:
                X1, X2, X3, m1, m2, m3, y = get_pair_3(Gs, classes, M, cls_st, target_cls_ind=st, mode=mode)
            epoch_data.append( (X1,X2,X3,m1,m2,m3,y) )
            st += M
            #break
    else:   ## Load from previous id data
        id_data = load_id
        for id_pair in id_data:
            X1, X2, X3, m1, m2, m3, y = get_pair_3(Gs, classes, M, load_id=id_pair, mode=mode)
            epoch_data.append( (X1, X2, X3, m1, m2, m3, y) )

    if output_id:
        return epoch_data, id_data, ids
    else:
        return epoch_data


def get_pair_3(Gs, classes, M, cls_st, target_cls_ind = -1, output_id = False, load_id = None, mode = None):
    if load_id is None:
        C = len(classes)

        #if (st + M > C): #len(Gs)):
        #    M = C - st
        #ed = st + M

        pos_ids = []
        neg_ids = []
        ids = []

        #for cls_id in range(st, ed):
        cls_id = target_cls_ind
        #g0 = classes[cls_id]
        cls = classes[cls_id] #g0 #.label
        tot_g = len(cls) #len(classes[cls])
        if (len(cls) == 2):

            g1_id = -1
            g2_id = -1
            g3_id = -1
            for i in range(0, len(cls)):
              if Gs[cls[i]].transformation == ["N/A"]:
                  g1_id = cls[i]
              elif Gs[cls[i]].transformation == ["Flatten"]: #["EncodeArithmetic, Flatten"]: #
                  g2_id = cls[i]
              elif Gs[cls[i]].transformation == ["Virtualize"]: #["EncodeArithmetic, Virtualize"]: #
                  g3_id = cls[i]

        if len(cls) != 2:
            print ("class lenght != 2 -> {}".format(Gs[cls[0]].function))

        cls_end = cls_st+M
        if cls_end > C:
            cls_end = C
        
        for j in range(cls_st, cls_end): #C): M -> batch_size = 500
          if j != cls_id:
           cls2 = classes[j]

           h1_id = -1
           h2_id = -1
           h3_id = -1

           for i in range(0, len(cls2)):

              if Gs[cls2[i]].transformation == ["N/A"]:
                  h1_id = cls2[i]
              elif Gs[cls2[i]].transformation == ["Flatten"]: #["EncodeArithmetic, Flatten"]: #
                  h2_id = cls2[i]
              elif Gs[cls2[i]].transformation == ["Virtualize"]: #["EncodeArithmetic, Virtualize"]: #
                  h3_id = cls2[i]

           pos_ids.append( (g2_id, g1_id, h1_id))
           #neg_ids.append( (g3_id, h1_id, g1_id))

           #print (f"[+] func: g3 -> {Gs[g3_id].function}, g1 -> {Gs[g1_id].function}, h1 -> {Gs[h1_id].function}")
           #print (f"[-] trs : g3 -> {Gs[g3_id].transformation}, g1 -> {Gs[g1_id].transformation}, h1 -> {Gs[h1_id].transformation}")

           ids.append((g2_id, g1_id, h1_id))
           #ids.append((g3_id, h1_id, g1_id))

    else:
        pos_ids = load_id[0]
        neg_ids = load_id[1]
        
    M_pos = len(pos_ids)
    M_neg = len(neg_ids)
    M = M_pos + M_neg

    maxN1 = 0
    maxN2 = 0
    maxN3 = 0
    for pair in pos_ids:
        maxN1 = max(maxN1, Gs[pair[0]].node_num)
        maxN2 = max(maxN2, Gs[pair[1]].node_num)
        maxN3 = max(maxN3, Gs[pair[2]].node_num)
    for pair in neg_ids:
        maxN1 = max(maxN1, Gs[pair[0]].node_num)
        maxN2 = max(maxN2, Gs[pair[1]].node_num)
        maxN3 = max(maxN3, Gs[pair[2]].node_num)

    feature_dim = len(Gs[0].features[0])
    X1_input = np.zeros((M, maxN1, feature_dim))
    X2_input = np.zeros((M, maxN2, feature_dim))
    X3_input = np.zeros((M, maxN3, feature_dim))
    node1_mask = np.zeros((M, maxN1, maxN1))
    node2_mask = np.zeros((M, maxN2, maxN2))
    node3_mask = np.zeros((M, maxN3, maxN3))
    y_input = np.zeros((M))
    
    for i in range(M_pos):
        y_input[i] = 1
        g1 = Gs[pos_ids[i][0]]
        g2 = Gs[pos_ids[i][1]]
        g3 = Gs[pos_ids[i][2]]
        for u in range(g1.node_num):
            X1_input[i, u, :] = np.array( g1.features[u] )
            for v in g1.succs[u]:
                node1_mask[i, u, v] = 1
        for u in range(g2.node_num):
            X2_input[i, u, :] = np.array( g2.features[u] )
            for v in g2.succs[u]:
                node2_mask[i, u, v] = 1
        for u in range(g3.node_num):
            X3_input[i, u, :] = np.array( g3.features[u] )
            for v in g3.succs[u]:
                node3_mask[i, u, v] = 1
        
    for i in range(M_pos, M_pos + M_neg):
        y_input[i] = -1
        g1 = Gs[neg_ids[i-M_pos][0]]
        g2 = Gs[neg_ids[i-M_pos][1]]
        g3 = Gs[neg_ids[i-M_pos][2]]
        for u in range(g1.node_num):
            X1_input[i, u, :] = np.array( g1.features[u] )
            for v in g1.succs[u]:
                node1_mask[i, u, v] = 1
        for u in range(g2.node_num):
            X2_input[i, u, :] = np.array( g2.features[u] )
            for v in g2.succs[u]:
                node2_mask[i, u, v] = 1
        for u in range(g3.node_num):
            X3_input[i, u, :] = np.array( g3.features[u] )
            for v in g3.succs[u]:
                node3_mask[i, u, v] = 1

    #print ("ids: {}".format(ids))
    if output_id:
        return X1_input,X2_input,X3_input,node1_mask,node2_mask,node3_mask,y_input,pos_ids,neg_ids,ids
    else:
        return X1_input,X2_input,X3_input,node1_mask,node2_mask,node3_mask,y_input

def train_epoch(model, graphs, classes, batch_size, load_data=None):
    if load_data is None:
        epoch_data = generate_epoch_pair_3(graphs, classes, batch_size)
    else:
        epoch_data = load_data

    perm = np.random.permutation(len(epoch_data))

    cum_loss = 0.0
    for index in perm:
        cur_data = epoch_data[index]
        X1, X2, X3, mask1, mask2, mask3, y = cur_data
        loss = model.train(X1, X2, X3, mask1, mask2, mask3, y)
        cum_loss += loss

    return cum_loss / len(perm)


def get_auc_epoch(model, graphs, classes, batch_size, cls_st, load_data=None, ids=None, output_id=False, mode=None, st_ind=-1, end_ind=-1):
    tot_diff = []
    tot_truth = []

    if load_data is None:
        epoch_data, id_data, ids = generate_epoch_pair_3(graphs, classes, batch_size, cls_st, st_ind, end_ind, output_id=True)
    else:
        epoch_data, ids = load_data, ids

    acc = 0
    acc_cnt = 0

    acc_1 = 0
    #acc_2 = 0
    acc_3 = 0
    acc_4 = 0
    acc_cnt_1 = 0
    #acc_cnt_2 = 0
    false_pred_cnt = 0
    top_1 = 0
    top_2 = 0
    top_3 = 0
    top_4 = 0
    top_5 = 0
    top_10 = 0
    top_20 = 0
    top_25 = 0
    top_50 = 0
    top_100 = 0

    diff_lst_1 = {}
    diff_lst_2 = {}
    y_lst = {}
    
    if mode == "Testing":
      for i in range(0, len(epoch_data)):
        
        false_pred_cnt_lcl = 0
    
        X1, X2, X3, m1, m2, m3, y  = epoch_data[i]
        diff = model.calc_diff(X1, X2, X3, m1, m2, m3)
        diff1 = model.calc_diff1(X1, X2, X3, m1, m2, m3)
        #print ("diff: {}, y: {}".format(diff, y))
        #print ("diff1: {}, y: {}".format(diff1, y))
        
        #tot_diff += list(diff)
        #tot_truth += list(y > 0)

        diff_lst_1[i] = diff.tolist()
        diff_lst_2[i] = diff1.tolist()
        y_lst[i] = y.tolist()

        '''
        pred1 = False
        pred2 = False         
        hf = len(diff) #int(len(diff)/2)
        
        for j in range(0, hf):
          if (-1 * diff[j]) > (-1 * diff1[j]):
            acc = acc + 1
            acc_1 = acc_1 + 1
            pred1 = True
          else:
            print ("{} -> false pred at {}: diff -> {}, diff1 -> {}".format(i, j, diff[j], diff1[j]))
            false_pred_cnt = false_pred_cnt + 1
            false_pred_cnt_lcl = false_pred_cnt_lcl + 1
            
          #if (-1 * diff1[hf+j]) > (-1 * diff[hf+j]):
          #  acc = acc + 1
          #  acc_2 = acc_2 + 1
          #  pred2 = True

          acc_cnt = acc_cnt + 1
          acc_cnt_1 = acc_cnt_1 + 1
          #acc_cnt_2 = acc_cnt_2 + 1
      
        if false_pred_cnt_lcl == 0:
          top_1 = top_1 + 1
          top_2 = top_2 + 1
          top_3 = top_3 + 1
          top_4 = top_4 + 1
          top_5 = top_5 + 1
          top_10 = top_10 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl == 1:
          top_2 = top_2 + 1
          top_3 = top_3 + 1
          top_4 = top_4 + 1
          top_5 = top_5 + 1
          top_10 = top_10 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl == 2:
          top_3 = top_3 + 1
          top_4 = top_4 + 1
          top_5 = top_5 + 1
          top_10 = top_10 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl == 3:
          top_4 = top_4 + 1
          top_5 = top_5 + 1
          top_10 = top_10 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl == 4:
          top_5 = top_5 + 1
          top_10 = top_10 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl > 4 and false_pred_cnt_lcl < 10:
          top_10 = top_10 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl > 10 and false_pred_cnt_lcl < 20:
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl > 20 and false_pred_cnt_lcl < 25:
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl > 25 and false_pred_cnt_lcl < 50:
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl > 50 and false_pred_cnt_lcl < 100:
          top_100 = top_100 + 1
        '''
    else:
        i = 0
        for cur_data in epoch_data:
            i = i + 1
            X1, X2, X3, m1, m2, m3, y  = cur_data
            diff = model.calc_diff(X1, X2, X3, m1, m2, m3)

            diff1 = model.calc_diff1(X1, X2, X3, m1, m2, m3)

            tot_diff += list(diff)
            tot_truth += list(y > 0)
            
    #diff = np.array(tot_diff)
    #truth = np.array(tot_truth)
    
    #fpr, tpr, thres = roc_curve(truth, (1-diff)/2)
    #model_auc = auc(fpr, tpr)

    if mode == "Testing":
        '''
        auc2 = (acc/acc_cnt) * 100
        auc2_1 = (acc_1/acc_cnt_1) * 100
        #auc2_2 = (acc_2/acc_cnt_2) * 100
        fls_pd = (false_pred_cnt/acc_cnt) * 100
        
        print ("accuracy 2 : {}".format(auc2))
        print ("accuracy 2 case 1: {}".format(auc2_1))
        #print ("accuracy 2 case 2: {}".format(auc2_2))
        print ("fls_pd_cnt: {}, per: {}".format(false_pred_cnt, fls_pd))

        print (f"top_1 -> {top_1}")
        print (f"top_2 -> {top_2}")
        print (f"top_3 -> {top_3}")
        print (f"top_4 -> {top_4}")
        print (f"top_5 -> {top_5}")
        print (f"top_10 -> {top_10}")
        print (f"top_20 -> {top_20}")
        print (f"top_25 -> {top_25}")
        print (f"top_50 -> {top_50}")
        print (f"top_100 -> {top_100}")
        '''
        
        return diff_lst_1, diff_lst_2, y_lst #, auc2_3, auc2_4
    else:
        return #model_auc, fpr, tpr, thres
        
       
def get_auc_epoch_from_db(data_lst, mode=None):
    tot_diff = []
    tot_truth = []

    acc = 0
    acc_cnt = 0

    acc_1 = 0
    acc_2 = 0
    acc_3 = 0
    acc_4 = 0
    acc_cnt_1 = 0
    acc_cnt_2 = 0
    false_pred_cnt = 0
    top_1 = 0
    top_2 = 0
    top_3 = 0
    top_4 = 0
    top_5 = 0
    top_10 = 0
    top_20 = 0
    top_25 = 0
    top_50 = 0
    top_100 = 0

    print ("Inside get_auc_epoch_from_db !")
    
    for i, func, model, trs, diff, diff1, y in data_lst: 
      if i >= 0:
        st = 0
        false_pred_cnt_lcl = 0

        y = np.array(y)
        diff = np.array(diff)
        diff1 = np.array(diff1)

        tot_diff += list(diff)
        tot_truth += list(y > 0)

        pred1 = False
        pred2 = False         
        hf = int(len(diff)/2)

        for j in range(0, hf):
          if (-1 * diff[j]) > (-1 * diff1[j]):
            acc = acc + 1
            acc_1 = acc_1 + 1
            pred1 = True
          else:
            print ("{} -> false pred at {}: diff -> {}, diff1 -> {}".format(i-st, j, diff[j], diff1[j]))
            false_pred_cnt = false_pred_cnt + 1
            false_pred_cnt_lcl = false_pred_cnt_lcl + 1

          if (-1 * diff1[hf+j]) > (-1 * diff[hf+j]):
            acc = acc + 1
            acc_2 = acc_2 + 1
            pred2 = True

          acc_cnt = acc_cnt + 2
          acc_cnt_1 = acc_cnt_1 + 1
          acc_cnt_2 = acc_cnt_2 + 1

        if false_pred_cnt_lcl == 0:
          top_1 = top_1 + 1
          top_2 = top_2 + 1
          top_3 = top_3 + 1
          top_4 = top_4 + 1
          top_5 = top_5 + 1
          top_10 = top_10 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl == 1:
          top_2 = top_2 + 1
          top_3 = top_3 + 1
          top_4 = top_4 + 1
          top_5 = top_5 + 1
          top_10 = top_10 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl == 2:
          top_3 = top_3 + 1
          top_4 = top_4 + 1
          top_5 = top_5 + 1
          top_10 = top_10 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl == 3:
          top_4 = top_4 + 1
          top_5 = top_5 + 1
          top_10 = top_10 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl == 4:
          top_5 = top_5 + 1
          top_10 = top_10 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl > 4 and false_pred_cnt_lcl < 10:
          top_10 = top_10 + 1
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl > 10 and false_pred_cnt_lcl < 20:
          top_20 = top_20 + 1
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl > 20 and false_pred_cnt_lcl < 25:
          top_25 = top_25 + 1
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl > 25 and false_pred_cnt_lcl < 50:
          top_50 = top_50 + 1
          top_100 = top_100 + 1
        elif false_pred_cnt_lcl > 50 and false_pred_cnt_lcl < 100:
          top_100 = top_100 + 1
      
    
    diff = np.array(tot_diff)
    truth = np.array(tot_truth)
    
    fpr, tpr, thres = roc_curve(truth, (1-diff)/2)
    model_auc = auc(fpr, tpr)

    auc2 = (acc/acc_cnt) * 100
    auc2_1 = (acc_1/acc_cnt_2) * 100
    auc2_2 = (acc_2/acc_cnt_2) * 100
    fls_pd = (false_pred_cnt/acc_cnt) * 100

    print ("accuracy 2 : {}".format(auc2))
    print ("accuracy 2 case 1: {}".format(auc2_1))
    print ("accuracy 2 case 2: {}".format(auc2_2))
    print ("fls_pd_cnt: {}, per: {}".format(false_pred_cnt, fls_pd))

    print (f"top_1 -> {top_1}")
    print (f"top_2 -> {top_2}")
    print (f"top_3 -> {top_3}")
    print (f"top_4 -> {top_4}")
    print (f"top_5 -> {top_5}")
    print (f"top_10 -> {top_10}")
    print (f"top_20 -> {top_20}")
    print (f"top_25 -> {top_25}")
    print (f"top_50 -> {top_50}")
    print (f"top_100 -> {top_100}")


    return model_auc, fpr, tpr, thres, auc2, auc2_1, auc2_2
