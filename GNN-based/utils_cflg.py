import numpy as np
from sklearn.metrics import auc, roc_curve
from siamese_triplet import graphnn
import json, pymongo
from settings import *

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


def generate_epoch_pair_3(Gs, classes, M, output_id = False, load_id = None, mode = None):
    epoch_data = []
    id_data = []  
    ids = []

    if load_id is None:
        st = 0
        while st < len(classes) : #len(Gs):
            if output_id:
                X1, X2, X3, m1, m2, m3, y, pos_id, neg_id, gid = get_pair_3(Gs, classes,
                        M, st=st, output_id=True, mode=mode)
                id_data.append( (pos_id, neg_id) )
                ids = ids + gid
            else:
                X1, X2, X3, m1, m2, m3, y = get_pair_3(Gs, classes, M, st=st, mode=mode)
            epoch_data.append( (X1,X2,X3,m1,m2,m3,y) )
            st += M
    else:   ## Load from previous id data
        id_data = load_id
        for id_pair in id_data:
            X1, X2, X3, m1, m2, m3, y = get_pair_3(Gs, classes, M, load_id=id_pair, mode=mode)
            epoch_data.append( (X1, X2, X3, m1, m2, m3, y) )

    if output_id:
        return epoch_data, id_data, ids
    else:
        return epoch_data


def get_pair_3(Gs, classes, M, st = -1, output_id = False, load_id = None, mode = None):
    if load_id is None:
        C = len(classes)

        if (st + M > C): #len(Gs)):
            M = C - st
        ed = st + M

        pos_ids = []
        neg_ids = []
        ids = []

        for cls_id in range(st, ed):
            g0 = classes[cls_id]
            cls = g0 #.label
            tot_g = len(cls) #len(classes[cls])
            if (len(cls) == 2): #>= 2):
                    
                g1_id = -1
                g2_id = -1
                g3_id = -1
                g4_id = -1
                for i in range(0, len(cls)):
                  if Gs[cls[i]].compiler_flag == "O0":
                      g1_id = cls[i]
                  elif Gs[cls[i]].compiler_flag == "O1":
                      g2_id = cls[i]
                  elif Gs[cls[i]].compiler_flag == "O2":
                      g3_id = cls[i]
                  elif Gs[cls[i]].compiler_flag == "O3":
                      g4_id = cls[i]

            if len(cls) != 2:
                print ("class lenght != 2 -> {}".format(Gs[g0[0]].function))
                continue
               
            cls2 = np.random.randint(C)
            while (len(classes[cls2]) == 0) or (cls2 == cls_id):
                cls2 = np.random.randint(C)

            tot_g2 = len(classes[cls2])
            cls2 = classes[cls2]
             

            h1_id = -1
            h2_id = -1
            h3_id = -1
            h4_id = -1
            for i in range(0, len(cls2)):
                 if Gs[cls2[i]].compiler_flag == "O0":
                     h1_id = cls2[i]
                 elif Gs[cls2[i]].compiler_flag == "O1":
                     h2_id = cls2[i]
                 elif Gs[cls2[i]].compiler_flag == "O2":
                     h3_id = cls2[i]
                 elif Gs[cls2[i]].compiler_flag == "O3":
                     h4_id = cls2[i]

            pos_ids.append( (g4_id, g1_id, h1_id) )
            neg_ids.append( (g4_id, h1_id, g1_id) )
            
            ids.append((g4_id, g1_id, h1_id))
            ids.append((g4_id, h1_id, g1_id))

    else:
        pos_ids = load_id[0]
        neg_ids = load_id[1]
        #return
    
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


def get_auc_epoch(model, graphs, classes, batch_size, load_data=None, ids=None, output_id=False, mode=None):
    tot_diff = []
    tot_truth = []

    if load_data is None:
        epoch_data, id_data, ids = generate_epoch_pair_3(graphs, classes, batch_size, output_id=True)
    else:
        epoch_data, ids = load_data, ids

    acc = 0
    acc_cnt = 0

    acc_1 = 0
    acc_2 = 0
    acc_3 = 0
    acc_4 = 0
    acc_cnt_1 = 0
    acc_cnt_2 = 0


    if mode == "Testing":
      for i in range(0, len(epoch_data)):
        X1, X2, X3, m1, m2, m3, y  = epoch_data[i]
        diff = model.calc_diff(X1, X2, X3, m1, m2, m3)
        diff1 = model.calc_diff1(X1, X2, X3, m1, m2, m3)

        tot_diff += list(diff)
        tot_truth += list(y > 0)

        
        pred1 = False
        pred2 = False         
        
        if (-1 * diff[0]) > (-1 * diff1[0]):
            acc = acc + 1
            acc_1 = acc_1 + 1
            pred1 = True
        
        if (-1 * diff1[1]) > (-1 * diff[1]):
            acc = acc + 1
            acc_2 = acc_2 + 1
            pred2 = True
        
        acc_cnt = acc_cnt + 2
        acc_cnt_1 = acc_cnt_1 + 1
        acc_cnt_2 = acc_cnt_2 + 1

    else:
        i = 0
        for cur_data in epoch_data:
            i = i + 1
            X1, X2, X3, m1, m2, m3, y  = cur_data
            diff = model.calc_diff(X1, X2, X3, m1, m2, m3)

            tot_diff += list(diff)
            tot_truth += list(y > 0)
    
    diff = np.array(tot_diff)
    truth = np.array(tot_truth)

    fpr, tpr, thres = roc_curve(truth, (1-diff)/2)
    model_auc = auc(fpr, tpr)

    if mode == "Testing":    
        auc2 = (acc/acc_cnt) * 100
        auc2_1 = (acc_1/acc_cnt_2) * 100
        auc2_2 = (acc_2/acc_cnt_2) * 100
        print ("accuracy 2 : {}".format(auc2))
        print ("accuracy 2 case 1: {}".format((acc_1/acc_cnt_1) * 100))
        print ("accuracy 2 case 2: {}".format((acc_2/acc_cnt_2) * 100))

        return model_auc, fpr, tpr, thres, auc2, auc2_1, auc2_2
    else:
        return model_auc, fpr, tpr, thres
