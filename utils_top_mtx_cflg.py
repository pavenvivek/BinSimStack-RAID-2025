import numpy as np
from sklearn.metrics import auc, roc_curve
from siamese_nonuplet import graphnn
import json, sys
from settings import *

np.set_printoptions(threshold=sys.maxsize)

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
        self.model = ""
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

        
def read_graph(F_NAME, FUNC_NAME_DICT):
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
                cur_graph.model = g_info['model']
                for u in range(g_info['n_num']):
                    cur_graph.features[u] = np.array(g_info['features'][u])
                    for v in g_info['succs'][u]:
                        cur_graph.add_edge(u, v)
                graphs.append(cur_graph)

    return graphs, classes


def partition_data(Gs, classes, partitions, perm):
    C = len(classes)
    ret = []
    cur_g = []
    cur_c = []

    for cls in range(0, C):
        prev_class = classes[perm[cls]]
        cur_c.append([])
        for i in range(len(prev_class)):
            cur_g.append(Gs[prev_class[i]])
            cur_g[-1].label = len(cur_c)-1
            cur_c[-1].append(len(cur_g)-1)

    ret.append(cur_g)
    ret.append(cur_c)
    
    return ret


#def generate_epoch_pair(Gs, classes, M, output_id = False, load_id = None, mode = None):
def generate_epoch_pair(Gs, classes, M, cls_st, st_ind, end_ind, output_id = False, load_id = None, mode = None):
    epoch_data = []
    id_data = []  
    ids = []

    if len(Gs) < end_ind:
        end_ind = len(Gs)
    
    if load_id is None:
        st = st_ind
        while st < len(classes) and st < end_ind:
            if output_id:
                X1, X2, X3, m1, m2, m3, y1, X4, X5, X6, m4, m5, m6, y2, X7, X8, X9, m7, m8, m9, y3, pos_id, neg_id, gid = get_pair(Gs, classes, M, cls_st, target_cls_ind=st, output_id=True, mode=mode)
                id_data.append( (pos_id, neg_id) )
                ids = ids + gid # (pos_id, neg_id) )
            else:
                X1, X2, X3, m1, m2, m3, y1, X4, X5, X6, m4, m5, m6, y2, X7, X8, X9, m7, m8, m9, y3 = get_pair(Gs, classes, M, cls_st, target_cls_ind=st, mode=mode)
            epoch_data.append( (X1,X2,X3,m1,m2,m3,y1, X4,X5,X6,m4,m5,m6,y2, X7,X8,X9,m7,m8,m9,y3) )
            st += M
    else:   ## Load from previous id data
        id_data = load_id
        for id_pair in id_data:
            X1, X2, X3, m1, m2, m3, y1, X4, X5, X6, m4, m5, m6, y2, X7, X8, X9, m7, m8, m9, y3 = get_pair(Gs, classes, M, load_id=id_pair, mode=mode)
            epoch_data.append( (X1, X2, X3, m1, m2, m3, y1, X4, X5, X6, m4, m5, m6, y2, X7, X8, X9, m7, m8, m9, y3) )

    if output_id:
        return epoch_data, id_data, ids
    else:
        return epoch_data


#def get_pair(Gs, classes, M, st = -1, output_id = False, load_id = None, mode = None):
def get_pair(Gs, classes, M, cls_st, target_cls_ind = -1, output_id = False, load_id = None, mode = None):

    if load_id is None:
        C = len(classes)

        #if (st + M > C): #len(Gs)):
        #    M = C - st
        #ed = st + M

        pos_ids = [] # [(G_0, G_1)]
        neg_ids = [] # [(G_0, H_0)]
        ids = []

        #for cls_id in range(st, ed):

        cls_id = target_cls_ind
        #g0 = classes[cls_id]
        cls = classes[cls_id] #g0 #.label
        tot_g = len(cls) #len(classes[cls])
        if (len(cls) == 6):

            g1_id = -1
            g2_id = -1
            g3_id = -1
            g4_id = -1
            c1_id = -1
            c2_id = -1
            c3_id = -1
            c4_id = -1
            p1_id = -1
            p2_id = -1
            p3_id = -1
            p4_id = -1

            for i in range(0, len(cls)):

                if Gs[cls[i]].model == "Cirrina":
                    if Gs[cls[i]].compiler_flag == "O0":
                        c1_id = cls[i]
                    elif Gs[cls[i]].compiler_flag == "O1":
                        c2_id = cls[i]
                    elif Gs[cls[i]].compiler_flag == "O2":
                        c3_id = cls[i]
                    elif Gs[cls[i]].compiler_flag == "O3":
                        c4_id = cls[i]

                elif Gs[cls[i]].model == "Gemini_trp":
                    if Gs[cls[i]].compiler_flag == "O0":
                        g1_id = cls[i]
                    elif Gs[cls[i]].compiler_flag == "O1":
                        g2_id = cls[i]
                    elif Gs[cls[i]].compiler_flag == "O2":
                        g3_id = cls[i]
                    elif Gs[cls[i]].compiler_flag == "O3":
                        g4_id = cls[i]
                elif Gs[cls[i]].model == "Palmtree_trp":
                    if Gs[cls[i]].compiler_flag == "O0":
                        p1_id = cls[i]
                    elif Gs[cls[i]].compiler_flag == "O1":
                        p2_id = cls[i]
                    elif Gs[cls[i]].compiler_flag == "O2":
                        p3_id = cls[i]
                    elif Gs[cls[i]].compiler_flag == "O3":
                        p4_id = cls[i]



        if len(cls) != 6:
            print ("class lenght != 6 -> {}".format(Gs[g0[0]].function))
            #continue

        cls_end = cls_st+M
        if cls_end > C:
            cls_end = C
        
        for j in range(cls_st, cls_end):
            if j != cls_id:
                cls2 = classes[j]

                h1_id = -1
                h2_id = -1
                h3_id = -1
                h4_id = -1
                k1_id = -1
                k2_id = -1
                k3_id = -1
                k4_id = -1
                l1_id = -1
                l2_id = -1
                l3_id = -1
                l4_id = -1

                for i in range(0, len(cls2)):

                     if Gs[cls2[i]].model == "Cirrina":
                         if Gs[cls2[i]].compiler_flag == "O0":
                             h1_id = cls2[i]
                         elif Gs[cls2[i]].compiler_flag == "O1":
                             h2_id = cls2[i]
                         elif Gs[cls2[i]].compiler_flag == "O2":
                             h3_id = cls2[i]
                         elif Gs[cls2[i]].compiler_flag == "O3":
                             h4_id = cls2[i]

                     elif Gs[cls2[i]].model == "Gemini_trp":
                         if Gs[cls2[i]].compiler_flag == "O0":
                             k1_id = cls2[i]
                         elif Gs[cls2[i]].compiler_flag == "O1":
                             k2_id = cls2[i]
                         elif Gs[cls2[i]].compiler_flag == "O2":
                             k3_id = cls2[i]
                         elif Gs[cls2[i]].compiler_flag == "O3":
                             k4_id = cls2[i]

                     elif Gs[cls2[i]].model == "Palmtree_trp":
                         if Gs[cls2[i]].compiler_flag == "O0":
                             l1_id = cls2[i]
                         elif Gs[cls2[i]].compiler_flag == "O1":
                             l2_id = cls2[i]
                         elif Gs[cls2[i]].compiler_flag == "O2":
                             l3_id = cls2[i]
                         elif Gs[cls2[i]].compiler_flag == "O3":
                             l4_id = cls2[i]



                pos_ids.append( (c4_id, c1_id, h1_id, g4_id, g1_id, k1_id, p4_id, p1_id, l1_id) )
                #neg_ids.append( (c3_id, h1_id, c1_id, g3_id, k1_id, g1_id, p3_id, l1_id, p1_id) )

                #print (f"[+] func: c2 -> {Gs[c2_id].function}, c1 -> {Gs[c1_id].function}, h1 -> {Gs[h1_id].function}, g2 -> {Gs[g2_id].function}, g1 -> {Gs[g1_id].function}, k1 -> {Gs[k1_id].function}, p2 -> {Gs[p2_id].function}, p1 -> {Gs[p1_id].function}, l1 -> {Gs[l1_id].function}")

                ids.append((c4_id, c1_id, h1_id, g4_id, g1_id, k1_id, p4_id, p1_id, l1_id))
                #ids.append((c3_id, h1_id, c1_id, g3_id, k1_id, g1_id, p3_id, l1_id, p1_id))

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

    maxN4 = 0
    maxN5 = 0
    maxN6 = 0

    maxN7 = 0
    maxN8 = 0
    maxN9 = 0

    for pair in pos_ids:
        maxN1 = max(maxN1, Gs[pair[0]].node_num)
        maxN2 = max(maxN2, Gs[pair[1]].node_num)
        maxN3 = max(maxN3, Gs[pair[2]].node_num)

        feature_dim1 = len(Gs[pair[0]].features[0])

        maxN4 = max(maxN4, Gs[pair[3]].node_num)
        maxN5 = max(maxN5, Gs[pair[4]].node_num)
        maxN6 = max(maxN6, Gs[pair[5]].node_num)

        feature_dim2 = len(Gs[pair[3]].features[0])

        maxN7 = max(maxN7, Gs[pair[6]].node_num)
        maxN8 = max(maxN8, Gs[pair[7]].node_num)
        maxN9 = max(maxN9, Gs[pair[8]].node_num)

        feature_dim3 = len(Gs[pair[6]].features[0])
        
    for pair in neg_ids:
        maxN1 = max(maxN1, Gs[pair[0]].node_num)
        maxN2 = max(maxN2, Gs[pair[1]].node_num)
        maxN3 = max(maxN3, Gs[pair[2]].node_num)

        feature_dim1 = len(Gs[pair[0]].features[0])

        maxN4 = max(maxN4, Gs[pair[3]].node_num)
        maxN5 = max(maxN5, Gs[pair[4]].node_num)
        maxN6 = max(maxN6, Gs[pair[5]].node_num)

        feature_dim2 = len(Gs[pair[3]].features[0])

        maxN7 = max(maxN7, Gs[pair[6]].node_num)
        maxN8 = max(maxN8, Gs[pair[7]].node_num)
        maxN9 = max(maxN9, Gs[pair[8]].node_num)

        feature_dim3 = len(Gs[pair[6]].features[0])

        
    #print (f"fdim1 -> {feature_dim1}, fdim2 -> {feature_dim2}")    
    X1_input = np.zeros((M, maxN1, feature_dim1))
    X2_input = np.zeros((M, maxN2, feature_dim1))
    X3_input = np.zeros((M, maxN3, feature_dim1))
    node1_mask = np.zeros((M, maxN1, maxN1))
    node2_mask = np.zeros((M, maxN2, maxN2))
    node3_mask = np.zeros((M, maxN3, maxN3))
    y1_input = np.zeros((M))

    X4_input = np.zeros((M, maxN4, feature_dim2))
    X5_input = np.zeros((M, maxN5, feature_dim2))
    X6_input = np.zeros((M, maxN6, feature_dim2))
    node4_mask = np.zeros((M, maxN4, maxN4))
    node5_mask = np.zeros((M, maxN5, maxN5))
    node6_mask = np.zeros((M, maxN6, maxN6))
    y2_input = np.zeros((M))

    X7_input = np.zeros((M, maxN7, feature_dim3))
    X8_input = np.zeros((M, maxN8, feature_dim3))
    X9_input = np.zeros((M, maxN9, feature_dim3))
    node7_mask = np.zeros((M, maxN7, maxN7))
    node8_mask = np.zeros((M, maxN8, maxN8))
    node9_mask = np.zeros((M, maxN9, maxN9))
    y3_input = np.zeros((M))
    
    for i in range(M_pos):
        y1_input[i] = 1
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

        y2_input[i] = 1
        g1 = Gs[pos_ids[i][3]]
        g2 = Gs[pos_ids[i][4]]
        g3 = Gs[pos_ids[i][5]]
        for u in range(g1.node_num):
            X4_input[i, u, :] = np.array( g1.features[u] )
            for v in g1.succs[u]:
                node4_mask[i, u, v] = 1
        for u in range(g2.node_num):
            X5_input[i, u, :] = np.array( g2.features[u] )
            for v in g2.succs[u]:
                node5_mask[i, u, v] = 1
        for u in range(g3.node_num):
            X6_input[i, u, :] = np.array( g3.features[u] )
            for v in g3.succs[u]:
                node6_mask[i, u, v] = 1

        y3_input[i] = 1
        g1 = Gs[pos_ids[i][6]]
        g2 = Gs[pos_ids[i][7]]
        g3 = Gs[pos_ids[i][8]]
        for u in range(g1.node_num):
            X7_input[i, u, :] = np.array( g1.features[u] )
            for v in g1.succs[u]:
                node7_mask[i, u, v] = 1
        for u in range(g2.node_num):
            X8_input[i, u, :] = np.array( g2.features[u] )
            for v in g2.succs[u]:
                node8_mask[i, u, v] = 1
        for u in range(g3.node_num):
            X9_input[i, u, :] = np.array( g3.features[u] )
            for v in g3.succs[u]:
                node9_mask[i, u, v] = 1

                
                
    for i in range(M_pos, M_pos + M_neg):
        y1_input[i] = -1
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

        y2_input[i] = -1
        g1 = Gs[neg_ids[i-M_pos][3]]
        g2 = Gs[neg_ids[i-M_pos][4]]
        g3 = Gs[neg_ids[i-M_pos][5]]
        for u in range(g1.node_num):
            X4_input[i, u, :] = np.array( g1.features[u] )
            for v in g1.succs[u]:
                node4_mask[i, u, v] = 1
        for u in range(g2.node_num):
            X5_input[i, u, :] = np.array( g2.features[u] )
            for v in g2.succs[u]:
                node5_mask[i, u, v] = 1
        for u in range(g3.node_num):
            X6_input[i, u, :] = np.array( g3.features[u] )
            for v in g3.succs[u]:
                node6_mask[i, u, v] = 1

        y3_input[i] = -1
        g1 = Gs[neg_ids[i-M_pos][6]]
        g2 = Gs[neg_ids[i-M_pos][7]]
        g3 = Gs[neg_ids[i-M_pos][8]]
        for u in range(g1.node_num):
            X7_input[i, u, :] = np.array( g1.features[u] )
            for v in g1.succs[u]:
                node7_mask[i, u, v] = 1
        for u in range(g2.node_num):
            X8_input[i, u, :] = np.array( g2.features[u] )
            for v in g2.succs[u]:
                node8_mask[i, u, v] = 1
        for u in range(g3.node_num):
            X9_input[i, u, :] = np.array( g3.features[u] )
            for v in g3.succs[u]:
                node9_mask[i, u, v] = 1

                
    if output_id:
        return X1_input,X2_input,X3_input,node1_mask,node2_mask,node3_mask,y1_input,X4_input,X5_input,X6_input,node4_mask,node5_mask,node6_mask,y2_input, \
               X7_input,X8_input,X9_input,node7_mask,node8_mask,node9_mask,y3_input,pos_ids,neg_ids,ids
    else:
        return X1_input,X2_input,X3_input,node1_mask,node2_mask,node3_mask,y1_input,X4_input,X5_input,X6_input,node4_mask,node5_mask,node6_mask,y2_input, \
               X7_input,X8_input,X9_input,node7_mask,node8_mask,node9_mask,y3_input

    
def train_epoch(model, graphs, classes, batch_size, load_data=None):
    if load_data is None:
        epoch_data = generate_epoch_pair(graphs, classes, batch_size)
    else:
        epoch_data = load_data

    perm = np.random.permutation(len(epoch_data))

    cum_loss = 0.0
    for index in perm:
        cur_data = epoch_data[index]
        X1, X2, X3, mask1, mask2, mask3, y1, X4, X5, X6, mask4, mask5, mask6, y2, X7, X8, X9, mask7, mask8, mask9, y3 = cur_data
        loss = model.train(X1, X2, X3, mask1, mask2, mask3, y1, X4, X5, X6, mask4, mask5, mask6, y2, X7, X8, X9, mask7, mask8, mask9, y3)
        cum_loss += loss

    return cum_loss / len(perm)


def get_auc_epoch(model, graphs, classes, batch_size, cls_st, load_data=None, ids=None, output_id=False, mode=None):

    epoch_data, ids = load_data, ids

    diff_lst_1 = {}
    diff_lst_2 = {}
    y1_lst = {}
    diff_lst_3 = {}
    diff_lst_4 = {}
    y2_lst = {}
    diff_lst_5 = {}
    diff_lst_6 = {}
    y3_lst = {}
    
    #if mode == "Testing":
    for i in range(0, len(epoch_data)):

        X1, X2, X3, m1, m2, m3, y1, X4, X5, X6, m4, m5, m6, y2, X7, X8, X9, m7, m8, m9, y3  = epoch_data[i]
        diff1 = model.calc_diff1(X1, X2, X3, m1, m2, m3)
        diff2 = model.calc_diff2(X1, X2, X3, m1, m2, m3)
        diff3 = model.calc_diff3(X4, X5, X6, m4, m5, m6)
        diff4 = model.calc_diff4(X4, X5, X6, m4, m5, m6)
        diff5 = model.calc_diff5(X7, X8, X9, m7, m8, m9)
        diff6 = model.calc_diff6(X7, X8, X9, m7, m8, m9)

        diff_lst_1[i] = diff1.tolist()
        diff_lst_2[i] = diff2.tolist()
        diff_lst_3[i] = diff3.tolist()
        diff_lst_4[i] = diff4.tolist()
        diff_lst_5[i] = diff5.tolist()
        diff_lst_6[i] = diff6.tolist()
        y1_lst[i] = y1.tolist()
        y2_lst[i] = y2.tolist()
        y3_lst[i] = y3.tolist()
        
    return diff_lst_1, diff_lst_2, y1_lst, diff_lst_3, diff_lst_4, y2_lst, diff_lst_5, diff_lst_6, y3_lst
        
       
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
          #if (-1 * diff[0]) > (-1 * diff[3]):
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
    
    #print ("diff: {}, truth: {}".format(diff, truth))

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
