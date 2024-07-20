import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from siamese_nonuplet import graphnn
import json
from settings import *

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
        self.model = ""
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
                #print(f"line -> {g_info['fname']}, {g_info['model']}")
                label = FUNC_NAME_DICT[g_info['fname']]
                classes[label].append(len(graphs))
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




def generate_epoch_pair_3(Gs, classes, M, output_id = False, load_id = None, mode = None):
    epoch_data = []
    id_data = []  
    ids = []

    if load_id is None:
        st = 0
        while st < len(classes) : #len(Gs):
            if output_id:
                X1, X2, X3, m1, m2, m3, y1, X4, X5, X6, m4, m5, m6, y2, X7, X8, X9, m7, m8, m9, y3, pos_id, neg_id, gid = get_pair_3(Gs, classes,
                        M, st=st, output_id=True, mode=mode)
                id_data.append( (pos_id, neg_id) )
                ids = ids + gid # (pos_id, neg_id) )
            else:
                X1, X2, X3, m1, m2, m3, y1, X4, X5, X6, m4, m5, m6, y2, X7, X8, X9, m7, m8, m9, y3 = get_pair_3(Gs, classes, M, st=st, mode=mode)
            epoch_data.append( (X1,X2,X3,m1,m2,m3,y1, X4,X5,X6,m4,m5,m6,y2, X7,X8,X9,m7,m8,m9,y3) )
            st += M
    else:   ## Load from previous id data
        id_data = load_id
        for id_pair in id_data:
            X1, X2, X3, m1, m2, m3, y1, X4, X5, X6, m4, m5, m6, y2, X7, X8, X9, m7, m8, m9, y3 = get_pair_3(Gs, classes, M, load_id=id_pair, mode=mode)
            epoch_data.append( (X1, X2, X3, m1, m2, m3, y1, X4, X5, X6, m4, m5, m6, y2, X7, X8, X9, m7, m8, m9, y3) )

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
                continue
               
            #if mode != "Testing":
            cls2 = np.random.randint(C)
            while (len(classes[cls2]) == 0) or (cls2 == cls_id):
                cls2 = np.random.randint(C)

            tot_g2 = len(classes[cls2])
            cls2 = classes[cls2]
             

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
            neg_ids.append( (c4_id, h1_id, c1_id, g4_id, k1_id, g1_id, p4_id, l1_id, p1_id) )
            

            ids.append((c4_id, c1_id, h1_id, g4_id, g1_id, k1_id, p4_id, p1_id, l1_id))
            ids.append((c4_id, h1_id, c1_id, g4_id, k1_id, g1_id, p4_id, l1_id, p1_id))

    else:
        #print ("pos : {}, neg : {}".format(load_id[0], load_id[1]))
        pos_ids = load_id[0]
        neg_ids = load_id[1]
        #return
    
        
    M_pos = len(pos_ids)
    M_neg = len(neg_ids)
    M = M_pos + M_neg

    #print ("pos : {}, neg : {}, M : {}".format(pos_ids, neg_ids, M))

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
        #print (f"pair -> {pair}, {Gs[pair[0]].model}, {Gs[pair[1]].function}, {Gs[pair[2]].function}, {Gs[pair[3]].model}, {Gs[pair[4]].function}, {Gs[pair[5]].function}")
        maxN1 = max(maxN1, Gs[pair[0]].node_num)
        maxN2 = max(maxN2, Gs[pair[1]].node_num)
        maxN3 = max(maxN3, Gs[pair[2]].node_num)

        maxN4 = max(maxN4, Gs[pair[3]].node_num)
        maxN5 = max(maxN5, Gs[pair[4]].node_num)
        maxN6 = max(maxN6, Gs[pair[5]].node_num)

        maxN7 = max(maxN7, Gs[pair[6]].node_num)
        maxN8 = max(maxN8, Gs[pair[7]].node_num)
        maxN9 = max(maxN9, Gs[pair[8]].node_num)
        
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
        epoch_data = generate_epoch_pair_3(graphs, classes, batch_size)
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
        
        X1, X2, X3, m1, m2, m3, y1, X4, X5, X6, m4, m5, m6, y2, X7, X8, X9, m7, m8, m9, y3  = epoch_data[i]
        diff1 = model.calc_diff1(X1, X2, X3, m1, m2, m3)
        diff2 = model.calc_diff2(X1, X2, X3, m1, m2, m3)
        diff3 = model.calc_diff3(X4, X5, X6, m4, m5, m6)
        diff4 = model.calc_diff4(X4, X5, X6, m4, m5, m6)
        diff5 = model.calc_diff5(X7, X8, X9, m7, m8, m9)
        diff6 = model.calc_diff6(X7, X8, X9, m7, m8, m9)

        tot_diff += list(diff1)
        tot_truth += list(y1 > 0)

        
        pred1 = False
        pred2 = False         
        
        if ((-1 * diff1[0]) + (-1 * diff3[0]) + (-1 * diff5[0])) > ((-1 * diff2[0]) + (-1 * diff4[0]) + (-1 * diff6[0])):
            acc = acc + 1
            acc_1 = acc_1 + 1
            pred1 = True
        
        if ((-1 * diff2[1]) + (-1 * diff4[1]) + (-1 * diff6[1])) > ((-1 * diff1[1]) + (-1 * diff3[1]) + (-1 * diff5[1])):
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
            X1, X2, X3, m1, m2, m3, y1, X4, X5, X6, m4, m5, m6, y2, X7, X8, X9, m7, m8, m9, y3  = cur_data
            diff1 = model.calc_diff1(X1, X2, X3, m1, m2, m3)

            tot_diff += list(diff1)
            tot_truth += list(y1 > 0)
    
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
