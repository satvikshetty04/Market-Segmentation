import pandas as pd
import sys
import igraph as ig
import numpy as np
from scipy import spatial

if len(sys.argv) != 2:
    print('sac1.py <alpha value>')
    sys.exit(1)

alpha = float(sys.argv[1])

attributes = pd.read_csv('data/fb_caltech_small_attrlist.csv').as_matrix()
g = ig.Graph.Read_Edgelist('data/fb_caltech_small_edgelist.txt', directed=False)
#print(g.summary())

membership = list(range(g.vcount()))
#print(membership)


#To calculate the cosine similarity
def cos_sim(i, j, values, attr):
    #indices = [k for k, x in enumerate(values) if x == j]
    arr = np.array(values)
    indices = np.where(arr == j)[0]
    similarity = 0
    for each in indices:
        similarity += spatial.distance.cosine(attr[i],attr[each])
    similarity /= len(indices)
    return similarity


#Phase 1 of SAC1
def phase1(g, membership):
    print("Executing phase 1...")
    v = len(set(membership))
    for k in range(15):
        gain = 0
        membership_original = membership.copy()
        for i in range(v):
            max_pos_gain = 0.0
            max_index = -1
            for j in range(v):
                if i == j or membership[i] == membership[j]:
                    continue
                membership_copy = membership.copy()
                q_newman = g.modularity(membership)
                membership_copy[i] = membership[j]
                q_newman_updated = g.modularity(membership_copy)
                delta_q_newman = q_newman_updated - q_newman
                delta_q_attr = cos_sim(i, membership[j], membership, attributes)
                delta_q = alpha * delta_q_newman + float(1 - alpha) * delta_q_attr

                if delta_q > max_pos_gain:
                    max_pos_gain = delta_q
                    max_index = j

            if max_pos_gain > 0 and max_index>=0:
                membership[i] = membership[max_index]
            gain = max(gain, max_pos_gain)

        if gain <= 0:
            break
    return membership


#Phase 2 of SAC1
def phase2(communities):
    print("Executing phase 2...")
    g.contract_vertices(membership)
    g.simplify(multiple = True, loops = True)
    communities = phase1(g, communities)
    return communities

#SAC1 Algorithm
membership = phase1(g, membership)
communities = phase2(membership)

dict_communities = {}
for i in range(324):
    if dict_communities.get(communities[i]):
        dict_communities[communities[i]].append(i)
    else:
        dict_communities[communities[i]] = [i]

#To create the output file
file_name = "communities_"+str(sys.argv[1])+".txt"
output_file = open(file_name, "w+")
for d in dict_communities.values():
    s = str(d)
    output_file.write("%s" % s[1:-1])
    output_file.write("\n")
output_file.close()