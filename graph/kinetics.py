import numpy as np
import sys
from . import hypergraph_utils as hgut
sys.path.extend(['../'])
from graph import tools
import networkx as nx

# Joint index:
# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8,  "RHip"},
# {9,  "RKnee"},
# {10, "RAnkle"},
# {11, "LHip"},
# {12, "LKnee"},
# {13, "LAnkle"},
# {14, "REye"},
# {15, "LEye"},
# {16, "REar"},
# {17, "LEar"},

# Edge format: (origin, neighbor)
num_node = 18
self_link = [(i, i) for i in range(num_node)]
inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
          (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
          (16, 14)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        
        
        self.A_binary = tools.edge2mat(neighbor, num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

       
        
        


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            X = tools.edge2mat(neighbor, num_node) #图邻接矩阵
            #A2 = tools.get_spatial_graph(num_node, self_link, inward, outward)[2]+tools.get_spatial_graph(num_node, self_link, inward, outward)[1]
            A_binary = tools.edge2mat(neighbor, num_node)
          #  A5 = tools.normalize_adjacency_matrix(A_binary + 2*np.eye(num_node))
            A5=X
            A5 = hgut.generate_G_from_H(A5)
                    
            A3 = tools.gen_knn_hg(X, n_neighbors=3)

            
            A4 = tools.gen_clustering_hg(X, n_clusters=4)  
            
            A4 = hgut.generate_G_from_H(A4)
           # A1=np.expand_dims(A1,axis=0)
            #A2=np.expand_dims(A2,axis=0)
            A3=np.expand_dims(A3,axis=0)
            A4=np.expand_dims(A4,axis=0)
            A5=np.expand_dims(A5,axis=0)
            #A = np.concatenate((A4,A3,A2,A5))
            A = np.concatenate((A4,A3,A5))
        else:
            raise ValueError()
        return A
