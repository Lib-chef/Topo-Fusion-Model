import networkx as nx
import numpy as np
import torch,util
import torch.utils.data

class GraphConstruction(torch.utils.data.Dataset):
    def __init__(self, G_list, features='default', normalize=True, trans_feat='default', max_num_nodes=0):        
        self.adj_all = []
        self.adj_edgeWeight_all=[]
        self.len_all = []
        self.feature_all = []
        self.label_all = []
        self.inFID_nodeNum_all=[]
        self.inFID_graphIndicator_all=[]
        self.node_coord_all=[]
        self.trans_feat_all = []

        if max_num_nodes == 0:
            self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        else:
            self.max_num_nodes = max_num_nodes

        self.feat_dim = util.node_infor(G_list[0])[0]['feat'].shape[0]

        for G in G_list:
            adj = nx.linalg.graphmatrix.adjacency_matrix(G).todense()  
            adj = np.array(adj)            
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            self.adj_all.append(adj)
            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])
            if 'inFID_nodeNum' in G.graph:
                self.inFID_nodeNum_all.append(G.graph['inFID_nodeNum'])
            if 'inFID_graphIndicator' in G.graph:
                self.inFID_graphIndicator_all.append(G.graph['inFID_graphIndicator'])                
            if 'nodes_coords' in G.graph:
                self.node_coord_all.append(G.graph['nodes_coords'])

            if features == 'default':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(G.nodes()):
                    f[i,:] = util.node_infor(G)[u]['feat']
                self.feature_all.append(f)
            elif features == 'id':
                self.feature_all.append(np.identity(self.max_num_nodes))
            elif features == 'deg-num':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()], 0),
                                      axis=1)
                self.feature_all.append(degs)
            elif features == 'deg':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>self.max_deg] = self.max_deg
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                feat = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(util.node_s(G)):
                    f[i,:] = util.node_infor(G)[u]['feat']

                feat = np.concatenate((feat, f), axis=1)

                self.feature_all.append(feat)
            elif features == 'struct':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>10] = 10
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                degs = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(np.pad(clusterings, 
                                                    [0, self.max_num_nodes - G.number_of_nodes()],
                                                    'constant'),
                                             axis=1)
                g_feat = np.hstack([degs, clusterings])
                if 'feat' in util.node_infor(G)[0]:
                    node_feas = np.array([util.node_infor(G)[i]['feat'] for i in range(G.number_of_nodes())])
                    node_feas = np.pad(node_feas, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                                        'constant')
                    g_feat = np.hstack([g_feat, node_feas])

                self.feature_all.append(g_feat)

            if trans_feat == 'id':
                self.trans_feat_all.append(
                        np.hstack((np.identity(self.max_num_nodes), self.feature_all[-1])) )
            else:
                self.trans_feat_all.append(self.feature_all[-1])
            
        self.feat_dim = self.feature_all[0].shape[1]
        self.trans_feat_dim = self.trans_feat_all[0].shape[1]

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        if len(self.inFID_nodeNum_all)>0 and len(self.node_coord_all)>0 and len(self.inFID_graphIndicator_all)>0:
            return {'adj':adj_padded,
                'feas':self.feature_all[idx].copy(),
                'label':self.label_all[idx],
                'num_nodes': num_nodes,
                'trans_feas':self.trans_feat_all[idx].copy(),
                'inFIDs_nodesNum':self.inFID_nodeNum_all[idx],
                'inFID_graphIndicator':self.inFID_graphIndicator_all[idx],
                'nodes_coords':self.node_coord_all[idx]
                }
        elif len(self.inFID_nodeNum_all)>0 and len(self.inFID_graphIndicator_all)>0:
            return {'adj':adj_padded,
                'feas':self.feature_all[idx].copy(),
                'label':self.label_all[idx],
                'num_nodes': num_nodes,
                'trans_feas':self.trans_feat_all[idx].copy(),
                'inFIDs_nodesNum':self.inFID_nodeNum_all[idx],
                'inFID_graphIndicator':self.inFID_graphIndicator_all[idx],
                }
        elif len(self.inFID_graphIndicator_all)>0:
            return {'adj':adj_padded,
                'feas':self.feature_all[idx].copy(),
                'label':self.label_all[idx],
                'num_nodes': num_nodes,
                'trans_feas':self.trans_feat_all[idx].copy(),
                # 'inFIDs_nodesNum':self.inFID_nodeNum_all[idx],
                'inFID_graphIndicator':self.inFID_graphIndicator_all[idx],
                }
        elif len(self.node_coord_all)>0:
            return {'adj':adj_padded,
                'feas':self.feature_all[idx].copy(),
                'label':self.label_all[idx],
                'num_nodes': num_nodes,
                'trans_feas':self.trans_feat_all[idx].copy(),
                'nodes_coords':self.node_coord_all[idx]
                }
        else:                
            return {'adj':adj_padded,
                    'feas':self.feature_all[idx].copy(),
                    'label':self.label_all[idx],
                    'num_nodes': num_nodes,
                    'trans_feas':self.trans_feat_all[idx].copy()
                    }

