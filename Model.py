import torch,os
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device('cpu')

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super(GraphConvolution, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).to(device))
        self.normalize_embedding = normalize_embedding
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).to(device))
        else:
            self.bias = None

    def forward(self, x, adj):# x: [batchsize*max_num_nodes*num_node_attributes] adj: [batchsize*max_num_nodes*max_num_nodes]
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)# Matrix multiplication  y: [batchsize*max_num_nodes*num_node_attributes]
        if self.add_self:
            y += x
        y = torch.matmul(y,self.weight)# self.weight: ?[num_node_attributes*hiddendim]   out y: [batchsize*max_num_nodes*hiddendim]
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y# [batchsize*max_num_nodes*batchsize]
        
class GCGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GCGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_1, self.convolution_mid, self.convolution_end = self.convolution_lys(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.prediction_lys(self.pred_input_dim, pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)

        for item in self.modules():# ：conv_1、convolution_mid、convolution_end；  act：ReLU、pred_model：Linear
            if isinstance(item, GraphConvolution):
                item.weight.data = init.xavier_normal(item.weight.data, gain=nn.init.calculate_gain('relu'))
                if item.bias is not None:
                    item.bias.data = init.constant(item.bias.data, 0.0)
    def convolution_lys(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_1 = GraphConvolution(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        convolution_mid = nn.ModuleList(
                [GraphConvolution(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias) 
                 for i in range(num_layers-2)])
        convolution_end = GraphConvolution(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        return conv_1, convolution_mid, convolution_end

    def prediction_lys(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_lys = []
            for pred_dim in pred_hidden_dims:
                pred_lys.append(nn.Linear(pred_input_dim, pred_dim))
                pred_lys.append(self.act)
                pred_input_dim = pred_dim
            pred_lys.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_lys)
        return pred_model

    def mask_operation(self, max_nodes, batch_num_nodes): 
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).to(device)

    def batch_normal(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1]).to(device)
        return bn_module(x)

    def GCN_forward(self, x, adj, conv_1, convolution_mid, convolution_end, embedding_mask=None):# GCN_DP 中forward处调用

        x = conv_1(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.batch_normal(x)
        fea_all = [x]

        for i in range(len(convolution_mid)):
            x = convolution_mid[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.batch_normal(x)
            fea_all.append(x)
        x = convolution_end(x,adj)
        fea_all.append(x)
        x_tensor = torch.cat(fea_all, dim=2)
        x_tensor_lst1,x_tensor_lst2=[],[]
        for i,x_tensor_i in enumerate(x_tensor):#
            x_tensor_lst1.append(x_tensor_i.detach().cpu().numpy())
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask 
        for j,x_tensor_j in enumerate(x_tensor):
            x_tensor_lst2.append(x_tensor_j.detach().cpu().numpy())
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.mask_operation(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        x = self.conv_1(x, adj)
        x = self.act(x)# Activation function
        if self.bn:
            x = self.batch_normal(x)# batch normalization
        out_all = []
        out, _ = torch.max(x, dim=1)#
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.convolution_mid[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.batch_normal(x)
            out,_ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.convolution_end(x,adj)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)#
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, type='softmax'):
        if type == 'softmax':
            return F.cross_entropy(pred, label, reduction='mean')
        elif type == 'margin':
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().to(device)
            label_onehot.scatter_(1, label.view(-1,1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)
            
class GCN_DP(GCGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            trans_hidden_dim, trans_ratio=0.25, trans_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
            trans_input_dim=-1, args=None):
        super(GCN_DP, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.trans_ent = True

        self.conv_1_after_pool = nn.ModuleList()
        self.convolution_mid_after_pool = nn.ModuleList()
        self.convolution_end_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            conv_12, convolution_mid2, convolution_end2 = self.convolution_lys(
                    self.pred_input_dim, hidden_dim, embedding_dim, num_layers, 
                    add_self, normalize=True, dropout=dropout)
            self.conv_1_after_pool.append(conv_12)
            self.convolution_mid_after_pool.append(convolution_mid2)
            self.convolution_end_after_pool.append(convolution_end2)

        trans_dims = []
        if trans_num_layers == -1:
            trans_num_layers = num_layers
        if trans_input_dim == -1:
            trans_input_dim = input_dim

        self.trans_conv_1_modules = nn.ModuleList()
        self.trans_convolution_mid_modules = nn.ModuleList()
        self.trans_convolution_end_modules = nn.ModuleList()
        self.trans_pred_modules = nn.ModuleList()
        trans_dim = int(max_num_nodes * trans_ratio)
        for i in range(num_pooling):
            trans_dims.append(trans_dim)
            trans_conv_1, trans_convolution_mid, trans_convolution_end = self.convolution_lys(
                    trans_input_dim, trans_hidden_dim, trans_dim, trans_num_layers, 
                    add_self, normalize=True)
            trans_pred_input_dim = trans_hidden_dim * (num_layers - 1) + trans_dim if concat else trans_dim
            trans_pred = self.prediction_lys(trans_pred_input_dim, [], trans_dim, num_aggs=1)

            # pooling layer
            trans_input_dim = self.pred_input_dim
            trans_dim = int(trans_dim * trans_ratio)
            if trans_dim==0:# prevent trans_dim being 0 
                trans_dim=1

            self.trans_conv_1_modules.append(trans_conv_1)
            self.trans_convolution_mid_modules.append(trans_convolution_mid)
            self.trans_convolution_end_modules.append(trans_convolution_end)
            self.trans_pred_modules.append(trans_pred)

        self.pred_model = self.prediction_lys(self.pred_input_dim * (num_pooling+1), pred_hidden_dims, 
                label_dim, num_aggs=self.num_aggs)

        for item in self.modules():
            if isinstance(item, GraphConvolution):
                item.weight.data = init.xavier_normal(item.weight.data, gain=nn.init.calculate_gain('relu'))
                if item.bias is not None:
                    item.bias.data = init.constant(item.bias.data, 0.0)
    # x is ['feas'],shape(batch_size*max node number*node feature dimension)
    def forward(self, x, adj, batch_num_nodes, feat_pca, **kwargs):
        if 'trans_x' in kwargs:
            x_a = kwargs['trans_x']
        else:
            x_a = x
        
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.mask_operation(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None
        embedding_mask_lst=[]#
        for i,embedding_mask_i in enumerate(embedding_mask):#
            embedding_mask_lst.append(embedding_mask_i.detach().cpu().numpy())
        out_all = []

        # [核心GCN逻辑] - 计算第一层GCN嵌入
        embedding_tensor = self.GCN_forward(x, adj,
                self.conv_1, self.convolution_mid, self.convolution_end, embedding_mask)
        
        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        # [核心DiffPool逻辑] - 进行可微分池化
        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.mask_operation(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None
            self.trans_tensor = self.GCN_forward(x_a, adj, 
                    self.trans_conv_1_modules[i], self.trans_convolution_mid_modules[i], self.trans_convolution_end_modules[i],
                    embedding_mask)
            self.trans_tensor = nn.Softmax(dim=-1)(self.trans_pred_modules[i](self.trans_tensor))
            if embedding_mask is not None:
                self.trans_tensor = self.trans_tensor * embedding_mask

            x = torch.matmul(torch.transpose(self.trans_tensor, 1, 2), embedding_tensor) 
            adj = torch.transpose(self.trans_tensor, 1, 2) @ adj @ self.trans_tensor
            x_a = x
        
            embedding_tensor = self.GCN_forward(x, adj, 
                    self.conv_1_after_pool[i], self.convolution_mid_after_pool[i],
                    self.convolution_end_after_pool[i])

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        # 得到GCN部分的嵌入向量
        if self.concat:
            output_gcn = torch.cat(out_all, dim=1)
        else:
            output_gcn = out

        # ----------- 新增：在這裡與PCA特徵融合 -----------
        # 確保 feat_pca 维度匹配 batch size
        # feat_pca 應為 [batch_size, pca_feature_dim]
        emb_fused = torch.cat([output_gcn, feat_pca], dim=1)
        
        # 移除原有的 ypred = self.pred_model(output)
        # 直接返回融合後的嵌入向量
        return emb_fused
    
    def extract_features(self, x, adj, batch_num_nodes=None):
        """
        Extract feature matrix from input graph data.

        Args:
            x (torch.Tensor): Node features with shape [batch_size, max_num_nodes, node_feature_dim].
            adj (torch.Tensor): Adjacency matrices with shape [batch_size, max_num_nodes, max_num_nodes].
            batch_num_nodes (list): List of numbers of nodes in each graph.

        Returns:
            torch.Tensor: Extracted feature matrix.
        """
        # Switch to evaluation mode
        self.eval()
        
        # Disable gradient computation to save memory and speed up computation
        with torch.no_grad():
            # Forward pass to get features
            _, output = self.forward(x, adj, batch_num_nodes)
        
        return output

    def final_loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        eps = 1e-7
        loss = super(GCN_DP, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.trans_tensor @ torch.transpose(self.trans_tensor, 1, 2) 
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop-1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).to(device))
            #torch.cuda.empty_cache()
            adj = adj.to(device); pred_adj = pred_adj.to(device)
            self.link_loss = -adj * torch.log(pred_adj+eps) - (1-adj) * torch.log(1-pred_adj+eps)
            #torch.cuda.empty_cache()
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print('!loss without mask!')
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.mask_operation(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[(1-adj_mask).bool()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            return loss + self.link_loss
        return loss
    
    def Compsimilarity(self, sample1, sample2):
        # Switch to evaluation mode
        self.eval()
    
        # Disable gradient computation to save memory and speed up computation
        with torch.no_grad():
            # Extract features for both samples
            adj1, h0_1, batch_num_nodes1 = sample1
            adj2, h0_2, batch_num_nodes2 = sample2
        
            # Convert to tensors
            adj1 = torch.tensor(adj1, dtype=torch.float32).unsqueeze(0).to(device)
            h0_1 = torch.tensor(h0_1, dtype=torch.float32).unsqueeze(0).to(device)
            adj2 = torch.tensor(adj2, dtype=torch.float32).unsqueeze(0).to(device)
            h0_2 = torch.tensor(h0_2, dtype=torch.float32).unsqueeze(0).to(device)
        
            # Ensure batch_num_nodes is a tensor with batch dimension
            batch_num_nodes1 = torch.tensor([batch_num_nodes1], dtype=torch.int).to(device)
            batch_num_nodes2 = torch.tensor([batch_num_nodes2], dtype=torch.int).to(device)
            # Extract features
            output1 = self.extract_features(h0_1, adj1, batch_num_nodes1)
            output2 = self.extract_features(h0_2, adj2, batch_num_nodes2)
        
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(output1, output2, dim=1)
        
            return similarity.item()
        
        
class TerrainSimilarityModel(nn.Module):
    def __init__(self, gcn_dp_backbone, mlp_hidden_dim=32, dropout=0.5):
        """
        頂層模型，用於比較一對地形的相似度。
        :param gcn_dp_backbone: 預先實例化的、已修改的 GCN_DP 模型。
        :param mlp_hidden_dim: 最終融合MLP的隱藏層維度。
        """
        super(TerrainSimilarityModel, self).__init__()
        self.gcn_dp_backbone = gcn_dp_backbone
        
        # 餘弦相似度計算層
        self.similarity_function = nn.CosineSimilarity(dim=1,eps=1e-6)

        # 分數級融合 MLP (方案二)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden_dim), # 輸入維度為2: [Sim_Struct_PCA, Sim_DTW]
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden_dim, 1) # 輸出維度為1，代表最終的相似度 logits
        )

    def forward(self, data1, data2):
        """
        處理一對數據。data1 和 data2 是包含各自所需信息的字典。
        """
        # --- 處理第一個樣本 ---
        x1 = data1['feas']
        adj1 = data1['adj']
        batch_num_nodes1 = data1['num_nodes']
        feat_pca1 = data1['feat_pca']
        
        # --- 處理第二個樣本 ---
        x2 = data2['feas']
        adj2 = data2['adj']
        batch_num_nodes2 = data2['num_nodes']
        feat_pca2 = data2['feat_pca']

        # --- DTW 相似度 ---
        sim_dtw = data1['sim_dtw'] # 假設 DTW 分數隨第一個樣本傳入

        # 1. 使用骨幹網絡提取融合嵌入
        emb_fused1 = self.gcn_dp_backbone(x1, adj1, batch_num_nodes1, feat_pca1)
        emb_fused2 = self.gcn_dp_backbone(x2, adj2, batch_num_nodes2, feat_pca2)

        # 2. 計算 GCN-PCA 結構相似度
        sim_struct_pca = self.similarity_function(emb_fused1, emb_fused2).unsqueeze(1) # 保持維度 [B, 1]

        # 3. 拼接兩個相似度分數
        # 確保 sim_dtw 也是 [B, 1] 的形狀
        score_input = torch.cat([sim_struct_pca, sim_dtw], dim=1)

        # 4. 輸入到融合MLP中得到 logits
        logits = self.fusion_mlp(score_input)

        # 5. 通過 Sigmoid 函數得到最終的相似概率
        probability = torch.sigmoid(logits)

        return probability.squeeze(1)