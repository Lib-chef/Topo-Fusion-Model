import networkx as nx
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix,classification_report
import torch.nn as nn
from torch.autograd import Variable
from PIL.Image import NONE
import matplotlib,torch,tensorboardX,argparse,os,warnings,itertools,time
import torch
from torch.utils.data import Dataset, DataLoader
import SplitData,Model,Load_GraphTxt,util
import produce.feat as OutFeas
from torch.optim.optimizer import  required
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = ""
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import util
import produce.feat as OutFeas
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
import re

device = torch.device('cpu')

def extract_pid_from_filename(file_name):
    match = re.search(r'(\d+)', file_name)
    return match.group(1) if match else "unknown"

class PairedDataset(Dataset):
    def __init__(self, pair_list, graph_map, pca_map, dtw_map):
        self.pair_list = pair_list
        self.graph_map = graph_map
        self.pca_map = pca_map
        self.dtw_map = dtw_map

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        pid1, pid2, label = self.pair_list[idx]

        data1 = self.graph_map[pid1].copy() 
        data2 = self.graph_map[pid2].copy()

        data1['feat_pca'] = torch.tensor(self.pca_map.get(pid1, []), dtype=torch.float)
        data2['feat_pca'] = torch.tensor(self.pca_map.get(pid2, []), dtype=torch.float)
        
        pair_key = tuple(sorted((pid1, pid2)))
        data1['sim_dtw'] = torch.tensor([self.dtw_map.get(pair_key, 0.0)], dtype=torch.float)
        
        return data1, data2, torch.tensor(label, dtype=torch.float)
    
def collate_paired_data(batch):

    data1_list, data2_list, label_list = zip(*batch)

    collated_data1 = default_collate(data1_list)
    collated_data2 = default_collate(data2_list)
    
    collated_labels = torch.stack(label_list, 0)

    return collated_data1, collated_data2, collated_labels

# draw confusion matrix
def plot_confusion_matrix_percent(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,confusionMatrixPngPath='./confusionMatrixPng.png'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    else:
        print()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    plt.savefig(confusionMatrixPngPath)
    plt.close()
    
def Eval(dataset_loader, model, args):
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (data1, data2, labels) in enumerate(dataset_loader):
            labels = labels.to(device)
            for key in data1: data1[key] = data1[key].to(device)
            for key in data2: data2[key] = data2[key].to(device)

            ypred_prob = model(data1, data2)
            loss = criterion(ypred_prob, labels)
            total_loss += loss.item()

            preds = (ypred_prob > 0.5).long()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(dataset_loader)
    accuracy = metrics.accuracy_score(all_labels, all_preds)
    f1 = metrics.f1_score(all_labels, all_preds)
    print(f"Validation: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}, F1-Score = {f1:.4f}")


def get_directory(args,name=None):
    if name is not None:
        a=1
    elif args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method
    if args.method == 'soft-trans':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name+='_Lr'+str(args.lr)
        name += '_ar' + str(args.trans_ratio)
        if args.linkpred:
            name += '_lp'
    else:
        name += '_l' + str(args.num_gc_layers)
    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if args.batch_size is not NONE:#  
        name+='_batchsize'+str(args.batch_size)
    if args.cross_val_current_num is not NONE:#  
        name+='_cross_val_num'+str(args.cross_val_current_num)
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    name=name+str(time.time())
    return 'results/'+'/'+args.bmname+'/'+ name + '.png'

def Train(dataset_loader, model, args, val_dataset_loader=None, start_epoch = 0, checkpoint=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("優化器狀態已成功加載。")

    criterion = nn.BCELoss()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join('AAAI_Training_results', f'training_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"模型檢查點將保存在: {save_dir}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss = 0
        all_labels = []
        all_preds = []

        for batch_idx, (data1, data2, labels) in enumerate(dataset_loader):
            model.zero_grad()
            
            labels = labels.to(device)
            for key in data1: data1[key] = data1[key].to(device)
            for key in data2: data2[key] = data2[key].to(device)

            ypred_prob = model(data1, data2)
            loss = criterion(ypred_prob, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            total_loss += loss.item()
            
            preds = (ypred_prob > 0.5).long()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        avg_loss = total_loss / len(dataset_loader)
        accuracy = metrics.accuracy_score(all_labels, all_preds)
        if (epoch + 1) % 50 == 0 or (epoch + 1) == args.num_epochs:
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Train Accuracy = {accuracy:.4f}")

        if val_dataset_loader is not None :
            if (epoch + 1) % 50 == 0 or (epoch + 1) == args.num_epochs:
                Eval(val_dataset_loader, model, args)

        if (epoch + 1) % 100 == 0 or (epoch + 1) == args.num_epochs:
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(state, checkpoint_path)
            print(f"--- 模型已保存至: {checkpoint_path} ---")

def draw_avg_my(x, decay=0.9):
    shadow = x[0]
    a = [shadow]
    for v in x[1:]:
        shadow -= (1-decay) * (shadow-v)
        a.append(shadow)
    return a

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')
    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--trans-ratio', dest='trans_ratio', type=float,
            help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', dest='num_pool', type=int,
            help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
            const=True, default=False,
            help='Whether link prediction side objective is used')

    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--cuda', dest='cuda', default='cpu',
                    help='Device to use. Default: CPU.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
            help='weight_decay.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to Train.')
    parser.add_argument('--current_epoch', dest='current_epoch', type=int,
            help='current_epoch of Train.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')



    parser.set_defaults(datadir='data',
                        logdir='log',
                        max_nodes=1500,
                        cuda='0',
                        feature_type='default',
                        lr=0.001,#  0.001
                        clip=2.0,
                        batch_size=50,
                        weight_decay=5e-4,
                        num_epochs=2000,
                        current_epoch=0, 
                        num_workers=0,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=5,
                        num_gc_layers=3,
                        dropout=0.5,
                        train_ratio=0.05,
                        num_pool=1,
                        linkpred=True,
                        cross_val_num=1,
                        cross_val_current_num=10
                       )
    return parser.parse_args()

def extract_pid_from_graph(G):
    if 'inFID_graphIndicator' in G.graph and G.graph['inFID_graphIndicator']:
        return G.graph['inFID_graphIndicator'][0]
    raise ValueError("無法從圖對象中提取 PID")

def transform_graph_to_input(G, max_nodes):
    adj = np.zeros((max_nodes, max_nodes))
    feas = np.zeros((max_nodes, G.nodes[next(iter(G.nodes()))]['feat'].shape[0]))
    
    node_map = {node: i for i, node in enumerate(G.nodes())}
    
    for u, v in G.edges():
        if u in node_map and v in node_map:
            u_idx, v_idx = node_map[u], node_map[v]
            adj[u_idx, v_idx] = 1
            adj[v_idx, u_idx] = 1 

    for node, data in G.nodes(data=True):
        if 'feat' in data and node in node_map:
            feas[node_map[node]] = data['feat']
            
    num_nodes = G.number_of_nodes()
    
    return {
        'adj': torch.from_numpy(adj).float(),
        'feas': torch.from_numpy(feas).float(),
        'num_nodes': torch.tensor(num_nodes).int(),
        'label': torch.tensor(G.graph['label']) if 'label' in G.graph else torch.tensor(0) # 單圖的標籤
    }

def load_all_data_and_pairs(config):

    print("步骤 1/4: 加载原始图数据...")
    graphs = Load_GraphTxt.Load_GraphTxt(config['datadir'], config['bmname'], max_nodes=config['max_nodes'])
    print(f"診斷信息：Load_GraphTxt 初始加載了 {len(graphs)} 個圖。")
    print("步骤 1.1: 生成节点特征...")
    feat_type = 'node-label' 
    if feat_type != 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('使用节点自身特征 (node features)。')
        input_dim = graphs[0].graph['feat_dim']
    elif feat_type == 'node-label' and 'label' in util.node_infor(graphs[0])[next(iter(graphs[0].nodes()))]:
        print('节点标签作为特征 (node labels)。')
        for G in graphs:
            for u in G.nodes():
                util.node_infor(G)[u]['feat'] = np.array(util.node_infor(G)[u]['label'])
        input_dim = util.node_infor(graphs[0])[next(iter(graphs[0].nodes()))]['feat'].shape[0]
    else:
        print('常数标签作为特征 (constant labels)。')
        featgen_const = OutFeas.ConstFeatureGen(np.ones(config['input_dim'], dtype=float))
        for G in graphs:
            featgen_const.gen_node_features(G)
        input_dim = config['input_dim']

    print("步骤 1.2: 转换图格式并建立地图...")
    graph_data_map = {}
    for G in graphs:
        try:
            pid = extract_pid_from_graph(G) 
            pid = str(pid)
            graph_input_dict = transform_graph_to_input(G, config['max_nodes'])
            graph_data_map[pid] = graph_input_dict
        except Exception as e:
            print(f"处理发生错误，跳过：{e}")
    print(f"转换 {len(graph_data_map)} 个地形的图数据。")
    # 2. 加载预处理后的PCA信息
    pca_features_map = {}
    pca_files = glob.glob(os.path.join(config['pca_dir'], '*.xlsx'))
    for f in pca_files:
        pid = extract_pid_from_filename(f) 
        df_pca = pd.read_excel(f, header=None)
        pca_features_map[pid] = df_pca.values.ravel()
    print(f"已加载 {len(pca_features_map)} 个地形的PCA数据。")


    # 3. 加載所有預計算的 DTW 分數
    df_dtw = pd.read_excel(config['dtw_path'])
    dtw_scores_map = {}
    for _, row in df_dtw.iterrows():
        pid1, pid2, score = extract_pid_from_filename(row['pid1']), extract_pid_from_filename(row['pid2']), row['score']
        key = tuple(sorted((pid1, pid2)))
        dtw_scores_map[key] = score
    print(f"已加載 {len(dtw_scores_map)} 對地形的DTW分數。")

    # 4. 加載包含標籤的配對文件
    df_pairs = pd.read_csv(config['pairs_path'])
    all_pairs = []
    for _, row in df_pairs.iterrows():
        pid1, pid2, label = extract_pid_from_filename(row['PID1']), extract_pid_from_filename(row['PID2']), row['label']
        if pid1 in graph_data_map and pid2 in graph_data_map and pid1 in pca_features_map and pid2 in pca_features_map:
            all_pairs.append((pid1, pid2, label))
        else:
            print(f"警告：配對 ({pid1}, {pid2}) 的數據不完整，已跳過。")
    print(f"已加載 {len(all_pairs)} 個有效樣本對。")
    
    print("--- 所有數據加載完畢 ---")
    return graph_data_map, pca_features_map, dtw_scores_map, all_pairs, input_dim


def main():
    args = arg_parse()
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')

    config = {
        'datadir': 'D:\VS\Python\GCN-DP\data',
        'bmname': 'dtw&pca',
        'max_nodes': 3000,
        'input_dim': 5, 
        'pca_dir': 'D:\VS\Python\GCN-DP\PCA&DTW_Data\PCA_new',
        'dtw_path': 'D:\VS\Python\GCN-DP\PCA&DTW_Data\dtw_sim.xlsx',
        'pairs_path': 'D:\VS\Python\GCN-DP\data\pairs.csv'
    }
    args.num_epochs= 2000
    args.train_ratio=0.05#8
    args.test_ratio=0.9
    args.max_nodes=1500
    args.dropout=0.5
 
    graph_map, pca_map, dtw_map, all_pairs, input_dim = load_all_data_and_pairs(config)
    from sklearn.model_selection import train_test_split
    # 0.8,0.2比例划分
    train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42, stratify=[p[2] for p in all_pairs])
    print(f"數據集劃分完畢: 訓練集 {len(train_pairs)} 對, 驗證集 {len(val_pairs)} 對")

    train_dataset = PairedDataset(train_pairs, graph_map, pca_map, dtw_map)
    val_dataset = PairedDataset(val_pairs, graph_map, pca_map, dtw_map)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_paired_data, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_paired_data, num_workers=args.num_workers)

    is_inherit = False
    start_epoch = 0
    checkpoint = None

    gcn_dp_backbone = Model.GCN_DP(max_num_nodes = config['max_nodes'], 
                                        input_dim = input_dim, 
                                        hidden_dim = args.hidden_dim, 
                                        embedding_dim= args.output_dim, 
                                        label_dim = 2,
                                        num_layers =args.num_gc_layers,
                                        trans_hidden_dim = args.hidden_dim,
                                        trans_ratio=args.trans_ratio, 
                                        num_pooling=args.num_pool,
                                        bn=args.bn, 
                                        dropout=args.dropout, 
                                        linkpred=args.linkpred, 
                                        args=args,
                                        trans_input_dim = input_dim).to(device)
    model = Model.TerrainSimilarityModel(gcn_dp_backbone, mlp_hidden_dim = 64).to(device)

    if is_inherit:
        print("--- 继承上一检查点信息 ---")
        checkpoint_path = r'D:\VS\Python\AAAI_Training_results\training_20250716-110110\checkpoint_epoch_1000.pth' 
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"将从epoch {start_epoch} 开始训练")
        else:
            print(f"无法找到 {checkpoint_path}，从头开始训练")
    else:
        print("--- 从头训练 ---")

    print(model)

    Train(train_loader, model, args, val_dataset_loader=val_loader, start_epoch=start_epoch, checkpoint=checkpoint)

if __name__ == "__main__":
    main()