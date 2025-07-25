import networkx as nx
import numpy as np
import torch,os,random

from GraphConstruction import GraphConstruction

def SplitData(graphs, args, val_idx, max_nodes=0,train_graphIndicatorpath='',val_graphIndicatorpath='',test_graphIndicatorpath=''):
        train_graphs,val_graphs,test_graphs = [],[],[]
        if os.path.isfile(train_graphIndicatorpath) and os.path.isfile(val_graphIndicatorpath) and os.path.isfile(test_graphIndicatorpath):
                train_graphIndicator_lst,val_graphIndicator_lst,test_graphIndicator_lst=np.load(train_graphIndicatorpath),np.load(val_graphIndicatorpath),np.load(test_graphIndicatorpath)
                for i,singleGraph in enumerate(graphs):
                        graphIndicator=singleGraph.graph['inFID_graphIndicator'][1]-1
                        if graphIndicator in train_graphIndicator_lst:
                                train_graphs.append(singleGraph)
                        elif graphIndicator in val_graphIndicator_lst:
                                val_graphs.append(singleGraph)
                        elif graphIndicator in test_graphIndicator_lst:
                                test_graphs.append(singleGraph)
                        else:
                                print(str(graphIndicator))
        else:                        
                random.shuffle(graphs)#  random shuffle the samples
                #  split train,val ,test data
                labelsDic,classes={},[]#  labelsDic record the number of each type samples
                for i,singleGraph in enumerate(graphs):
                        label=singleGraph.graph['label']
                        if labelsDic.get(label)==None:
                                labelsDic[label]=1
                        else:
                                labelsDic[label]+=1
                dataSeparation=[args.train_ratio,1-args.train_ratio-args.test_ratio,args.test_ratio]
                for k in labelsDic:
                        oneClass = [k, 
                                round(labelsDic[k] * dataSeparation[0]), 
                                round(labelsDic[k] * dataSeparation[1]),
                                int(labelsDic[k]) - round(labelsDic[k] * dataSeparation[0]) - round(labelsDic[k] * dataSeparation[1])]
                        classes.append(oneClass)
                classes=np.array(classes)    
                for i,singleGraph in enumerate(graphs):
                        label=singleGraph.graph['label']
                        index=np.argwhere(classes[:,0]==label)[0][0].astype(np.int64)
                        if (classes[index][1] > 0):
                                train_graphs.append(singleGraph)
                                classes[index][1] = classes[index][1]-1
                        elif (classes[index][2] > 0):
                                val_graphs.append(singleGraph)
                                classes[index][2] = classes[index][2]-1
                        else:
                                test_graphs.append(singleGraph)
                                classes[index][3] = classes[index][3]-1

        dataset_sampler = GraphConstruction(train_graphs, normalize=False, max_num_nodes=max_nodes,
                features=args.feature_type)
        train_dset_loader = torch.utils.data.DataLoader(
                dataset_sampler, 
                batch_size=args.batch_size, 
                shuffle=True,
                num_workers=args.num_workers)
        if len(val_graphs)>0:
                dataset_sampler = GraphConstruction(val_graphs, normalize=False, max_num_nodes=max_nodes,
                        features=args.feature_type)
                val_dset_loader = torch.utils.data.DataLoader(
                        dataset_sampler, 
                        batch_size=args.batch_size, 
                        shuffle=False,
                        num_workers=args.num_workers)
        else:
               val_dset_loader=None
        if len(test_graphs)>0: 
                dataset_sampler = GraphConstruction(test_graphs, normalize=False, max_num_nodes=max_nodes,
                        features=args.feature_type)
                test_dataset_loader = torch.utils.data.DataLoader(
                        dataset_sampler, 
                        batch_size=args.batch_size, 
                        shuffle=False,
                        num_workers=args.num_workers)
        else:
                test_dataset_loader=None
        return train_dset_loader, val_dset_loader, test_dataset_loader,\
                dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.trans_feat_dim
