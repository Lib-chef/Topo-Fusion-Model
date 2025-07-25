import networkx as nx
import numpy as np
import os,re,util

#print('Current working directory:', os.getcwd())

def Load_GraphTxt(datadir, dataname, max_nodes=None):
    directory = os.path.join(datadir, dataname, dataname)
    fgi = directory + '_graph_indicator.txt'   
    graph_indic={}
    with open(fgi) as f:
        i=1
        for line in f:
            line=line.strip("\n")
            graph_indic[i]=int(line)
            i+=1

    file_attributes=directory + '_node_attributes.txt'
    node_attrs=[]
    with open(file_attributes) as f:
        for line in f:
            line = line.strip("\s\n")
            attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
            node_attrs.append(np.array(attrs))

       
    label_has_zero = False
    filename_graphs=directory + '_graph_labels.txt'
    graph_labels=[]

    with open(filename_graphs) as f:
        for line in f:
            line=line.strip("\n")
            val = int(line)
            graph_labels.append(val-1)
    graph_labels = np.array(graph_labels)
 
    filename_inFID_graphIndicator=directory + '_graph_indicator_JOIN_FID.txt'
    inFID_graphIndicator_lst=[]
    with open(filename_inFID_graphIndicator) as f:
        for line in f:
            line=line.strip("\n").split(" ")
            inFID_graphIndicator_lst_i=[int(line[0]),int(line[1])]
            inFID_graphIndicator_lst.append(inFID_graphIndicator_lst_i)

        
    filename_adj=directory + '_w_A.txt'#_w_A

    adj_list={i:[] for i in range(1,len(graph_labels)+1)} 
    adj_edgeWeight_list={i:[] for i in range(1,len(graph_labels)+1)} 
    index_graph={i:[] for i in range(1,len(graph_labels)+1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line=line.strip("\n").split(",")
            e0,e1,e2=(int(line[0].strip(" ")),int(line[1].strip(" ")),float(line[2].strip(" ")))
            adj_list[graph_indic[e0]].append((e0,e1))
            adj_edgeWeight_list[graph_indic[e0]].append((e0,e1,e2))
            index_graph[graph_indic[e0]]+=[e0,e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k]=[u-1 for u in set(index_graph[k])]

    graphs=[]
    ggggg,ggg=[],0
    for i in range(1,1+len(adj_list)):
        ggg+=1
        if ggg==42:
            t=1
        G=nx.DiGraph()
        G.add_weighted_edges_from(adj_edgeWeight_list[i])
        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue
        
        G.graph['label'] = graph_labels[i-1]
        for u in util.node_s(G):
            if len(node_attrs) > 0:
                util.node_infor(G)[u]['feat'] = node_attrs[u-1]

        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]
        if len(inFID_graphIndicator_lst)>0:
            G.graph['inFID_graphIndicator']=inFID_graphIndicator_lst[i-1]        
        mapping={}
        it=0
        for n in util.node_s(G):
            mapping[n]=it
            it+=1
            
        graphs.append(nx.relabel_nodes(G, mapping))
        ggggg.append(ggg)
    return graphs