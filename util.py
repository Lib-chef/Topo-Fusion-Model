import community
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def node_s(G):
    if float(nx.__version__)<2.0:
        return G.nodes()
    else:
        return G.nodes

def node_infor(G):
    if float(nx.__version__)>2.1:
        node_infor = G.nodes
    else:
        node_infor = G.node
    return node_infor

def draw_graph(plt, G):
    plt.title('num of nodes: '+str(G.number_of_nodes()), fontsize = 4)
    parts = community.best_partition(G)
    values = [parts.get(node) for node in G.nodes()]
    colors = ['black','orange','pink','yellow','blue','green','red']
    plt.axis("off")
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, with_labels=True, node_size=4, width=0.3, font_size = 3, node_color=colors,pos=pos)

def draw_avg(x, decay=0.9):
    shadow = x[0]
    a = [shadow]
    for v in x[1:]:
        shadow -= (1-decay) * (shadow-v)
        a.append(shadow)
    return a


