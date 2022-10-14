import numpy as np
from scipy.linalg import sqrtm
import math

from tqdm.auto import tqdm

from graphviz import Digraph
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

class Network:
    '''
    A simple class holding the nodes, edges, tensors, edge and bond dimensions etc.
    This class is introduced to keep the main name space clean.
    '''
    
    def __init__(self, nodes, edges, edge_dims, bond_dims,
                 tensors={}):
        self.nodes = nodes
        self.tensors = tensors
        self.edges = edges
        self.edge_dims = edge_dims
        self.bond_dims = bond_dims
        self.costs = {}
        self.max_id = len(nodes) - 1

        self.dots = []

        self.compute_costs()

    def compute_costs(self):
        '''
        Compute the cost of cotraction for each edge
        '''
        costs = {}
        for edge_id in self.edges.keys():
            adjoining_edges = set(sum([self.nodes[n] for n in self.edges[edge_id]], []))
            adjoining_bond_dims = [self.bond_dims[e] for e in adjoining_edges]
            self.costs[edge_id] = math.prod(adjoining_bond_dims)

    def contract(self, edge_id):
        '''
        Contract an edge and update the tensor network
        '''        
        n1, n2 = self.edges[edge_id]
        d1, d2 = self.edge_dims[edge_id]
        
        if n1 == n2:
    #         print('Self-loop detected, using contract_self_loop instead')
            self.contract_self_loop(edge_id)
            return
        
        # collect old edges and remove the edge to be contracted 
        new_node = list(set([e for e in (self.nodes[n1] + self.nodes[n2]) if e != edge_id]))

        # contract and update tensors

        # input tensors
        a, b = self.tensors[n1], self.tensors[n2]

        # contracted tesnor
        c = np.tensordot(a, b, axes=[d1, d2])
        self.tensors[self.max_id + 1] = c

        # re-route old edges to the new node and add edge dimensions

        a_inds = [i for i in range(len(a.shape)) if i != d1]
        a_mask = {j:i for i, j in enumerate(a_inds)}

        b_inds = [i for i in range(len(b.shape)) if i != d2]
        b_mask = {j: i + len(a.shape) - 1 for i, j in enumerate(b_inds)}
        # change edge dimensions 
        for e in new_node:

            if len(set(self.edges[e])) == 1:
                '''
                handle self-loop in one of the nodes
                '''
                n = self.edges[e][0]

                if n == n1:
                    self.edge_dims[e][0] = a_mask[self.edge_dims[e][0]]
                    self.edge_dims[e][1] = a_mask[self.edge_dims[e][1]]
                elif n == n2:
                    self.edge_dims[e][0] = b_mask[self.edge_dims[e][0]]
                    self.edge_dims[e][1] = b_mask[self.edge_dims[e][1]]

            elif set(self.edges[e]) == set([n1, n2]):
                '''
                create self-loop to the new node from the 
                addtional edge between the same nodes
                '''
                ind = self.edges[e].index(n1)
                self.edge_dims[e][ind] = a_mask[self.edge_dims[e][ind]]
                ind = self.edges[e].index(n2)
                self.edge_dims[e][ind] = b_mask[self.edge_dims[e][ind]]

            elif n1 in self.edges[e]:
                ind = self.edges[e].index(n1)
                self.edge_dims[e][ind] = a_mask[self.edge_dims[e][ind]]

            elif n2 in self.edges[e]:
                ind = self.edges[e].index(n2)
                self.edge_dims[e][ind] = b_mask[self.edge_dims[e][ind]]

            # re-route edges to the new node
            edge = [n if (n not in [n1, n2]) else self.max_id+1 for n in self.edges[e]]
            self.edges[e] = edge

        self.nodes[self.max_id+1] = list(set(new_node))
        self.max_id += 1

        # delete the nodes and edges
        del self.edges[edge_id]
        del self.edge_dims[edge_id]
        del self.bond_dims[edge_id]

        del self.nodes[n1], self.nodes[n2]

        del self.tensors[n1], self.tensors[n2]

        self.compute_costs()

    def contract_self_loop(self, edge_id):
        '''
        Contract a self-loop and update the tensor network,
        seperated from contract for simplicity
        '''
        n = self.edges[edge_id][0]
        d1, d2 = self.edge_dims[edge_id]
        a = self.tensors[n]
        b = np.trace(a, axis1=d1, axis2=d2)

        self.tensors[n] = b

        new_node = [e for e in self.nodes[n] if e != edge_id]

        inds = [i for i in range(len(a.shape)) if i not in [d1, d2]]
        mask = {j:i for i,j in enumerate(inds)}

        for e in new_node:
            self.edge_dims[e] = [mask[x] for x in self.edge_dims[e]]

        # delete the edge
        del self.edges[edge_id]
        self.nodes[n].remove(edge_id)
        del self.edge_dims[edge_id]
        del self.bond_dims[edge_id]

        self.compute_costs()

    def contract_edges(self, contraction_order=None):
        '''
        Contract all the edges of the tensor network to give a scalar
        '''
        if not contraction_order:
            contraction_order = list(self.edges.keys()) 
        for e in tqdm(contraction_order, desc='Contracting edges', leave=False):
            self.contract(e)

    def __repr__(self):
        return f'Tensor Network \n Number of Nodes = {len(self.nodes)} \n Number of Edges = {len(self.edges)}'
    
def IsingTensorNetwork(N, beta):
    '''
    Create and return a tensor network (instance of the network class previously
    defined) that on contraction yields the partition function of an N*N 
    Ising lattice with an inverse temperature beta.
    '''

    S = np.array([[np.exp(beta), np.exp(-beta)], [np.exp(-beta), np.exp(beta)]])
    S_root = sqrtm(S)
    T = np.einsum('ia,ib,ic,id->abcd', S_root, S_root, S_root, S_root)

    nodes = {i:[] for i in range(N**2)}
    tensors = {i:T for i in range(N**2)}
    bond_dims = {'e'+str(i):2 for i in range(2*N**2)}
    edge_dims = {}
    edges = {}

    cordinates2node = {}
    k = 0
    for i in range(N):
        for j in range(N):
            cordinates2node[f'{i},{j}'] = k
            k += 1

    k = 0
    for i in range(N):
        for j in range(N):

            # horizontal edge
            edges['e'+str(k)] = [cordinates2node[f'{i},{j}'], 
                                 cordinates2node[f'{(i+1)%N},{j}']]
            edge_dims['e'+str(k)] = [0, 2]

            for n in edges['e'+str(k)]:
                nodes[n].append('e'+str(k))
            k += 1

            #vertical edge
            edges['e'+str(k)] = [cordinates2node[f'{i},{j}'], 
                                 cordinates2node[f'{i},{(j+1)%N}']]

            edge_dims['e'+str(k)] = [1, 3]

            for n in edges['e'+str(k)]:
                nodes[n].append('e'+str(k))
            k +=1

    return Network(nodes, edges, edge_dims, bond_dims, tensors)

    
def detailed_plot(tn):

    dot = Digraph(format='jpg', engine='dot')

    dot.attr('node', shape='circle', style=None)
    dot.attr(size='4,4!')

    for n in tn.nodes.keys():
        dot.node(name = str(n), 
                 color = 'black',
                 label= str(n),
                 **{'width':str(0.1), 'height':str(0.1), 'nodesep':str(1)})

    for l, e in tn.edges.items():
            dot.edge(str(e[0]), str(e[1]), 
                     label= f'{l}',#{tn.edge_dims[l]}', 
                     color='grey',
                     **{'penwidth':str(1)})

    return dot

def plot(tn):
    dot = Digraph(format='jpg', engine='dot')

    dot.attr('node', shape='circle', style=None)
    dot.attr(size='4,4!')

    for n in tn.nodes.keys():
        dot.node(name = str(n), 
                 color = 'black',
                 label= '',
                 **{'width':str(0.1), 'height':str(0.1)})

    norm = plt.Normalize()
    costs = np.array(list(tn.costs.values()))
    colors = [rgb2hex(x) for x in plt.cm.rainbow(norm(costs))]
    color =  {e: c for e, c in zip(tn.edges.keys(), colors)}

    for l, e in tn.edges.items():
            dot.edge(str(e[0]), str(e[1]), 
                     label= None, #color=color[l],
                     **{'penwidth':str(1),'arrowhead':'none'})

    return dot