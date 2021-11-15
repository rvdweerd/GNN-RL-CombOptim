import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
from scipy.spatial import distance_matrix
from asp import Solve_MinVertexCover_ASP
from graph import Graph

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # if you are using multi-GPU.
    np.random.seed(seed)
    # Numpy module.
    random.seed(seed)
    # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def state2tensor(state):
    """ Creates a Pytorch tensor representing the history of visited nodes, from a (single) state tuple.
        
        Returns a (Nx5) tensor, where for each node we store whether this node is in the sequence,
        whether it is first or last, and its (x,y) coordinates.
    """
    solution = set(state.partial_solution)
    #sol_last_node = state.partial_solution[-1] if len(state.partial_solution) > 0 else -1
    #sol_first_node = state.partial_solution[0] if len(state.partial_solution) > 0 else -1
    #coords = state.coords
    #nr_nodes = coords.shape[0]

    xv = [[(1 if i in solution else 0),
           #(1 if i == sol_first_node else 0),
           #(1 if i == sol_last_node else 0),
           #coords[i,0],
           #coords[i,1]
           state.G.n_in[i]
          ] for i in range(state.num_nodes)]
    
    return torch.tensor(xv, dtype=torch.float32, requires_grad=False, device=device)



# Note: we store state tensors in experience to compute these tensors only once later on
from collections import namedtuple
Experience = namedtuple('Experience', ('state', 'state_tsr', 'action', 'reward', 'next_state', 'next_state_tsr'))

class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.nr_inserts = 0
        
    def remember(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        self.nr_inserts += 1
        
    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return min(self.nr_inserts, self.capacity)

def total_distance(solution, W):
    if len(solution) < 2:
        return 0  # there is no travel
    
    total_dist = 0
    for i in range(len(solution) - 1):
        total_dist += W[solution[i], solution[i+1]].item()
        
    # if this solution is "complete", go back to initial point
    if len(solution) == W.shape[0]:
        total_dist += W[solution[-1], solution[0]].item()

    return total_dist

def all_edges_covered(partial_solution, out_edges, true_min_vertex_cover):
    if len(partial_solution) < true_min_vertex_cover:
        return False
    else:
        solset=set(partial_solution)
        for k,v in out_edges.items():
            for e in v:
                if not (k in partial_solution or e[0] in partial_solution):
                    return False
        return True

def is_state_final(state,G):
    return all_edges_covered(state.partial_solution, G.out_edges, state.min_VC)

def get_next_neighbor_random(state):
    solution, G, candidates = state.partial_solution, state.G, state.candidates
    
    if len(solution) == 0:
        return random.choice(range(state.num_nodes))
    #already_in = set(solution)
    #candidates = list(filter(lambda n: n.item() not in already_in, W[solution[-1]].nonzero()))
    if len(candidates) == 0:
        return None
    return random.choice(candidates)#.item()

def get_graph_mat(n=10, size=1):
    """ Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
        Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
    """
    coords = size * np.random.uniform(size=(n,2))
    dist_mat = distance_matrix(coords, coords)
    return coords, dist_mat

def get_graph0(n=6):
    G=Graph([0,1,2,3,4,5],[[0,4,1],[0,1,1],[1,5,1],[1,2,1],[2,5,1],[2,3,1]],Reflexive=False,Directed=True)
    solutions=Solve_MinVertexCover_ASP(G)
    return G, solutions

def get_graph(n=6, p_edge=0.4):
    nodelist=[i for i in range(n)]
    edgelist=[]
    for node1 in range(n):
        for node2 in range(n):
            eps=random.random()
            if eps<=p_edge and node2 != node1:
                edgelist.append([node1,node2,1])
    G=Graph(nodelist,edgelist,Reflexive=False,Directed=True)
    solutions=Solve_MinVertexCover_ASP(G)
    return G, solutions



def TestTorch():
    alpha = Variable(torch.tensor(0,dtype=torch.float,device=device),requires_grad=True)
    lr=0.1
    optimizer = optim.Adam([alpha], lr=lr)
    phist=[]
    lhist=[]
    for i in range(200):
        loss = (1-alpha)**2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        phist.append(alpha.detach().item())
        lhist.append(loss)

    plt.plot(phist)
    plt.title('alpha')
    plt.figure()
    plt.plot(lhist)
    plt.title('loss')
    plt.show()

def NaiveEmbedding():
    # Naive embedding
    p=5
    T=4
    theta1 = Variable(torch.ones((p,1), dtype=torch.float, device=device),requires_grad=True)*0.1
    theta2 = Variable(torch.ones((p,p), dtype=torch.float, device=device),requires_grad=True)*0.1
    theta3 = Variable(torch.ones((p,p), dtype=torch.float, device=device),requires_grad=True)*0.1
    theta4 = Variable(torch.ones((p,1), dtype=torch.float, device=device),requires_grad=True)*0.1
    theta5 = Variable(torch.ones((2*p,1), dtype=torch.float, device=device),requires_grad=True)*0.1
    theta6 = Variable(torch.ones((p,p), dtype=torch.float, device=device),requires_grad=True)*0.1
    theta7 = Variable(torch.ones((p,p), dtype=torch.float, device=device),requires_grad=True)*0.1
    n=G.num_nodes
    S=set()
    in_set = torch.zeros(n)
    mu = torch.zeros((p,n))
    for node in G.node_list:
        for t in range(T):
            F1=theta1*in_set[node]

            F2=torch.zeros(p)
            for e in G.out_edges[node]:
                F2+=mu[:,e[0]]
            F2=theta2@(F2.unsqueeze(1))

            F3=torch.zeros((p,1))
            for e in G.out_edges[node]:
                F3+=nn.ReLU()(theta4*e[1])
            F3=theta3@F3
            mu[:,node]+= nn.ReLU()(F1+F2+F3).squeeze()
        
    print(mu)