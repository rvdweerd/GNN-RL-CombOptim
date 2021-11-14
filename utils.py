import torch
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
from scipy.spatial import distance_matrix

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
    sol_last_node = state.partial_solution[-1] if len(state.partial_solution) > 0 else -1
    sol_first_node = state.partial_solution[0] if len(state.partial_solution) > 0 else -1
    #coords = state.coords
    #nr_nodes = coords.shape[0]

    xv = [[(1 if i in solution else 0),
           (1 if i == sol_first_node else 0),
           (1 if i == sol_last_node else 0),
           #coords[i,0],
           #coords[i,1]
           i
          ] for i in range(state.W.shape[0])]
    
    return torch.tensor(xv, dtype=torch.float32, requires_grad=False, device=device)

def get_graph_mat(n=10, size=1):
    """ Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
        Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
    """
    coords = size * np.random.uniform(size=(n,2))
    dist_mat = distance_matrix(coords, coords)
    return coords, dist_mat

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