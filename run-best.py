import os
import torch
#from main import NR_NODES
from qnet import *
from asp import Solve_MinVertexCover_ASP
from graph import Graph
from collections import namedtuple
from utils import *
from itertools import product
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

EMBEDDING_DIMENSIONS = 8 #5  # Embedding dimension D
EMBEDDING_ITERATIONS_T = 4  # Number of embedding iterations T
FOLDER_NAME = './models'  # where to checkpoint the best models
NR_NODES=6
State = namedtuple('State',('G', 'partial_solution', 'num_nodes', 'min_VC', 'candidates'))

all_lengths_fnames = [f for f in os.listdir(FOLDER_NAME) if f.endswith('.tar')]
shortest_fname = sorted(all_lengths_fnames, key=lambda s: float(s.split('.tar')[0].split('_')[-1]))[0]
print('shortest avg length found: {}'.format(shortest_fname.split('.tar')[0].split('_')[-1]))

def is_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def all_symmetric_adj_matrices(n):
    all_g = []
    for vals in product([0, 1], repeat=(n*n-n)//2):
        arr = np.zeros((n,n))
        i_up=np.triu_indices(6,1) # non reflexive (offset from diag = 1)
        arr[i_up]=vals
        arr=arr+arr.T
        all_g.append(arr)
    return all_g

def init_model_from_file(fname=None):
    """ Create a new model. If fname is defined, load the model from the specified file.
    """
    Q_net = QNet(EMBEDDING_DIMENSIONS, T=EMBEDDING_ITERATIONS_T).to(device)
    optimizer = optim.Adam(Q_net.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    if fname is not None:
        checkpoint = torch.load(fname)
        Q_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    Q_func = QFunction(Q_net, optimizer, lr_scheduler)
    return Q_func, Q_net, optimizer, lr_scheduler

Q_func, Q_net, optimizer, lr_scheduler = init_model_from_file(os.path.join(FOLDER_NAME, shortest_fname))

def TestInstance(G):
    solution = []
    remaining_candidates = [i for i in range(NR_NODES) if i not in solution]
    # current state (tuple and tensor)
    current_state = State(partial_solution=solution, G=G, num_nodes=NR_NODES, min_VC=true_minVC, candidates=remaining_candidates)
    current_state_tsr = state2tensor(current_state)
    while not is_state_final(current_state,G):
        next_node, est_reward = Q_func.get_best_action(current_state_tsr, current_state)
        solution=solution+[next_node]
        remaining_candidates = [i for i in range(NR_NODES) if i not in solution]
        current_state = State(partial_solution=solution, G=G, num_nodes=NR_NODES, min_VC=true_minVC, candidates=remaining_candidates)
        current_state_tsr = state2tensor(current_state)
    return solution

for i in range(10):
    print('Random example\n==============')
    print('Adjacency matrix:')
    G,solutions=get_graph()
    W = torch.tensor(G.W, dtype=torch.float32, requires_grad=False, device=device)
    true_minVC=len(solutions[0])
    print(G.W)
    print('True solution(s):\n',solutions)
    print('True Minimum Vertex Cover:',true_minVC)
    solution = TestInstance(G)
    print('Predicted solution by trained Qnet:',solution)
    print("Ratio MVC_predicted / MVC_true: {:.2f}".format(len(solution)/true_minVC))
    print('\n\n')

def create_graphs(all_W):
    all_G=[]
    for W in all_W:
        N=W.shape[0]
        nodelist=[i for i in range(N)]
        edgelist=[]
        for i in range(N):
            for j in range(N):
                if W[i,j]==1:
                    edgelist.append([i,j,1])
        G=Graph(nodelist,edgelist)
        all_G.append(G)
    return all_G

print('Performance on all possible graphs, N=',(NR_NODES**2-NR_NODES)//2)
print('=================================================')
all_W=all_symmetric_adj_matrices(NR_NODES)
all_G=create_graphs(all_W)
ratios_minvc={'all':[]}#0:[],1:[],2:[],3:[],4:[],5:[],6:[],'all':[]}
ratios_num_minvc={'all':[]}#{0:[],1:[],2:[],3:[],4:[],5:[],6:[],'all':[]}
correct_minvc={'all':0}
correct_num_minvc={'all':0}
i=0
for G in all_G:
    solutions=Solve_MinVertexCover_ASP(G)
    W = torch.tensor(G.W, dtype=torch.float32, requires_grad=False, device=device)
    true_minVC=len(solutions[0])
    num_minVC=len(solutions)
    solution = TestInstance(G)
    if true_minVC == 0:
        print('*')
        Ratio = 1+100*len(solution)
    else:
        Ratio=len(solution)/true_minVC
    if true_minVC not in ratios_minvc.keys():
        ratios_minvc[true_minVC]=[]
    if num_minVC not in ratios_num_minvc.keys():
        ratios_num_minvc[num_minVC]=[]
    if true_minVC not in correct_minvc.keys():
        correct_minvc[true_minVC]=0
    if num_minVC not in correct_num_minvc.keys():
        correct_num_minvc[num_minVC]=0
    ratios_minvc[true_minVC].append(Ratio)
    ratios_minvc['all'].append(Ratio)
    ratios_num_minvc[num_minVC].append(Ratio)
    ratios_num_minvc['all'].append(Ratio)
    if len(solution)==true_minVC:
        correct_minvc[true_minVC]+=1
        correct_minvc['all']+=1
        correct_num_minvc[num_minVC]+=1
        correct_num_minvc['all']+=1
    if (i+1)%2500 == 0 or (i+1)==len(all_G):
        print('\ngraph count',i+1)
        print('Ratios per minVC (number of nodes in minVC):')
        for k,v in ratios_minvc.items():
            print("{:4s}{:6} {:5s}{:.3f} {:5s} {:.2f}".format(str(k),len(v), \
                str(int(100*len(v)/len(ratios_minvc['all'])))+'%',np.mean(v), \
                str(correct_minvc[k]), correct_minvc[k]/len(v)
                    ))
        print('------------------------------------')
        print('Ratios per nSol (number of minVC solutions):')
        for k,v in ratios_num_minvc.items():
            #print(k,np.mean(v))
            print("{:4s}{:6} {:5s}{:.3f} {:5s} {:.2f}".format(str(k),len(v), \
                str(int(100*len(v)/len(ratios_num_minvc['all'])))+'%',np.mean(v), \
                str(correct_num_minvc[k]), correct_num_minvc[k]/len(v)
                    ))
    i+=1





    
