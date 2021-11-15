import os
import torch
#from main import NR_NODES
from qnet import *
from asp import Solve_MinVertexCover_ASP
from graph import Graph
from collections import namedtuple
from utils import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

EMBEDDING_DIMENSIONS = 8#5  # Embedding dimension D
EMBEDDING_ITERATIONS_T = 4  # Number of embedding iterations T
INIT_LR = 5e-3
LR_DECAY_RATE = 0.9995#1. - 2e-5  # learning rate decay
FOLDER_NAME = './models'  # where to checkpoint the best models
NR_NODES=6
State = namedtuple('State',('W', 'partial_solution', 'num_nodes', 'min_VC', 'candidates'))

all_lengths_fnames = [f for f in os.listdir(FOLDER_NAME) if f.endswith('.tar')]
shortest_fname = sorted(all_lengths_fnames, key=lambda s: float(s.split('.tar')[0].split('_')[-1]))[0]
print('shortest avg length found: {}'.format(shortest_fname.split('.tar')[0].split('_')[-1]))

def init_model_from_file(fname=None):
    """ Create a new model. If fname is defined, load the model from the specified file.
    """
    Q_net = QNet(EMBEDDING_DIMENSIONS, T=EMBEDDING_ITERATIONS_T).to(device)
    optimizer = optim.Adam(Q_net.parameters(), lr=INIT_LR)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_RATE)
    
    if fname is not None:
        checkpoint = torch.load(fname)
        Q_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    Q_func = QFunction(Q_net, optimizer, lr_scheduler)
    return Q_func, Q_net, optimizer, lr_scheduler

Q_func, Q_net, optimizer, lr_scheduler = init_model_from_file(os.path.join(FOLDER_NAME, shortest_fname))
#G=Graph([0,1,2,3,4,5],[[0,4,1],[0,1,1],[1,5,1],[1,2,1],[2,5,1],[2,3,1]],Reflexive=False,Directed=True)
#solutions=Solve_MinVertexCover_ASP(G)
G,solutions=get_graph()
print(G.W)
print(solutions)
true_minVC=len(solutions[0])
W = torch.tensor(G.W, dtype=torch.float32, requires_grad=False, device=device)

# current partial solution - a list of node index
#solution = [random.randint(0, NR_NODES-1)]
solution = []
remaining_candidates = [i for i in range(NR_NODES) if i not in solution]

# current state (tuple and tensor)
current_state = State(partial_solution=solution, W=W, num_nodes=NR_NODES, min_VC=true_minVC, candidates=remaining_candidates)
current_state_tsr = state2tensor(current_state)
while not is_state_final(current_state,G):
    next_node, est_reward = Q_func.get_best_action(current_state_tsr, current_state)
    solution=solution+[next_node]
    print(solution)
    remaining_candidates = [i for i in range(NR_NODES) if i not in solution]
    current_state = State(partial_solution=solution, W=W, num_nodes=NR_NODES, min_VC=true_minVC, candidates=remaining_candidates)
    current_state_tsr = state2tensor(current_state)

