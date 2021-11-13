class Graph(object):
    def __init__(
        self,
        node_list,  # [n1,n2,...]
        edge_list, # [[n1,n2,w],...]
        Reflexive = False,
        Directed = True
    ):
        self.num_nodes = len(node_list)
        self.node_list = node_list
        self.out_edges = {}
        self.in_edges = {}
        self.n_out = {}
        self.n_in = {}
        self.max_outdegree = 0
        self.max_indegree = 0
        for n in node_list:
            self.out_edges[n]=[]
            self.in_edges[n]=[]
            self.n_out[n]=0
            self.n_in[n]=0
        for e in edge_list: # Edges in one direction
            if [e[1],e[2]] not in self.out_edges[e[0]]:
                self.out_edges[e[0]].append([e[1],e[2]])
                self.in_edges[e[1]].append([e[0],e[2]])
                self.n_out[e[0]]+=1
                self.n_in[e[1]]+=1
                if self.n_out[e[0]] > self.max_outdegree: 
                    self.max_outdegree = self.n_out[e[0]]
                if self.n_in[e[1]] > self.max_indegree:
                    self.max_indegree = self.n_in[e[1]]
                
        if not Directed:
            for e in edge_list: # Edges in opposite direction
                if [e[0],e[2]] not in self.out_edges[e[1]]:
                    self.out_edges[e[1]].append([e[0],e[2]])
                    self.in_edges[e[0]].append([e[1],e[2]])
                    self.n_out[e[1]]+=1
                    self.n_in[e[0]]+=1
                    if self.n_out[e[1]] > self.max_outdegree: 
                        self.max_outdegree = self.n_out[e[1]]
                    if self.n_in[e[0]] > self.max_indegree:
                        self.max_indegree = self.n_in[e[0]]

        if Reflexive:
            for n in self.node_list:
                if [n,1] not in self.out_edges[n]:
                    self.out_edges[n].append([n,1])
                    self.in_edges[n].append([n,1])
                    self.n_in[n]+=1
                    self.n_out[n]+=1
                    if self.n_out[n] > self.max_outdegree:
                        self.max_outdegree = self.n_out[n]
                    if self.n_in[n] > self.max_indegree:
                        self.max_indegree = self.n_in[n]



#import networkx as nx
#import networkx.algorithms.approximation
#G=nx.DiGraph()
#G.add_nodes_from([0,1,2,3,4,5])
#G.add_edges_from([(0,1),(0,4),(1,5),(1,2),(2,5),(2,3)])
#G=G.to_undirected()
#a=networkx.algorithms.approximation.vertex_cover.min_weighted_vertex_cover(G)
#A=nx.adjacency_matrix(G)
#AA = nx.convert_matrix.to_numpy_matrix(G)