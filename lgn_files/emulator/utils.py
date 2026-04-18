from itertools import islice

import networkx as nx


# def unique_path(prev_path_set: set, set_curr_path: set) -> bool:
#      return (set_curr_path.difference(prev_path_set) >= 1)


def k_shortest_paths(G, k, source, target, weight):
    """Returns the k shortest paths in a graph

    Parameters
    ----------
        G : NetworkX Graph
            NetworkX Graph based on a DFN
        k : int
            Number of requested paths
        source : node
            Starting node
        target : node
            Ending node
        weight : string
            Edge weight used for finding the shortest path

    Returns
    -------
        paths : sets of nodes
            a list of lists of nodes in the k shortest paths
    Notes
    -----
    Edge weights must be numerical and non-negative
"""
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def k_shortest_paths_backbone(G, k, source='s', target='t', weight=None):
    """Returns the subgraph made up of the k shortest paths in a graph 
   
    Parameters
    ----------
        G : NetworkX Graph
            NetworkX Graph based on a DFN 
        k : int
            Number of requested paths
        source : node 
            Starting node
        target : node
            Ending node
        weight : string
            Edge weight used for finding the shortest path

    Returns 
    -------
        H : NetworkX Graph
            Subgraph of G made up of the k shortest paths 

    Notes
    -----
        See Hyman et al. 2017 "Predictions of first passage times in sparse discrete fracture networks using graph-based reductions" Physical Review E for more details
"""

    print("\n--> Determining %d shortest paths in the network" % k)
    H = G.copy()
    k_shortest = set([])
    for path in k_shortest_paths(G, k, source, target, weight):
        k_shortest |= set(path)
    k_shortest.remove('s')
    k_shortest.remove('t')
    path_nodes = sorted(list(k_shortest))
    path_nodes.append('s')
    path_nodes.append('t')
    nodes = list(G.nodes())
    secondary = list(set(nodes) - set(path_nodes))
    for n in secondary:
        H.remove_node(n)
    return H
    print("--> Complete\n")


def greedy_edge_disjoint(G, source='s', target='t', weight='None', k=''):
    """
    Greedy Algorithm to find edge disjoint subgraph from s to t. 
    See Hyman et al. 2018 SIAM MMS

    Parameters
    ----------
        G : NetworkX graph
            NetworkX Graph based on the DFN
        source : node 
            Starting node
        target : node
            Ending node
        weight : string
            Edge weight used for finding the shortest path
        k : int
            Number of edge disjoint paths requested
    
    Returns
    -------
        H : NetworkX Graph
            Subgraph of G made up of the k shortest of all edge-disjoint paths from source to target

    Notes
    -----
        1. Edge weights must be numerical and non-negative.
        2. See Hyman et al. 2018 "Identifying Backbones in Three-Dimensional Discrete Fracture Networks: A Bipartite Graph-Based Approach" SIAM Multiscale Modeling and Simulation for more details 

    """
    print("--> Identifying edge disjoint paths")
    if G.graph['representation'] != "intersection":
        print(
            "--> ERROR!!! Wrong type of DFN graph representation\nRepresentation must be intersection\nReturning Empty Graph\n"
        )
        return nx.Graph()
    Gprime = G.copy()
    Hprime = nx.Graph()
    Hprime.graph['representation'] = G.graph['representation']
    cnt = 0

    # if a number of paths in not provided k will equal the min cut between s and t
    min_cut = len(nx.minimum_edge_cut(G, 's', 't'))
    if k == '' or k > min_cut:
        k = min_cut

    while nx.has_path(Gprime, source, target):
        path = nx.shortest_path(Gprime, source, target, weight=weight)
        H = Gprime.subgraph(path)
        Hprime.add_edges_from(H.edges(data=True))
        Gprime.remove_edges_from(list(H.edges()))

        cnt += 1
        if cnt > k:
            break
    print("--> Complete")
    return Hprime


def filter_k(path_list, G_list, k):
    y = []
    for j in range(len(path_list)):
        temp = k_shortest_paths(G_list[j], k, 's', 't', weight=None)

        if len(temp) < k:
            y.append(path_list[j])

    return y
