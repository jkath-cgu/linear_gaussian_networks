import os
import sys

import networkx as nx
import numpy as np
import scipy.sparse.linalg

# from emulator.utils import greedy_edge_disjoint
# from emulator.utils import filter_k


class crawler:
    """

    """

    def __init__(self, data_source, quantile='50 percent', fidelity='high', k_shortest_path=1):
        """
        :param data_source: specify if the data source is a folder, a torrent file or an api
        :param args: arguments for getting data from the data_source
        """
        self.graph_list, self.breakthrough_times = [], []  # key lists to update

        self.data_source = data_source
        self.quantile = quantile
        self.fidelity = fidelity
        self.k_shortest_path = k_shortest_path

    def crawl_folder(self, folder_path):
        """

        :param folder_path:
        :return:
        """
        train_path = self._get_train_path(folder_path)
        train_path.sort()

        start_index = 0     #   0, 100, 200, 300, 400
        end_index   = 100   # 100, 200, 300, 400, 500

        # if self.fidelity == 'high':
        #     train_path = train_path[start_index: end_index]

        # if self.fidelity == 'low':
        #     train_path = train_path[start_index: end_index]

        # process network graphs
        G_list, y = self._get_train_graphs(train_path)

        # ?? how to remove cycles / cliques of edges from IG ??
        # G_list_subgraph_k = []
        # # return subgraph containing transport backbone
        # for j in range(len(G_list)):
        #     # G_list_filter_k.append(k_shortest_paths_backbone(G_list[j], k, source='s', target='t', weight=None))
        #     G_list_subgraph_k.append(greedy_edge_disjoint(G_list[j], source='s', target='t', weight='None', k=''))

        # # remove stuff if necessary
        # delete = filter_k(train_path, G_list, self.k_shortest_path)

        # G_list_filter_k, y_filter_k = [], []
        # for j in range(len(train_path)):
        #     if train_path[j] not in delete:
        #         G_list_filter_k.append(G_list[j])
        #         y_filter_k.append(y[j])

        self.graph_list = G_list
        self.breakthrough_times = y

        return self.graph_list, self.breakthrough_times

    def _get_train_graphs(self, train_path):
        """
        :param train_path:
        :return: list of graphs
        """
        G_list = []
        y = []
        for sample in train_path:

            # # get a fracture graph
            # sample_g = create_fracture_graph(inflow=sample + "/left",
            #                                  outflow=sample + "/right",
            #                                  topology_file=sample + "/connectivity.dat",
            #                                  fracture_info=sample + "/fracture_info.dat")
           
            # get an intersection graph
            sample_g = create_intersection_graph(inflow = "left",
                                                 outflow = "right",
                                                 intersection_file = sample + "/intersection_list.dat",
                                                 fracture_info = sample + "/fracture_info.dat")

            # compute flow and transport on graph
            for v in nx.nodes(sample_g):
                sample_g.nodes[v]['inletflag'] = False
                sample_g.nodes[v]['outletflag'] = False

            for v in nx.neighbors(sample_g, 's'):
                sample_g.nodes[v]['inletflag'] = True

            for v in nx.neighbors(sample_g, 't'):
                sample_g.nodes[v]['outletflag'] = True

            # added
            mapping = {'s':12345678910}
            sample_g = nx.relabel_nodes(sample_g, mapping)
            mapping = {'t':1234567891011}
            sample_g = nx.relabel_nodes(sample_g, mapping)
            # sample_g.remove_node('s')
            # sample_g.remove_node('t')

            sample_g = nx.convert_node_labels_to_integers(sample_g,
                                                          first_label=0,
                                                          ordering="sorted",
                                                          label_attribute="old_label")
            
            # defining viscosity
            fluid_viscosity = 8.9e-4          
                                               
            # input the value of the pressure (in Pa) at outlet and inlet
            Pin = 0.3
            Pout = 0.21            
            
            # preparing to solve the sparse laplacian system
            Inlet = [v for v in nx.nodes(sample_g) if sample_g.nodes[v]['inletflag']]
            Outlet = [v for v in nx.nodes(sample_g) if sample_g.nodes[v]['outletflag']]

            if not set(Inlet).isdisjoint(set(Outlet)):
                error = "Incompatible graph: Vertex connected to both source and target\n"
                sys.stderr.write(error)
                sys.exit(1)

            # set up the sparse laplacian system
            D, A = get_laplacian_sparse_mat(sample_g, weight='weight', format='lil')
            rhs = np.zeros(sample_g.number_of_nodes())

            for v in Inlet:
                rhs[v] = Pin
                A[v, :] = 0
                D[v, v] = 1.0
                
            for v in Outlet:
                rhs[v] = Pout
                A[v, :] = 0
                D[v, v] = 1.0
            L = D - A  # automatically converts to csr when returning L
            # solve the sparse laplacian system
            print("Solving sparse system")
            Phat = scipy.sparse.linalg.spsolve(L, rhs)

            # update pressure at nodes
            print("Updating graph nodes features with flow solution")
            for v in nx.nodes(sample_g):
                sample_g.nodes[v]['pressure'] = Phat[v]

            print("Updating graph edges with flow solution")

            for u, v in nx.edges(sample_g):
                delta_p = abs(sample_g.nodes[u]['pressure'] -
                                sample_g.nodes[v]['pressure'])
                if delta_p > np.spacing(sample_g.nodes[u]['pressure']):
                    sample_g.edges[u, v]['flux'] = (
                        sample_g.edges[u, v]['perm'] / fluid_viscosity
                        ) * abs(sample_g.nodes[u]['pressure'] -
                                sample_g.nodes[v]['pressure']) / sample_g.edges[u, v]['length']
                    
                    sample_g.edges[u, v]['time'] = sample_g.edges[
                            u, v]['length'] / sample_g.edges[u, v]['flux']
                
                else:
                    sample_g.edges[u, v]['flux'] = 0       
            
            # added
            source_n = [v for v in nx.nodes(sample_g) if sample_g.nodes[v]['old_label'] == 12345678910]
            target_n = [v for v in nx.nodes(sample_g) if sample_g.nodes[v]['old_label'] == 1234567891011]
            
            # print(source_n)
            # print(target_n)

            mapping = {source_n[0]:'s'}
            sample_g = nx.relabel_nodes(sample_g, mapping)
            mapping = {target_n[0]:'t'}
            sample_g = nx.relabel_nodes(sample_g, mapping)
            
            print("graph flow complete")

            G_list.append(sample_g)

            # get a y output
            # pts_name_dict = {'0 percent': 1, '20 percent': 2, '50 percent': 3, '70 percent': 4, '90 percent': 5,
            #                  'peak': 6}

            pts_name_dict = {'0 percent': 0, '10 percent': 1,'20 percent': 2,'30 percent': 3,'40 percent': 4,
                             '50 percent': 5,'60 percent': 6, '70 percent': 7,'80 percent': 8, '90 percent': 9,
                             'peak': 10}

            # with open(sample + "/pts.dat") as f:
            #     L = list([line.split() for line in f][pts_name_dict[self.quantile]][0].split(','))
            #     if self.fidelity == 'high' and len(L) == 4:
            #         y.append(L[1])
            #     elif self.fidelity == 'low' and len(L) == 4:
            #         y.append(L[2])
            #     elif self.fidelity == 'low' and len(L) == 2:
            #         y.append(L[1])
            
            ds_fname = ""
            if self.fidelity == 'high':
                ds_fname = "/dfn_data.dat"
            elif self.fidelity == 'low':
                ds_fname = "/graph_data.dat"
                # ds_fname = "/pts.dat"
            if ds_fname != "":
                with open(sample + ds_fname) as f:
                    L = list([line.split() for line in f][1][0].split(','))
                    if len(L) == 11:
                        y.append(L[pts_name_dict[self.quantile]])

        return G_list, y

    @staticmethod
    def _get_train_path(folder_path):
        """
        takes in the data folder and output tall train_paths (relative)
        :param folder_path:
        :return: list of train_paths
        """
        train_path = []
        for j in os.walk(folder_path):
            train_path.append(j[0])
        train_path = train_path[1:]

        return list(set(train_path))  # remove duplicate path entries


# fracture graph parser
def create_fracture_graph(inflow,
                          outflow,
                          topology_file="connectivity.dat",
                          fracture_info="fracture_info.dat"):
    """ Create a graph based on topology of network. Fractures
    are represented as nodes and if two fractures intersect
    there is an edge between them in the graph.

    Source and Target node are added to the graph.

    Parameters
    ----------
        inflow : string
            Name of inflow boundary (connect to source)
        outflow : string
            Name of outflow boundary (connect to target)
        topology_file : string
            Name of adjacency matrix file for a DFN default=connectivity.dat
        fracture_infor : str
                filename for fracture information

    Returns
    -------
        G : NetworkX Graph
            NetworkX Graph where vertices in the graph correspond to fractures and edges indicated two fractures intersect

    Notes
    -----
    :param inflow:
    :param outflow:
    :param topology_file:
    :param fracture_info:
    """
    print("--> Loading Graph based on topology in " + topology_file)
    G = nx.Graph(representation="fracture")
    with open(topology_file, "r") as infile:
        for i, line in enumerate(infile):
            conn = [int(n) for n in line.split()]
            for j in conn:
                G.add_edge(i + 1, j)
    ## Create Source and Target and add edges
    inflow_filename = inflow + ".dat"
    outflow_filename = outflow + ".dat"
    inflow = np.genfromtxt(inflow_filename).astype(int)
    outflow = np.genfromtxt(outflow_filename).astype(int)

    try:
        if len(inflow) > 1:
            inflow = list(inflow)
    except:
        inflow = [inflow.tolist()]

    try:
        if len(outflow) > 1:
            outflow = list(outflow)
    except:
        outflow = [outflow.tolist()]

    G.add_node('s')
    G.add_node('t')
    G.add_edges_from(zip(['s'] * (len(inflow)), inflow))
    G.add_edges_from(zip(outflow, ['t'] * (len(outflow))))
    add_perm(G, fracture_info)
    print("--> Graph loaded")
    return G


# intersection graph parser
def create_intersection_graph(inflow,
                              outflow,
                              intersection_file="intersection_list.dat",
                              fracture_info="fracture_info.dat"):
    """ Create a graph based on topology of network.
    Edges are represented as nodes and if two intersections
    are on the same fracture, there is an edge between them in the graph. 
    
    Source and Target node are added to the graph. 
   
    Parameters
    ----------
        inflow : string
            Name of inflow boundary
        outflow : string
            Name of outflow boundary
        intersection_file :
        string
             File containing intersection information
             File Format:
             fracture 1, fracture 2, x center, y center, z center, intersection length

        fracture_infor : str
                filename for fracture information
    Returns
    -------
        G : NetworkX Graph
            Vertices have attributes x,y,z location and length. Edges has attribute length

    Notes
    -----
    Aperture and Perm on edges can be added using add_app and add_perm functions
    """

    print("Creating Graph Based on DFN")
    print("Intersections being mapped to nodes and fractures to edges")
    inflow_index = boundary_index(inflow)
    outflow_index = boundary_index(outflow)

    f = open(intersection_file)
    f.readline()
    frac_edges = []
    for line in f:
        frac_edges.append(line.rstrip().split())
    f.close()

    # Tag mapping
    G = nx.Graph(representation="intersection")
    remove_list = []

    # each edge in the DFN is a node in the graph
    for i in range(len(frac_edges)):
        f1 = int(frac_edges[i][0])
        keep = True
        if frac_edges[i][1] == 's' or frac_edges[i][1] == 't':
            f2 = frac_edges[i][1]
        elif int(frac_edges[i][1]) > 0:
            f2 = int(frac_edges[i][1])
        elif int(frac_edges[i][1]) == inflow_index:
            f2 = 's'
        elif int(frac_edges[i][1]) == outflow_index:
            f2 = 't'
        elif int(frac_edges[i][1]) < 0:
            keep = False

        if keep:
            # note fractures of the intersection
            G.add_node(i, frac=(f1, f2))
            # keep intersection location and length
            G.nodes[i]['x'] = float(frac_edges[i][2])
            G.nodes[i]['y'] = float(frac_edges[i][3])
            G.nodes[i]['z'] = float(frac_edges[i][4])
            G.nodes[i]['length'] = float(frac_edges[i][5])

    nodes = list(nx.nodes(G))
    f1 = nx.get_node_attributes(G, 'frac')
    # identify which edges are on whcih fractures
    for i in nodes:
        e = set(f1[i])
        for j in nodes:
            if i != j:
                tmp = set(f1[j])
                x = e.intersection(tmp)
                if len(x) > 0:
                    x = list(x)[0]
                    # Check for Boundary Intersections
                    # This stops boundary fractures from being incorrectly
                    # connected
                    # If not, add edge between
                    if x != 's' and x != 't':
                        xi = G.nodes[i]['x']
                        yi = G.nodes[i]['y']
                        zi = G.nodes[i]['z']

                        xj = G.nodes[j]['x']
                        yj = G.nodes[j]['y']
                        zj = G.nodes[j]['z']

                        distance = np.sqrt((xi - xj)**2 + (yi - yj)**2 +
                                           (zi - zj)**2)
                        G.add_edge(i, j, frac=x, length=distance)

    # Add Sink and Source nodes
    G.add_node('s')
    G.add_node('t')

    for i in nodes:
        e = set(f1[i])
        if len(e.intersection(set('s'))) > 0 or len(e.intersection(set(
            [-1]))) > 0:
            G.add_edge(i, 's', frac='s', length=0.0)
        if len(e.intersection(set('t'))) > 0 or len(e.intersection(set(
            [-2]))) > 0:
            G.add_edge(i, 't', frac='t', length=0.0)
    add_perm(G, fracture_info)
    add_area(G, fracture_info)  # added
    add_weight(G)               # added
    print("Graph Construction Complete")
    return G


# boundary index setup
def boundary_index(bc_name):
    """Determines boundary index in intersections_list.dat from name

    Parameters
    ----------
        bc_name : string
            Boundary condition name

    Returns
    -------
        bc_index : int
            integer indexing of cube faces

    Notes
    -----
    top = 1
    bottom = 2
    left = 3
    front = 4
    right = 5
    back = 6
    """
    bc_dict = {
        "top": -1,
        "bottom": -2,
        "left": -3,
        "front": -4,
        "right": -5,
        "back": -6
    }
    try:
        return bc_dict[bc_name]
    except:
        error = "Unknown boundary condition: %s\nExiting\n" % bc_name
        sys.stderr.write(error)
        sys.exit(1)


# addtribute adder
def add_perm(G, fracture_info="fracture_info.dat"):
    """ Add fracture permeability to Graph. If Graph representation is
    fracture, then permeability is a node attribute. If graph representation
    is intersection, then permeability is an edge attribute


    Parameters
    ----------
        G :networkX graph
            NetworkX Graph based on the DFN

        fracture_infor : str
                filename for fracture information
    Returns
    -------

    Notes
    -----
    :param fracture_info:

"""

    perm = np.genfromtxt(fracture_info, skip_header=1)[:, 1]
    # aperture = np.genfromtxt(fracture_info, skip_header=1)[:, 2]  # added
    if G.graph['representation'] == "fracture":
        nodes = list(nx.nodes(G))
        for n in nodes:
            if n != 's' and n != 't':
                G.nodes[n]['perm'] = perm[n - 1]
                G.nodes[n]['iperm'] = 1.0 / perm[n - 1]
                # G.nodes[n]['aperture'] = aperture[n - 1] # added
            else:
                G.nodes[n]['perm'] = 1.0
                G.nodes[n]['iperm'] = 1.0
                # G.nodes[n]['aperture'] = 1.0 # added

    elif G.graph['representation'] == "intersection":
        edges = list(nx.edges(G))
        for u, v in edges:
            x = G[u][v]['frac']
            if x != 's' and x != 't':
                G[u][v]['perm'] = perm[x - 1]
                G[u][v]['iperm'] = 1.0 / perm[x - 1]
            else:
                G[u][v]['perm'] = 1.0
                G[u][v]['iperm'] = 1.0

    elif G.graph['representation'] == "bipartite":
        # add fracture info
        with open(fracture_info) as f:
            header = f.readline()
            data = f.read().strip()
            for fracture, line in enumerate(data.split('\n'), 1):
                c, perm, aperture = line.split(' ')
                G.nodes[fracture]['perm'] = float(perm)
                G.nodes[fracture]['iperm'] = 1.0 / float(perm)
                G.nodes[fracture]['aperture'] = float(aperture)


def add_area(G, fracture_info="fracture_info.dat"):
    ''' Read Fracture aperture from fracture_info.dat and
    load on the edges in the graph. Graph must be intersection to node
    representation

    Parameters
    ----------
        G : NetworkX Graph
            networkX graph
        fracture_info : str
            filename for fracture information

    Returns
    -------
        None
'''

    aperture = np.genfromtxt(fracture_info, skip_header=1)[:, 2]
    edges = list(nx.edges(G))
    for u, v in edges:
        x = G.edges[u, v]['frac']
        if x != 's' and x != 't':
            G.edges[u,v]['area'] = aperture[x - 1] * (G.nodes[u]['length'] +
                                                    G.nodes[v]['length']) / 2.0
        else:
            G.edges[u, v]['area'] = 1.0
    return


def add_weight(G):
    """Compute weight w = K*A/L associated with each edge
    Parameters
    ----------
        G : NetworkX Graph
            networkX graph

    Returns
    -------
        None
"""
    edges = list(nx.edges(G))
    for u, v in edges:
        if G.edges[u, v]['length'] > 0:
            G.edges[u, v]['weight'] = G.edges[u, v]['perm'] * G.edges[
                u, v]['area'] / G.edges[u, v]['length']
    return


# get sparse laplacian matrix
def get_laplacian_sparse_mat(G,
                             nodelist=None,
                             weight=None,
                             dtype=None,
                             format='lil'):
    """ Get the matrices D, A that make up the Laplacian sparse matrix in desired sparsity format. Used to enforce boundary conditions by modifying rows of L = D - A

    Parameters
    ----------
        G : object
            NetworkX graph equipped with weight attribute

        nodelist : list
            list of nodes of G for which laplacian is desired. Default is None in which case, all the nodes
        
        weight : string
            For weighted Laplacian, else all weights assumed unity
        
        dtype :  default is None, cooresponds to float
        
        format: string
            sparse matrix format, csr, csc, coo, lil_matrix with default being lil

    Returns
    -------
        D : sparse 2d float array       
            Diagonal part of Laplacian
            
        A : sparse 2d float array
            Adjacency matrix of graph
    """

    # A = nx.to_scipy_sparse_matrix(G,
    A = nx.to_scipy_sparse_array(G,
                                  nodelist=nodelist,
                                  weight=weight,
                                  dtype=dtype,
                                  format=format)

    (n, n) = A.shape
    data = np.asarray(A.sum(axis=1).T)
    D = scipy.sparse.spdiags(data, 0, n, n, format=format)
    return D, A

def debug():
    G_list = folder_traversal("../" + "data/")
    print("list has shape: {}".format(len(G_list)))
    print(G_list[0], "\n", G_list[1])


def folder_traversal(folder_path, y_quantile='50 percent', fidelity='high', delete=None):
    """
    takes in the folder and outputs the list of graphs
    :param delete: sublist to be deleted from the train_path
    :param fidelity: fidelity of the label to be returned
    :param y_quantile: quantile to use as emulator label
    :param folder_path: point within the data/ folder
    :return: list of fracture graphs generated
    """
    train_path = get_train_path(folder_path)
    if delete is not None:
        train_path = [x for x in train_path if x not in delete]

    return get_train_graphs(train_path, y_quantile, fidelity)
