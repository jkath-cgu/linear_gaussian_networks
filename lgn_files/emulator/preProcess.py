from .featureProcess import *
from .crawler import *

from emulator.utils import greedy_edge_disjoint
from emulator.utils import k_shortest_paths_backbone
from emulator.utils import k_shortest_paths


class preProcess:
    """

    """

    def __init__(self, quantile, fidelity, k_shortest_path=1, verbose=True):
        self.quantile = quantile
        self.fidelity = fidelity
        self.k_shortest_path = k_shortest_path
        self.verbose = verbose

        self.worker = None
        self.featWorker = None

        self.X = []
        self.y = []
        self.X_filter_k = []
        self.y_filter_k = []
        self.graph_list = []

    def fit_transform(self, data_folder):
        self.worker = crawler(data_folder, self.quantile, self.fidelity, self.k_shortest_path)

        # crawl folder to obtain the graph list
        self.graph_list, self.y = self.worker.crawl_folder(data_folder)
        self.y = np.array(self.y, dtype='float64')

        # feature engineering
        featWorker = featureProcess(k_shortest_path=self.k_shortest_path, verbose=self.verbose)
        self.X = featWorker.fit_transform(self.graph_list)

        return self.X, self.y

    def build_graphs_from_dfn_data(self, data_folder):
        self.worker = crawler(data_folder, self.quantile, self.fidelity, self.k_shortest_path)

        # crawl folder to obtain the graph list
        self.graph_list, self.y = self.worker.crawl_folder(data_folder)
        self.y = np.array(self.y, dtype='float64')

        # # ?? how to remove cycles / cliques of edges from IG ??
        # G_list_subgraph_k = []
        # # return subgraph containing transport backbone
        # for j in range(len(self.graph_list)):
        #     G_list_subgraph_k.append(k_shortest_paths_backbone(self.graph_list[j], self.k_shortest_path, source='s', target='t', weight=None))
        #     # G_list_subgraph_k.append(greedy_edge_disjoint(self.graph_list[j], source='s', target='t', weight='None', k=''))
        # self.graph_list = G_list_subgraph_k

    def extract_features_from_graphs(self, k_shortest_path, flux_calc_pct=42, travel_calc_pct=85, verbose=False):

        self.G_list_filter_k, self.y_filter_k = [], []
        for j in range(len(self.graph_list)):
            temp = k_shortest_paths(self.graph_list[j], k_shortest_path, 's', 't', weight=None)
    
            if not len(temp) < k_shortest_path:
                self.G_list_filter_k.append(self.graph_list[j])
                self.y_filter_k.append(self.y[j])
        
        self.y_filter_k = np.array(self.y_filter_k, dtype='float64')

        # feature engineering
        self.featWorker = featureProcess(k_shortest_path=k_shortest_path, flux_calc_pct=flux_calc_pct, travel_calc_pct=travel_calc_pct, verbose=verbose)
        self.X_filter_k = self.featWorker.fit_transform(self.G_list_filter_k)

        return self.X_filter_k, self.y_filter_k
