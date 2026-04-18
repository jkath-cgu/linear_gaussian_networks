import numpy as np
import collections
# from networkx import degree_centrality, betweenness_centrality, \
#     current_flow_betweenness_centrality

# from itertools import islice
# from itertools import takewhile
# import networkx as nx

from emulator.utils import k_shortest_paths


class featureProcess:
    """

    """

    def __init__(self, k_shortest_path=1, flux_calc_pct=50, travel_calc_pct=50, verbose=False):
    # def __init__(self, k_shortest_path=1, flux_calc_pct=42, travel_calc_pct=85, verbose=False):
        self.k_shortest_path = k_shortest_path
        self.flux_calc_pct = flux_calc_pct
        self.travel_calc_pct = travel_calc_pct
        self.verbose = verbose

        self.graph_list = []
        self.output = []
        self.output_append = []

    def fit_transform(self, graph_list):
        """
        :param graph_list:
        :return:
        """
        assert len(graph_list) > 0
        self.graph_list = graph_list
        self.output = self._fit_transform(self.graph_list[0])
        # print(self.output)

        if len(self.graph_list) > 1:
            counter = 0
            if self.verbose:
                print("in total {} graphs to be processed: \n".format(len(self.graph_list)))
            for graph in self.graph_list[1:]:

                self.output_append = self._fit_transform(graph)
                for index in range(len(self.output)):
                    self.output[index] = np.vstack((self.output[index], self.output_append[index]))

                # self.output = np.vstack((self.output, self._fit_transform(graph)))

                counter += 1
                if self.verbose:
                    print("{} has been processed and remaining {} to be done \n".format(counter,
                                                                                        len(self.graph_list) - counter))

        # # compute weighted (perm) sum of travel time
        # max_perm = self.output.min(axis=0)[0]
        # self.output[:,3] / max_perm

        return self.output

    def _fit_transform(self, graph):
        """
        :param g:
        :return:
        """
        # # pre-compute the centrality features to avoid duplicates
        # feat_deg = degree_centrality(graph)
        # feat_bet = betweenness_centrality(graph)
        # feat_cur = current_flow_betweenness_centrality(graph)

        # prepare the output handler
        output = np.array([])
        output_agg = np.array([])
        output_agg_pfc = np.array([])
        output_agg_nfc = np.array([])
        output_list = []
        path_list = []

        # slice k-shortest path
        path_list = k_shortest_paths(graph, self.k_shortest_path, 's', 't', weight=None)
        # path_list = k_shortest_paths(graph, self.k_shortest_path, 's', 't', weight='iperm')


        # path_list = list(islice(nx.edge_disjoint_paths(graph, 's', 't'), self.k_shortest_path))

        # test = len(list(nx.edge_disjoint_paths(graph, 's', 't', flow_func=None, cutoff=self.k_shortest_path, auxiliary=None, residual=None)))
        # test = len(list(nx.edge_disjoint_paths(graph, 's', 't')))
        # print(test)

        # it = nx.shortest_simple_paths(graph, 's', 't', weight=None)
        # set_diff_len = 2
        # count = 0
        # prev_path_set = set()
        # for curr_path in takewhile(lambda curr_path: count < self.k_shortest_path, it):
        # # for path in takewhile(lambda curr_path: len(set(curr_path).difference(prev_path_set)) >= set_diff_len and count < self.k_shortest_path, it):

        #     if (len(set(curr_path).difference(prev_path_set)) >= set_diff_len):

        #         path_list.append(curr_path)

        #         # print(curr_path)
        #         count+=1

        #     prev_path_set = set(curr_path)

        # # print("")


        for path_index in range(self.k_shortest_path):
            path = path_list[path_index][1: -1]  # excluding node s and t

            # print(path)

            path_feat = self._path_feat(graph=graph, path=path)
                                        # args=[path_index])                    # calculate features along the path
                                        # args=[feat_deg, feat_bet, feat_cur])  # calculate features along the path

            # print(path_feat)

            output = np.concatenate((output, path_feat), axis=0)

        output = output.reshape(len(path_list), -1)
        # output = output.reshape(self.k_shortest_path, -1)

        # for sum_end in range(len(output)):
        #     output_sum_end = output[0:sum_end+1]

        #     backbone_length = 5 #10
        #     if len(output_sum_end) > backbone_length:
        #         output_agg_pfc = np.sum(output_sum_end[0:backbone_length], axis=0)
        #         output_agg_nfc = np.sum(output_sum_end[backbone_length:], axis=0)
        #         output_agg = np.concatenate((output_agg_pfc, output_agg_nfc), axis=0)
        #     else:
        #         output_agg = np.sum(output_sum_end, axis=0)

        #     # output_agg = np.array([])
        #     # backbone_step = 100 #25
        #     # for i in range(0, len(output_sum_end), backbone_step):
        #     #     output_agg = np.concatenate((output_agg, np.sum(output_sum_end[i:i+backbone_step], axis=0)), axis=0)

        #     output_list.append(output_agg)

        for sum_end in range(len(output)):
            output_list.append(np.sum(output[0:sum_end+1], axis=0))

        # print(output_list)

        return output_list


        # backbone_step = 25
        # for i in range(0, len(output), backbone_step):
        #     output_agg = np.concatenate((output_agg, np.sum(output[i:i+backbone_step], axis=0)), axis=0)

        # backbone_length = 10
        # if len(output) > backbone_length:
        #     output_agg_pfc = np.sum(output[0:backbone_length], axis=0)
        #     output_agg_nfc = np.sum(output[backbone_length:], axis=0)
        #     output_agg = np.concatenate((output_agg_pfc, output_agg_nfc), axis=0)
        # else:
        #     output_agg = np.sum(output, axis=0)
        
        # print(output_agg)


        # return output_agg

    def _path_feat(self, graph, path):
    # def _path_feat(self, graph, path, args):
        """
        :param path:
        :param args:
        :return:
        """
        it1 = iter(path)
        it2 = iter(path[1:])
        path_dict = collections.defaultdict(list)
        for u, v in zip(it1, it2):
            edge_data = graph.get_edge_data(u, v)
            # path_dict['frac'].append(edge_data['frac'])
            path_dict['length'].append(edge_data['length'])
            path_dict['iperm'].append(edge_data['iperm'])
            path_dict['flux'].append(edge_data['flux'])
            path_dict['time'].append(edge_data['time'])

        # topological features
        path_len = len(path)                                                                        # path length
        length_sum = np.sum(path_dict['length'])                                                  # sum fracture length
        # length_median = np.median(path_dict['length'])                                              # median fracture length
        # length_min = np.min(path_dict['length'])                                                  # min fracture length
        # hydrological features
        iperm_sum = np.sum(path_dict['iperm'])                                                      # sum inverse perm
        flux_pct = np.percentile(path_dict['flux'], self.flux_calc_pct)                             # percentile mass flux
        time_pct = np.percentile(path_dict['time'], self.travel_calc_pct)                      # percentile travel time
        
        # features in order of MAPE significance
        feature_list = [path_len, flux_pct, iperm_sum, time_pct, length_sum] # length_sum length_min
        # feature_list = [path_len]

        output = np.array(feature_list)

        return output


# %%
