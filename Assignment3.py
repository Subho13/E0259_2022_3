import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import time

def import_facebook_data(filename):
    # Data is already in ideal format
    # if i->j is in list, j->i isn't there
    # nothing extra to do with the data
    # So simply read and return
    return np.genfromtxt(filename, dtype=np.int32)

def spectralDecomp_OneIter(connectivity):
    def getAdjAndFiedlerVector(nodes, connectivity):
        def createAdjacencyMatrix2(nodes, connectivity):
            # Obtain sorted list of unique nodes
            # nodes = np.unique(connectivity)
            # Reverse mapping of nodes to their indices in the list of nodes
            nodes_i = { node: i for i, node in enumerate(nodes) }
            adj = np.zeros((len(nodes), len(nodes)), dtype=np.int32)

            # Traverse over edges and add them to adjacency list
            for n1, n2 in connectivity:
                row = nodes_i[n1]
                col = nodes_i[n2]

                adj[row, col] = adj[col, row] = 1
            
            return adj
        
        def createDegreeMatrix2(adj):
            # Sum of adjacency matrix along rows gives the degree of nodes
            diag_vec = np.sum(adj, axis=0, dtype=np.int32) # This is a 1-d array
            diag_mat = np.diag(diag_vec) # Convert to 2-d square matrix (diagonal matrix)
            return diag_mat

        # Reverse mapping from nodes to their index in the list of nodes
        nodes_i = { node: i for i, node in enumerate(nodes) }

        adj_mat = createAdjacencyMatrix2(nodes, connectivity)
        deg_mat = createDegreeMatrix2(adj_mat)
        # Calculate Laplacian matrix from degree and adjacency matrix
        lap_mat = deg_mat - adj_mat

        # Eigen values and Eigen vectors
        w, v = np.linalg.eigh(lap_mat)
        # Neglecting negative eigen values
        w[w<0] = 0

        min_i = np.argmin(w) # Position of minimum Eigen value
        min_val = w[min_i] # Minimum Eigen value (possibly zero)
        min2_eig = w[w != min_val] # List of eigen values without minimum one
        min2_eig_v = v[w != min_val] # List of eigen vectors without corresponding to minimum eigen values
        min2_i = np.argmin(min2_eig) # Second minimum eigen value index
        min2_val = min2_eig[min2_i] # Second minimum eigen value

        fiedler_vec = min2_eig_v[min2_i] # Second minimum eigen vector
        return adj_mat, fiedler_vec

    # Obtain sorted list of unique nodes
    nodes = np.unique(connectivity)
    # Reverse mapping of nodes to index in the list of nodes
    nodes_i = { node: i for i, node in enumerate(nodes) }

    adj_mat, fiedler_vec = getAdjAndFiedlerVector(nodes, connectivity)
    # Sign of fiedler vector values determine the two partitions
    partition = np.sign(fiedler_vec)
    # Separate the two partitions
    partition1 = nodes[partition > 0]
    partition2 = nodes[partition <= 0]

    # If cannot partition into two, return as it is
    if len(partition1) == 0:
        return fiedler_vec, adj_mat, np.array(list(zip(partition2, [np.min(partition2)] * len(partition2))))
    elif len(partition2) == 0:
        return fiedler_vec, adj_mat, np.array(list(zip(partition1, [np.min(partition1)] * len(partition1))))

    # plt.plot(range(len(fiedler_vec)), np.sort(fiedler_vec), 'o')
    # plt.show()

    # Format of components as per given template
    new_components = np.array(list(zip(partition1, [np.min(partition1)] * len(partition1))) + list(zip(partition2, [np.min(partition2)] * len(partition2))))

    # Remove the disconnected edges from the adjacency matrix
    for n1, n2 in connectivity:
        i1 = nodes_i[n1]
        i2 = nodes_i[n2]
        if partition[i1] != partition[i2]:
            adj_mat[i1, i2] = adj_mat[i2, i1] = 0

    return fiedler_vec, adj_mat, new_components

def spectralDecomposition(connectivity):
    # List of possible communities that can be broken down
    edgelist_of_components = [ connectivity ]
    # List of components after breaking down
    components_ret = []
    # List of components that can no longer be broken
    community_ret_final = []
    # Iteration counter
    i = 0
    # while there are more possible communities that can be broken
    while len(edgelist_of_components) > 0:
        new_edgelist_of_components = []
        new_components_ret = []
        # For every such community
        for edgelist in edgelist_of_components:
            if len(edgelist) == 0: continue # Safety check, just in case
            # Use the previous function
            f_m, a_m, community = spectralDecomp_OneIter(edgelist)
            com_dict = dict(community) # Dictionary for fast lookup

            # Divide into two communities
            first_com = (community[:,1] == community[0,1])
            nodeset1 = community[first_com, 0]
            nodeset2 = community[~first_com, 0]

            # Condition for stopping iterations
            if len(nodeset1) < 20 or len(nodeset2) < 20:
                # Do not split this community, keep as it is
                component_parent = np.min(community, axis=0)[1]
                # Append this to the final array, so that
                # it does not appear in next iteration
                community_ret_final.extend([np.array([co_t[0], component_parent]) for co_t in community])
            else:
                # Split into two communities
                edgelist1 = list()
                edgelist2 = list()

                for edge in edgelist:
                    # Use the community dictionary for fast lookup
                    if com_dict[edge[0]] == com_dict[edge[1]]: # Same community
                        if edge[0] in nodeset1:
                            edgelist1.append(edge)
                        else:
                            edgelist2.append(edge)

                if len(edgelist1) > 0: new_edgelist_of_components.append(np.array(edgelist1)) # New sub community might have further possibility of being broken down
                if len(edgelist2) > 0: new_edgelist_of_components.append(np.array(edgelist2)) # Other new sub community might also have further possibility of being broken down
                # new_components_ret.extend(community)

        # if len(edgelist_of_components) == len(new_edgelist_of_components):
        #     # print('No more partitioning possible')
        #     break

        # Try breaking down the new list of components
        edgelist_of_components = new_edgelist_of_components
        i+=1
    
    # print(i, "iterations")
    # print("edgelist_of_components", len(new_components_ret + community_ret_final))
    # min_len = np.inf
    # max_len = -min_len
    # for i in edgelist_of_components:
    #     if len(i) < min_len:
    #         min_len = len(i)
    #     elif len(i) > max_len:
    #         max_len = len(i)
    # print("max", max_len, "min", min_len)
    # return edgelist_of_components

    # return np.array(new_components_ret + community_ret_final)
    return np.array(community_ret_final)

def createSortedAdjMat(partition, connectivity):
    def getAdjAndFiedlerVector(nodes, connectivity):
        def createAdjacencyMatrix2(nodes, connectivity):
            # Obtain sorted list of unique nodes
            # nodes = np.unique(connectivity)
            # Reverse mapping of nodes to their indices in the list of nodes
            nodes_i = { node: i for i, node in enumerate(nodes) }
            adj = np.zeros((len(nodes), len(nodes)), dtype=np.int32)

            # Traverse over edges and add them to adjacency list
            for n1, n2 in connectivity:
                row = nodes_i[n1]
                col = nodes_i[n2]

                adj[row, col] = adj[col, row] = 1
            
            return adj
        
        def createDegreeMatrix2(adj):
            # Sum of adjacency matrix along rows gives the degree of nodes
            diag_vec = np.sum(adj, axis=0, dtype=np.int32) # This is a 1-d array
            diag_mat = np.diag(diag_vec) # Convert to 2-d square matrix (diagonal matrix)
            return diag_mat

        # Reverse mapping from nodes to their index in the list of nodes
        nodes_i = { node: i for i, node in enumerate(nodes) }

        adj_mat = createAdjacencyMatrix2(nodes, connectivity)
        deg_mat = createDegreeMatrix2(adj_mat)
        # Calculate Laplacian matrix from degree and adjacency matrix
        lap_mat = deg_mat - adj_mat

        # Eigen values and Eigen vectors
        w, v = np.linalg.eigh(lap_mat)
        # Neglecting negative eigen values
        w[w<0] = 0

        min_i = np.argmin(w) # Position of minimum Eigen value
        min_val = w[min_i] # Minimum Eigen value (possibly zero)
        min2_eig = w[w != min_val] # List of eigen values without minimum one
        min2_eig_v = v[w != min_val] # List of eigen vectors without corresponding to minimum eigen values
        min2_i = np.argmin(min2_eig) # Second minimum eigen value index
        min2_val = min2_eig[min2_i] # Second minimum eigen value

        fiedler_vec = min2_eig_v[min2_i] # Second minimum eigen vector
        return adj_mat, fiedler_vec

    def createAdjacencyMatrix2(nodes, connectivity):
        # Obtain sorted list of unique nodes
        # nodes = np.unique(connectivity)
        # Reverse mapping of nodes to their indices in the list of nodes
        nodes_i = { node: i for i, node in enumerate(nodes) }
        adj = np.zeros((len(nodes), len(nodes)), dtype=np.int32)

        # Traverse over edges and add them to adjacency list
        for n1, n2 in connectivity:
            row = nodes_i[n1]
            col = nodes_i[n2]

            adj[row, col] = adj[col, row] = 1
        
        return adj

    nodes = np.unique(connectivity)
    _, f_vec = getAdjAndFiedlerVector(nodes, connectivity)

    # Create tuples of nodes and corresponding fiedler vector values
    tup = zip(nodes, f_vec)

    # Sort the tuples according to the fiedler vector values
    sorted_tup = sorted(tup, key=lambda x: x[1])
    # Extract the nodes from the sorted tuples
    # to get the desired order of nodes
    sorted_nodes = [node[0] for node in sorted_tup]

    # create corresponding adjacency matrix
    # adj_mat_sorted = createAdjacencyMatrix_nodes(connectivity, sorted_nodes)
    adj_mat_sorted = createAdjacencyMatrix2(sorted_nodes, connectivity)
    return adj_mat_sorted

def louvain_one_iter(connectivity):
    def createAdjacencyMatrix2(nodes, connectivity):
        # Obtain sorted list of unique nodes
        # nodes = np.unique(connectivity)
        # Reverse mapping of nodes to their indices in the list of nodes
        nodes_i = { node: i for i, node in enumerate(nodes) }
        adj = np.zeros((len(nodes), len(nodes)), dtype=np.int32)

        # Traverse over edges and add them to adjacency list
        for n1, n2 in connectivity:
            row = nodes_i[n1]
            col = nodes_i[n2]

            adj[row, col] = adj[col, row] = 1
        
        return adj

    edges = len(connectivity)
    nodes = np.unique(connectivity)
    nodes_index = { node: index for index, node in enumerate(nodes) }

    adj_mat = createAdjacencyMatrix2(nodes, connectivity)
    diag_vec = np.sum(adj_mat, axis=0, dtype=np.int32) # required to calculate the degree of a vertex

    # Data structure to keep record of which node belong to which community
    coms = { node: node for node in nodes }
    # Data structure to hold total internal degree of the community to which node belongs to
    degs = dict(zip(nodes, diag_vec))
    # Dictionary to hold members of the community
    # com_members = dict()

    # for node in nodes:
    #     com_members[node] = [node]
    com_members = { node: [node] for node in nodes }

    for node, com in coms.items():
        # If node has already merged with previous nodes, do not consider that node
        if node != com: continue

        node_i = nodes_index[node]
        d_i = degs[node]

        # Variables to store parameters which give maximum change in modularity
        max_q = -1
        max_node = np.min(nodes) # Initialize with min value
        max_com = np.min(nodes) # Initialize with min value
        for node2, com2 in coms.items():
            if node2 != com2: continue

            # List of nodes in the community
            com_mems = com_members[com2]
            new_nodes_i = [nodes_index[i] for i in com_mems]

            # Calculate d_ij: sum of weights of links from current node to nodes in community
            d_ij = 0
            for new_node_i in new_nodes_i:
                d_ij += adj_mat[node_i, new_node_i]
            
            # d_ij = 0 means they are not neighbours (not connected)
            if d_ij == 0: continue

            # Sum of degrees of internal nodes of the community
            d_j = degs[com2]

            # Formula of modularity
            del_q = ((2 * d_ij) - ((d_i * d_j) / edges)) / (2 * edges)

            if del_q > max_q: # Find the maximum modularity
                max_q = del_q
                max_node = node2
                max_com = com2
                # print("max_node", max_node)
        
        # Cannot merge because merging gives negative modularity
        # which means it is a bad merge
        if max_q < 0: continue

        # Otherwise make the node a member of the community
        coms[node] = coms[max_node]
        # Update the internal degree of this community
        degs[max_com] += degs[com]
        # Add this node to the community
        com_members[max_com].append(node)

    # return coms as np array
    return np.array([(key, value) for key, value in coms.items()])


def import_bitcoin_data(filename):
    data = np.genfromtxt(filename, delimiter=',', dtype=np.int32)
    data_edges = data[:,:2]
    final_data_edges = []
    for i, j in data_edges:
        if [j, i] in final_data_edges: continue
        final_data_edges.append([i, j])

    return final_data_edges

if __name__ == "__main__":

    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")
    # nodes_connectivity_list_fb = import_facebook_data("../data/0.edges")
    # nodes_connectivity_list_fb = import_facebook_data("../data/test.edges")
    # t1 = time.time()

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)

    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)
    # print(time.time() - t1, "seconds")
    # nodes = np.unique(graph_partition_fb[:, 0])
    # partitions = np.unique(graph_partition_fb[:, 1])
    # print(nodes, len(nodes))
    # print(partitions, len(partitions))

    # print(graph_partition_fb)
    # exit(0)

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    # plt.figure(figsize=(10,10),dpi=80)
    # plt.scatter(range(len(fielder_vec_fb)), np.sort(fielder_vec_fb), marker='o')
    # plt.savefig('fiedler.png')

    # plt.imshow(clustered_adj_mat_fb*255, vmin=0, vmax=1)
    # plt.imshow(adj_mat_fb*255, vmin=0, vmax=1)
    # plt.savefig('adj.png')

    # first_com = (graph_partition_fb[:,1] == graph_partition_fb[0,1])
    # nodeset1 = graph_partition_fb[first_com, 0]
    # nodeset2 = graph_partition_fb[~first_com, 0]

    # communities = np.unique(graph_partition_fb[:,1])
    # print("comms", len(communities))
    # partitions_abcd = dict()
    # for com in graph_partition_fb:
    #     if com[1] in partitions_abcd.keys():
    #         partitions_abcd[com[1]].append(com[0])
    #     else:
    #         partitions_abcd[com[1]] = [com[0]]
    
    # print(partitions_abcd)
    # print("dict keys", len(partitions_abcd.keys()))

    # G = nx.from_edgelist(nodes_connectivity_list_fb)
    # pos = nx.spring_layout(G, seed=42069)
    # options = {"edgecolors": "tab:gray", "node_size": 20, "alpha": 1}
    # col = [ "tab:red", "tab:blue", "tab:green", "tab:orange", "tab:yellow", "tab:purple", "tab:black", "tab:brown", "tab:pink", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue" ]
    # col_i = 0
    # for com in partitions_abcd.keys():
    #     # nx.draw_networkx_nodes(G, pos, nodelist=partitions_abcd[com], node_color=col[col_i], **options)
    #     nx.draw_networkx_nodes(G, pos, nodelist=partitions_abcd[com], node_color=col_i, **options)
    #     col_i += 1
    # # nx.draw_networkx_nodes(G, pos, nodelist=nodeset1, node_color="tab:red", **options)
    # # nx.draw_networkx_nodes(G, pos, nodelist=nodeset2, node_color="tab:blue", **options)
    # nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.6)
    # # plt.savefig('graph_fin.png')
    # print('done')

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)
    # # print(graph_partition_louvain_fb)

    # nodes = np.unique(nodes_connectivity_list_fb)

    # com_dict = { node: com for node, com in graph_partition_louvain_fb}
    # node_i = { node: index for index, node in  enumerate(nodes) }
    # adj, _ = getAdjAndFiedlerVector(np.unique(nodes_connectivity_list_fb), nodes_connectivity_list_fb)
    # for n1, n2 in nodes_connectivity_list_fb:
    #     if com_dict[n1] != com_dict[n2]:
    #         adj[node_i[n1],node_i[n2]] = adj[node_i[n2],node_i[n1]] = 0
    # plt.imshow(adj*255, vmin=0, vmax=1)
    # plt.savefig('adj_lou_btc.png')

    # print(time.time() - t1, "seconds")
    # print(len(np.unique(graph_partition_louvain_fb[:, 1])))
    # exit(0)
    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")
    # print("edges", len(nodes_connectivity_list_btc))
    # print("nodes", len(np.unique(nodes_connectivity_list_btc)))
    # t1 = time.time()
    # Question 1
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)

    # print(time.time() - t1, "seconds")
    # nodes = np.unique(graph_partition_btc[:, 0])
    # partitions = np.unique(graph_partition_btc[:, 1])
    # # print(nodes, len(nodes))
    # print(partitions, len(partitions))
    # exit(0)

    # plt.figure(figsize=(10,10),dpi=80)
    # plt.scatter(range(len(fielder_vec_btc)), np.sort(fielder_vec_btc), marker='o')
    # plt.savefig('fiedler_btc_one_i.png')

    # plt.imshow(clustered_adj_mat_btc*255, vmin=0, vmax=1)
    # plt.imshow(adj_mat_btc*255, vmin=0, vmax=1)
    # plt.savefig('adj_cl_btc.png')

    # partitions_abcd = dict()
    # for com in graph_partition_btc:
    #     if com[1] in partitions_abcd.keys():
    #         partitions_abcd[com[1]].append(com[0])
    #     else:
    #         partitions_abcd[com[1]] = [com[0]]
    
    # # print(partitions_abcd)
    # print("dict keys", len(partitions_abcd.keys()))

    # G = nx.from_edgelist(nodes_connectivity_list_btc)
    # pos = nx.spring_layout(G, seed=42069)
    # options = {"edgecolors": "tab:gray", "node_size": 20, "alpha": 1}
    # for partition in partitions_abcd:
    #     nx.draw_networkx_nodes(G, pos, nodelist=partition, node_color="tab:red", **options)
    # nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.6)
    # plt.savefig('graph_btc_louvain.png')

    # Question 2
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)

    # print(time.time() - t1, "seconds")
    # nodes = np.unique(graph_partition_btc[:, 0])
    # partitions = np.unique(graph_partition_btc[:, 1])
    # # print(nodes, len(nodes))
    # print(partitions, len(partitions))

    # Question 3
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    # plt.scatter(range(len(fielder_vec_btc)), np.sort(fielder_vec_btc), marker='o')
    # plt.savefig('fiedler.png')

    # plt.imshow(clustered_adj_mat_btc*255, vmin=0, vmax=1)
    # plt.imshow(adj_mat_btc*255, vmin=0, vmax=1)
    # plt.savefig('adj_cl_btc.png')

    # first_com = (graph_partition_btc[:,1] == graph_partition_btc[0,1])
    # nodeset1 = graph_partition_btc[first_com, 0]
    # nodeset2 = graph_partition_btc[~first_com, 0]
    # G = nx.from_edgelist(nodes_connectivity_list_btc)
    # pos = nx.spring_layout(G, seed=69420)
    # options = {"edgecolors": "tab:gray", "node_size": 20, "alpha": 1}
    # nx.draw_networkx_nodes(G, pos, nodelist=nodeset1, node_color="tab:red", **options)
    # nx.draw_networkx_nodes(G, pos, nodelist=nodeset2, node_color="tab:blue", **options)
    # nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.6)
    # plt.savefig('graph_btc.png')

    # Question 4
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)
    # print(graph_partition_louvain_fb)

    # print(time.time() - t1, "seconds")
    # print(len(np.unique(graph_partition_louvain_btc[:, 1])))

    # nodes = np.unique(nodes_connectivity_list_btc)

    # com_dict = { node: com for node, com in graph_partition_louvain_btc}
    # node_i = { node: index for index, node in  enumerate(nodes) }
    # adj, _ = getAdjAndFiedlerVector(np.unique(nodes_connectivity_list_btc), nodes_connectivity_list_btc)
    # for n1, n2 in nodes_connectivity_list_btc:
    #     if com_dict[n1] != com_dict[n2]:
    #         adj[node_i[n1],node_i[n2]] = adj[node_i[n2],node_i[n1]] = 0
    # plt.imshow(adj*255, vmin=0, vmax=1)
    # plt.savefig('adj_lou_btc.png')

    # partitions_abcd = dict()
    # for com in graph_partition_louvain_btc:
    #     if com[1] in partitions_abcd.keys():
    #         partitions_abcd[com[1]].append(com[0])
    #     else:
    #         partitions_abcd[com[1]] = [com[0]]
    
    # # print(partitions_abcd)
    # comms_leng = len(partitions_abcd.keys())
    # print("dict keys", comms_leng)

    # G = nx.from_edgelist(nodes_connectivity_list_btc)
    # pos = nx.spring_layout(G, seed=42069)
    # options = {"edgecolors": "tab:gray", "node_size": 20, "alpha": 1}
    # i = 0
    # for partition in partitions_abcd.values():
    #     # nx.draw_networkx_nodes(G, pos, nodelist=partition, node_color="tab:red", **options)
    #     nx.draw_networkx_nodes(G, pos, nodelist=partition, node_color=(i * 255) / comms_leng, **options)
    #     i += 1
    # nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.6)
    # plt.savefig('graph_btc_louvain.png')

    # print(time.time() - t1, "seconds")
    # exit(0)
