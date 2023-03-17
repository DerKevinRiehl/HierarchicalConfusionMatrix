# Imports
import networkx as nx
import numpy as np




# Methods
"""
This method determines the hierarchical confusion matrix for a given problem with structure "graph", true labels "true_labels" and prediction "pred_labels".
"""
def determineHierarchicalConfusionMatrix(graph, true_labels, pred_labels):
    confusion_hk = []
    pred_labels, w_dj = generateSortedPredictions(graph, true_labels, pred_labels) # Includes Step 1 and 2
    for pred_path in pred_labels:  # Step 3
        if (len(w_dj.keys())==0): # Step 3.1
            confusion_hk.append([0, 0, len(pred_path)-1, 0])
        else:
            m_max = -1
            sel_true_path = ""
            sel_true_label = ""
            for n in w_dj:
                for p in w_dj[n]:
                    m_val = len(getCommonPath(p,pred_path))
                    if(m_val > m_max):
                        m_max = m_val
                        sel_true_label = n # Step 3.2
                        sel_true_path  = p
            confusion_hk.append(getHierarchicalConfusion_Tree_SPL_MLNP(graph, sel_true_path, pred_path)) # Step 3.3
            del w_dj[sel_true_label] # Step 3.4
    confusion_matrix = np.sum(np.asarray(confusion_hk), axis=0) # Step 4
    if(len(w_dj.keys())!=0):
        for key in w_dj:
            confusion_matrix[3] += getShortestPathLength(w_dj[key])
    return confusion_matrix
    
"""
This method returns the predictions in sorted order with true paths w_dj and M values.
"""
def generateSortedPredictions(graph, true_labels, pred_labels):
    w_dj = determineTruePathSet(graph, true_labels)
    m    = determine_M_Values(pred_labels, w_dj) # Step 1, 2
    pred_labels = [x for _, x in sorted(zip(m, pred_labels))]
    m.sort()
    pred_labels.reverse()
    m.reverse()
    return pred_labels, w_dj

"""
This method determines a set of true paths w_dj from the true_labels.
"""
def determineTruePathSet(graph, true_labels):
    w_dj = {}
    for node in true_labels:
        w_dj[node] = []
        for path in nx.all_simple_paths(graph, source="root", target=node):
            w_dj[node].append(path)
    return w_dj

"""
This method determines the M values for predictions and true paths w_dj.
"""
def determine_M_Values(pred_labels, w_dj):
    m_values = []
    for path in pred_labels:
        m_max = -1
        for n in w_dj:
            for p in w_dj[n]:
                m_val = len(getCommonPath(p,path))
                if(m_val > m_max):
                    m_max = m_val
        m_values.append(m_max)
    return m_values

"""
This method calculates the common path between true_path and pred_path.
"""
def getCommonPath(true_path, pred_path):
    common_path = []
    for node in true_path:
        if node in pred_path:
            common_path.append(node)
        else:
            break
    return common_path

"""
This method determines the four values of the confusion matrix TP, TN, FP, FN for two paths path_true and path_pred
"""
def getHierarchicalConfusion_Tree_SPL_MLNP(graph, path_true, path_pred):
    tp_h = len(getCommonPath(path_true, path_pred))-1
    tn_h = getHierarchical_TrueNegative(graph, path_true, path_pred)
    fp_h = len(deleteFromPath(path_pred, path_true))
    fn_h = len(deleteFromPath(path_true, path_pred))
    return [tp_h, tn_h, fp_h, fn_h]

"""
This method determines the true negative for two paths path_true and path_pred.
"""
def getHierarchical_TrueNegative(graph, path_true, path_pred):
    common_path = getCommonPath(path_true, path_pred)
    common_path_nodes  = []
    for node in common_path:
        common_path_nodes += getNeighbors(graph, node)
    common_path_nodes = list(set(deleteFromPath(common_path_nodes, path_true)))
    relevant_descendants = deleteFromPath(getDescendants(graph, getLeafNode(common_path)), path_true+path_pred)
    return len(common_path_nodes)+len(relevant_descendants)

"""
This method determines a new_path from a given path by removing a certain to_delete node.
"""
def deleteFromPath(path, to_delete):
    new_path = path.copy()
    for node in to_delete:
        while node in new_path:
            new_path.remove(node)
    return new_path

"""
This method determines the shortest path length amongst a list of given paths.
"""
def getShortestPathLength(paths):
    min_l = -1
    shortest_path = []
    for path in paths:
        if((min_l == -1) or (len(path)<min_l)):
            min_l = len(path)
            shortest_path = path
    return len(shortest_path)-1

"""
This method prints the hierarchical confusion matrix to the console.
"""
def printHierarchicalConfusionMatrix(mat, title=""):
    if(not title==""):
        print(">>> HierarchicalConfusionMatrix ",title)
    print("\tTP\t",mat[0])
    print("\tTN\t",mat[1])
    print("\tFP\t",mat[2])
    print("\tFN\t",mat[3])

"""
This method returns a list of descendants of a given node t in a graph.
"""
def getDescendants(graph, t):
    nodes = []
    for node in graph.neighbors(t):
        nodes.append(node)
    return nodes

"""
This method returns a list of ancestors of a given node t in a graph.
"""
def getAncestors(graph, t):
    nodes = []
    for node in graph.predecessors(t):
        nodes.append(node)
    return nodes

"""
This method returns a list of neighbors of a given node t in a graph.
"""
def getNeighbors(graph, t):
    ancestors = getAncestors(graph, t)
    nodes = []
    for ancestor in ancestors:
        nodes += getDescendants(graph, ancestor)
    while(t in nodes):
        nodes.remove(t)
    return nodes

"""
This method returns the leaf node (deepest node) in a path.
"""
def getLeafNode(path):
    return path[-1]
        
"""
This method 
"""
def drawPath(graph, pos, path, color, width):
    for i in range(0,len(path)-1):
        nx.draw_networkx_edges(graph, pos=pos,edgelist=[(path[i],path[i+1])], edge_color=color, width=width)
