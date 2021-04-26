# Imports
import networkx as nx
import numpy as np

# Methods
def drawPath(g1, pos, path, color, width):
    for i in range(0,len(path)-1):
        nx.draw_networkx_edges(g1, pos=pos,edgelist=[(path[i],path[i+1])], edge_color=color, width=width)

def getIntersection(truePath, predPath):
    commonPath = []
    for n in truePath:
        if n in predPath:
            commonPath.append(n)
    return commonPath

def getCommonPath(truePath, predPath):
    commonPath = []
    for n in truePath:
        if n in predPath:
            commonPath.append(n)
        else:
            break
    return commonPath

def getDescendants(G, t):
    nodes = []
    for n in G.neighbors(t):
        nodes.append(n)
    return nodes

def getAncestors(G, t):
    nodes = []
    for n in G.predecessors(t):
        nodes.append(n)
    return nodes

def getNeighbors(G, t):
    ancestors = getAncestors(G, t)
    nodes = []
    for a in ancestors:
        nodes += getDescendants(G, a)
    while(t in nodes):
        nodes.remove(t)
    return nodes

def deleteFromPath(path, toDelete):
    newPath = path.copy()
    for n in toDelete:
        while n in newPath:
            newPath.remove(n)
    return newPath

def getLeafNode(path):
    return path[-1]

def determineTruePathSet(G, trueLabels):
    W_dj = {}
    for n in trueLabels:
        W_dj[n] = []
        for path in nx.all_simple_paths(G, source="root", target=n):
            W_dj[n].append(path)
    return W_dj

def determine_M_Values(P_d, W_dj):
    M_values = []
    for path in P_d:
        m_max = -1
        for n in W_dj:
            for p in W_dj[n]:
                m_val = len(getCommonPath(p,path))
                if(m_val > m_max):
                    m_max = m_val
        M_values.append(m_max)
    return M_values
    
def generateSortedPredictions(G, trueLabels, P_d):
    W_dj = determineTruePathSet(G, trueLabels)
    M    = determine_M_Values(P_d, W_dj) # Step 1, 2
    P_d = [x for _, x in sorted(zip(M, P_d))]
    M.sort()
    P_d.reverse()
    M.reverse()
    return P_d, W_dj

def getHierarchical_TrueNegative(G, P_true, P_pred):
    commonPath = getCommonPath(P_true, P_pred)
    commonPathNodes  = []
    for n in commonPath:
        commonPathNodes += getNeighbors(G, n)
    commonPathNodes = list(set(deleteFromPath(commonPathNodes, P_true)))
    relevantDescendants = deleteFromPath(getDescendants(G, getLeafNode(commonPath)), P_true+P_pred)
    return len(commonPathNodes)+len(relevantDescendants)

def getHierarchicalConfusion_Tree_SPL_MLNP(G, P_true, P_pred):
    TP_h = len(getIntersection(P_true, P_pred))-1
    TN_h = getHierarchical_TrueNegative(G, P_true, P_pred)
    FP_h = len(deleteFromPath(P_pred, P_true))
    FN_h = len(deleteFromPath(P_true, P_pred))
    return [TP_h, TN_h, FP_h, FN_h]

def getShortestPathLength(paths):
    minL = -1
    shortestPath = []
    for path in paths:
        if((minL == -1) or (len(path)<minL)):
            minL = len(path)
            shortestPath = path
    return len(shortestPath)-1

def determineHierarchicalConfusionMatrix(G, trueLabels, P_d):
    Confusion_hk = []
    P_d, W_dj = generateSortedPredictions(G, trueLabels, P_d) # Includes Step 1 and 2
    for predPath in P_d:  # Step 3
        if (len(W_dj.keys())==0): # Step 3.1
            Confusion_hk.append([0, 0, len(predPath)-1, 0])
        else:
            m_max = -1
            selTruePath = ""
            selTrueLabel = ""
            for n in W_dj:
                for p in W_dj[n]:
                    m_val = len(getCommonPath(p,predPath))
                    if(m_val > m_max):
                        m_max = m_val
                        selTrueLabel = n # Step 3.2
                        selTruePath  = p
            Confusion_hk.append(getHierarchicalConfusion_Tree_SPL_MLNP(G, selTruePath, predPath)) # Step 3.3
            del W_dj[selTrueLabel] # Step 3.4
    Confusion_matrix = np.sum(np.asarray(Confusion_hk), axis=0) # Step 4
    if(len(W_dj.keys())!=0):
        for key in W_dj:
            Confusion_matrix[3] += getShortestPathLength(W_dj[key])
    return Confusion_matrix

def printHierarchicalConfusionMatrix(mat, title=""):
    if(not title==""):
        print(">>> HierarchicalConfusionMatrix ",title)
    print("\tTP\t",mat[0])
    print("\tTN\t",mat[1])
    print("\tFP\t",mat[2])
    print("\tFN\t",mat[3])
    