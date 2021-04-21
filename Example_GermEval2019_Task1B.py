############################################################################
##### Hierarchical confusion matrix ########################################
##### Kevin Riehl (kevin.riehl.de@gmail.com, 2021) #########################
############################################################################

# GermEval2019 Competition on hierarchical classification of texts
# Task 1B: (DAG, MPL, NMLNP) classification problem
# More infos can be found here: https://2019.konvens.org/germeval

# Imports
import os
import numpy as np
import networkx as nx
from HierarchicalConfusion import determineHierarchicalConfusionMatrix, getLeafNode

# Methods
def loadHierarchy(file, level=-1):
    # Load GermEval2019 Hierarchy
    f = open(file, "r", encoding="utf8")
    edges = []
    for l in f.readlines():
        edges.append(l.replace("\n","").split("\t"))
    f.close()
    # Determine root nodes
    rootNodes = []
    for i in range(0,len(edges)):
        cat = edges[i][0]
        if(cat in rootNodes):
            continue
        found = False
        for j in range(0,len(edges)):
            if(cat == edges[j][1] and i != j):
                found = True
                break
        if(not found):
            rootNodes.append(cat)
    # Add root node connection
    if(level==1):
        edges = []
    for n in rootNodes:
        edges.append(["root",n])
    # Convert to Networkx Graph
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G

def loadEvaluationData_GermEval2019_Task1B(trueLabel_file, predLabel_file):    # Load True Labels of task A (Tree(1Level), MPL, MLNP)
    trueLabel_data = {}
    predLabel_data = {}
    evalLabel_data = {}
    # Load data from true label file
    f = open(trueLabel_file, "r", encoding="utf8")
    line = f.readline()
    while not line.startswith("subtask_b"):
        line = f.readline()
    line = f.readline()
    while line!="":
        parts = line.replace("\n","").split("\t")
        lList = []
        for p in parts[1:]:
            if(p!=""):
                lList.append(p)
        trueLabel_data[parts[0]] = lList
        line = f.readline()
    f.close()    
    # Load data from prediction label file
    f = open(predLabel_file, "r", encoding="utf8")
    line = f.readline()
    while not line.startswith("subtask_b"):
        line = f.readline()
    line = f.readline()
    while line!="":
        parts = line.replace("\n","").split("\t")
        lList = []
        for p in parts[1:]:
            if(p!=""):
                lList.append(p)
        predLabel_data[parts[0]] = lList
        line = f.readline()
    f.close()    
    # Convert data for hierarchical classification paths
    for key in trueLabel_data:
        minimPaths = createMinimPathsFromLabels(G, trueLabel_data[key])
        trueLabelCorrected = []
        for p in minimPaths:
            trueLabelCorrected.append(getLeafNode(p))
        trueLabel_data[key] = trueLabelCorrected
    # Convert data for hierarchical classification paths
    for key in predLabel_data:
        minimPaths = createMinimPathsFromLabels(G, predLabel_data[key])
        if(len(minimPaths)==0):
            minimPaths = [["root"]]
        predLabel_data[key] = minimPaths
    # Process evaluation data results
    for key in trueLabel_data:
        if(key in predLabel_data):
            evalLabel_data[key] = {}
            evalLabel_data[key]["true"] = trueLabel_data[key]
            evalLabel_data[key]["pred"] = predLabel_data[key]
    n_nopredictions = 0
    for key in trueLabel_data:
        if(key not in predLabel_data):
            evalLabel_data[key] = {}
            evalLabel_data[key]["true"] = trueLabel_data[key]
            evalLabel_data[key]["pred"] = [["root"]]
            n_nopredictions += 1
    # Return results
    return evalLabel_data, n_nopredictions

def countNodesOnPath(path, nodeList):
    ctr = 0
    intersect = []
    for n in nodeList:
        if(n in path):
            ctr += 1
            intersect.append(n)
    return ctr, intersect

def createMinimPathsFromLabels(G, labelData):
    if(len(labelData)==0):
        return []
    selected_Nodes = []
    selected_Paths = []
    remaining_Nodes = labelData.copy()
    remaining_Paths = []
    for val in labelData:
        for p in nx.all_simple_paths(G, "root", val):
            remaining_Paths.append(p)
    while True:
        # Select path that covers most of remaining nodes
        maxN = -1
        sNodes = []
        sPath  = []
        for path in remaining_Paths:
            ctr, nod = countNodesOnPath(path, remaining_Nodes)
            if(ctr>maxN):
                maxN   = ctr
                sNodes = nod.copy()
                sPath  = path.copy()
        # Add Path to selected Paths and remove from remaining
        selected_Paths.append(sPath)
        for n in sNodes:
            remaining_Nodes.remove(n)
            selected_Nodes.append(n)
        remaining_Paths.remove(sPath)
        # Break loop if all nodes covered
        if(len(remaining_Nodes)==0):
            break
    return selected_Paths

# Load GermEval2019 Hierarchy
path = "CaseStudies/GermEval2019"

hierarchy_file = os.path.join(path,"hierarchy.txt")
G = loadHierarchy(hierarchy_file) 

trueLabel_file = os.path.join(path,"blurbs_test_label.txt")
algoFolder = os.listdir(os.path.join(path, "system-submissions/test-phase-txt"))
#print("algo\tF1\tPPV\tREC\tACC\tMCC\tTP\tTN\tFP\tFN")
print("algo\tTP\tTN\tFP\tFN")
for algo in algoFolder:
    predLabel_file = os.path.join(path, "system-submissions/test-phase-txt", algo)
    f = open(predLabel_file, "r", encoding="utf8")
    lines = f.readlines()
    f.close()
    if("subtask_b\n" not in lines):
        continue
    evalLabel_data, nn = loadEvaluationData_GermEval2019_Task1B(trueLabel_file, predLabel_file)
    # Predict Confusion Matrix
    hConfusion = {}
    hConfusionTotal = []
    for key in evalLabel_data:
        hConfusion[key] = determineHierarchicalConfusionMatrix(G, evalLabel_data[key]["true"], evalLabel_data[key]["pred"])
        hConfusionTotal.append(hConfusion[key])
    hConfusionTotal = np.sum(np.asarray(hConfusionTotal),axis=0)
#    F1 = 2*hConfusionTotal[0]/(2*hConfusionTotal[0]+hConfusionTotal[2]+hConfusionTotal[3])
#    PPV = hConfusionTotal[0]/(hConfusionTotal[0]+hConfusionTotal[2])
#    REC = (hConfusionTotal[0])/(hConfusionTotal[0]+hConfusionTotal[3])
#    ACC = (hConfusionTotal[0]+hConfusionTotal[1])/(hConfusionTotal[0]+hConfusionTotal[1]+hConfusionTotal[2]+hConfusionTotal[3])
#    MCC = (hConfusionTotal[0]*hConfusionTotal[1]-hConfusionTotal[2]*hConfusionTotal[3])/np.sqrt((hConfusionTotal[0]+hConfusionTotal[2])*(hConfusionTotal[0]+hConfusionTotal[3])*(hConfusionTotal[1]+hConfusionTotal[2])*(hConfusionTotal[1]+hConfusionTotal[3]))
#    print(algo, "\t", F1, "\t", PPV, "\t", REC, "\t", ACC, "\t", MCC, "\t", hConfusionTotal[0], "\t", hConfusionTotal[1], "\t", hConfusionTotal[2], "\t", hConfusionTotal[3])    
    print(algo, "\t", hConfusionTotal[0], "\t", hConfusionTotal[1], "\t", hConfusionTotal[2], "\t", hConfusionTotal[3])    

## Draw Graph
#pos = nx.planar_layout(G)
#nx.draw_networkx(G, pos=pos, arrows=True)