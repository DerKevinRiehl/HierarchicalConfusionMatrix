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
"""
This method loads the structure / classification hierarchy for GermEval2019 data from a given file,
and returns it as a graph object.
"""
def loadHierarchy(file, level=-1):
    # Load GermEval2019 Hierarchy
    f = open(file, "r", encoding="utf8")
    edges = []
    for l in f.readlines():
        edges.append(l.replace("\n","").split("\t"))
    f.close()
    # Determine root nodes
    root_nodes = []
    for i in range(0,len(edges)):
        cat = edges[i][0]
        if(cat in root_nodes):
            continue
        found = False
        for j in range(0,len(edges)):
            if(cat == edges[j][1] and i != j):
                found = True
                break
        if(not found):
            root_nodes.append(cat)
    # Add root node connection
    if(level==1):
        edges = []
    for n in root_nodes:
        edges.append(["root",n])
    # Convert to Networkx Graph
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    return graph

"""
This method loads the evaluation data from GermEval2019_Task1B (true labels and prediction labels)
"""
def loadEvaluationData_GermEval2019_Task1B(true_label_file, pred_label_file):    # Load True Labels of task A (Tree(1Level), MPL, MLNP)
    true_label_data = {}
    pred_label_data = {}
    eval_label_data = {}
    # Load data from true label file
    f = open(true_label_file, "r", encoding="utf8")
    line = f.readline()
    while not line.startswith("subtask_b"):
        line = f.readline()
    line = f.readline()
    while line!="":
        parts = line.replace("\n","").split("\t")
        l_list = []
        for p in parts[1:]:
            if(p!=""):
                l_list.append(p)
        true_label_data[parts[0]] = l_list
        line = f.readline()
    f.close()    
    # Load data from prediction label file
    f = open(pred_label_file, "r", encoding="utf8")
    line = f.readline()
    while not line.startswith("subtask_b"):
        line = f.readline()
    line = f.readline()
    while line!="":
        parts = line.replace("\n","").split("\t")
        l_list = []
        for p in parts[1:]:
            if(p!=""):
                l_list.append(p)
        pred_label_data[parts[0]] = l_list
        line = f.readline()
    f.close()    
    # Convert data for hierarchical classification paths
    for key in true_label_data:
        minim_paths = createMinimPathsFromLabels(graph, true_label_data[key])
        true_labels_corrected = []
        for p in minim_paths:
            true_labels_corrected.append(getLeafNode(p))
        true_label_data[key] = true_labels_corrected
    # Convert data for hierarchical classification paths
    for key in pred_label_data:
        minim_paths = createMinimPathsFromLabels(graph, pred_label_data[key])
        if(len(minim_paths)==0):
            minim_paths = [["root"]]
        pred_label_data[key] = minim_paths
    # Process evaluation data results
    for key in true_label_data:
        if(key in pred_label_data):
            eval_label_data[key] = {}
            eval_label_data[key]["true"] = true_label_data[key]
            eval_label_data[key]["pred"] = pred_label_data[key]
    n_nopredictions = 0
    for key in true_label_data:
        if(key not in pred_label_data):
            eval_label_data[key] = {}
            eval_label_data[key]["true"] = true_label_data[key]
            eval_label_data[key]["pred"] = [["root"]]
            n_nopredictions += 1
    # Return results
    return eval_label_data, n_nopredictions

"""
This method counts the number of nodes from node_list that appear in a path.
"""
def countNodesOnPath(path, node_list):
    ctr = 0
    intersect = []
    for node in node_list:
        if(node in path):
            ctr += 1
            intersect.append(node)
    return ctr, intersect

"""
This method creates the minimum length path to a given node from labels in label_data.
"""
def createMinimPathsFromLabels(graph, label_data):
    if(len(label_data)==0):
        return []
    selected_nodes = []
    selected_paths = []
    remaining_nodes = label_data.copy()
    remaining_paths = []
    for val in label_data:
        for p in nx.all_simple_paths(graph, "root", val):
            remaining_paths.append(p)
    while True:
        # Select path that covers most of remaining nodes
        max_n = -1
        s_nodes = []
        s_path  = []
        for path in remaining_paths:
            ctr, nod = countNodesOnPath(path, remaining_nodes)
            if(ctr>max_n):
                max_n   = ctr
                s_nodes = nod.copy()
                s_path  = path.copy()
        # Add Path to selected Paths and remove from remaining
        selected_paths.append(s_path)
        for n in s_nodes:
            remaining_nodes.remove(n)
            selected_nodes.append(n)
        remaining_paths.remove(s_path)
        # Break loop if all nodes covered
        if(len(remaining_nodes)==0):
            break
    return selected_paths




# Load GermEval2019 Hierarchy
path = "CaseStudies/GermEval2019"
hierarchy_file = os.path.join(path,"hierarchy.txt")
graph = loadHierarchy(hierarchy_file) 

# List all available algorithms
true_label_file = os.path.join(path,"blurbs_test_label.txt")
algo_folder = os.listdir(os.path.join(path, "system-submissions/test-phase-txt"))

# For each algorithm determine hierarchical confusion matrix
print("algo\tF1\tPPV\tREC\tACC\tMCC\tTP\tTN\tFP\tFN")
for algo in algo_folder:
    pred_label_file = os.path.join(path, "system-submissions/test-phase-txt", algo)
    f = open(pred_label_file, "r", encoding="utf8")
    lines = f.readlines()
    f.close()
    if("subtask_b\n" not in lines):
        continue
    eval_label_data, nn = loadEvaluationData_GermEval2019_Task1B(true_label_file, pred_label_file)
    # Predict Confusion Matrix
    h_confusion = {}
    h_confusion_total = []
    for key in eval_label_data:
        h_confusion[key] = determineHierarchicalConfusionMatrix(graph, eval_label_data[key]["true"], eval_label_data[key]["pred"])
        h_confusion_total.append(h_confusion[key])
    h_confusion_total = np.sum(np.asarray(h_confusion_total),axis=0)
    F1 = 2*h_confusion_total[0]/(2*h_confusion_total[0]+h_confusion_total[2]+h_confusion_total[3])
    PPV = h_confusion_total[0]/(h_confusion_total[0]+h_confusion_total[2])
    REC = (h_confusion_total[0])/(h_confusion_total[0]+h_confusion_total[3])
    ACC = (h_confusion_total[0]+h_confusion_total[1])/(h_confusion_total[0]+h_confusion_total[1]+h_confusion_total[2]+h_confusion_total[3])
    MCC = (h_confusion_total[0]*h_confusion_total[1]-h_confusion_total[2]*h_confusion_total[3])/np.sqrt((h_confusion_total[0]+h_confusion_total[2])*(h_confusion_total[0]+h_confusion_total[3])*(h_confusion_total[1]+h_confusion_total[2])*(h_confusion_total[1]+h_confusion_total[3]))
    print(algo, "\t", F1, "\t", PPV, "\t", REC, "\t", ACC, "\t", MCC, "\t", h_confusion_total[0], "\t", h_confusion_total[1], "\t", h_confusion_total[2], "\t", h_confusion_total[3])    
