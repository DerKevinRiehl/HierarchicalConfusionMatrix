############################################################################
##### Hierarchical confusion matrix ########################################
##### Kevin Riehl (kevin.riehl.de@gmail.com, 2021) #########################
############################################################################

# GermEval2019 Competition on hierarchical classification of texts
# Task 1A: (1-Level-Tree, MPL, MLNP) classification problem
# More infos can be found here: https://2019.konvens.org/germeval

# Imports
import os
import numpy as np
import networkx as nx
from HierarchicalConfusion import determineHierarchicalConfusionMatrix

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
This method loads the evaluation data from GermEval2019_Task1A (true labels and prediction labels)
"""
def loadEvaluationData_GermEval2019_Task1A(true_label_file, pred_label_file):    # Load True Labels of task A (Tree(1Level), MPL, MLNP)
    true_label_data = {}
    pred_label_data = {}
    eval_label_data = {}
    # Load true_label_data
    f = open(true_label_file, "r", encoding="utf8")
    f.readline()
    line = f.readline()
    while not line.startswith("subtask_b"):
        parts = line.replace("\n","").split("\t")
        true_label_data[parts[0]] = parts[1:]
        line = f.readline()
    f.close()
    # load pred_label_data
    f = open(pred_label_file, "r", encoding="utf8")
    f.readline()
    line = f.readline()
    while not line.startswith("subtask_b") and not line=="":
        parts = line.replace("\n","").split("\t")
        predPaths = []
        for n in parts[1:]:
            predPaths.append(["root",n])
        pred_label_data[parts[0]] = predPaths
        line = f.readline()
    f.close()
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
    return eval_label_data, n_nopredictions




# Load GermEval2019 hierarchy
path = "CaseStudies/GermEval2019"
hierarchy_file = os.path.join(path,"hierarchy.txt")
graph = loadHierarchy(hierarchy_file, level=1)

# List all available algorithms
true_label_file = os.path.join(path,"blurbs_test_label.txt")
algo_folder = os.listdir(os.path.join(path, "system-submissions/test-phase-txt"))

# For each algorithm determine hierarchical confusion matrix
print("algo\tF1\tPPV\tREC\tACC\tMCC\tTP\tTN\tFP\tFN")
for algo in algo_folder:
    pred_label_file = os.path.join(path, "system-submissions/test-phase-txt", algo)
    eval_label_data, nn = loadEvaluationData_GermEval2019_Task1A(true_label_file, pred_label_file)
    # Determine Confusion Matrix
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
