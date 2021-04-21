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

def loadEvaluationData_GermEval2019_Task1A(trueLabel_file, predLabel_file):    # Load True Labels of task A (Tree(1Level), MPL, MLNP)
    trueLabel_data = {}
    predLabel_data = {}
    evalLabel_data = {}
    
    f = open(trueLabel_file, "r", encoding="utf8")
    f.readline()
    line = f.readline()
    while not line.startswith("subtask_b"):
        parts = line.replace("\n","").split("\t")
        trueLabel_data[parts[0]] = parts[1:]
        line = f.readline()
    f.close()
    
    f = open(predLabel_file, "r", encoding="utf8")
    f.readline()
    line = f.readline()
    while not line.startswith("subtask_b") and not line=="":
        parts = line.replace("\n","").split("\t")
        predPaths = []
        for n in parts[1:]:
            predPaths.append(["root",n])
        predLabel_data[parts[0]] = predPaths
        line = f.readline()
    f.close()
    
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

    return evalLabel_data, n_nopredictions

# Load GermEval2019 Hierarchy
path = "CaseStudies/GermEval2019"

hierarchy_file = os.path.join(path,"hierarchy.txt")
G = loadHierarchy(hierarchy_file, level=1)

trueLabel_file = os.path.join(path,"blurbs_test_label.txt")
algoFolder = os.listdir(os.path.join(path, "system-submissions/test-phase-txt"))
#print("algo\tF1\tPPV\tREC\tACC\tMCC\tTP\tTN\tFP\tFN")
print("algo\tTP\tTN\tFP\tFN")
for algo in algoFolder:
    predLabel_file = os.path.join(path, "system-submissions/test-phase-txt", algo)
    evalLabel_data, nn = loadEvaluationData_GermEval2019_Task1A(trueLabel_file, predLabel_file)
    
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