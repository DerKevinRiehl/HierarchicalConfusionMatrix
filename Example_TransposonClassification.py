############################################################################
##### Hierarchical confusion matrix ########################################
##### Kevin Riehl (kevin.riehl.de@gmail.com, 2021) #########################
############################################################################

# Transposon Classification, Benchmark of different classifiers
# Transposon Classification: (Tree, SPL, MLNP) classification problem

# Imports
import os
import numpy as np
import networkx as nx
from hierarchical_confusion_matrix import determineHierarchicalConfusionMatrix

# Methods
"""
This method generates the structure / hierarchical taxonomy used in this transposon classification example.
It returns a graph object, a list of all node labels "classes", a list of the node levels "levels",
and a list of all superior nodes "s_nodes"
"""
def generateStructure():
    # Add edges
    edges = []
    edges.append(["root","1"])
    edges.append(["1","1/1"])
    edges.append(["1/1","1/1/1"])
    edges.append(["1/1","1/1/2"])
    edges.append(["1/1","1/1/3"])
    edges.append(["1","1/2"])
    edges.append(["1/2","1/2/1"])
    edges.append(["1/2","1/2/2"])
    edges.append(["root","2"])
    edges.append(["2","2/1"])
    edges.append(["2/1","2/1/1"])
    edges.append(["2/1","2/1/2"])
    edges.append(["2/1","2/1/3"])
    edges.append(["2/1","2/1/4"])
    edges.append(["2/1","2/1/5"])
    edges.append(["2/1","2/1/6"])
    edges.append(["2","2/2"])
    edges.append(["2","2/3"])
    # Convert to Networkx Graph
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    # Set of classes and levels
    classes = ["1","1/1","1/1/1","1/1/2","1/1/3","1/2","1/2/1","1/2/2","2","2/1","2/1/1","2/1/2","2/1/3","2/1/4","2/1/5","2/1/6","2/2","2/3"]
    levels  = [1,   2,    3,      3,      3,      2,    3,      3,      1,  2,    3,      3,      3,      3,      3,      3,      2,    2]    
    s_nodes = getSuperiorNodes(classes)
    return graph, classes, levels, s_nodes

"""
This method determines the superior nodes of all nodes in a given list "classes".
"""
def getSuperiorNodes(classes):
    s_nodes = list()
    for i in range(0,len(classes)):
        if(not "/" in classes[i]):
            s_nodes.append(list())
        else:
            l = list()
            sup = getSuperiorNode(classes[i])
            while(sup!=""):
                l.append(sup)
                sup = getSuperiorNode(sup)
            s_nodes.append(l)
    return s_nodes

"""
This method determines the superior node of a given node "c".
"c" represents the label of a node, e.g. "1/2/1".
The superior node therefore can be determined by splitting the last "/" part away.
Finally, the result would be "1/2".
"""
def getSuperiorNode(c):
    if(not "/" in c):
        return ""
    else:
        parts = c.split("/")
        new_c = ""
        for p in range(0, len(parts)-1):
            new_c = new_c + parts[p] + "/"
        return new_c[:-1]
    
"""
This method loads the transposon classification data from two given files "true_label_file" and "pred_label_file".
"""
def loadEvaluationData_TransposonClassification(graph, true_label_file, pred_label_file):
    true_label_data = loadInferenceData(classes, levels, true_label_file)
    pred_label_data = loadInferenceData(classes, levels, pred_label_file)
    eval_label_data = {}
    for key in range(0,len(true_label_data)):
        eval_label_data[key] = {}
        eval_label_data[key]["true"] = [true_label_data[key]]
        paths = []
        for p in nx.all_simple_paths(graph, source="root", target=pred_label_data[key]):
            paths.append(p)
        eval_label_data[key]["pred"] = [p]
    return eval_label_data

"""
This method loads the probability data from a given "file" considering "classes" and "labels".
"""
def loadInferenceData(classes, levels, file):
    labels = []
    f = open(file,"r")
    line = f.readline()
    while line!="" and line!="\n":
        binary_line_parts = convertProbabilityToBinaryLabel(line, classes, levels, sNodes)
        calculated_label = convertBinaryToPredictionLabel(classes, levels, binary_line_parts)
        if(calculated_label != "-"):
            labels.append(calculated_label)
        line = f.readline()
    f.close()
    return labels

"""
This method converts the probability data from the TransposonClassification Dataset (values between 0.0 and 1.0 for each node) to binary predictions (1 or 0).
It does this selecting always the highest probabilites on each level, and all other nodes outside of the path get "0".
Example input:  "0.9817518248175182 0.9753649635036497 0.0 0.45468369829683697 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
Example output: [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
"""
def convertProbabilityToBinaryLabel(line, classes, levels, sNodes):    
    threshold = 0
    probs = readProbabilitiesFromLine(line)
    if(len(probs)==0):
        return "-"
    preds = list()
    for i in range(0,len(probs)):
        preds.append(0)        
    maxLevel = max(levels)
    lastNode = ""
    for l in range(1, maxLevel+1): # for each level
        # determine candidates on that level
        candidates = list() 
        candidatesIdx = list()
        for i in range(0, len(classes)): 
            if(levels[i]==l):
                if(lastNode==""):
                    candidates.append(classes[i])
                    candidatesIdx.append(i)
                else:
                    if(lastNode in sNodes[i]):
                        candidates.append(classes[i])
                        candidatesIdx.append(i)
        # determine candidate with highest probability
        mx = -1
        ix = -1
        cx = -1
        for c in range(0, len(candidates)):
            if(probs[candidatesIdx[c]]>mx):
                ix = candidatesIdx[c]
                cx = c
                mx = probs[candidatesIdx[c]]
        # add prediction
        if(mx >= threshold):
            preds[ix] = 1
            lastNode = candidates[cx]
        else:
            break
    return preds


"""
This method converts a input string "line" into a list of float values by splitting the numbers by whitespace characters.
Example input:  "0.9817518248175182 0.9753649635036497 0.0 0.45468369829683697 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0"
Example output: [0.9817518248175182, 0.9753649635036497, 0.0, 0.45468369829683697, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
"""
def readProbabilitiesFromLine(line):
    parts = []
    for p in line.replace("\n","").split(" "):
        if(p!=""):
            parts.append(float(p))
    return parts

"""
This method converts the binary data from the function convertProbabilityToBinaryLabel() (1s and 0s) to the label names of the taxonomy.
Example input:  [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Example output: "1/1/2"
"""
def convertBinaryToPredictionLabel(classes, levels, parts):
    idx = -1
    level = -1
    for c in range(0,len(classes)):
        if(parts[c]==1):
            if(levels[c]>level):
                level = levels[c]
                idx = c
    return classes[idx]


    


# Generate Structure
graph, classes, levels, sNodes = generateStructure()

# List all available algorithms in the dataset
path = "CaseStudies/TransposonClassification"
algo_folder = os.listdir(os.path.join(path))

# For each algorithm determine hierarchical confusion matrix and evaluation measures (F1, PPV, REC, ACC, MCC)
print("algo\tF1\tPPV\tREC\tACC\tMCC\tTP\tTN\tFP\tFN")
for algo in algo_folder: 
    true_label_file = os.path.join(path,algo, "ALL_small/inference10", "truelabels.txt")
    pred_label_file = os.path.join(path,algo, "ALL_small/inference10", "predictions.txt")
    evalLabel_data = loadEvaluationData_TransposonClassification(graph, true_label_file, pred_label_file)
    
    # Predict Confusion Matrix
    h_confusion = {}
    h_confusion_total = []
    for key in evalLabel_data:
        h_confusion[key] = determineHierarchicalConfusionMatrix(graph, evalLabel_data[key]["true"], evalLabel_data[key]["pred"])
        h_confusion_total.append(h_confusion[key])
    h_confusion_total = np.sum(np.asarray(h_confusion_total),axis=0)
    F1 = 2*h_confusion_total[0]/(2*h_confusion_total[0]+h_confusion_total[2]+h_confusion_total[3])
    PPV = h_confusion_total[0]/(h_confusion_total[0]+h_confusion_total[2])
    REC = (h_confusion_total[0])/(h_confusion_total[0]+h_confusion_total[3])
    ACC = (h_confusion_total[0]+h_confusion_total[1])/(h_confusion_total[0]+h_confusion_total[1]+h_confusion_total[2]+h_confusion_total[3])
    MCC = (h_confusion_total[0]*h_confusion_total[1]-h_confusion_total[2]*h_confusion_total[3])/np.sqrt((h_confusion_total[0]+h_confusion_total[2])*(h_confusion_total[0]+h_confusion_total[3])*(h_confusion_total[1]+h_confusion_total[2])*(h_confusion_total[1]+h_confusion_total[3]))
    print(algo, "\t", F1, "\t", PPV, "\t", REC, "\t", ACC, "\t", MCC, "\t", h_confusion_total[0], "\t", h_confusion_total[1], "\t", h_confusion_total[2], "\t", h_confusion_total[3])    
