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
from HierarchicalConfusion import determineHierarchicalConfusionMatrix

def getSuperiorNode(c):
    if(not "/" in c):
        return ""
    else:
        parts = c.split("/")
        newC = ""
        for p in range(0, len(parts)-1):
            newC = newC + parts[p] + "/"
        return newC[:-1]
    
def getSuperiorNodes(classes, levels):
    sNodes = list()
    for i in range(0,len(classes)):
        if(not "/" in classes[i]):
            sNodes.append(list())
        else:
            l = list()
            sup = getSuperiorNode(classes[i])
            while(sup!=""):
                l.append(sup)
                sup = getSuperiorNode(sup)
            sNodes.append(l)
    return sNodes

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
    G = nx.DiGraph()
    G.add_edges_from(edges)
    # Set of classes and levels
    classes = ["1","1/1","1/1/1","1/1/2","1/1/3","1/2","1/2/1","1/2/2","2","2/1","2/1/1","2/1/2","2/1/3","2/1/4","2/1/5","2/1/6","2/2","2/3"]
    levels  = [1,   2,    3,      3,      3,      2,    3,      3,      1,  2,    3,      3,      3,      3,      3,      3,      2,    2]    
    sNodes = getSuperiorNodes(classes,levels)
    return G, classes, levels, sNodes

def convertLineToBinary(line):
    parts = []
    for p in line.replace("\n","").split(" "):
        if(p!=""):
            parts.append(float(p))
    return parts
    
def convertBinaryToPredictionLabel(classes, levels, parts):
    idx = -1
    level = -1
    for c in range(0,len(classes)):
        if(parts[c]==1):
            if(levels[c]>level):
                level = levels[c]
                idx = c
    return classes[idx]

def convertProbabilityToBinaryLabel(line, classes, levels, sNodes):    
    threshold = 0
    probs = convertLineToBinary(line)
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

def loadProbabilityData(classes, levels, file):
    labels = []
    f = open(file,"r")
    line = f.readline()
    while line!="" and line!="\n":
        calculatedLabel = convertBinaryToPredictionLabel(classes, levels, convertProbabilityToBinaryLabel(line, classes, levels, sNodes))
        if(calculatedLabel != "-"):
            labels.append(calculatedLabel)
        line = f.readline()
    f.close()
    return labels

def loadEvaluationData_TransposonClassification(trueLabel_file, predLabel_file):
    trueLabel_data = loadProbabilityData(classes, levels, trueLabel_file)
    predLabel_data = loadProbabilityData(classes, levels, predLabel_file)
    evalLabel_data = {}
    for key in range(0,len(trueLabel_data)):
        evalLabel_data[key] = {}
        evalLabel_data[key]["true"] = [trueLabel_data[key]]
        paths = []
        for p in nx.all_simple_paths(G, source="root", target=predLabel_data[key]):
            paths.append(p)
        evalLabel_data[key]["pred"] = [p]
    return evalLabel_data

# Generate Structure
G, classes, levels, sNodes = generateStructure()

#
path = "CaseStudies/TransposonClassification"
algoFolder = os.listdir(os.path.join(path))

dataSet = "ALL_small/inference10"
#print("algo\tF1\tPPV\tREC\tACC\tMCC\tTP\tTN\tFP\tFN")
print("algo\tTP\tTN\tFP\tFN")
for algo in algoFolder:
    trueLabel_file = os.path.join(path,algo,dataSet,"truelabels.txt")
    predLabel_file = os.path.join(path,algo,dataSet,"predictions.txt")
    evalLabel_data = loadEvaluationData_TransposonClassification(trueLabel_file, predLabel_file)
    
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