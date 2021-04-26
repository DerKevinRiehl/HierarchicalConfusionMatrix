# Hierarchical Confusion Matrix
This GitHub repository includes the implementation of the hierarchical confusion matrix in Python, that was proposed in [CITATION STILL IN PROGRESS].

![Hierarchical Confusion Matrix Examples](https://github.com/DerKevinRiehl/HierarchicalConfusionMatrix/blob/main/ExampleProblems.png)

## Table of Contents
* File structure of this GitHub
* Dependencies
* Exemplary use
* Citations

## File structure of this GitHub
This GitHub consists of following parts...
* *HierarchicalConfusion.py* that includes the implementation of hierarchical confusion matrix.
* Four examples (*Example_Paper.py*, *Example_TransposonClassification.py*, *Example_GermEval2019_Task1A.py*, *Example_GermEval2019_Task1B.py*) that show how to use the implementation and calculate evaluation measures based on the hierarchical confusion matrix.
* A folder *CaseStudies* that include the classification model predictions for the different examples.

## Dependencies
* We use the [networkx](https://anaconda.org/anaconda/networkx) package for all graph related tasks. It can be easily installed using conda or pip.
```
conda install -c anaconda networkx 
```
```
pip install networkx
```
* Moreover we use [numpy](https://anaconda.org/anaconda/numpy) package. It can be easily installed using conda or pip.
```
conda install -c anaconda numpy 
```
```
pip install numpy
```

## Exemplary use
Let us take the four examples from the figure above that was taken from the paper.
The core method of the implementation *determineHierarchicalConfusionMatrix(G, trueLabels, P_d)* in HierarchicalConfusion.py needs three arguments, a graph, a list of true labels and a list of prediction paths. As a result, it returns a numpy array with four elements, including true positives (TP), true negatives (TN), false positives (FP) and false negatives (FN) in this order.
For the first problem shown in Fig. (a), in which the true label is node "I", we can calculate the hierarchical confusion matrix for different predictions (x, 1, and 2) as follows:
```
# Imports
import networkx as nx
from HierarchicalConfusion import determineHierarchicalConfusionMatrix, printHierarchicalConfusionMatrix

# Generate strucutre = tree graph
G_A = nx.DiGraph()
G_A.add_edges_from([("root", "A"), ("root", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("C", "H"), ("D","I"), ("D","J"), ("D","K"), ("E","L"), ("E","M"), ("B","F"), ("B","G")])

# Set true label(s)
trueLabels = ["I"]

# Set prediction path(s)
P_d0 = [["root","A","D","I"]]
P_d1 = [["root","A","D","J"]]
P_d2 = [["root","A","E","L"]]

# Calculate hierarchical confusion matrix
confusion_matrix_A0 = determineHierarchicalConfusionMatrix(G_A, trueLabels, P_d0)
confusion_matrix_A1 = determineHierarchicalConfusionMatrix(G_A, trueLabels, P_d1)
confusion_matrix_A2 = determineHierarchicalConfusionMatrix(G_A, trueLabels, P_d2)

# Print results
printHierarchicalConfusionMatrix(Confusion_matrix_A0, "Problem A, Prediction 0")
printHierarchicalConfusionMatrix(Confusion_matrix_A1, "Problem A, Prediction 1")
printHierarchicalConfusionMatrix(Confusion_matrix_A2, "Problem A, Prediction 2")
```
Further examples for the other three problems of the figure can be found in *Example_Paper.py*.
These examples show the calculation of the hierarchical confusion matrix for one single object of a dataset.

Examples for the calculation of the hierarchical confusion matrix for multiple object predictions can be found in *Example_TransposonClassification.py*, *Example_GermEval2019_Task1A.py*, and *Example_GermEval2019_Task1B.py*.
We suggest to aggregate the single object confusion matrixes by summing them up.

## Citations
Please cite our paper if you find hierarchical confusion matrix useful: [CITATION STILL IN PROGRESS].
