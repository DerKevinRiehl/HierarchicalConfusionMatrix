# Hierarchical Confusion Matrix
This GitHub repository includes the implementation of the hierarchical confusion matrix in Python, that was proposed in [Riehl et al., 2023] Riehl, K., Neunteufel, M., Hemberg, M. (2023). Hierarchical confusion matrix for classification performance evaluation. In revision at Journal of the Royal Statistical Society:Series C..

![Hierarchical Confusion Matrix Examples](https://github.com/DerKevinRiehl/HierarchicalConfusionMatrix/blob/main/ExampleProblems.png)

## Table of Contents
* File structure of this GitHub
* Installation using Pip
* Exemplary use
* Citations

## File structure of this GitHub
This GitHub consists of following parts...
* *HierarchicalConfusion.py* that includes the implementation of hierarchical confusion matrix.
* Four examples from the paper that show how to use the implementation and calculate evaluation measures based on the hierarchical confusion matrix.
  * ![Example_Figure4](https://github.com/DerKevinRiehl/HierarchicalConfusionMatrix/blob/main/JupyterNotebooks/Example_Figure4.ipynb) (*Example_Figure4.py*)
  * ![Example_TransposonClassification](https://github.com/DerKevinRiehl/HierarchicalConfusionMatrix/blob/main/JupyterNotebooks/Example_TransposonClassification.ipynb) (*Example_TransposonClassification.py*)
  * ![Example_GermEval2019_Task1A](https://github.com/DerKevinRiehl/HierarchicalConfusionMatrix/blob/main/JupyterNotebooks/Example_GermEval2019_Task1A.ipynb) (*Example_GermEval2019_Task1A.py*)
  * ![Example_GermEval2019_Task1B](https://github.com/DerKevinRiehl/HierarchicalConfusionMatrix/blob/main/JupyterNotebooks/Example_GermEval2019_Task1B.ipynb) (*Example_GermEval2019_Task1B.py*)
  * A folder *CaseStudies* that include the classification model predictions for the different examples.

## Installation using Pip
```
pip install numpy, networkx
```

## Exemplary use
Let us take the four examples from the figure above that was taken from the paper.
The core method of the implementation *determineHierarchicalConfusionMatrix(G, trueLabels, P_d)* in HierarchicalConfusion.py needs three arguments, a graph, a list of true labels and a list of prediction paths. As a result, it returns a numpy array with four elements, including true positives (TP), true negatives (TN), false positives (FP) and false negatives (FN) in this order.
For the first problem shown in Fig.4 (a), in which the true label is node "I", we can calculate the hierarchical confusion matrix for different predictions (x, 1, and 2) as follows:
```python
# Imports
import networkx as nx
from hierarchical_confusion_matrix import determineHierarchicalConfusionMatrix, printHierarchicalConfusionMatrix

# Generate strucutre = tree graph
graph = nx.DiGraph()
graph.add_edges_from([("root", "A"), ("root", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("C", "H"), ("D","I"), ("D","J"), ("D","K"), ("E","L"), ("E","M"), ("B","F"), ("B","G")])

# Set true label(s)
true_labels = ["I"]

# Set prediction path(s)
p_d0 = [["root","A","D","I"]]
p_d1 = [["root","A","D","J"]]
p_d2 = [["root","A","E","L"]]

# Calculate hierarchical confusion matrix
confusion_matrix_a0 = determineHierarchicalConfusionMatrix(graph, true_labels, p_d0)
confusion_matrix_a1 = determineHierarchicalConfusionMatrix(graph, true_labels, p_d1)
confusion_matrix_a2 = determineHierarchicalConfusionMatrix(graph, true_labels, p_d2)

# Print results
printHierarchicalConfusionMatrix(Confusion_matrix_a0, "Problem A, Prediction 0")
printHierarchicalConfusionMatrix(Confusion_matrix_a1, "Problem A, Prediction 1")
printHierarchicalConfusionMatrix(Confusion_matrix_a2, "Problem A, Prediction 2")
```
**Console Output:**
```console
HierarchicalConfusionMatrix  Problem A, Prediction 0
	TP	 3
	TN	 5
	FP	 0
	FN	 0
HierarchicalConfusionMatrix  Problem A, Prediction 1
	TP	 2
	TN	 4
	FP	 1
	FN	 1
HierarchicalConfusionMatrix  Problem A, Prediction 2
	TP	 1
	TN	 2
	FP	 2
	FN	 2
```

Further examples for the other three problems of the Fig.4 can be found in ![Example_Figure4](https://github.com/DerKevinRiehl/HierarchicalConfusionMatrix/blob/main/JupyterNotebooks/Example_Figure4.ipynb).
These examples show the calculation of the hierarchical confusion matrix for one single object of a dataset.

Examples for the calculation of the hierarchical confusion matrix for multiple object predictions can be found in ![Example_TransposonClassification](https://github.com/DerKevinRiehl/HierarchicalConfusionMatrix/blob/main/JupyterNotebooks/Example_TransposonClassification.ipynb), ![Example_GermEval2019_Task1A](https://github.com/DerKevinRiehl/HierarchicalConfusionMatrix/blob/main/JupyterNotebooks/Example_GermEval2019_Task1A.ipynb), and ![Example_GermEval2019_Task1B](https://github.com/DerKevinRiehl/HierarchicalConfusionMatrix/blob/main/JupyterNotebooks/Example_GermEval2019_Task1B.ipynb).

## Citations
Please cite our paper if you find hierarchical confusion matrix useful: [Riehl et al., 2023] Riehl, K., Neunteufel, M., Hemberg, M. (2023). Hierarchical confusion matrix for classification performance evaluation. In revision at Journal of the Royal Statistical Society:Series C..
