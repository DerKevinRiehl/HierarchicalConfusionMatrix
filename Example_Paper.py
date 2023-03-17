# Imports
import networkx as nx
import matplotlib.pyplot as plt
from hierarchical_confusion_matrix import determineHierarchicalConfusionMatrix, printHierarchicalConfusionMatrix

######################################################
####### (A) Exemplary (T, SPL, MLNP) problem #########
######################################################

##### Input:
    # Structure = Exemplary graph from paper
g_a = nx.DiGraph()
g_a.add_edges_from([("root", "A"), ("root", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("C", "H"), ("D","I"), ("D","J"), ("D","K"), ("E","L"), ("E","M"), ("B","F"), ("B","G")])
    # True labels
true_labels = ["I"]
    # Prediction Paths, Example 1
p_d0 = [["root","A","D","I"]]
p_d1 = [["root","A","D","J"]]
p_d2 = [["root","A","E","L"]]

###### Output:
confusion_matrix_a0 = determineHierarchicalConfusionMatrix(g_a, true_labels, p_d0)
confusion_matrix_a1 = determineHierarchicalConfusionMatrix(g_a, true_labels, p_d1)
confusion_matrix_a2 = determineHierarchicalConfusionMatrix(g_a, true_labels, p_d2)
printHierarchicalConfusionMatrix(confusion_matrix_a0, "Problem A, Prediction 0")
printHierarchicalConfusionMatrix(confusion_matrix_a1, "Problem A, Prediction 1")
printHierarchicalConfusionMatrix(confusion_matrix_a2, "Problem A, Prediction 2")




######################################################
###### (B) Exemplary (DAG, SPL, MLNP) problem ########
######################################################

##### Input:
    # Structure = Exemplary graph from paper
g_b = nx.DiGraph()
g_b.add_edges_from([("root", "A"), ("root", "B"), ("root", "D"), ("A", "C"), ("A", "D"), ("A", "E"), ("C", "H"), ("D","I"), ("D","J"), ("D","K"), ("E", "K"), ("E","L"), ("E","M"), ("B","F"), ("B","G")])
    # True labels
true_labels = ["I"]
    # Prediction Paths, Example 1
p_d0 = [["root","A","D","I"]]
p_d1 = [["root","D","J"]]
p_d2 = [["root","A","E","L"]]

###### Output:
confusion_matrix_b0 = determineHierarchicalConfusionMatrix(g_b, true_labels, p_d0)
confusion_matrix_b1 = determineHierarchicalConfusionMatrix(g_b, true_labels, p_d1)
confusion_matrix_b2 = determineHierarchicalConfusionMatrix(g_b, true_labels, p_d2)
printHierarchicalConfusionMatrix(confusion_matrix_b0, "Problem B, Prediction 0")
printHierarchicalConfusionMatrix(confusion_matrix_b1, "Problem B, Prediction 1")
printHierarchicalConfusionMatrix(confusion_matrix_b2, "Problem B, Prediction 2")




######################################################
###### (C) Exemplary (DAG, MPL, MLNP) problem ########
######################################################

##### Input:
    # Structure = Exemplary graph from paper
g_c = nx.DiGraph()
g_c.add_edges_from([("root", "A"), ("root", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("C", "H"), ("D","I"), ("D","J"),("D","K"),("E","L"),("E","M"),("B","E"),("B","F"),("B","G")])
    # True labels
true_labels = ["I", "F"]
    # Prediction Paths, Example 1
p_d0 = [["root","A","D","I"],["root","B","F"]]
p_d1 = [["root","A","D","I"],["root","A","D","J"]]
p_d2 = [["root","A","C","H"],["root","B","E","M"]]

###### Output:
confusion_matrix_c0 = determineHierarchicalConfusionMatrix(g_c, true_labels, p_d0)
confusion_matrix_c1 = determineHierarchicalConfusionMatrix(g_c, true_labels, p_d1)
confusion_matrix_c2 = determineHierarchicalConfusionMatrix(g_c, true_labels, p_d2)
printHierarchicalConfusionMatrix(confusion_matrix_c0, "Problem C, Prediction 0")
printHierarchicalConfusionMatrix(confusion_matrix_c1, "Problem C, Prediction 1")
printHierarchicalConfusionMatrix(confusion_matrix_c2, "Problem C, Prediction 2")




######################################################
###### (D) Exemplary (DAG, MPL, NMLNP) problem #######
######################################################

##### Input:
    # Structure = Exemplary graph from paper
g_d = nx.DiGraph()
g_d.add_edges_from([("root", "A"), ("root", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("C", "H"), ("D","I"), ("D","J"),("D","K"),("E","L"),("E","M"),("B","E"),("B","F"),("B","G")])
    # True labels
true_labels = ["D", "F"]
    # Prediction Paths, Example 1
P_d0 = [["root","A","D"],["root","B","F"]]
P_d1 = [["root","A","D","J"],["root","B"]]
P_d2 = [["root","A","C"],["root","B","E"]]

###### Output:
confusion_matrix_d0 = determineHierarchicalConfusionMatrix(g_d, true_labels, p_d0)
confusion_matrix_d1 = determineHierarchicalConfusionMatrix(g_d, true_labels, p_d1)
confusion_matrix_d2 = determineHierarchicalConfusionMatrix(g_d, true_labels, p_d2)
printHierarchicalConfusionMatrix(confusion_matrix_d0, "Problem D, Prediction 0")
printHierarchicalConfusionMatrix(confusion_matrix_d1, "Problem D, Prediction 1")
printHierarchicalConfusionMatrix(confusion_matrix_d2, "Problem D, Prediction 2")




######################################################
###### Draw Graphs ###################################
######################################################
plt.rcParams['figure.figsize'] = [20, 20]
plt.subplot(2,2,1)
plt.title("Problem A")
pos_a = nx.kamada_kawai_layout(g_a)
nx.draw_networkx(g_a, pos=pos_a, arrows=True)
plt.subplot(2,2,2)
plt.title("Problem B")
pos_b = nx.kamada_kawai_layout(g_b)
nx.draw_networkx(g_b, pos=pos_b, arrows=True)
plt.subplot(2,2,3)
plt.title("Problem C")
pos_c = nx.kamada_kawai_layout(g_c)
nx.draw_networkx(g_c, pos=pos_c, arrows=True)
plt.subplot(2,2,4)
plt.title("Problem D")
pos_d = nx.kamada_kawai_layout(g_d)
nx.draw_networkx(g_d, pos=pos_d, arrows=True)
