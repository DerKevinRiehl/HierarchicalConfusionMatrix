# Imports
import networkx as nx
import matplotlib.pyplot as plt
from HierarchicalConfusion import determineHierarchicalConfusionMatrix, printHierarchicalConfusionMatrix

######################################################
####### (A) Exemplary (T, SPL, MLNP) problem #########
######################################################

##### Input:
    # Structure = Exemplary graph from paper
G_A = nx.DiGraph()
G_A.add_edges_from([("root", "A"), ("root", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("C", "H"), ("D","I"), ("D","J"), ("D","K"), ("E","L"), ("E","M"), ("B","F"), ("B","G")])
    # True labels
trueLabels = ["I"]
    # Prediction Paths, Example 1
P_d0 = [["root","A","D","I"]]
P_d1 = [["root","A","D","J"]]
P_d2 = [["root","A","E","L"]]

###### Output:
Confusion_matrix_A0 = determineHierarchicalConfusionMatrix(G_A, trueLabels, P_d0)
Confusion_matrix_A1 = determineHierarchicalConfusionMatrix(G_A, trueLabels, P_d1)
Confusion_matrix_A2 = determineHierarchicalConfusionMatrix(G_A, trueLabels, P_d2)





######################################################
###### (B) Exemplary (DAG, SPL, MLNP) problem ########
######################################################

##### Input:
    # Structure = Exemplary graph from paper
G_B = nx.DiGraph()
G_B.add_edges_from([("root", "A"), ("root", "B"), ("root", "D"), ("A", "C"), ("A", "D"), ("A", "E"), ("C", "H"), ("D","I"), ("D","J"), ("D","K"), ("E", "K"), ("E","L"), ("E","M"), ("B","F"), ("B","G")])
    # True labels
trueLabels = ["I"]
    # Prediction Paths, Example 1
P_d0 = [["root","A","D","I"]]
P_d1 = [["root","D","J"]]
P_d2 = [["root","A","E","L"]]

###### Output:
Confusion_matrix_B0 = determineHierarchicalConfusionMatrix(G_B, trueLabels, P_d0)
Confusion_matrix_B1 = determineHierarchicalConfusionMatrix(G_B, trueLabels, P_d1)
Confusion_matrix_B2 = determineHierarchicalConfusionMatrix(G_B, trueLabels, P_d2)




######################################################
###### (C) Exemplary (DAG, MPL, MLNP) problem ########
######################################################

##### Input:
    # Structure = Exemplary graph from paper
G_C = nx.DiGraph()
G_C.add_edges_from([("root", "A"), ("root", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("C", "H"), ("D","I"), ("D","J"),("D","K"),("E","L"),("E","M"),("B","E"),("B","F"),("B","G")])
    # True labels
trueLabels = ["I", "F"]
    # Prediction Paths, Example 1
P_d0 = [["root","A","D","I"],["root","B","F"]]
P_d1 = [["root","A","D","I"],["root","A","D","J"]]
P_d2 = [["root","A","C","H"],["root","B","E","M"]]

###### Output:
Confusion_matrix_C0 = determineHierarchicalConfusionMatrix(G_C, trueLabels, P_d0)
Confusion_matrix_C1 = determineHierarchicalConfusionMatrix(G_C, trueLabels, P_d1)
Confusion_matrix_C2 = determineHierarchicalConfusionMatrix(G_C, trueLabels, P_d2)




######################################################
###### (D) Exemplary (DAG, MPL, NMLNP) problem #######
######################################################

##### Input:
    # Structure = Exemplary graph from paper
G_D = nx.DiGraph()
G_D.add_edges_from([("root", "A"), ("root", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("C", "H"), ("D","I"), ("D","J"),("D","K"),("E","L"),("E","M"),("B","E"),("B","F"),("B","G")])
    # True labels
trueLabels = ["D", "F"]
    # Prediction Paths, Example 1
P_d0 = [["root","A","D"],["root","B","F"]]
P_d1 = [["root","A","D","J"],["root","B"]]
P_d2 = [["root","A","C"],["root","B","E"]]

###### Output:
Confusion_matrix_D0 = determineHierarchicalConfusionMatrix(G_D, trueLabels, P_d0)
Confusion_matrix_D1 = determineHierarchicalConfusionMatrix(G_D, trueLabels, P_d1)
Confusion_matrix_D2 = determineHierarchicalConfusionMatrix(G_D, trueLabels, P_d2)




######################################################
###### Print Hierarchical Confusion Matrices #########
######################################################
printHierarchicalConfusionMatrix(Confusion_matrix_A0, "Problem A, Prediction 0")
printHierarchicalConfusionMatrix(Confusion_matrix_A1, "Problem A, Prediction 1")
printHierarchicalConfusionMatrix(Confusion_matrix_A2, "Problem A, Prediction 2")
printHierarchicalConfusionMatrix(Confusion_matrix_B0, "Problem B, Prediction 0")
printHierarchicalConfusionMatrix(Confusion_matrix_B1, "Problem B, Prediction 1")
printHierarchicalConfusionMatrix(Confusion_matrix_B2, "Problem B, Prediction 2")
printHierarchicalConfusionMatrix(Confusion_matrix_C0, "Problem C, Prediction 0")
printHierarchicalConfusionMatrix(Confusion_matrix_C1, "Problem C, Prediction 1")
printHierarchicalConfusionMatrix(Confusion_matrix_C2, "Problem C, Prediction 2")
printHierarchicalConfusionMatrix(Confusion_matrix_D0, "Problem D, Prediction 0")
printHierarchicalConfusionMatrix(Confusion_matrix_D1, "Problem D, Prediction 1")
printHierarchicalConfusionMatrix(Confusion_matrix_D2, "Problem D, Prediction 2")



######################################################
###### Draw Graphs ###################################
######################################################
plt.subplot(2,2,1)
plt.title("Problem A")
posA = nx.planar_layout(G_A)
nx.draw_networkx(G_A, pos=posA, arrows=True)
plt.subplot(2,2,2)
plt.title("Problem A")
posB = nx.planar_layout(G_B)
nx.draw_networkx(G_B, pos=posB, arrows=True)
plt.subplot(2,2,3)
plt.title("Problem A")
posC = nx.planar_layout(G_C)
nx.draw_networkx(G_C, pos=posC, arrows=True)
plt.subplot(2,2,4)
plt.title("Problem A")
posD = nx.planar_layout(G_D)
nx.draw_networkx(G_D, pos=posD, arrows=True)