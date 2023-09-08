#-------------------------------------------------------------------------------
# Hubness HSP graph construction
#
# released under the MIT opensource software license 
# Copyright 2023 - Stephane Marchand-Maillet
# Email: stephane.marchand-maillet@unige.ch

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻

import numpy as np
import igraph as ig  # we use igraph as base structure for the graph

#⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
# 'step' in the comments refer to the description of the algorithm in the article:
# HubHSP graph: capturing local geometrical and statistical data properties via spanning graphs
# Stephane Marchand-Maillet and Edgar Chávez
# <Information Systems reference if accepted for publication>
#
#⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
# Construction queue: a set of marked neighbors
# the set ensures randomness in the progression
# could be a stack to mimic a DFS-like progression
# could be a DEQueue to mimic a BFS-like progression

def push(lQ,i):  # insert data[i] into q                         [steps 3 and 6]
    lQ[i] = True
    return lQ

def pop(lQ): # gets the first data from q                        [step 5]
    n = len(lQ)
    i = 0
    while (i<n) and (not(lQ[i])):
        i += 1
    if i<n:
        lQ[i] = False
    return lQ, i

#⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
# (squared) Euclidean distance: only sorting is used throughout

def distance(x,y):                               
    return ((x-y)**2).sum()

#⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
# Get edge IDs built from the "start" vertex to the "ends" vertices

def getEid(lG, start, ends):
    ans = []
    for l in range(len(ends)):
        ans.append((start,ends[l]))
    return lG.get_eids(ans)

#⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
# Basic procedure to construct kNN graph as a igraph structure
# !!! may be slow !!!!
# could be replaced by faster algorithm or precomputed 

def compute_knn(lData,lK):
    print(f"Building {lK}NN graph...")
    
    lN = lData.shape[0]
    lKnn = ig.Graph(lN,directed=True)  # kNN is a directed graph

    dist = np.zeros(lN)
    lE = [[] for _ in range(lN*lK)]  # lM = lN*k edges
    lEWeight = np.zeros(lN*lK)       # we keep the distance values
    lM = 0
    for i in range(lN):
        if not(i % 1000):
            print(f"node {i}/{lN}   ",end='\r')
            
        for j in range(lN):
            dist[j] = distance(lData[i],lData[j])  # distance values
        idx = np.argsort(dist)                   # sorted

        for j in range(1,lK+1): # we skip i itself 
            #lKnn.add_edge(i,idx[j])     # this would be slower 
            lE[lM] = (i,idx[j])          # build the edges
            lEWeight[lM] = dist[idx[j]]  # keep the distances
            lM += 1

    lKnn.add_edges(lE)  # build the graph at once
    
    print(f"  ---> {lKnn.vcount()} nodes / {lKnn.ecount()} edges ")
    
    return lKnn, lEWeight

#⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
# Main procedure: HubHSP construction
#   lData: NxD np.array of data
#   lKnn: the kNN igraph structure
#         used here as a proximity index eg for steps 7 and 13
#   lEWeight: the corresponding kNN edge weights (distance values)
#   lStart: index of the the start vertex

def compute_hubhsp(lData,lKnn,lEWeight,lStart):
    print(f"Building HubHSP graph from kNN graph...")

    # initialize local parameters 
    lN = lData.shape[0]
    lHHSP = ig.Graph(lN,directed=True)    # HubHSP is a directed graph
    lHWeight = []

    lH = np.zeros(lN)                     # to store the hubness values
    Q = np.zeros(lN, dtype=bool)          # construction queue

    # initialize the procedure
    push(Q,lStart)                        # start from the start vertex

    mark = np.zeros(lN, dtype=bool)       # mark the node so as to explore them once only

    # Main iterative construction procedure
    while Q.sum() > 0 and mark.sum()<lN:
        Q,i = pop(Q)                         # step 5
        while i<lN and mark[i]:
            Q,i = pop(Q)
        
        if i==lN:   # Q was empty => disconnected kNN graph (DFS-like procedure)
            i = 0
            while mark[i]: # => re-start from the first unmarked vertex
                i += 1
            print(f"**** Restarting from {i}")
            
        print(f"Currently at {i}",end='')
        
        neighb = lKnn.neighbors(i,mode='out')  # forward neighbors of i
        nNeighb = len(neighb)                  # their number (should be k, implicit here)
        eid = getEid(lKnn,i,neighb)            # the edges between i and its neighbors
        push(Q,neighb)                         # step 6: these are next candidates for the construction

        rho = np.min(lEWeight[eid]) # step 7: distance to closest neighbor

        discard = np.zeros(nNeighb, dtype=bool) # discarded data (none at initialization)
        
        while discard.sum() < nNeighb:     # step 8: while not all neighbors have been discarded
            sH = lH[neighb]                # step 9: current hubness of neighbors
            idx = np.argsort(sH)           
            idx = idx[::-1]                # step 9: sorting indices (decreasing order)
            idxJ = 0
            while idxJ < nNeighb and discard[idx[idxJ]]: # step 9: find strongest non-discarded hub
                idxJ += 1
            j = neighb[idx[idxJ]]                        # step 9: j is its index
            distJ = lEWeight[eid[idx[idxJ]]]             # precomputed distance i -> j
            lHHSP.add_edge(i,j)                          # step 10: and becomes a HubHSP neighbor
            lHWeight.append(distJ)                       # step 10: keep the edge weight (distance)
            lH[j] += 1                                   # step 11: and its hubness increases
            discard[idx[idxJ]]= True                     # and gets discarded as a neighbor
            
            xJtilde = lData[i] + rho*(lData[j]-lData[i])/distJ  # step 12: projection over C_i
            for l in range(nNeighb):                  # step 13: HSP rule to disard neighbors if any
                if not(discard[l]):
                    if lEWeight[eid[l]] > distance(xJtilde,lData[neighb[l]]):
                        discard[l] = True

        mark[i] = True                     # done with vertex i
        print(f"   -> {mark.sum()}   ",end='\r')       # How many done

    print("\n done")
    
    return lHHSP,lHWeight  # as a igraph structure

#⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻ THIS IS THE END ⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
