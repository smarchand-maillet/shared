#-------------------------------------------------------------------------------
# Hubness HSP graph construction test
#
# released under the MIT opensource software license 
# Copyright 2023 - Stephane Marchand-Maillet
# Email: stephane.marchand-maillet@unige.ch

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻

import numpy as np
import matplotlib.pyplot as plt
import hubhsp.hubhsp as hhsp

#-------------------------------------------------------------------------------- 
if __name__ == '__main__':
    
# data file is D columns of N data features, possibly with header (skip lines)
    N = 10000
    D = 2
    k = 50      # parameter  for the kNN

    # data = np.genfromtxt('dataFileName',delimiter=',',skip_header=0) # when reading CSV
    # data = np.random.randn(N,D) # generate random gaussian data
    data = np.random.rand(N,D) # generate random uniform data

    print(f"data: {data.shape[0]} x {data.shape[1]}D")

    knn,kWeight = hhsp.compute_knn(data,k)  # computes the kNN igraph structure

    lStart = 0   # starting data index
    hubHSP,hWeight = hhsp.compute_hubhsp(data,knn,kWeight,lStart) # get the HubHSP
    
    h = np.array(hubHSP.degree(mode='in'))           # hubness (in degree)  
    plt.hist(h,bins=h.max().astype(int))             # hubness histograms (mostly zeros)
    
    outDegree = np.array(hubHSP.degree(mode='out'))           # out degree
    plt.hist(outDegree,bins=outDegree.max().astype(int))      # out degree histogram

    plt.title("hubness and out-degree")
    
    if data.shape[1]==2:  
        plt.figure()  # if 2D: plot the hubness over the data (as size and color)
        idx = np.argsort(h)
        plt.scatter(data[idx,0],data[idx,1],s=0.2+10*h[idx]/h.max(),c='r') 
        plt.scatter(data[idx,0],data[idx,1],s=0.1+10*h[idx]/h.max(),c=h[idx],cmap='Reds')
        plt.colorbar() 
        plt.title(f"hubness ({N} data)")
        
        n=200

        plt.figure()   # select representatives from highest hubness
        plt.scatter(data[:,0],data[:,1],s=0.1,c='c') 
        plt.scatter(data[idx[-n:],0],data[idx[-n:],1],s=5,c='r') 
        plt.title(f"selection from hubness: {n} data/{N}")
        
        plt.figure()   # compare with random selection
        idx = np.argsort(np.random.rand(N))
        plt.scatter(data[:,0],data[:,1],s=0.1,c='c') 
        plt.scatter(data[idx[-200:],0],data[idx[-200:],1],s=5,c='m') 
        plt.title(f"selection from random: {n} data/{N}")
        
    plt.show()
#⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻ THIS IS THE END ⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻


