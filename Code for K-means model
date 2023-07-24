# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:20:01 2023

@author: georg
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import scipy.io

# Load the .mat file
data = scipy.io.loadmat(r'C:\Users\georg\Neuroscience\NiN internship\Pavlov_Pellet.mat')

data = data['DA_allStrLocs']

list_of_labels = []
DATA_matrices = []

counter = 0 

vms_data = data[0,0]
dms_data = data[0,1]
dls_data = data[0,2]   





track_of_indeces = []

"""
track_of_indeces = []
####  VMS DATA  #### 
for k in range(0,20):
    
    if len(vms_data[0,k]) == 0:
        continue
    else:
        if len((vms_data[0,k])[0,:]) < 50:
            
            for i in range(0,len((vms_data[0,k])[0,:])):
                DATA_matrices.append((vms_data[0,k])[:,i])
                
                list_of_labels.append([1,0,0])
                track_of_indeces.append([0,k])
                counter += 1
        else: 
            for m in range(0,50):
                      
                DATA_matrices.append((vms_data[0,k])[:,m])
                track_of_indeces.append([0,k])     
                list_of_labels.append([1,0,0])
                counter += 1   
"""
#session_number = data['Session']
counter = 0
trial_number = 0                
for i in range(0,60):
      
    trial_number += 1
    if len((vms_data[0,i])) == 0:
           continue
    else:
            
        for k in range(0,len((vms_data[0,i])[0])):
            
            
                              
            DATA_matrices.append((vms_data[0,i])[:,k])
                    
            track_of_indeces.append([0,k,counter,i+1,1,counter])    
     
          
            counter += 1   

x_p = np.arange(0,len(DATA_matrices[0])/float(10),0.1)
plt.plot(x_p, np.mean(DATA_matrices[0:50],axis = 0))
plt.show()

plt.plot(x_p, np.mean(DATA_matrices[15:17],axis = 0))
plt.show()




counter = 0
trial_number = 0                
for i in range(0,60):
      
    trial_number += 1
    if len((dms_data[0,i])) == 0:
           continue
    else:
        for k in range(0,len((dms_data[0,i])[0])):
            
      
                    
                              
              DATA_matrices.append((dms_data[0,i])[:,k])
                  
              track_of_indeces.append([1,k,counter,i+1,2,counter])      
            
           
              counter += 1   



counter = 0
trial_number = 0                
for i in range(0,60):
      
    trial_number += 1
    if len((dls_data[0,i])) == 0:
           continue
    else:
        for k in range(0,len((dls_data[0,i])[0])):
            
      
                    
                              
              DATA_matrices.append((dls_data[0,i])[:,k])
                  
              track_of_indeces.append([2,k,counter,i+1,3,counter])      
       
              counter += 1   






print(len(DATA_matrices))
print(len(track_of_indeces))
list_of_deleted = []
new_i = -1
for i in range(0,len(DATA_matrices)):
    new_i += 1
    for k in range(0,350):
        if np.isnan((DATA_matrices[new_i])[k]) == True:
            del track_of_indeces[new_i]
            del DATA_matrices[new_i]
           
            new_i -= 1
            break
        else:
            continue
print(len(DATA_matrices))  
print(len(track_of_indeces)) 

"""
new_data_matrices = []
new_data_indeces = []
for i in range(0,len(DATA_matrices)):
    add = 0
    for k in range(0,7):
        
        new_data_matrices.append((DATA_matrices[i])[add:add+50])
        add += 50
        new_data_indeces.append(track_of_indeces[i])

print(len(new_data_matrices))
print(len(new_data_indeces))

DATA_matrices = new_data_matrices
track_of_indeces = new_data_indeces

print(np.array(track_of_indeces).shape)
"""


import numpy as np
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.cluster import KMeans

# Scale the selected time series data
X = TimeSeriesScalerMeanVariance().fit_transform(DATA_matrices)

# Select the number of clusters based on the SSE curve
k = 3
# Initialize the KMeans object
kmeans = KMeans(n_clusters=k)

# Fit the KMeans model to the data
kmeans.fit(X.reshape(X.shape[0], X.shape[1]))




colors = ['blue','darkblue','cyan']
cluster_indices = []
p_values_list = []
rat_id = []
for t in range(0,1):    
 
    p_values =[]  
    for m in range(3,4):    
        # Select the number of clusters based on the SSE curve
        k = m
        # Initialize the KMeans object
        kmeans = KMeans(n_clusters=k)
        
        # Fit the KMeans model to the data
        kmeans.fit(X.reshape(X.shape[0], X.shape[1]))
        
        
        # Initialize an empty list for each cluster
        cluster_indices = [[] for _ in range(k)]
        
        # Iterate over the labels and append the corresponding index to the list for each cluster
        for i, label in enumerate(kmeans.labels_):
            cluster_indices[label].append(i)
        
        # Print the indices for each cluster
        #for i, indices in enumerate(cluster_indices):
         #   print(f"Cluster {i}: {indices}")
            
        """
        n_vms_cluster = []
        n_dls_cluster = []
        
        for m in range(0,k):
                
            n_vms = 0
            n_dms = 0
            n_dls = 0
        
            for i in cluster_indices[m]:
            
                if (track_of_indeces[i])[0] == 0:
                    
                    n_vms += 1
                    
                elif (track_of_indeces[i])[0] == 1:
                    
                    n_dms += 1
                else:
                    
                    n_dls += 1
                
            n_vms_cluster.append(n_vms)
            n_dls_cluster.append(n_dls)
            
        print(n_vms_cluster)
        print(n_dls_cluster) 
        
       
        correlation_matrix = []
        
        for m in range(0,k):
                
            
            correlation_counter = 0
            for i in cluster_indices[m]:
            
                
            
                for t in cluster_indices[m]:
                    
                    if (track_of_indeces[i])[0] == 0 and (track_of_indeces[t])[0] == 2 and (track_of_indeces[i])[1] == (track_of_indeces[t])[1] and (track_of_indeces[i])[3] == (track_of_indeces[t])[3]:
                        correlation_counter += 1 
                        
            correlation_matrix.append(correlation_counter)        
                    
        
                    
        
        
        print(correlation_matrix)
        """         
        track = 0
        meanx = []
        cluster_indicess = []
        cluster_rat_id = []
        cluster_rat_order = []
        cluster_regions = []
        for m in range(0,k):
            trial_counter_list = []    
            trial_counter = 0 
            rat_id = []
            rat_order = []
            regions = []
        
            for i in cluster_indices[m]:
            
                trial_counter += ((track_of_indeces[i])[1])/float(len(cluster_indices[m]))
                trial_counter_list.append((track_of_indeces[i])[1])
                track += 1
                rat_id.append((track_of_indeces[i])[3])
                rat_order.append((track_of_indeces[i])[5])
                regions.append((track_of_indeces[i])[4])
             
            cluster_rat_id.append(rat_id)
            cluster_regions.append(regions)
            cluster_indicess.append(trial_counter_list)
            cluster_rat_order.append(rat_order)
            meanx.append(trial_counter)   
            
        
        
        list_of_means =[]
        list_of_std =[]
        # Plot the indices for each cluster and calculate mean and std
        for i in range(0,k):
            
            x_values = np.random.uniform(i, i+0.9, size=len(cluster_indicess[i]))
            plt.scatter(x_values, cluster_indicess[i],color = colors[i] , s=10, alpha=0.5)
            mean = np.mean(cluster_indicess[i])
            list_of_means.append(mean)
            std = np.std(cluster_indicess[i])
            list_of_std.append(std)
            plt.errorbar(i+0.5, mean, std, fmt='ko',markersize=4, ecolor='red', capsize=5)
            
            # Print mean and std for the current cluster
            #print(f"Cluster {i} mean: {mean:.2f}, std: {std:.2f}")
        
        # Set the plot title and labels
     
        plt.xlabel("Cluster",fontsize = 14, fontweight='bold')
        plt.ylabel("Trial Index",fontsize = 14, fontweight='bold')
        
        # Set the x-axis ticks at the center of each cluster
        plt.xticks(np.arange(k)+0.45, np.arange(k),fontsize=14)
        plt.yticks(fontsize=14)
        
        # Show the plot
        plt.show()
    
        
        
        
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        from scipy.stats import f_oneway
        # Perform one-way ANOVA
        f_stat, p_value = f_oneway(*cluster_indicess)
        
        p_values.append(p_value)
        print("F statistic:", f_stat)
        print("P value:", p_value)
        
        # Perform Tukey's HSD test
        n_samples = [len(group) for group in cluster_indicess]
        labels = np.repeat(np.arange(1, len(n_samples)+1), n_samples)
        tukey_results = pairwise_tukeyhsd(np.concatenate(cluster_indicess), labels)
        print("Tukey's HSD test:")
        print(tukey_results)
        print(k)
        
    
    p_values_list.append(p_values)




from scipy.stats import sem
# Iterate over the labels and append the corresponding index to the list for each cluster
for i, label in enumerate(kmeans.labels_):
    cluster_indices[label].append(i)

# Create a new figure with subplots for each cluster
fig, axs = plt.subplots(1, k, figsize=(8, 6), sharex=True)

for m in range(k):
    cluster_data = [DATA_matrices[i] for i in cluster_indices[m]]

    # Calculate mean and standard error of the mean for the current cluster
    cluster_mean = np.mean(cluster_data, axis=0)
    cluster_sem = sem(cluster_data, axis=0)

    # Plot the mean with the standard error of the mean for the current cluster
    axs[m].plot(cluster_mean, color='blue', linewidth=2)
    axs[m].fill_between(
        range(len(cluster_mean)),
        cluster_mean - cluster_sem,
        cluster_mean + cluster_sem,
        alpha=0.3
    )

    # Set the subplot title and labels
    axs[m].set_title(f"Cluster {m}")
    axs[m].set_ylabel("Value")

# Set the x-axis label for the last subplot
axs[-1].set_xlabel("Time")

plt.tight_layout()
plt.show()




rat_and_trial = []
correlated_regions = []

for q in range(0,k):
    rat_and_trial_loop = []
    correlated_regions_loop = []
    indexes = []
    for i in range(len(cluster_rat_id[q])):
        for j in range(i + 1, len(cluster_rat_id[q])):
            if (cluster_rat_id[q])[i] == (cluster_rat_id[q])[j]:
                indexes.append((i, j))
    
    for index_pair in indexes:
        if (cluster_indicess[q])[index_pair[0]] == (cluster_indicess[q])[index_pair[1]]:
            rat_and_trial_loop.append([(cluster_rat_id[q])[index_pair[0]],(cluster_indicess[q])[index_pair[0]]])
            correlated_regions_loop.append([(cluster_regions[q])[index_pair[0]],(cluster_regions[q])[index_pair[1]]])
    rat_and_trial.append(rat_and_trial_loop)
    correlated_regions.append(correlated_regions_loop)


#### Plotting the individual rat correlation hits for each cluster #####
all_rat_ids = np.array([16, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 58, 60])
hits_for_each_rat = np.zeros(32)
cluster_hits_for_each_rat = []
correlated_regions_for_each_rat = [[[] for _ in range(32)] for _ in range(k)]
for u in range(0,k):
    adder = 0
    hits_for_each_rat_loop = np.zeros(32)
    correlated_regions_for_each_rat_loop = []
    for i in all_rat_ids:
        
        for y in range(0,len((rat_and_trial[u]))):
            
            if i == ((rat_and_trial[u])[y])[0]:
                
                hits_for_each_rat[adder] += 1
                hits_for_each_rat_loop[adder] += 1
                
                ((correlated_regions_for_each_rat[u])[adder]).append(((correlated_regions[u])[y])[0] + ((correlated_regions[u])[y])[1])
                
                
        adder += 1       
    cluster_hits_for_each_rat.append(hits_for_each_rat_loop)       
            



# Create a figure with three subplots
fig, axs = plt.subplots(1, k, figsize=(16, 4),sharey=True)
# Loop over each subplot and plot the corresponding data
for i, ax in enumerate(axs):
    
    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    ax.bar(all_rat_ids, cluster_hits_for_each_rat[i],color = color[i],width=1)
    ax.set_title(f"Cluster {i}",fontsize = 16, fontweight='bold')
    ax.set_xlabel("Rat Ids",fontsize = 16, fontweight='bold')
    ax.set_xticklabels(ax.get_xticks().astype(int),fontsize = 16, fontweight='bold')
    ax.set_yticklabels(ax.get_yticks().astype(int),fontsize = 16, fontweight='bold')
    
    if i == 0:
        ax.set_ylabel("# of correlated trials",fontsize = 16, fontweight='bold')
# Show the plot

plt.show()

numbers_of_correlations_for_each_rat =  [[[0,0,0] for _ in range(32)] for _ in range(k)]
####### Fixing empty matrices and changing values to VMS,DMS,DLS ##############
for i in range(0,len(correlated_regions_for_each_rat)):
    
    for j in range(0,len(correlated_regions_for_each_rat[i])):
        
        if len((correlated_regions_for_each_rat[i])[j]) == 0:
            
            ((correlated_regions_for_each_rat[i])[j]).append(0)
            
        else:
            
            for h in range(0,len((correlated_regions_for_each_rat[i])[j])):
                
                if ((correlated_regions_for_each_rat[i])[j])[h] == 3:
                    
                    (((numbers_of_correlations_for_each_rat[i])[j])[0]) += 1
                    
                elif ((correlated_regions_for_each_rat[i])[j])[h] == 4:
                    
                    (((numbers_of_correlations_for_each_rat[i])[j])[1]) += 1
                
                elif ((correlated_regions_for_each_rat[i])[j])[h] == 5:
                    
                    (((numbers_of_correlations_for_each_rat[i])[j])[2]) += 1
                    
            
      
# Create a figure with three subplots
fig, axs = plt.subplots(1, k, figsize=(16, 4),sharey = True)
# Loop over each subplot and plot the corresponding data
for i, ax in enumerate(axs):
    
    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for c in range(0,len(all_rat_ids)):
            
        for w in range(0,k):
            
            ax.bar(all_rat_ids[c], ((numbers_of_correlations_for_each_rat[i])[c])[w],color = color[w+4],width=1)
    ax.set_title(f"Cluster {i}",fontsize =16, fontweight='bold')
    ax.set_xlabel("Rat Ids",fontsize = 16, fontweight='bold')
    ax.set_xticklabels(ax.get_xticks().astype(int),fontsize = 16, fontweight='bold')
    ax.set_yticklabels(ax.get_yticks().astype(int),fontsize = 16, fontweight='bold')
    if i == 0:
        ax.set_ylabel("# of correlated trials",fontsize =16, fontweight='bold')
# Show the plot
plt.show()

rat_and_trial = []
correlated_regions = []
vms_dls_correlated_signals = []
vms_dms_correlated_signals = []
dls_vms_correlated_signals = []
dms_vms_correlated_signals = []
dms_dls_correlated_signals = []
dls_dms_correlated_signals = []


vms_next_trial = []
dms_next_trial = []
dls_next_trial = []

vms_dls_next_trial = []
vms_dms_next_trial = []
dls_vms_next_trial = []
dms_vms_next_trial = []
dms_dls_next_trial = []
dls_dms_next_trial = []

vms_dls_trial_index = []
vms_dms_trial_index = []
dls_vms_trial_index = []
dms_vms_trial_index = []
dms_dls_trial_index = []
dls_dms_trial_index = []

vms_dls_rat_order = []
vms_dms_rat_order = []
dms_dls_rat_order = []

    



k = 3
for q in range(0,k):
    rat_and_trial_loop = []
    correlated_regions_loop = []
    indexes = []
    vms_dls_correlated_signals_loop = []
    vms_dms_correlated_signals_loop = []
    dls_vms_correlated_signals_loop = []
    dms_vms_correlated_signals_loop = []
    dms_dls_correlated_signals_loop = []
    dls_dms_correlated_signals_loop = []
    
    vms_dls_next_trial_loop = []
    vms_dms_next_trial_loop = []
    dls_vms_next_trial_loop = []
    dms_vms_next_trial_loop = []
    dms_dls_next_trial_loop = []
    dls_dms_next_trial_loop = []
    
    vms_dls_trial_index_loop = []
    vms_dms_trial_index_loop = []
    dls_vms_trial_index_loop = []
    dms_vms_trial_index_loop = []
    dms_dls_trial_index_loop = []
    dls_dms_trial_index_loop = []
    
    vms_dls_trial_index_loop = []
    vms_dms_trial_index_loop = []
    dls_vms_trial_index_loop = []
    dms_vms_trial_index_loop = []
    dms_dls_trial_index_loop = []
    dls_dms_trial_index_loop = []
    
    
    vms_dls_rat_order_loop = []
    vms_dms_rat_order_loop = []
    dms_dls_rat_order_loop = []

    
    
    for i in range(len(cluster_rat_id[q])):
        for j in range(i + 1, len(cluster_rat_id[q])):
            if (cluster_rat_id[q])[i] == (cluster_rat_id[q])[j]:
                indexes.append((i, j))
    
    for index_pair in indexes:
        if (cluster_indicess[q])[index_pair[0]] == (cluster_indicess[q])[index_pair[1]]:
            rat_and_trial_loop.append([(cluster_rat_id[q])[index_pair[0]],(cluster_indicess[q])[index_pair[0]]])
            correlated_regions_loop.append([(cluster_regions[q])[index_pair[0]],(cluster_regions[q])[index_pair[1]]])
            
            if ((cluster_regions[q])[index_pair[0]] == 1 and (cluster_regions[q])[index_pair[1]] == 3) or ((cluster_regions[q])[index_pair[0]] == 3 and (cluster_regions[q])[index_pair[1]] == 1):
                
                if ((cluster_regions[q])[index_pair[0]]) == 1:
                    
                    vms_dls_correlated_signals_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[0]]])
                    vms_dls_next_trial_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[0]] + 1])
                    vms_dls_trial_index_loop.append((track_of_indeces[(cluster_indices[q])[index_pair[0]]])[1])
                    vms_dls_rat_order_loop.append([(track_of_indeces[(cluster_indices[q])[index_pair[0]]])[5],(cluster_indicess[q])[index_pair[0]]])
                    
                elif ((cluster_regions[q])[index_pair[1]]) == 1:
                    
                    vms_dls_correlated_signals_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[1]]])
                    vms_dls_next_trial_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[1]] + 1])
                    vms_dls_trial_index_loop.append((track_of_indeces[(cluster_indices[q])[index_pair[1]]])[1])
                    vms_dls_rat_order_loop.append([(track_of_indeces[(cluster_indices[q])[index_pair[0]]])[5],(cluster_indicess[q])[index_pair[0]]])
                    
                if (cluster_regions[q])[index_pair[1]] == 3:
                    if ((cluster_indices[q])[index_pair[1]] + 1) == 2755:
                        continue 
                    else:
                        dls_vms_correlated_signals_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[1]]])
                        dls_vms_next_trial_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[1]] + 1])
                        dls_vms_trial_index_loop.append((track_of_indeces[(cluster_indices[q])[index_pair[1]]])[1])
                        
                elif (cluster_regions[q])[index_pair[0]] == 3:
                    if ((cluster_indices[q])[index_pair[1]] + 1) == 2755:
                        continue 
                    else:
                        dls_vms_correlated_signals_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[0]]])
                        dls_vms_next_trial_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[0]] + 1])
                        dls_vms_trial_index_loop.append((track_of_indeces[(cluster_indices[q])[index_pair[0]]])[1])
                        
            if ((cluster_regions[q])[index_pair[0]] == 1 and (cluster_regions[q])[index_pair[1]] == 2) or ((cluster_regions[q])[index_pair[0]] == 2 and (cluster_regions[q])[index_pair[1]] == 1):
                    
                if ((cluster_regions[q])[index_pair[0]]) == 1:
                    
                        
                    vms_dms_correlated_signals_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[0]]])
                    vms_dms_next_trial_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[0]] + 1])
                    vms_dms_trial_index_loop.append((track_of_indeces[(cluster_indices[q])[index_pair[0]]])[1])
                    vms_dms_rat_order_loop.append([(track_of_indeces[(cluster_indices[q])[index_pair[0]]])[5],(cluster_indicess[q])[index_pair[0]]])
                elif ((cluster_regions[q])[index_pair[1]]) == 1:
                        
                    vms_dms_correlated_signals_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[1]]])
                    vms_dms_next_trial_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[1]] + 1])    
                    vms_dms_trial_index_loop.append((track_of_indeces[(cluster_indices[q])[index_pair[1]]])[1])  
                    vms_dms_rat_order_loop.append([(track_of_indeces[(cluster_indices[q])[index_pair[0]]])[5],(cluster_indicess[q])[index_pair[0]]])
                if (cluster_regions[q])[index_pair[1]] == 2:
                        
                    dms_vms_correlated_signals_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[1]]])
                    dms_vms_next_trial_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[1]] + 1])    
                    dms_vms_trial_index_loop.append((track_of_indeces[(cluster_indices[q])[index_pair[1]]])[1])
                    
                elif (cluster_regions[q])[index_pair[0]] == 2:
                        
                    dms_vms_correlated_signals_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[0]]])
                    dms_vms_next_trial_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[0]] + 1])
                    dms_vms_trial_index_loop.append((track_of_indeces[(cluster_indices[q])[index_pair[0]]])[1])
                    
            if ((cluster_regions[q])[index_pair[0]] == 2 and (cluster_regions[q])[index_pair[1]] == 3) or ((cluster_regions[q])[index_pair[0]] == 3 and (cluster_regions[q])[index_pair[1]] == 2):
                    
                if ((cluster_regions[q])[index_pair[0]]) == 2:
                    
                        
                    dms_dls_correlated_signals_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[0]]])
                    dms_dls_next_trial_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[0]] + 1])
                    dms_dls_trial_index_loop.append((track_of_indeces[(cluster_indices[q])[index_pair[0]]])[1])
                    dms_dls_rat_order_loop.append([(track_of_indeces[(cluster_indices[q])[index_pair[0]]])[5],(cluster_indicess[q])[index_pair[0]]])
                elif ((cluster_regions[q])[index_pair[1]]) == 2:
                        
                    dms_dls_correlated_signals_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[1]]])
                    dms_dls_next_trial_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[1]] + 1])
                    dms_dls_trial_index_loop.append((track_of_indeces[(cluster_indices[q])[index_pair[1]]])[1])
                    dms_dls_rat_order_loop.append([(track_of_indeces[(cluster_indices[q])[index_pair[0]]])[5],(cluster_indicess[q])[index_pair[0]]])
                if (cluster_regions[q])[index_pair[1]] == 3:
                        
                    dls_dms_correlated_signals_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[1]]])
                    dls_dms_next_trial_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[1]] + 1])
                    dls_dms_trial_index_loop.append((track_of_indeces[(cluster_indices[q])[index_pair[1]]])[1])
        
                elif (cluster_regions[q])[index_pair[0]] == 3:
                        
                    dls_dms_correlated_signals_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[0]]])
                    dls_dms_next_trial_loop.append(DATA_matrices[(cluster_indices[q])[index_pair[0]] + 1])
                    dls_dms_trial_index_loop.append((track_of_indeces[(cluster_indices[q])[index_pair[0]]])[1])
                                        
                    
    vms_dls_correlated_signals.append(vms_dls_correlated_signals_loop) 
    vms_dms_correlated_signals.append(vms_dms_correlated_signals_loop)
    dls_vms_correlated_signals.append(dls_vms_correlated_signals_loop)                                                           
    dms_vms_correlated_signals.append(dms_vms_correlated_signals_loop)
    dms_dls_correlated_signals.append(dms_dls_correlated_signals_loop)
    dls_dms_correlated_signals.append(dls_dms_correlated_signals_loop)

    
    vms_next_trial.append(vms_dls_next_trial_loop)
    vms_next_trial.append(vms_dms_next_trial_loop)
    dls_next_trial.append(dls_vms_next_trial_loop)
    dms_next_trial.append(dms_vms_next_trial_loop)
    dms_next_trial.append(dms_dls_next_trial_loop)
    dls_next_trial.append(dls_dms_next_trial_loop)

    vms_dls_next_trial.append(vms_dls_next_trial_loop)
    vms_dms_next_trial.append(vms_dms_next_trial_loop)
    dls_vms_next_trial.append(dls_vms_next_trial_loop)
    dms_vms_next_trial.append(dms_vms_next_trial_loop)
    dms_dls_next_trial.append(dms_dls_next_trial_loop)
    dls_dms_next_trial.append(dls_dms_next_trial_loop)
    
    
    vms_dls_trial_index.append(vms_dls_trial_index_loop)
    vms_dms_trial_index.append(vms_dms_trial_index_loop)
    dls_vms_trial_index.append(dls_vms_trial_index_loop)
    dms_vms_trial_index.append(dms_vms_trial_index_loop)
    dms_dls_trial_index.append(dms_dls_trial_index_loop)
    dls_dms_trial_index.append(dls_dms_trial_index_loop)

    vms_dls_rat_order.append(vms_dls_rat_order_loop)
    vms_dms_rat_order.append(vms_dms_rat_order_loop)
    dms_dls_rat_order.append(dms_dls_rat_order_loop)
    
    
    rat_and_trial.append(rat_and_trial_loop)
    correlated_regions.append(correlated_regions_loop)
    


print(len(rat_and_trial))

#### To plot the correlated trials per region #### 
import numpy as np
import matplotlib.pyplot as plt


nrows = 1
ncols = 3

titles = ["VMS - DMS correlation -> VMS activity", "VMS - DMS correlation -> DMS activity",
          "VMS - DLS correlation -> VMS activity","VMS - DLS correlation -> DLS activity",
          "DMS - DLS correlation -> DMS activity","DMS - DLS correlation -> DLS activity"]

secondary_titles = ["VMS - DMS correlation", "VMS - DMS correlation -> DMS activity",
          "VMS - DLS correlation -> VMS activity","VMS - DLS correlation -> DLS activity",
          "DMS - DLS correlation -> DMS activity","DMS - DLS correlation -> DLS activity"]


data = [vms_dms_correlated_signals,dms_vms_correlated_signals,
        vms_dls_correlated_signals,dls_vms_correlated_signals,
        dms_dls_correlated_signals,dls_dms_correlated_signals]


##### To plot the correlated trials together ########

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 18), sharey=True)  # Create three figures with three subplots each

colors = ['blue', 'green', 'blue', 'red', 'green', 'red', 'pink', 'gray', 'olive', 'cyan',
          'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
          'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']



for r in range(0, 6):
    fig_index = 0  # Initialize the figure index
    if nrows == 1:
        for i in range(0, ncols):
            cluster_data = (data[r])[i]
            means = np.mean(cluster_data, axis=0)
            stds = np.std(cluster_data, axis=0)
            stds = stds / np.sqrt(350)
            x = np.arange(-9.5, 25.5, 0.1)

            ax_index = (r // 2)  # Calculate the subplot index based on row (r) divided by 2
            ax = axs[fig_index, ax_index]  # Select the corresponding subplot

            ax.set_title(f"Cluster {i}", fontsize=20, fontweight='bold')
            ax.plot(x, means, color=colors[r], linewidth=2)
            lower_bound = (means - stds).ravel()
            upper_bound = (means + stds).ravel()

            ax.fill_between(x, lower_bound, upper_bound, color=colors[r], alpha=0.1)

            ax.set_xlabel('Time(sec)',fontsize = 20, fontweight='bold')
            ax.set_yticklabels(ax.get_yticks(), fontsize=20, fontweight='bold')
            ax.set_xticklabels(ax.get_xticks(), fontsize = 16, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', which='both', length=0)

            fig_index += 1  # Increment the figure index

        
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


######## To plot the next trial after the correlation #############

import numpy as np
import matplotlib.pyplot as plt


nrows = 1
ncols = 3

titles = ["VMS - DMS correlation -> VMS activity", "VMS - DMS correlation -> DMS activity",
          "VMS - DLS correlation -> VMS activity","VMS - DLS correlation -> DLS activity",
          "DMS - DLS correlation -> DMS activity","DMS - DLS correlation -> DLS activity"]

secondary_titles = ["Trial After VMS - DMS correlation", "Trial After VMS - DMS correlation",
          "Trial After VMS - DLS correlation","Trial After VMS - DLS correlation",
          "Trial After DMS - DLS correlation","Trial After DMS - DLS correlation"]


data = [vms_dms_next_trial,dms_vms_next_trial,
        vms_dls_next_trial,dls_vms_next_trial,
        dms_dls_next_trial,dls_dms_next_trial]


##### To plot the correlated trials together ########

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 18), sharey=True)  # Create three figures with three subplots each

colors = ['blue', 'green', 'blue', 'red', 'green', 'red', 'pink', 'gray', 'olive', 'cyan',
          'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
          'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']



for r in range(0, 6):
    fig_index = 0  # Initialize the figure index
    if nrows == 1:
        for i in range(0, ncols):
            cluster_data = (data[r])[i]
            means = np.mean(cluster_data, axis=0)
            stds = np.std(cluster_data, axis=0)
            stds = stds / np.sqrt(350)
            x = np.arange(-9.5, 25.5, 0.1)

            ax_index = (r // 2)  # Calculate the subplot index based on row (r) divided by 2
            ax = axs[fig_index, ax_index]  # Select the corresponding subplot

            ax.set_title(f"Cluster {i}", fontsize=20, fontweight='bold')
            ax.plot(x, means, color=colors[r], linewidth=2)
            lower_bound = (means - stds).ravel()
            upper_bound = (means + stds).ravel()

            ax.fill_between(x, lower_bound, upper_bound, color=colors[r], alpha=0.1)

            ax.set_xlabel('Time(sec)',fontsize = 20, fontweight='bold')
            ax.set_yticklabels(ax.get_yticks(), fontsize=20, fontweight='bold')
            ax.set_xticklabels(ax.get_xticks(), fontsize = 16, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_ylim(-25, 32)
            fig_index += 1  # Increment the figure index

        
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


############ To plot trial index and check when correlation occur ##############

import numpy as np
import matplotlib.pyplot as plt


nrows = 1
ncols = 3

titles = ["VMS - DMS correlation -> VMS activity", "VMS - DMS correlation -> DMS activity",
          "VMS - DLS correlation -> VMS activity","VMS - DLS correlation -> DLS activity",
          "DMS - DLS correlation -> DMS activity","Change it"]

secondary_titles = ["Trial After VMS - DMS correlation", "Trial After VMS - DMS correlation",
          "Trial After VMS - DLS correlation","Trial After VMS - DLS correlation",
          "Trial After DMS - DLS correlation","Trial After DMS - DLS correlation"]


data = [vms_dms_trial_index,
        vms_dls_trial_index,
        dms_dls_trial_index]



#colors = ['blue', 'green', 'blue', 'red', 'green', 'red', 'pink', 'gray', 'olive', 'cyan',
#          'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
#          'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway

data = [
    vms_dms_trial_index,
    vms_dls_trial_index,
    dms_dls_trial_index
]



fig, axs = plt.subplots(1, 3, figsize=(12, 4))

for i in range(3):
    sublist_data = data[i]  # Get the sublist data
    
    # Create a list to collect all the boxplots for the subplot
    boxplots = []
    
    for j in range(3):
        # Scatter plot
        x = np.random.normal(j, 0.02, size=len(sublist_data[j]))
        y = sublist_data[j]
        axs[i].scatter(x, y, color='gray', alpha=0.5)

        # Boxplot
        box_x = [j]
        box_y = sublist_data[j]
        boxplot = axs[i].boxplot([box_y], positions=box_x, widths=0.6)
        boxplots.append(boxplot)

        # Mean
        mean_x = j
        mean_y = np.mean(box_y)
        axs[i].plot(mean_x, mean_y, 'r+', markersize=10)

    # Perform ANOVA
    stat, p_value = f_oneway(*sublist_data)
    
    # Check if there are significant differences
    if p_value < 0.05:
        # Find the indices of the boxplots with significant differences
        significant_indices = np.where(np.array(p_values) < 0.05)[0]

        # Add lines and asterisks on top of the significant boxplots
        for index in significant_indices:
            boxplot = boxplots[index]
            box_x = boxplot['boxes'][0].get_xdata()
            box_y = boxplot['boxes'][0].get_ydata()
            axs[i].plot(box_x[[0, 1, 1, 0, 0]], box_y[[0, 0, 1, 1, 0]], color='red')
            axs[i].text(np.mean(box_x), np.max(box_y) + 0.05, '*', color='red', ha='center')
    if i == 0:
        
        axs[i].set_ylabel('Trial Index',fontsize = 20, fontweight='bold')
    axs[i].set_xlabel('Cluster Index',fontsize = 20, fontweight='bold')
    axs[i].set_xticklabels(axs[i].get_xticks(),fontsize = 16, fontweight='bold')
    axs[i].set_yticklabels(axs[i].get_yticks(),fontsize = 16, fontweight='bold')
    # Print statistics
    print(f'Subfigure {i+1} - ANOVA p-value: {p_value:.4f}')
    
    # Perform posthoc test
    data_flat = np.concatenate(sublist_data)
    groups = np.repeat(np.arange(3), [len(sublist_data[j]) for j in range(3)])
    posthoc = pairwise_tukeyhsd(data_flat, groups)
    print(posthoc)

plt.tight_layout()
plt.show()




print(rat_and_trial)


values_list = [16, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 58, 60]
### to save trials per rat, from left to right in the values_list parameter ###


trials_for_behavior_all = []

for clust in range(0,3):
    trial_and_behavior_loop2 = []
    for q in range(0,len(values_list)):
        trials_for_behavior_loop = []
        for i in range(0,len(rat_and_trial[clust])):
            
            if ((rat_and_trial[clust])[i])[0] == values_list[q]:
                
                trials_for_behavior_loop.append(((rat_and_trial[clust])[i])[1])
                
        trial_and_behavior_loop2.append(trials_for_behavior_loop)   
    
    

    trials_for_behavior_all.append(trial_and_behavior_loop2)


print(trials_for_behavior_all[0])
print(len(trials_for_behavior_all[0]))



### Check trials for one rat ###
cluster_num = 0
for i in range(0,len(rat_and_trial[cluster_num])):
    if ((rat_and_trial[cluster_num])[i])[0] == 28:
        
        print(((rat_and_trial[cluster_num])[i])[1])
        #print((correlated_regions[cluster_num])[i])


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
import pandas as pd
import glob
import numpy as np


folder_path = r"C:/Users/georg/Neuroscience/NiN internship/Behavioral analysis/Pavlovian"
values_list = [16, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 58, 60]

csv_files = glob.glob('C:/Users/georg/Neuroscience/NiN internship/Behavioral analysis/Pavlovian/CSV files/*.csv')
csv_array = []

for file in csv_files:
    df = pd.read_csv(file)
    df = df.iloc[2:]
    selected_columns = df.iloc[:, [4, 5, 1, 2, -12,-11, 13, 14]].values.astype(float).tolist()  # Convert to float
    csv_array.append(selected_columns)
    


csv_array = np.array(csv_array)

print(len(csv_array))


import csv
import os

folder_path = r'C:/Users/georg/Neuroscience/NiN internship/Behavioral analysis/Pavlovian/Event times video - all trials/Traylight&Pellet'  # Replace with the actual folder path

import xlrd
import numpy as np
import os

####################################################################################


trial_start_times = []

for filename in os.listdir(folder_path):
    if filename.endswith('.xls') or filename.endswith('.xlsx'):
        file_path = os.path.join(folder_path, filename)

        workbook = xlrd.open_workbook(file_path)
        sheet = workbook.sheet_by_index(0)

        # Extracting data from the first column
        column_values = sheet.col_values(0)

        # Converting the column data to a NumPy array
        array = np.array(column_values)

        trial_start_times.append(array)

print((len(trial_start_times)))

print(trial_start_times[15]/float(30))



correlated_trials_index_for_csv = []

for clust in range(0,3):
    correlated_trials_index_for_csv_loop2 = []
    for y in range(0,len(trials_for_behavior_all[clust])-1):
        correlated_trials_index_for_csv_loop = []
        
        for p in (trials_for_behavior_all[clust])[y]:
            print(p)
            if p >= len(trial_start_times[y]):
                continue
            else:
                
                correlated_trials_index_for_csv_loop.append((trial_start_times[y])[p])
                
        correlated_trials_index_for_csv_loop2.append(correlated_trials_index_for_csv_loop)
    
    correlated_trials_index_for_csv.append(correlated_trials_index_for_csv_loop2)
   
    
import random

non_correlated_trials_index_for_csv = []

for clust in range(0, 3):
    non_correlated_trials_index_for_csv_loop2 = []
    for y in range(0, len(trials_for_behavior_all[clust])-1):
        non_correlated_trials_index_for_csv_loop = []

        for p in trials_for_behavior_all[clust][y]:
          
            if p >= len(trial_start_times[y]):
                continue
            else:
                # Generate a random trial index different from p
                random_trial_index = random.choice(
                    [i for i in range(len(trial_start_times[y])) if i != p]
                )
                print(random_trial_index,p)
                non_correlated_trials_index_for_csv_loop.append(trial_start_times[y][random_trial_index])

        non_correlated_trials_index_for_csv_loop2.append(non_correlated_trials_index_for_csv_loop)

    non_correlated_trials_index_for_csv.append(non_correlated_trials_index_for_csv_loop2)




after_correlated_trials_index_for_csv = []

for clust in range(0, 3):
    after_correlated_trials_index_for_csv_loop2 = []
    for y in range(0, len(trials_for_behavior_all[clust])-1):
        after_correlated_trials_index_for_csv_loop = []

        for p in (trials_for_behavior_all[clust])[y]:
            
            if p >= len(trial_start_times[y]):
                continue
            else:
                if (p+1) >= 49:
                    continue
                after_correlated_trials_index_for_csv_loop.append(trial_start_times[y][p+1])
                

        after_correlated_trials_index_for_csv_loop2.append(after_correlated_trials_index_for_csv_loop)

    after_correlated_trials_index_for_csv.append(after_correlated_trials_index_for_csv_loop2)

after2_correlated_trials_index_for_csv = []

for clust in range(0, 3):
    after2_correlated_trials_index_for_csv_loop2 = []
    for y in range(0, len(trials_for_behavior_all[clust])-1):
        after2_correlated_trials_index_for_csv_loop = []

        for p in (trials_for_behavior_all[clust])[y]:
            
            if p >= len(trial_start_times[y]):
                continue
            else:
                if (p+1) >= 48:
                    continue
                after2_correlated_trials_index_for_csv_loop.append(trial_start_times[y][p+2])
                

        after2_correlated_trials_index_for_csv_loop2.append(after2_correlated_trials_index_for_csv_loop)

    after2_correlated_trials_index_for_csv.append(after2_correlated_trials_index_for_csv_loop2)



adder = 0
for i in range(0,len((correlated_trials_index_for_csv)[0])):
    
    adder += len(((correlated_trials_index_for_csv)[0])[i])

print(adder)


for sublist in correlated_trials_index_for_csv:
    for sub_sublist in sublist:
        if any(value < 0 for value in sub_sublist):
            sublist[sublist.index(sub_sublist)] = []
            
 
for sublist in after_correlated_trials_index_for_csv:
    for sub_sublist in sublist:
        if any(value < 0 for value in sub_sublist):
            sublist[sublist.index(sub_sublist)] = []
           

for sublist in non_correlated_trials_index_for_csv:
    for sub_sublist in sublist:
        if any(value < 0 for value in sub_sublist):
            sublist[sublist.index(sub_sublist)] = []
            
for sublist in after2_correlated_trials_index_for_csv:
    for sub_sublist in sublist:
        if any(value < 0 for value in sub_sublist):
            sublist[sublist.index(sub_sublist)] = []


print((correlated_trials_index_for_csv[0])[1])

middle_back_data_of_correlated_trials = []
snout_data_of_correlated_trials = []

tt_list = [0,2]
for tt in tt_list:
    for clust in range(0,1):
        middle_back_data_of_correlated_trials_loop = []
        snout_data_of_correlated_trials_loop = []
        for index in range(0,len(((correlated_trials_index_for_csv)[clust]))):
            
            if len(((correlated_trials_index_for_csv)[clust])[index]) == 0:
                continue
            
            else:
                
                for trial_time in (((correlated_trials_index_for_csv)[clust])[index]):
                    
                    time_window_data = ((csv_array[index])[int(trial_time):int(trial_time + 20*30)])
                    
                    x_selection = [sublist[tt] for sublist in time_window_data]
                    y_selection = [sublist[tt+1] for sublist in time_window_data]
                    
                    if tt == 0:
                        middle_back_data_of_correlated_trials_loop.append([x_selection,y_selection])
                    else:
                        snout_data_of_correlated_trials_loop.append([x_selection,y_selection])
               
            middle_back_data_of_correlated_trials.append(middle_back_data_of_correlated_trials_loop)
            snout_data_of_correlated_trials.append(snout_data_of_correlated_trials_loop)
            
            
            
print(len(middle_back_data_of_correlated_trials))
       

        
after_middle_back_data_of_correlated_trials = []
after_snout_data_of_correlated_trials = []

tt_list = [0,2]
for tt in tt_list:
    for clust in range(0,1):
        
        for index in range(0,len(((after_correlated_trials_index_for_csv)[clust]))):
            
            if len(((after_correlated_trials_index_for_csv)[clust])[index]) == 0:
                continue
            
            else:
                
                for trial_time in (((after_correlated_trials_index_for_csv)[clust])[index]):
                    
                    time_window_data = ((csv_array[index])[int(trial_time):int(trial_time + 20*30)])
                    
                    x_selection = [sublist[tt] for sublist in time_window_data]
                    y_selection = [sublist[tt+1] for sublist in time_window_data]
                    
                    if tt == 0:
                        after_middle_back_data_of_correlated_trials.append([x_selection,y_selection])
                    else:
                        after_snout_data_of_correlated_trials.append([x_selection,y_selection])
                
print(len(after_snout_data_of_correlated_trials))
            








non_middle_back_data_of_correlated_trials = []
non_snout_data_of_correlated_trials = []

tt_list = [0, 2]
for tt in tt_list:
    for clust in range(0, 1):
        
        for index in range(0, len(((non_correlated_trials_index_for_csv)[clust]))):
            
            if len(((non_correlated_trials_index_for_csv)[clust])[index]) == 0:
                continue
            
            else:
                
                for trial_time in (((non_correlated_trials_index_for_csv)[clust])[index]):
                    
                    time_window_data = ((csv_array[index])[int(trial_time):int(trial_time + 20 * 30)])
                    
                    x_selection = [sublist[tt] for sublist in time_window_data]
                    y_selection = [sublist[tt + 1] for sublist in time_window_data]
                    
                    if tt == 0:
                        non_middle_back_data_of_correlated_trials.append([x_selection, y_selection])
                    else:
                        non_snout_data_of_correlated_trials.append([x_selection, y_selection])

print(len(non_snout_data_of_correlated_trials),len(non_middle_back_data_of_correlated_trials[5]))


import numpy as np
import matplotlib.pyplot as plt
    
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_sublists(sublists1, sublists2, color='blue', alpha=0.05):
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111)

    speeds = []  # List to store calculated speeds

    for sublist in sublists1:
        x_values = sublist[0]
        y_values = sublist[1]

        filtered_data = [(x, y) for x, y in zip(x_values, y_values) if 50 <= x <= 250 and 100 <= y <= 425]

        if len(filtered_data) > 0:
            filtered_x, filtered_y = zip(*filtered_data)
            ax1.scatter(filtered_x, filtered_y, color='red', alpha=alpha)

            

    speed = []
    for sublist in sublists2:
        x_values = sublist[0]
        y_values = sublist[1]

        filtered_data = [(x, y) for x, y in zip(x_values, y_values) if 50 <= x <= 250 and 100 <= y <= 425]
        speed_loop = []
        if len(filtered_data) > 0:
            filtered_x, filtered_y = zip(*filtered_data)
            ax1.scatter(filtered_x, filtered_y, color=color, alpha=alpha)
            #ax1.plot(filtered_x, filtered_y, color='black')
            
            
            
            for i in range(1, len(filtered_data)):
                x1, y1 = filtered_data[i-1]
                x2, y2 = filtered_data[i]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                speed_loop.append(distance / 0.0333333)
               
        speed.append(speed_loop)

    
    
    
    
    
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    max_length = max(len(sublist) for sublist in speed)
    time = np.arange(0,max_length/float(30),1./30)
    print(len(time))
    
    for sublist in speed:
        if not sublist:  # Skip empty sublists
            continue
        ax3.plot(time[:len(sublist)], sublist, color='blue',alpha = 0.01)
    
    mean_speed = np.mean([np.interp(time, np.arange(len(sublist)), sublist) for sublist in speed if sublist], axis=0)

    ax3.plot(time, mean_speed, color='red', linestyle='--', label='Mean Speed')
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Speed')
    ax3.set_title('Line Plot of Speeds')
    ax3.legend()
    
    # Set y-axis limit to 500
    ax3.set_ylim(0, 500)
    
    plt.show()


    ax1.axvline(150, color='black', linestyle='--')
    ax1.axhline(250, color='black', linestyle='--')

    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_title('Scatter Plot')

    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')

    xpos = np.array([0, 1])
    ypos = np.array([0, 1])
    xpos, ypos = np.meshgrid(xpos, ypos)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()

    # Calculate time spent in each quadrant
    filtered_data = [(x, y) for sublist in sublists2 for x, y in zip(sublist[0], sublist[1]) if 50 <= x <= 250 and 100 <= y <= 425]
    total_points = len(filtered_data)
    quadrant4_points = sum(1 for x, y in filtered_data if x >= 150 and y >= 250)
    quadrant3_points = sum(1 for x, y in filtered_data if x < 150 and y >= 250)
    quadrant1_points = sum(1 for x, y in filtered_data if x < 150 and y < 250)
    quadrant2_points = sum(1 for x, y in filtered_data if x >= 150 and y < 250)

    time_quadrant1 = quadrant1_points * 0.0333333
    time_quadrant2 = quadrant2_points * 0.0333333
    time_quadrant3 = quadrant3_points * 0.0333333
    time_quadrant4 = quadrant4_points * 0.0333333

    dz = np.array([time_quadrant1, time_quadrant2, time_quadrant3, time_quadrant4])

    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color='grey')

    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.set_zlabel('Time (seconds)')
    ax2.set_title('3D Bar Plot')

    plt.show()

plot_sublists(after_snout_data_of_correlated_trials,after_middle_back_data_of_correlated_trials, color='blue', alpha=0.01)




















import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns







def choose_data(cluster_num,selection, comparison, time_before,time_after, cc):

    ################ To calculate quadrants and compare them ###################
    
    import numpy as np
    
    
    middle_back_data_of_correlated_trials = []
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((correlated_trials_index_for_csv)[clust]))):
            middle_back_data_of_correlated_trials_loop = []
             
            if len(((correlated_trials_index_for_csv)[clust])[index]) == 0:
                middle_back_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[0] for sublist in time_window_data]
                    y_selection = [sublist[1] for sublist in time_window_data]
                                       
                    middle_back_data_of_correlated_trials_loop.append([x_selection,y_selection])
                middle_back_data_of_correlated_trials.append(middle_back_data_of_correlated_trials_loop)
            
    
    
    
            
    after_middle_back_data_of_correlated_trials = []
    
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((after_correlated_trials_index_for_csv)[clust]))):
            after_middle_back_data_of_correlated_trials_loop = []
             
            if len(((after_correlated_trials_index_for_csv)[clust])[index]) == 0:
                after_middle_back_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((after_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[0] for sublist in time_window_data]
                    y_selection = [sublist[1] for sublist in time_window_data]
                                       
                    after_middle_back_data_of_correlated_trials_loop.append([x_selection,y_selection])
                after_middle_back_data_of_correlated_trials.append(after_middle_back_data_of_correlated_trials_loop)
            
    print(len((after_middle_back_data_of_correlated_trials)))
    
                
    
    
    non_middle_back_data_of_correlated_trials = []
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((non_correlated_trials_index_for_csv)[clust]))):
            non_middle_back_data_of_correlated_trials_loop = []
             
            if len(((non_correlated_trials_index_for_csv)[clust])[index]) == 0:
                non_middle_back_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((non_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[0] for sublist in time_window_data]
                    y_selection = [sublist[1] for sublist in time_window_data]
                                       
                    non_middle_back_data_of_correlated_trials_loop.append([x_selection,y_selection])
                non_middle_back_data_of_correlated_trials.append(non_middle_back_data_of_correlated_trials_loop)
            
            
    
    for sublist in correlated_trials_index_for_csv:
        for sub_sublist in sublist:
            if any(value < 0 for value in sub_sublist):
                sublist[sublist.index(sub_sublist)] = []
                
     
    for sublist in after_correlated_trials_index_for_csv:
        for sub_sublist in sublist:
            if any(value < 0 for value in sub_sublist):
                sublist[sublist.index(sub_sublist)] = []
               
    
    for sublist in non_correlated_trials_index_for_csv:
        for sub_sublist in sublist:
            if any(value < 0 for value in sub_sublist):
                sublist[sublist.index(sub_sublist)] = []
    
    
    def plot_speeds(data):   
        import math
        import matplotlib.pyplot as plt
        
        # Function to calculate Euclidean distance
        def euclidean_distance(x1, y1, x2, y2):
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # List to store mean speeds of each sample
        mean_speeds = []
        mean_speeds_per_animal_saver =[]
        data[3] = []
        data[2] = []
        data[1] = []
        data[0] = []
        # Iterate over each sample (animal) separately
        for sample_idx in range(len(data)):
            sample = data[sample_idx]
            
            # List to store mean values within the sample
            sample_means = []
            mean_speeds_loop = []
            # Iterate over sublists in the sample
            for sublist in sample:
                x_values, y_values = sublist
                
                # Check if sublist is not empty
                if x_values and y_values:
                    # Calculate mean of x-axis and y-axis values
                    mean_x = sum(x_values) / len(x_values)
                    mean_y = sum(y_values) / len(y_values)
                    
                    # Add mean values to the list
                    sample_means.append((mean_x, mean_y))
            
            # Calculate mean speed for each mean in sample_means
            for mean in sample_means:
                mean_x, mean_y = mean
                mean_speed = euclidean_distance(0, 0, mean_x, mean_y)
                mean_speeds_loop.append(mean_speed)
            mean_speeds.append(mean_speeds_loop)
        # Generate x-axis values (sample indices)
        x_values_scatter = list(range(1, len(mean_speeds) + 1))
        x_values_line = list(range(1, len(data) + 1))
        
        # Plotting the scatter points
        for i in range(0, len(mean_speeds)):
            mean_speeds_per_animal = np.mean(mean_speeds[i])
            mean_speeds_per_animal_saver.append(mean_speeds_per_animal)
            plt.scatter(x_values_scatter[i], mean_speeds_per_animal, color='blue', label='Sample Mean Speeds')
        
        ################ To plot the speed of individual rats ########################
        
        # Remove empty sublists from mean_speeds
        mean_speeds = [sublist for sublist in mean_speeds if sublist]
        mean_speeds = [sublist for sublist in mean_speeds if len(sublist) > 3]
        # Create a list to store the x-axis positions for each boxplot
        positions = range(1, len(mean_speeds) + 1)
        
        # Create a figure and axes
        fig, ax = plt.subplots()
        
        # Create boxplots for each sublist
        boxplot = ax.boxplot(mean_speeds, positions=positions, patch_artist=True)
        
        # Set the same color for all the boxplots
        color = 'lightblue'
        for patch in boxplot['boxes']:
            patch.set_facecolor(color)
        
        # Add individual points to the plot
        for i, sublist in enumerate(mean_speeds):
            x = [i + 1] * len(sublist)
            ax.plot(x, sublist, 'ko', alpha=0.5)  # 'ko' represents black dots
        
        # Set x-axis label
        ax.set_xlabel('Rat',fontsize = 20)
        
        # Set y-axis label
        ax.set_ylabel('Mean Speed(pixel/sec)',fontsize = 16)
        
        
        # Set x-axis tick labels
        ax.set_xticklabels(range(1, len(mean_speeds) + 1),fontsize = 14)
        ax.set_yticklabels(ax.get_yticks().astype(int),fontsize = 14)
        # Remove the right and top spines
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        
        # Remove the ticks on the right and top axes
        plt.gca().yaxis.tick_left()
        plt.gca().xaxis.tick_bottom()
        
        # Set the remaining ticks to be on the left and bottom axes
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        # Set the linewidth of the x-axis and y-axis
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)

        # Show the plot
        plt.show()
        
        return mean_speeds_per_animal_saver
    """
    Available data: non_middle_back_data_of_correlated_trials , middle_back_data_of_correlated_trials, 
            after_middle_back_data_of_correlated_trials
    
    """
    
    
    means_list = []
    for q in range(0,3):
        
        if q == 0:
            
            tracker = plot_speeds(non_middle_back_data_of_correlated_trials)
            
            
                
            means_list.append(tracker[selection:comparison])
            
            
        elif q == 1:
            
            tracker = plot_speeds(middle_back_data_of_correlated_trials)
            
                
            means_list.append(tracker[selection:comparison])
            
        elif q == 2:
            
            tracker = plot_speeds(after_middle_back_data_of_correlated_trials)
            
                
            means_list.append(tracker[selection:comparison])
            
    

    
    
    import scipy.stats as stats
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Calculate valid values for each sublist
    valid_values = []
    for sublist in means_list:
        valid_values.append([val for val in sublist if not np.isnan(val)])
    
    # Perform one-way ANOVA
    f_value, p_value = stats.f_oneway(*valid_values)
    
    # Plotting box plots
    labels = ['Random trial', 'Clustered trial', 'After trial']
    
    fig, ax = plt.subplots()
    ax.boxplot(valid_values,showfliers=False)
    ax.set_xticklabels(labels,fontsize =14)
    ax.set_yticklabels(ax.get_yticks().astype(int),fontsize = 14)
    ax.set_ylabel('Speed(Pixel/sec)',fontsize = 16)
    
    
    
    # Plotting scatter points
    for i, values in enumerate(valid_values):
        ax.scatter(np.ones(len(values)) * (i + 1), values, color=cc, alpha=0.5)
    
    
    # Print ANOVA results
    print('ANOVA Results:')
    print(f'F-value: {f_value:.2f}')
    print(f'p-value: {p_value:.4f}')
    # Remove the right and top spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    
    # Remove the ticks on the right and top axes
    plt.gca().yaxis.tick_left()
    plt.gca().xaxis.tick_bottom()
    
    # Set the remaining ticks to be on the left and bottom axes
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    # Set the linewidth of the x-axis and y-axis
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)

    plt.show()
    
    
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def calculate_square_time(samples):
        square_time = []
        for sample in samples:
            filtered_data = [(x, y) for sublist in sample for x, y in zip(sublist[0], sublist[1]) if 50 <= x <= 250 and 100 <= y <= 425]
            total_points = len(filtered_data)
            
            square_points = [[0 for _ in range(5)] for _ in range(8)]  # Initialize a 2D list to store the points in each square
            
            for x, y in filtered_data:
                square_x = int((x - 50) // 25)  # Calculate the x-coordinate of the square
                square_y = int((y - 100) // 65)  # Calculate the y-coordinate of the square
                
                if square_x < 0 or square_x >= 8 or square_y < 0 or square_y >= 5:
                    continue  # Skip points outside the valid range
                
                square_points[square_x][square_y] += 1  # Increment the point count for the square
            
            square_time_sample = [[points * 0.0333333 for points in row] for row in square_points]  # Calculate the time spent in each square
            
            square_time.append(square_time_sample)
    
        return square_time
    
    # Calculate square time for each dataset
    non_middle_back_square_time = calculate_square_time(non_middle_back_data_of_correlated_trials)
    middle_back_square_time = calculate_square_time(middle_back_data_of_correlated_trials)
    after_middle_back_square_time = calculate_square_time(after_middle_back_data_of_correlated_trials)
    
    # Calculate average square time for each dataset
    non_middle_back_avg = np.mean(non_middle_back_square_time[selection:comparison], axis=0)
    middle_back_avg = np.mean(middle_back_square_time[selection:comparison], axis=0)
    after_middle_back_avg = np.mean(after_middle_back_square_time[selection:comparison], axis=0)
    
    # Combine all average datasets
    combined_avg = np.concatenate([non_middle_back_avg, middle_back_avg, after_middle_back_avg])
    
    # Calculate the upper limit for the colorbar
    max_value = np.max(combined_avg)
    
    # Create a figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot average square time for each dataset in separate subplots
    titles = ['Non-Middle Back', 'Middle Back', 'After Middle Back']
    avg_datasets = [non_middle_back_avg, middle_back_avg, after_middle_back_avg]
    
    for i, (title, avg_dataset) in enumerate(zip(titles, avg_datasets)):
        ax = axs[i]
        im = ax.imshow(avg_dataset, cmap='hot', interpolation='nearest', vmin=0, vmax=max_value)
        ax.set_title(title)
        ax.set_xlabel('Y')
        ax.set_ylabel('X')
    
        # Add text annotations for square indices
        #for x in range(5):
            #for y in range(8):
                #text = ax.text(y, x, square_points[x][y], ha='center', va='center', color='black')
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Add a shared colorbar
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal')
    cbar.set_label('Time(sec)')
    
    # Display the plot
    plt.show()
    
    
    
    
    
    
                
    def calculate_quadrant_time(samples):
        quadrant_time = []
        for sample in samples:
            filtered_data = [(x, y) for sublist in sample for x, y in zip(sublist[0], sublist[1]) if 50 <= x <= 250 and 100 <= y <= 425]
            total_points = len(filtered_data)
            quadrant4_points = sum(1 for x, y in filtered_data if x >= 150 and y >= 250)
            quadrant3_points = sum(1 for x, y in filtered_data if x < 150 and y >= 250)
            quadrant1_points = sum(1 for x, y in filtered_data if x < 150 and y < 250)
            quadrant2_points = sum(1 for x, y in filtered_data if x >= 150 and y < 250)
    
            time_quadrant1 = quadrant1_points * 0.0333333
            time_quadrant2 = quadrant2_points * 0.0333333
            time_quadrant3 = quadrant3_points * 0.0333333
            time_quadrant4 = quadrant4_points * 0.0333333
    
            quadrant_time.append([time_quadrant1, time_quadrant2, time_quadrant3, time_quadrant4])
    
        return quadrant_time
    
    
    
    
    # Calculate quadrant time for each list
    non_middle_back_quadrant_time = calculate_quadrant_time(non_middle_back_data_of_correlated_trials)
    middle_back_quadrant_time = calculate_quadrant_time(middle_back_data_of_correlated_trials)
    after_middle_back_quadrant_time = calculate_quadrant_time(after_middle_back_data_of_correlated_trials)
    
    print(middle_back_quadrant_time)
    
    
    for ii in range(0,len(non_middle_back_quadrant_time)):
        summ1 = sum(non_middle_back_quadrant_time[ii])
        summ2 = sum(middle_back_quadrant_time[ii])
        summ3 = sum(after_middle_back_quadrant_time[ii])
        
    
            
        
        for i in range(0,4):
            if ii == 1:
                if summ1 == 0:
                    continue
                (non_middle_back_quadrant_time[ii])[i] = 0
                if summ2 == 0:
                    continue
                (middle_back_quadrant_time[ii])[i] = 0
                if summ3 == 0:
                    continue
                (after_middle_back_quadrant_time[ii])[i] = 0
         
            else:
                
                if summ1 == 0:
                    continue
                (non_middle_back_quadrant_time[ii])[i] = ((non_middle_back_quadrant_time[ii])[i])/float(summ1)
                if summ2 == 0:
                    continue
                (middle_back_quadrant_time[ii])[i] = ((middle_back_quadrant_time[ii]))[i]/float(summ2)
                if summ3 == 0:
                    continue
                (after_middle_back_quadrant_time[ii])[i] = ((after_middle_back_quadrant_time[ii]))[i]/float(summ3)
         
    print(middle_back_quadrant_time)
    
    (middle_back_quadrant_time[1])[0] = 0
    (middle_back_quadrant_time[1])[1] = 0
    (middle_back_quadrant_time[1])[2] = 0
    (middle_back_quadrant_time[1])[3] = 0
    
    (after_middle_back_quadrant_time[1])[0] = 0
    (after_middle_back_quadrant_time[1])[1] = 0
    (after_middle_back_quadrant_time[1])[2] = 0
    (after_middle_back_quadrant_time[1])[3] = 0
    
    
    
    def print_quadrant_times(quadrant_times):
        for index, times in enumerate(quadrant_times):
            print(f"Sample {index+1}: Quadrant Times -> Quadrant 1: {times[0]}, Quadrant 2: {times[1]}, Quadrant 3: {times[2]}, Quadrant 4: {times[3]}")
    
    # Print quadrant times for each list
    print("Non Middle Back Quadrant Times:")
    print_quadrant_times(non_middle_back_quadrant_time)
    
    print("\nMiddle Back Quadrant Times:")
    print_quadrant_times(middle_back_quadrant_time)
    
    print("\nAfter Middle Back Quadrant Times:")
    print_quadrant_times(after_middle_back_quadrant_time)
    
    
    
    
    
    # Define the figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    x_values = 0
    
    # Set titles for each subplot
    axs[0, 0].set_title('Quadrant 1')
    axs[0, 1].set_title('Quadrant 2')
    axs[1, 0].set_title('Quadrant 3')
    axs[1, 1].set_title('Quadrant 4')
    x_pos = [0,1,2]
    means_list = []
    # Define the original blue color
    original_blue = (0, 0, 1)  # RGB values for blue
    # Increase the brightness to make it lighter
    lighter_blue = (0.7, 0.7, 1)  # RGB values for a lighter blue
    # Iterate over each quadrant and plot the scatter points
    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(ax_row):
            quadrant_index = i * 2 + j  # Calculate the quadrant index
            quadrant_data = [non_middle_back_quadrant_time[selection:comparison], middle_back_quadrant_time[selection:comparison], after_middle_back_quadrant_time[selection:comparison]]
            mean_calculation = [[],[],[]]
            # Iterate over the three parameters and plot scatter points
            for indexx in range(0,len(quadrant_data[0])):
                
                data_for_plot = []
                for e in range(0,len(quadrant_data)):
     
                    
                    
                    
                    
                    data_for_plot.append(((quadrant_data[e])[indexx])[quadrant_index])
                    # Connect scatter points with a line
                    (mean_calculation[e]).append(((quadrant_data[e])[indexx])[quadrant_index])
                    ax.scatter(e, ((quadrant_data[e])[indexx])[quadrant_index],color = original_blue,zorder = 2)
                
                if data_for_plot[0] == 0 and data_for_plot[1] == 0 and data_for_plot[2] == 0:
                    continue
                    
                ax.plot(x_pos, data_for_plot, color='black', linewidth=0.5,zorder = 1)
                data_for_plot = []
                
            # Transpose the list of sublists
            transposed_calculation = list(map(list, zip(*mean_calculation)))
            
            # Remove rows (which were columns before transposing) with all values equal to 0
            transposed_calculation = [sublist for sublist in transposed_calculation if not all(value == 0 for value in sublist)]
            
            # Transpose the result back to get the modified list
            mean_calculation = list(map(list, zip(*transposed_calculation)))        
            means_list.append((np.mean(mean_calculation,axis = 1)))
            
            
    
            ax.bar(x_pos, np.mean(mean_calculation,axis = 1),color= lighter_blue,zorder = 0)
    
    # Set common x and y labels
    fig.text(0.5, 0.04, 'Sample', ha='center')
    fig.text(0.04, 0.5, 'Time', va='center', rotation='vertical')
    
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    
    # Show the plot
    plt.show()
    
    
    return


print(choose_data(0,0,17,0,600,'blue'))







means = [
    [[0.39548353, 0.35504461, 0.3819607 ],
    [0.11886946, 0.11835939, 0.09090331],
    [0.43813201, 0.42139015, 0.41492383],
    [0.04751499, 0.03853919, 0.04554549]],

    [[0.39587497, 0.41928682, 0.41874432], 
     [0.10061733, 0.11314818, 0.10442406],
     [0.44908427, 0.43210441, 0.42837724],
     [0.05442343, 0.03546058, 0.04845438]],
    
    [[0.36917044, 0.25451961, 0.2594839 ],
     [0.10452604, 0.0679254 , 0.11233579],
     [0.49490435, 0.55890557, 0.51484047],
     [0.03710811, 0.04931294, 0.04303798]]
]

print(means[0])


for q in range(0,3):
    plt.plot(x_pos,(means[q])[0])
    
plt.show()



# Define the figure and subplots
fig, axs = plt.subplots(1, 4, figsize=(24, 8),sharey = True)
x_values = 0

# Set titles for each subplot
axs[0].set_title('Quadrant 1')
axs[1].set_title('Quadrant 2')
axs[2].set_title('Quadrant 3')
axs[3].set_title('Quadrant 4')


for i in range(0,4):
    for q in range(0,3):
        axs[i].plot(x_pos,(means[q])[i])
   

























        
# Calculate the total width of each bar
bar_width = 0.35

# Set the x-axis labels
labels = np.arange(0,50,1)
#labels = ['1','2','3']

# Create a figure and axis object
fig, ax = plt.subplots()

k = 3
for r in range(0,k):
    
    # Create the first set of bars
    ax.bar(labels[r], len(cluster_rat_id[r]), width=bar_width, label='Array 1',color = 'blue')
    
    
    
    # Create the second set of bars
    ax.bar(labels[r], len(rat_and_trial[r]), width=bar_width, label='Array 2', color = 'red')
ax.set_xticklabels(ax.get_xticks(),fontsize = 16)
ax.set_yticklabels(ax.get_yticks(),fontsize = 16)
# Add a legend and title
ax.set_xlabel('Cluster index',fontsize = 16)
ax.set_ylabel('Number of trials',fontsize = 16)

# Remove the right and top spines
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# Remove the ticks on the right and top axes
plt.gca().yaxis.tick_left()
plt.gca().xaxis.tick_bottom()

# Set the remaining ticks to be on the left and bottom axes
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')
# Set the linewidth of the x-axis and y-axis
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
# Show the plot
plt.show()





import numpy as np
import matplotlib.pyplot as plt

# Calculate the total width of each bar
bar_width = 0.35

# Set the x-axis labels
labels = np.arange(0, 50, 1)

# Create a figure and axis object
fig, ax = plt.subplots()

k = 3
for r in range(0, k):
    # Calculate the percentage of red bars relative to blue bars
    percentage = len(rat_and_trial[r]) / len(cluster_rat_id[r]) * 100

    # Create the first set of bars (blue)
    #ax.bar(labels[r], len(cluster_rat_id[r]), width=bar_width, label='Array 1', color='blue')

    # Create the second set of bars (red)
    ax.bar(labels[r], percentage, width=bar_width, label='Array 2', color='red')

ax.set_xticklabels(ax.get_xticks().astype(int), fontsize=16)
ax.set_yticklabels(ax.get_yticks().astype(int), fontsize=16)

# Add a legend and title
ax.set_xlabel('Cluster index', fontsize=16)
ax.set_ylabel('Percentage of clustered trials/cluster', fontsize=12)

# Remove the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Remove the ticks on the right and top axes
ax.yaxis.tick_left()
ax.xaxis.tick_bottom()

# Set the remaining ticks to be on the left and bottom axes
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

# Set the linewidth of the x-axis and y-axis
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

# Show the plot
plt.show()



















##########################################################################################

######################### To zip correlated trials csv ###################################
values_list = [16, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 58, 60]

print(len((rat_and_trial[0])))


values_list = [16, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 58, 60]

new_rat_and_trial = []
for sublist in rat_and_trial:
    new_sublist = []
    for subsublist in sublist:
        value = subsublist[0]
        if value in values_list:
            index = values_list.index(value)
            new_subsublist = [index, subsublist[1]]
            new_sublist.append(new_subsublist)
        else:
            new_subsublist = []
            new_sublist.append(new_subsublist)
    new_rat_and_trial.append(new_sublist)



print(len((correlated_trials_index_for_csv[0])[6]))







def produce_data(cluster_num,selection, comparison,selection2, comparison2, time_before,time_after):


    
    import numpy as np
    
    
    middle_back_data_of_correlated_trials = []
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((correlated_trials_index_for_csv)[clust]))):
            middle_back_data_of_correlated_trials_loop = []
             
            if len(((correlated_trials_index_for_csv)[clust])[index]) == 0:
                middle_back_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[0] for sublist in time_window_data]
                    y_selection = [sublist[1] for sublist in time_window_data]
                                       
                    middle_back_data_of_correlated_trials_loop.append([x_selection,y_selection])
                middle_back_data_of_correlated_trials.append(middle_back_data_of_correlated_trials_loop)
            
    
    
    
            
    after_middle_back_data_of_correlated_trials = []
    
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((after_correlated_trials_index_for_csv)[clust]))):
            after_middle_back_data_of_correlated_trials_loop = []
             
            if len(((after_correlated_trials_index_for_csv)[clust])[index]) == 0:
                after_middle_back_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((after_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[0] for sublist in time_window_data]
                    y_selection = [sublist[1] for sublist in time_window_data]
                                       
                    after_middle_back_data_of_correlated_trials_loop.append([x_selection,y_selection])
                after_middle_back_data_of_correlated_trials.append(after_middle_back_data_of_correlated_trials_loop)
            

    after2_middle_back_data_of_correlated_trials = []
    
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((after2_correlated_trials_index_for_csv)[clust]))):
            after2_middle_back_data_of_correlated_trials_loop = []
             
            if len(((after2_correlated_trials_index_for_csv)[clust])[index]) == 0:
                after2_middle_back_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((after2_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[0] for sublist in time_window_data]
                    y_selection = [sublist[1] for sublist in time_window_data]
                                       
                    after2_middle_back_data_of_correlated_trials_loop.append([x_selection,y_selection])
                after2_middle_back_data_of_correlated_trials.append(after2_middle_back_data_of_correlated_trials_loop)
            

    
                
    
    
    non_middle_back_data_of_correlated_trials = []
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((non_correlated_trials_index_for_csv)[clust]))):
            non_middle_back_data_of_correlated_trials_loop = []
             
            if len(((non_correlated_trials_index_for_csv)[clust])[index]) == 0:
                non_middle_back_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((non_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[0] for sublist in time_window_data]
                    y_selection = [sublist[1] for sublist in time_window_data]
                                       
                    non_middle_back_data_of_correlated_trials_loop.append([x_selection,y_selection])
                non_middle_back_data_of_correlated_trials.append(non_middle_back_data_of_correlated_trials_loop)
            
    ############# Tail base #################
    
    tail_data_of_correlated_trials = []
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((correlated_trials_index_for_csv)[clust]))):
            tail_data_of_correlated_trials_loop = []
             
            if len(((correlated_trials_index_for_csv)[clust])[index]) == 0:
                tail_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[2] for sublist in time_window_data]
                    y_selection = [sublist[3] for sublist in time_window_data]
                                       
                    tail_data_of_correlated_trials_loop.append([x_selection,y_selection])
                tail_data_of_correlated_trials.append(tail_data_of_correlated_trials_loop)
            
    
    
    
            
    after_tail_data_of_correlated_trials = []
    
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((after_correlated_trials_index_for_csv)[clust]))):
            after_tail_data_of_correlated_trials_loop = []
             
            if len(((after_correlated_trials_index_for_csv)[clust])[index]) == 0:
                after_tail_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((after_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[2] for sublist in time_window_data]
                    y_selection = [sublist[3] for sublist in time_window_data]
                                       
                    after_tail_data_of_correlated_trials_loop.append([x_selection,y_selection])
                after_tail_data_of_correlated_trials.append(after_tail_data_of_correlated_trials_loop)
            

    after2_tail_data_of_correlated_trials = []
    
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((after2_correlated_trials_index_for_csv)[clust]))):
            after2_tail_data_of_correlated_trials_loop = []
             
            if len(((after2_correlated_trials_index_for_csv)[clust])[index]) == 0:
                after2_tail_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((after2_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[2] for sublist in time_window_data]
                    y_selection = [sublist[3] for sublist in time_window_data]
                                       
                    after2_tail_data_of_correlated_trials_loop.append([x_selection,y_selection])
                after2_tail_data_of_correlated_trials.append(after2_tail_data_of_correlated_trials_loop)
            

                
    
    
    non_tail_data_of_correlated_trials = []
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((non_correlated_trials_index_for_csv)[clust]))):
            non_tail_data_of_correlated_trials_loop = []
             
            if len(((non_correlated_trials_index_for_csv)[clust])[index]) == 0:
                non_tail_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((non_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[2] for sublist in time_window_data]
                    y_selection = [sublist[3] for sublist in time_window_data]
                                       
                    non_tail_data_of_correlated_trials_loop.append([x_selection,y_selection])
                non_tail_data_of_correlated_trials.append(non_tail_data_of_correlated_trials_loop)
            
                       
            
    ########## Snout ###############

    
    
    snoot_data_of_correlated_trials = []
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((correlated_trials_index_for_csv)[clust]))):
            snoot_data_of_correlated_trials_loop = []
             
            if len(((correlated_trials_index_for_csv)[clust])[index]) == 0:
                snoot_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[4] for sublist in time_window_data]
                    y_selection = [sublist[5] for sublist in time_window_data]
                                       
                    snoot_data_of_correlated_trials_loop.append([x_selection,y_selection])
                snoot_data_of_correlated_trials.append(snoot_data_of_correlated_trials_loop)
            
    
    
    
            
    after_snout_data_of_correlated_trials = []
    
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((after_correlated_trials_index_for_csv)[clust]))):
            after_snout_data_of_correlated_trials_loop = []
             
            if len(((after_correlated_trials_index_for_csv)[clust])[index]) == 0:
                after_snout_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((after_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[4] for sublist in time_window_data]
                    y_selection = [sublist[5] for sublist in time_window_data]
                                       
                    after_snout_data_of_correlated_trials_loop.append([x_selection,y_selection])
                after_snout_data_of_correlated_trials.append(after_snout_data_of_correlated_trials_loop)
   
    
   
    after2_snout_data_of_correlated_trials = []
    
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((after2_correlated_trials_index_for_csv)[clust]))):
            after2_snout_data_of_correlated_trials_loop = []
             
            if len(((after2_correlated_trials_index_for_csv)[clust])[index]) == 0:
                after2_snout_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((after2_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[4] for sublist in time_window_data]
                    y_selection = [sublist[5] for sublist in time_window_data]
                                       
                    after2_snout_data_of_correlated_trials_loop.append([x_selection,y_selection])
                after2_snout_data_of_correlated_trials.append(after2_snout_data_of_correlated_trials_loop)
            
          
    
    non_snout_data_of_correlated_trials = []
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((non_correlated_trials_index_for_csv)[clust]))):
            non_snout_data_of_correlated_trials_loop = []
             
            if len(((non_correlated_trials_index_for_csv)[clust])[index]) == 0:
                non_snout_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((non_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[4] for sublist in time_window_data]
                    y_selection = [sublist[5] for sublist in time_window_data]
                                       
                    non_snout_data_of_correlated_trials_loop.append([x_selection,y_selection])
                non_snout_data_of_correlated_trials.append(non_snout_data_of_correlated_trials_loop)
            
           
    ######## Neck ########
    neck_data_of_correlated_trials = []
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((correlated_trials_index_for_csv)[clust]))):
            neck_data_of_correlated_trials_loop = []
             
            if len(((correlated_trials_index_for_csv)[clust])[index]) == 0:
                neck_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[4] for sublist in time_window_data]
                    y_selection = [sublist[5] for sublist in time_window_data]
                                       
                    neck_data_of_correlated_trials_loop.append([x_selection,y_selection])
                neck_data_of_correlated_trials.append(neck_data_of_correlated_trials_loop)
            
    
    
    
            
    after_neck_data_of_correlated_trials = []
    
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((after_correlated_trials_index_for_csv)[clust]))):
            after_neck_data_of_correlated_trials_loop = []
             
            if len(((after_correlated_trials_index_for_csv)[clust])[index]) == 0:
                after_neck_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((after_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[4] for sublist in time_window_data]
                    y_selection = [sublist[5] for sublist in time_window_data]
                                       
                    after_neck_data_of_correlated_trials_loop.append([x_selection,y_selection])
                after_neck_data_of_correlated_trials.append(after_neck_data_of_correlated_trials_loop)
            

    after2_neck_data_of_correlated_trials = []
    
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((after2_correlated_trials_index_for_csv)[clust]))):
            after2_neck_data_of_correlated_trials_loop = []
             
            if len(((after2_correlated_trials_index_for_csv)[clust])[index]) == 0:
                after2_neck_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((after2_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[4] for sublist in time_window_data]
                    y_selection = [sublist[5] for sublist in time_window_data]
                                       
                    after2_neck_data_of_correlated_trials_loop.append([x_selection,y_selection])
                after2_neck_data_of_correlated_trials.append(after2_neck_data_of_correlated_trials_loop)
            

                
    
    
    non_neck_data_of_correlated_trials = []
    
    for clust in range(cluster_num,cluster_num + 1):
        
        for index in range(0,len(((non_correlated_trials_index_for_csv)[clust]))):
            non_neck_data_of_correlated_trials_loop = []
             
            if len(((non_correlated_trials_index_for_csv)[clust])[index]) == 0:
                non_neck_data_of_correlated_trials.append([])
                
            else:
                    
                for trial_time in (((non_correlated_trials_index_for_csv)[clust])[index]):
                        
                    time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                    x_selection = [sublist[6] for sublist in time_window_data]
                    y_selection = [sublist[7] for sublist in time_window_data]
                                       
                    non_neck_data_of_correlated_trials_loop.append([x_selection,y_selection])
                non_neck_data_of_correlated_trials.append(non_neck_data_of_correlated_trials_loop)
            
           
                    
            
            
    return non_tail_data_of_correlated_trials[selection:comparison] + non_tail_data_of_correlated_trials[selection2:comparison2], tail_data_of_correlated_trials[selection:comparison] + tail_data_of_correlated_trials[selection2:comparison2], after_tail_data_of_correlated_trials[selection:comparison] + after_tail_data_of_correlated_trials[selection2:comparison2], non_middle_back_data_of_correlated_trials[selection:comparison] + non_middle_back_data_of_correlated_trials[selection2:comparison2], middle_back_data_of_correlated_trials[selection:comparison] + middle_back_data_of_correlated_trials[selection2:comparison2], after_middle_back_data_of_correlated_trials[selection:comparison] + after_middle_back_data_of_correlated_trials[selection2:comparison2], non_snout_data_of_correlated_trials[selection:comparison] + non_snout_data_of_correlated_trials[selection2:comparison2], snoot_data_of_correlated_trials[selection:comparison] + snoot_data_of_correlated_trials[selection2:comparison2], after_snout_data_of_correlated_trials[selection:comparison] + after_snout_data_of_correlated_trials[selection2:comparison2], non_neck_data_of_correlated_trials[selection:comparison] +  non_neck_data_of_correlated_trials[selection2:comparison2], neck_data_of_correlated_trials[selection:comparison] + neck_data_of_correlated_trials[selection2:comparison2], after_neck_data_of_correlated_trials[selection:comparison] + after_neck_data_of_correlated_trials[selection2:comparison2], after2_tail_data_of_correlated_trials[selection:comparison] + after2_tail_data_of_correlated_trials[selection2:comparison2], after2_middle_back_data_of_correlated_trials[selection:comparison] + after2_middle_back_data_of_correlated_trials[selection2:comparison2],after2_snout_data_of_correlated_trials[selection:comparison] + after2_snout_data_of_correlated_trials[selection2:comparison2], after2_neck_data_of_correlated_trials[selection:comparison] + after2_neck_data_of_correlated_trials[selection2:comparison2]




#def zip_data(cluster_num,selection, comparison, time_before,time_after):
    
# VMS x DMS: 17,21,26,30
    
sim = (produce_data(0,16,17,26,27,0,600))

# Define each parameter as sim[i]
non_tail_data_of_correlated_trials = sim[0]
tail_data_of_correlated_trials = sim[1]
after_tail_data_of_correlated_trials = sim[2]
after2_tail_data_of_correlated_trials = sim[12]
non_middle_back_data_of_correlated_trials = sim[3]
middle_back_data_of_correlated_trials = sim[4]
after_middle_back_data_of_correlated_trials = sim[5]
after2_middle_back_data_of_correlated_trials = sim[13]
non_snout_data_of_correlated_trials = sim[6]
snoot_data_of_correlated_trials = sim[7]
after_snout_data_of_correlated_trials = sim[8]
after2_snout_data_of_correlated_trials = sim[14]
non_neck_data_of_correlated_trials = sim[9]
neck_data_of_correlated_trials = sim[10]
after_neck_data_of_correlated_trials = sim[11]
after2_neck_data_of_correlated_trials = sim[15]

print(len(after2_neck_data_of_correlated_trials))

def flatten_list(data_list):
    result = []
    for item in data_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

# Flatten and convert each dataset to an array
datasets = []
for i in range(16):
    flattened_data = flatten_list(sim[i])
    array_data = np.array(flattened_data).reshape(-1, 2)
    datasets.append(array_data)



# Define the parameter names
parameter_names = [
    "non_tail_data_of_correlated_trials",
    "tail_data_of_correlated_trials",
    "after_tail_data_of_correlated_trials",
    
    "non_middle_back_data_of_correlated_trials",
    "middle_back_data_of_correlated_trials",
    "after_middle_back_data_of_correlated_trials",
    
    "non_snout_data_of_correlated_trials",
    "snout_data_of_correlated_trials",
    "after_snout_data_of_correlated_trials",
    
    "non_neck_data_of_correlated_trials",
    "neck_data_of_correlated_trials",
    "after_neck_data_of_correlated_trials",
    "after2_tail_data_of_correlated_trials",
    "after2_middle_back_data_of_correlated_trials",
    "after2_snout_data_of_correlated_trials",
    "after2_neck_data_of_correlated_trials"
]

# Store the shapes in a dictionary
parameter_shapes = {}
for i in range(16):
    parameter_shapes[parameter_names[i]] = datasets[i].shape

# Print the shape of each parameter
for parameter_name, parameter_shape in parameter_shapes.items():
    print(parameter_name, "shape:", parameter_shape)

import numpy as np

# Create arrays with zeros of the desired shape
array_1 = np.zeros((parameter_shapes["tail_data_of_correlated_trials"][0], 8))
array_2 = np.zeros((parameter_shapes["non_tail_data_of_correlated_trials"][0], 8))
array_3 = np.zeros((parameter_shapes["after_tail_data_of_correlated_trials"][0], 8))
array_4 = np.zeros((parameter_shapes["after2_tail_data_of_correlated_trials"][0], 8))

# Assign values from the corresponding arrays
array_1[:, :2] = datasets[1]  # tail_data_of_correlated_trials
array_1[:, 2:4] = datasets[4]  # middle_back_data_of_correlated_trials
array_1[:, 4:6] = datasets[7]  # snout_data_of_correlated_trials
array_1[:, 6:] = datasets[10]  # neck_data_of_correlated_trials

array_2[:, :2] = datasets[0]  # non_tail_data_of_correlated_trials
array_2[:, 2:4] = datasets[3]  # non_middle_back_data_of_correlated_trials
array_2[:, 4:6] = datasets[6]  # non_snout_data_of_correlated_trials
array_2[:, 6:] = datasets[9]  # non_neck_data_of_correlated_trials

array_3[:, :2] = datasets[2]  # after_tail_data_of_correlated_trials
array_3[:, 2:4] = datasets[5]  # after_middle_back_data_of_correlated_trials
array_3[:, 4:6] = datasets[8]  # after_snout_data_of_correlated_trials
array_3[:, 6:] = datasets[11]  # after_neck_data_of_correlated_trials

array_4[:, :2] = datasets[12]  # after_tail_data_of_correlated_trials
array_4[:, 2:4] = datasets[13]  # after_middle_back_data_of_correlated_trials
array_4[:, 4:6] = datasets[14]  # after_snout_data_of_correlated_trials
array_4[:, 6:] = datasets[15]  # after_neck_data_of_correlated_trials

# Print the shapes of the new arrays
print("Array 1 shape:", array_1.shape)
print("Array 2 shape:", array_2.shape)
print("Array 3 shape:", array_3.shape)
print("Array 4 shape:", array_4.shape)
import pandas as pd


import pandas as pd

# Define the file paths
root_path = r'C:\Users\georg\Neuroscience\NiN internship\Behavioral analysis\Pavlovian\CSV Files from python code\Files to overwrite\\'
correlated_file = 'PavlCond_AP&AV_Vrec_55DLC_resnet50_pav_instrMar10shuffle1_1030000.csv'

# Read the 'Correlated' CSV file with the first 3 rows as header
correlated_data = pd.read_csv(root_path + correlated_file, header=[0, 1, 2])

# Get the number of rows in the file
num_rows = correlated_data.shape[0]

# Determine the length of the data from array_1
array_1_length = array_3.shape[0]

# Determine the number of rows to overwrite
num_rows_to_overwrite = min(num_rows, array_1_length)

# Overwrite the specified columns with the data from array_1
correlated_data.iloc[:num_rows_to_overwrite, 1] = array_3[:num_rows_to_overwrite, 0]  # Overwrite 2nd column
correlated_data.iloc[:num_rows_to_overwrite, 2] = array_3[:num_rows_to_overwrite, 1]  # Overwrite 3rd column
correlated_data.iloc[:num_rows_to_overwrite, 4] = array_3[:num_rows_to_overwrite, 2]  # Overwrite 5th column
correlated_data.iloc[:num_rows_to_overwrite, 5] = array_3[:num_rows_to_overwrite, 3]  # Overwrite 6th column
correlated_data.iloc[:num_rows_to_overwrite, -12] = array_3[:num_rows_to_overwrite, 4]  # Overwrite -12th column
correlated_data.iloc[:num_rows_to_overwrite, -11] = array_3[:num_rows_to_overwrite, 5]  # Overwrite -11th column
correlated_data.iloc[:num_rows_to_overwrite, 13] = array_3[:num_rows_to_overwrite, 6]  # Overwrite 13th column
correlated_data.iloc[:num_rows_to_overwrite, 14] = array_3[:num_rows_to_overwrite, 7]  # Overwrite 14th column



# Create a list of column indexes to keep
columns_to_keep = [0]  # Keep the first column
columns_to_keep += [1, 2, 3, 4, 5, 6, -12, -11, -10, 13, 14, 15]  # Keep the specified columns

# Drop all other columns
correlated_data = correlated_data.iloc[:, columns_to_keep]

# Truncate the DataFrame to the desired length
correlated_data = correlated_data.truncate(after=num_rows_to_overwrite-1)

# Save the updated 'Correlated' CSV file
correlated_data.to_csv(root_path + correlated_file, index=False)







print(rat_and_trial[2])






















nrows = 1
ncols = 3

if nrows == 1:

    # Create a plot for each cluster
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6), sharey=True)
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i in range(0,ncols):
        cluster_data = DATA_matrices[kmeans.labels_ == i]
        print(len(cluster_data))
        means = np.mean(cluster_data, axis=0)
        stds = np.std(cluster_data, axis=0)  # standard error of the mean
        #x = np.arange(0,35,0.1)
        x = np.arange(-9.5,25.5,0.1)
        axs[i].set_title(f"Cluster {i}", fontsize=14)
        axs[i].plot(x, means, color=colors[i], linewidth=2)
        lower_bound = (means - stds).ravel()
        upper_bound = (means + stds).ravel()
        
        axs[i].fill_between(x, lower_bound, upper_bound, color=colors[i], alpha=0.1)
        
        axs[i].set_xlabel('Time(sec)')
        
        axs[i].set_yticklabels(axs[i].get_yticks(), fontsize=12)
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)
        axs[i].tick_params(axis='both', which='both', length=0)
    plt.suptitle("Cluster Means and Standard Errors of the Mean", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
else:
    # Create a plot for each cluster
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 6), sharey=True)
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan','blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            cluster_data = X[kmeans.labels_ == idx]
            means = np.mean(cluster_data, axis=0)
            stds = np.std(cluster_data, axis=0)  # standard error of the mean
            x = np.arange(len(means))
            #axs[i,j].set_title(f"Cluster {idx}", fontsize=14)
            axs[i,j].plot(x, means, color=colors[idx], linewidth=2)
            lower_bound = (means - stds).ravel()
            upper_bound = (means + stds).ravel()
    
            axs[i,j].fill_between(x, lower_bound, upper_bound, color=colors[idx], alpha=0.1)
            axs[i,j].set_xticks(x)
            axs[i,j].set_xticklabels([])
            axs[i,j].set_yticklabels(axs[i,j].get_yticks(), fontsize=12)
            axs[i,j].grid(axis='y', linestyle='--', alpha=0.7)
            axs[i,j].tick_params(axis='both', which='both', length=0)
    
    plt.suptitle("Cluster Means and Standard Errors of the Mean", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    

    








import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the first file
df1 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\non\Jun-03-2023labels_pose_30HzNon Correlated c2 VMSxDLS.csv', header=0)
# Load the second file
df2 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\cor\Jun-03-2023labels_pose_30HzCorrelated c0 VMSxDLS.csv', header=0)
# Load the third file
df3 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\after\Jun-03-2023labels_pose_30HzAfter Correlated c0 VMSxDLS.csv', header=0)
# Load the fourth file
df4 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\cor\Jun-03-2023labels_pose_30HzCorrelated c2 VMSxDLS.csv', header=0)
# Load the fifth file
df5 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\after\Jun-03-2023labels_pose_30HzAfter Correlated c2 VMSxDMS.csv', header=0)

# Calculate occurrences of number 0 in bins of 30 values for df1
bin_size = 30
num_bins = len(df1) // bin_size
occurrences_in_bins_df1 = [df1['B-SOiD labels'].iloc[i * bin_size : (i + 1) * bin_size].eq(0).sum() for i in range(num_bins)]
occurrences_in_bins_df1.append(df1['B-SOiD labels'].iloc[num_bins * bin_size :].eq(0).sum())

bins_df1 = range(0, len(df1) + bin_size, bin_size)  # Generate bin edges
bin_labels_df1 = [f"({bins_df1[i]}, {bins_df1[i+1]}]" for i in range(num_bins)] + [f"({bins_df1[num_bins]}, {len(df1)}]"]
occurrences_in_bins_df1 = pd.Series(occurrences_in_bins_df1, index=bin_labels_df1)

print("Occurrences of number 0 in bins of 30 values for df1:")
print(occurrences_in_bins_df1)

# Calculate occurrences of number 0 in bins of 30 values for df2
bin_size = 30
num_bins = len(df2) // bin_size
occurrences_in_bins_df2 = [df2['B-SOiD labels'].iloc[i * bin_size : (i + 1) * bin_size].eq(0).sum() for i in range(num_bins)]
occurrences_in_bins_df2.append(df2['B-SOiD labels'].iloc[num_bins * bin_size :].eq(0).sum())

bins_df2 = range(0, len(df2) + bin_size, bin_size)  # Generate bin edges
bin_labels_df2 = [f"({bins_df2[i]}, {bins_df2[i+1]}]" for i in range(num_bins)] + [f"({bins_df2[num_bins]}, {len(df2)}]"]
occurrences_in_bins_df2 = pd.Series(occurrences_in_bins_df2, index=bin_labels_df2)

print("Occurrences of number 0 in bins of 30 values for df2:")
print(occurrences_in_bins_df2)

# Calculate occurrences of number 0 in bins of 30 values for df3
bin_size = 30
num_bins = len(df3) // bin_size
occurrences_in_bins_df3 = [df3['B-SOiD labels'].iloc[i * bin_size : (i + 1) * bin_size].eq(0).sum() for i in range(num_bins)]
occurrences_in_bins_df3.append(df3['B-SOiD labels'].iloc[num_bins * bin_size :].eq(0).sum())

bins_df3 = range(0, len(df3) + bin_size, bin_size)  # Generate bin edges
bin_labels_df3 = [f"({bins_df3[i]}, {bins_df3[i+1]}]" for i in range(num_bins)] + [f"({bins_df3[num_bins]}, {len(df3)}]"]
occurrences_in_bins_df3 = pd.Series(occurrences_in_bins_df3, index=bin_labels_df3)

print("Occurrences of number 0 in bins of 30 values for df3:")
print(occurrences_in_bins_df3)

# Calculate occurrences of number 0 in bins of 30 values for df4
bin_size = 30
num_bins = len(df4) // bin_size
occurrences_in_bins_df4 = [df4['B-SOiD labels'].iloc[i * bin_size : (i + 1) * bin_size].eq(0).sum() for i in range(num_bins)]
occurrences_in_bins_df4.append(df4['B-SOiD labels'].iloc[num_bins * bin_size :].eq(0).sum())

bins_df4 = range(0, len(df4) + bin_size, bin_size)  # Generate bin edges
bin_labels_df4 = [f"({bins_df4[i]}, {bins_df4[i+1]}]" for i in range(num_bins)] + [f"({bins_df4[num_bins]}, {len(df4)}]"]
occurrences_in_bins_df4 = pd.Series(occurrences_in_bins_df4, index=bin_labels_df4)

print("Occurrences of number 0 in bins of 30 values for df4:")
print(occurrences_in_bins_df4)

# Calculate occurrences of number 0 in bins of 30 values for df5
bin_size = 30
num_bins = len(df5) // bin_size
occurrences_in_bins_df5 = [df5['B-SOiD labels'].iloc[i * bin_size : (i + 1) * bin_size].eq(0).sum() for i in range(num_bins)]
occurrences_in_bins_df5.append(df5['B-SOiD labels'].iloc[num_bins * bin_size :].eq(0).sum())

bins_df5 = range(0, len(df5) + bin_size, bin_size)  # Generate bin edges
bin_labels_df5 = [f"({bins_df5[i]}, {bins_df5[i+1]}]" for i in range(num_bins)] + [f"({bins_df5[num_bins]}, {len(df5)}]"]
occurrences_in_bins_df5 = pd.Series(occurrences_in_bins_df5, index=bin_labels_df5)

print("Occurrences of number 0 in bins of 30 values for df5:")
print(occurrences_in_bins_df5)

# Extract the values from the Series
data_df1 = occurrences_in_bins_df1.values
data_df2 = occurrences_in_bins_df2.values
data_df3 = occurrences_in_bins_df3.values
data_df4 = occurrences_in_bins_df4.values
data_df5 = occurrences_in_bins_df5.values

# Divide data into bins of size 20 and calculate the average of each index across the bins
bin_size = 20
num_bins = len(data_df1) // bin_size
averaged_values_df1 = [data_df1[i : len(data_df1) : bin_size].mean() for i in range(bin_size)]
averaged_values_df2 = [data_df2[i : len(data_df2) : bin_size].mean() for i in range(bin_size)]
averaged_values_df3 = [data_df3[i : len(data_df3) : bin_size].mean() for i in range(bin_size)]
averaged_values_df4 = [data_df4[i : len(data_df4) : bin_size].mean() for i in range(bin_size)]
averaged_values_df5 = [data_df5[i : len(data_df5) : bin_size].mean() for i in range(bin_size)]

# Plotting
x_range_df1 = np.arange(0, 20, 1)
x_range_df2 = np.arange(0, 20, 1)
colors = ["grey", "orange", "darkorange", "blue", "darkblue"]

plt.plot(x_range_df1, averaged_values_df1, color=colors[0], label='df1')
plt.plot(x_range_df2, averaged_values_df2, color=colors[1], label='df2')
plt.plot(x_range_df2, averaged_values_df3, color=colors[2], label='df3')
plt.plot(x_range_df2, averaged_values_df4, color=colors[3], label='df4')
plt.plot(x_range_df2, averaged_values_df5, color=colors[4], label='df5')

plt.xlabel('Index')
plt.ylabel('Average Occurrences')
plt.title('Average Occurrences of Number 0 Across Bins')

plt.legend()
plt.show()






    
    ########### FOR VALUE 5 ##############
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    

def plot_occurancies(value):
    # Load the first file
    df1 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\cor\Jun-03-2023labels_pose_30HzCorrelated c0 VMSxDMS.csv', header=0)
    
    # Calculate occurrences of number 5 in bins of 30 values for df1
    bin_size = 30
    num_bins = len(df1) // bin_size
    occurrences_in_bins_df1 = [df1['B-SOiD labels'].iloc[i * bin_size : (i + 1) * bin_size].eq(value).sum() for i in range(num_bins)]
    occurrences_in_bins_df1.append(df1['B-SOiD labels'].iloc[num_bins * bin_size :].eq(value).sum())
    
    bins_df1 = range(0, len(df1) + bin_size, bin_size)  # Generate bin edges
    bin_labels_df1 = [f"({bins_df1[i]}, {bins_df1[i+1]}]" for i in range(num_bins)] + [f"({bins_df1[num_bins]}, {len(df1)}]"]
    occurrences_in_bins_df1 = pd.Series(occurrences_in_bins_df1, index=bin_labels_df1)
    
    print("Occurrences of number 5 in bins of 30 values for df1:")
    print(occurrences_in_bins_df1)
    
    # Load the second file
    df2 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\cor\Jun-03-2023labels_pose_30HzCorrelated c2 VMSxDMS.csv', header=0)
    
    # Calculate occurrences of number 5 in bins of 30 values for df2
    bin_size = 30
    num_bins = len(df2) // bin_size
    occurrences_in_bins_df2 = [df2['B-SOiD labels'].iloc[i * bin_size : (i + 1) * bin_size].eq(value).sum() for i in range(num_bins)]
    occurrences_in_bins_df2.append(df2['B-SOiD labels'].iloc[num_bins * bin_size :].eq(value).sum())
    
    bins_df2 = range(0, len(df2) + bin_size, bin_size)  # Generate bin edges
    bin_labels_df2 = [f"({bins_df2[i]}, {bins_df2[i+1]}]" for i in range(num_bins)] + [f"({bins_df2[num_bins]}, {len(df2)}]"]
    occurrences_in_bins_df2 = pd.Series(occurrences_in_bins_df2, index=bin_labels_df2)
    
    print("Occurrences of number 5 in bins of 30 values for df2:")
    print(occurrences_in_bins_df2)
    
    # Load the third file
    df3 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\cor\Jun-03-2023labels_pose_30HzCorrelated c0 VMSxDLS.csv', header=0)
    
    # Calculate occurrences of number 5 in bins of 30 values for df3
    bin_size = 30
    num_bins = len(df3) // bin_size
    occurrences_in_bins_df3 = [df3['B-SOiD labels'].iloc[i * bin_size : (i + 1) * bin_size].eq(value).sum() for i in range(num_bins)]
    occurrences_in_bins_df3.append(df3['B-SOiD labels'].iloc[num_bins * bin_size :].eq(value).sum())
    
    bins_df3 = range(0, len(df3) + bin_size, bin_size)  # Generate bin edges
    bin_labels_df3 = [f"({bins_df3[i]}, {bins_df3[i+1]}]" for i in range(num_bins)] + [f"({bins_df3[num_bins]}, {len(df3)}]"]
    occurrences_in_bins_df3 = pd.Series(occurrences_in_bins_df3, index=bin_labels_df3)
    
    print("Occurrences of number 5 in bins of 30 values for df3:")
    print(occurrences_in_bins_df3)
    
    # Load the fourth file
    df4 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\cor\Jun-03-2023labels_pose_30HzCorrelated c2 VMSxDLS.csv', header=0)
    
    # Calculate occurrences of number 5 in bins of 30 values for df4
    bin_size = 30
    num_bins = len(df4) // bin_size
    occurrences_in_bins_df4 = [df4['B-SOiD labels'].iloc[i * bin_size : (i + 1) * bin_size].eq(value).sum() for i in range(num_bins)]
    occurrences_in_bins_df4.append(df4['B-SOiD labels'].iloc[num_bins * bin_size :].eq(value).sum())
    
    bins_df4 = range(0, len(df4) + bin_size, bin_size)  # Generate bin edges
    bin_labels_df4 = [f"({bins_df4[i]}, {bins_df4[i+1]}]" for i in range(num_bins)] + [f"({bins_df4[num_bins]}, {len(df4)}]"]
    occurrences_in_bins_df4 = pd.Series(occurrences_in_bins_df4, index=bin_labels_df4)
    
    print("Occurrences of number 5 in bins of 30 values for df4:")
    print(occurrences_in_bins_df4)
    
    # Load the fifth file
    df5 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\non\Jun-03-2023labels_pose_30HzNon Correlated c2 VMSxDMS.csv', header=0)
    
    # Calculate occurrences of number 5 in bins of 30 values for df5
    bin_size = 30
    num_bins = len(df5) // bin_size
    occurrences_in_bins_df5 = [df5['B-SOiD labels'].iloc[i * bin_size : (i + 1) * bin_size].eq(value).sum() for i in range(num_bins)]
    occurrences_in_bins_df5.append(df5['B-SOiD labels'].iloc[num_bins * bin_size :].eq(value).sum())
    
    bins_df5 = range(0, len(df5) + bin_size, bin_size)  # Generate bin edges
    bin_labels_df5 = [f"({bins_df5[i]}, {bins_df5[i+1]}]" for i in range(num_bins)] + [f"({bins_df5[num_bins]}, {len(df5)}]"]
    occurrences_in_bins_df5 = pd.Series(occurrences_in_bins_df5, index=bin_labels_df5)
    
    print("Occurrences of number 5 in bins of 30 values for df5:")
    print(occurrences_in_bins_df5)
    
    # Load the fifth file
    df6 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\non\Jun-03-2023labels_pose_30HzNon Correlated c2 VMSxDLS.csv', header=0)
    
    # Calculate occurrences of number 5 in bins of 30 values for df6
    bin_size = 30
    num_bins = len(df6) // bin_size
    occurrences_in_bins_df6 = [df6['B-SOiD labels'].iloc[i * bin_size : (i + 1) * bin_size].eq(value).sum() for i in range(num_bins)]
    occurrences_in_bins_df6.append(df6['B-SOiD labels'].iloc[num_bins * bin_size :].eq(value).sum())
    
    bins_df6 = range(0, len(df6) + bin_size, bin_size)  # Generate bin edges
    bin_labels_df6 = [f"({bins_df6[i]}, {bins_df6[i+1]}]" for i in range(num_bins)] + [f"({bins_df6[num_bins]}, {len(df5)}]"]
    occurrences_in_bins_df6 = pd.Series(occurrences_in_bins_df6, index=bin_labels_df6)
    
    print("Occurrences of number 5 in bins of 30 values for df6:")
    print(occurrences_in_bins_df6)
        
    
    # Extract the values from the Series
    data_df1 = occurrences_in_bins_df1.values
    data_df2 = occurrences_in_bins_df2.values
    data_df3 = occurrences_in_bins_df3.values
    data_df4 = occurrences_in_bins_df4.values
    data_df5 = occurrences_in_bins_df5.values
    data_df6 = occurrences_in_bins_df6.values
    
    # Divide data into bins of size 20 and calculate the average and standard error of each index across the bins
    bin_size = 20
    num_bins = len(data_df1) // bin_size
    averaged_values_df1 = [data_df1[i: len(data_df1): bin_size].mean() for i in range(bin_size)]
    averaged_values_df2 = [data_df2[i: len(data_df2): bin_size].mean() for i in range(bin_size)]
    averaged_values_df3 = [data_df3[i: len(data_df3): bin_size].mean() for i in range(bin_size)]
    averaged_values_df4 = [data_df4[i: len(data_df4): bin_size].mean() for i in range(bin_size)]
    averaged_values_df5 = [data_df5[i: len(data_df5): bin_size].mean() for i in range(bin_size)]
    averaged_values_df6 = [data_df6[i: len(data_df6): bin_size].mean() for i in range(bin_size)]
    
    sample_size = len(data_df1) // bin_size
    standard_error_df1 = [data_df1[i: len(data_df1): bin_size].std() / np.sqrt(sample_size) for i in range(bin_size)]
    standard_error_df2 = [data_df2[i: len(data_df2): bin_size].std() / np.sqrt(sample_size) for i in range(bin_size)]
    standard_error_df3 = [data_df3[i: len(data_df3): bin_size].std() / np.sqrt(sample_size) for i in range(bin_size)]
    standard_error_df4 = [data_df4[i: len(data_df4): bin_size].std() / np.sqrt(sample_size) for i in range(bin_size)]
    standard_error_df5 = [data_df5[i: len(data_df5): bin_size].std() / np.sqrt(sample_size) for i in range(bin_size)]
    standard_error_df6 = [data_df6[i: len(data_df6): bin_size].std() / np.sqrt(sample_size) for i in range(bin_size)]
    
    print("Averaged values across indices for df1:")
    print(averaged_values_df1)
    
    print("Averaged values across indices for df2:")
    print(averaged_values_df2)
    
    print("Averaged values across indices for df3:")
    print(averaged_values_df3)
    
    print("Averaged values across indices for df4:")
    print(averaged_values_df4)
    
    print("Averaged values across indices for df5:")
    print(averaged_values_df5)
    
    # Plot the results
    x_range_df1 = np.arange(1, 21, 1)
    x_range_df2 = np.arange(1, 21, 1)
    
    # Scatter points with error bars
    plt.errorbar(x_range_df1, averaged_values_df1, yerr=standard_error_df1, fmt='o', color='brown', label='VMSxDMS c0')
    #plt.errorbar(x_range_df2, averaged_values_df2, yerr=standard_error_df2, fmt='o', color='darkviolet', label='VMSxDMS c2')
    plt.errorbar(x_range_df2, averaged_values_df3, yerr=standard_error_df3, fmt='o', color='purple', label='VMSxDLS c0')
    #plt.errorbar(x_range_df2, averaged_values_df4, yerr=standard_error_df4, fmt='o', color='green', label='VMSxDLS c2')
    plt.errorbar(x_range_df2, averaged_values_df5, yerr=standard_error_df5, fmt='o', color='black', label='Non VMSxDMS')
    plt.errorbar(x_range_df2, averaged_values_df6, yerr=standard_error_df6, fmt='o', color='grey', label='Non VMSxDLS')
    
    # Connect points with lines
    plt.plot(x_range_df1, averaged_values_df1, color='brown', linestyle='-', linewidth=1)
    #plt.plot(x_range_df2, averaged_values_df2, color='darkviolet', linestyle='-', linewidth=1)
    plt.plot(x_range_df2, averaged_values_df3, color='purple', linestyle='-', linewidth=1)
    #plt.plot(x_range_df2, averaged_values_df4, color='green', linestyle='-', linewidth=1)
    plt.plot(x_range_df2, averaged_values_df5, color='black', linestyle='-', linewidth=1)
    plt.plot(x_range_df2, averaged_values_df6, color='grey', linestyle='-', linewidth=1)
    
    plt.axvline(x=0, color='black', linestyle='--', label='traylight')
    plt.axvline(x=5, color='black', linestyle='--', label='reward')
    
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time(sec)')
    plt.ylabel('Average Occurrences(frames/sec)')
    plt.title('Occurrences of Syllable {}'.format(value))
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    
    plt.show()
    
    normalized_values_df1 = [value / averaged_values_df5[idx] for idx, value in enumerate(averaged_values_df1)]
    normalized_values_df2 = [value / averaged_values_df5[idx] for idx, value in enumerate(averaged_values_df2)]
    normalized_values_df3 = [value / averaged_values_df6[idx] for idx, value in enumerate(averaged_values_df3)]
    normalized_values_df4 = [value / averaged_values_df6[idx] for idx, value in enumerate(averaged_values_df4)]
    normalized_values_df5 = [value / averaged_values_df5[idx] for idx, value in enumerate(averaged_values_df5)]
    normalized_values_df6 = [value / averaged_values_df6[idx] for idx, value in enumerate(averaged_values_df6)]

    # Scatter points with error bars
    plt.errorbar(x_range_df2, normalized_values_df1, yerr=standard_error_df1, fmt='o', color='brown', label='VMSxDMS c0')
    #plt.errorbar(x_range_df2, normalized_values_df2, yerr=standard_error_df2, fmt='o', color='darkviolet', label='VMSxDMS c2')
    #plt.errorbar(x_range_df2, normalized_values_df3, yerr=standard_error_df3, fmt='o', color='blue', label='VMSxDMS c0')
    plt.errorbar(x_range_df2, normalized_values_df4, yerr=standard_error_df4, fmt='o', color='purple', label='VMSxDMS c2')
    #plt.errorbar(x_range_df2, normalized_values_df5, yerr=standard_error_df5, fmt='o', color='black', label='Non VMSxDMS')
    #plt.errorbar(x_range_df2, normalized_values_df6, yerr=standard_error_df6, fmt='o', color='grey', label='Non VMSxDMS')
    
    # Connect points with lines
    plt.plot(x_range_df2, normalized_values_df1, color='brown', linestyle='-', linewidth=1)
    #plt.plot(x_range_df2, normalized_values_df2, color='darkviolet', linestyle='-', linewidth=1)
    #plt.plot(x_range_df2, normalized_values_df3, color='blue', linestyle='-', linewidth=1)
    plt.plot(x_range_df2, normalized_values_df4, color='purple', linestyle='-', linewidth=1)
    plt.plot(x_range_df2, normalized_values_df5, color='black', linestyle='-', linewidth=1)
    plt.plot(x_range_df2, normalized_values_df6, color='grey', linestyle='-', linewidth=1)
    
    plt.axvline(x=0, color='black', linestyle='--', label='traylight')
    plt.axvline(x=5, color='black', linestyle='--', label='reward')
    
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time(sec)')
    plt.ylabel('Normalized Average Occurrences')
    plt.title('Occurrences of Syllable {}'.format(value))
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    
    plt.show()
 
plot_occurancies(0)
plot_occurancies(1)
plot_occurancies(2)
plot_occurancies(3)
plot_occurancies(4)
plot_occurancies(5)
plot_occurancies(6)



print(rat_and_trial[2])

#### To plot occurrence across time
   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the fifth file
df6 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\specific\Jun-03-2023labels_pose_30HzPavlCond_AP&AV_Vrec_55DLC_resnet50_pav_instrMar10shuffle1_1030000.csv', header=0)
bin_size = 30
c0_trial_start = (correlated_trials_index_for_csv[0][-3])
c2_trial_start = (correlated_trials_index_for_csv[2][-3])

c0_trial_start = [value / bin_size for value in c0_trial_start]
c2_trial_start = [value / bin_size for value in c2_trial_start]

# Define the value you want to count occurrences for
value = 4

# Calculate occurrences of value in bins

num_bins = len(df6) // bin_size
occurrences_in_bins_df6 = [df6['B-SOiD labels'].iloc[i * bin_size : (i + 1) * bin_size].eq(value).sum() for i in range(num_bins)]
occurrences_in_bins_df6.append(df6['B-SOiD labels'].iloc[num_bins * bin_size :].eq(value).sum())

# Divide the occurrences by the mean
mean_occurrences = np.mean(occurrences_in_bins_df6)
print(mean_occurrences)
normalized_data = [occurrence / mean_occurrences for occurrence in occurrences_in_bins_df6]

# Calculate the logarithm base 2 of the normalized data
log_normalized_data = np.log2(normalized_data)

x = np.arange(0, (num_bins + 1), 1)
# Plot the log-transformed data
plt.plot(x, occurrences_in_bins_df6)
plt.xlabel('Time(sec)')
plt.ylabel('Log2(Occurrences / Mean)')
plt.title(f'Logarithm Base 2 of Occurrences of {value} in Bins (Normalized)')
# Plot vertical lines at the divided values
for val in c0_trial_start:
    plt.axvline(x=val, color='yellow', linestyle='--')
# Plot vertical lines at the divided values
for val in c2_trial_start:
    plt.axvline(x=val, color='red', linestyle='--')
plt.xlim(1200,1600)
plt.show()








print((correlated_trials_index_for_csv[2][-3]))












import pandas as pd
import matplotlib.pyplot as plt

# Load the file
file_path = r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\specific\Jun-03-2023labels_pose_30HzPavlCond_AP&AV_Vrec_55DLC_resnet50_pav_instrMar10shuffle1_1030000.csv'
df6 = pd.read_csv(file_path, header=0)

# Define the value you want to count occurrences for
value = 0

# Calculate occurrences of value in bins
bin_size = 30
num_bins = len(df6) // bin_size
occurrences_in_bins_df6 = [df6['B-SOiD labels'].iloc[i * bin_size : (i + 1) * bin_size].eq(value).sum() for i in range(num_bins)]
occurrences_in_bins_df6.append(df6['B-SOiD labels'].iloc[num_bins * bin_size :].eq(value).sum())

# Smooth the occurrences data
smoothed = pd.Series(occurrences_in_bins_df6).rolling(window=5).mean()  # Adjust window size as needed

# Plot the smoothed occurrences
plt.plot(smoothed)
plt.xlabel('Bins')
plt.ylabel('Occurrences')
plt.title('Smoothed Occurrences of Value 0')
plt.xlim(0,100)
plt.show()





########## To calculate the entropy with clustering method #################



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_rel

# Load the datasets
df1 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\cor\Jun-03-2023labels_pose_30HzCorrelated c0 VMSxDMS.csv', header=0)
df2 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\cor\Jun-03-2023labels_pose_30HzCorrelated c0 VMSxDLS.csv', header=0)
df3 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\cor\Jun-03-2023labels_pose_30HzCorrelated c0 DMSxDLS.csv', header=0)
df4 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\after\Jun-03-2023labels_pose_30HzAfter Correlated c0 VMSxDMS.csv', header=0)
df5 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\after\Jun-03-2023labels_pose_30HzAfter Correlated c0 VMSxDLS.csv', header=0)
df6 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\after\Jun-03-2023labels_pose_30HzAfter Correlated c0 DMSxDLS.csv', header=0)

# Calculate entropy for each dataset
datasets = [df1, df2, df3, df4, df5, df6]
dataset_names = ['VMSxDMS c0', 'VMSxDLS c0', 'DMSxDLS c0', 'After VMSxDMS', 'After VMSxDLS', 'After DMSxDLS']
colors = ['brown', 'purple', 'pink', 'black', 'violet', 'orange']
all_entropies = []


fig, ax = plt.subplots()

# Spacing between bar graphs
spacing = 0.2

for i, dataset in enumerate(datasets):
    syllable_labels = dataset['B-SOiD labels']

    # Calculate the transition counts for each syllable
    transition_counts = pd.DataFrame(columns=syllable_labels.unique(), index=syllable_labels.unique(), data=0)
    for j in range(len(syllable_labels) - 1):
        current_syllable = syllable_labels.iloc[j]
        next_syllable = syllable_labels.iloc[j + 1]
        transition_counts.loc[current_syllable, next_syllable] += 1

    # Calculate the transition probabilities
    transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)

    # Calculate the entropy for each syllable
    entropy_per_syllable = -np.sum(transition_probs * np.log2(transition_probs), axis=1)

    # Append individual entropy values to the list
    all_entropies.append(entropy_per_syllable)

    # Plot individual entropy values as scatter points
    x_values = [i * spacing] * len(entropy_per_syllable)
    plt.scatter(x_values, entropy_per_syllable, color=colors[i], alpha=0.5, label=dataset_names[i])

# Calculate mean entropies
mean_entropies = [np.mean(entropies) for entropies in all_entropies]

# Perform paired t-tests for all combinations of pairs of datasets
pairwise_tests = list(combinations(range(len(datasets)), 2))
p_values = []
for test in pairwise_tests:
    paired_entropies = [all_entropies[test[0]], all_entropies[test[1]]]
    statistic, p_value = ttest_rel(*paired_entropies)
    p_values.append(p_value)

# Plotting the bar graphs for each dataset
x = np.arange(len(dataset_names)) * spacing  # Adjust the multiplier (spacing) to decrease the distance

bars = ax.bar(x, mean_entropies, 0.12, yerr=np.std(all_entropies, axis=1), capsize=5,
              label='Average Entropy', alpha=0.5, color=colors)

# Formatting the plot
ax.set_ylabel('Entropy')
ax.set_title('Average Entropy of Syllables per Dataset')
ax.set_xticks(x)
ax.set_xticklabels(dataset_names, rotation=45, ha='right')

"""
# Add horizontal lines and asterisks for significant results
significant_indices = np.where(np.array(p_values) < 0.05)[0]
for index in significant_indices:
    bar1_index, bar2_index = pairwise_tests[index]
    height = max(mean_entropies[bar1_index], mean_entropies[bar2_index]) + 0.1
    ax.plot([x[bar1_index], x[bar2_index]], [height, height], color='black', linewidth=1)
    ax.text((x[bar1_index] + x[bar2_index]) / 2, height + 0.05, '*', ha='center', va='bottom', fontsize=12)
"""

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

plt.tight_layout()
plt.show()

# Print the results of paired t-tests
print("Paired t-test results:")
for i, test in enumerate(pairwise_tests):
    dataset1 = dataset_names[test[0]]
    dataset2 = dataset_names[test[1]]
    p_value = p_values[i]
    if p_value < 0.05:
        print(f"{dataset1} vs {dataset2}: p-value = {p_value:.4f} (Significant)")
    else:
        print(f"{dataset1} vs {dataset2}: p-value = {p_value:.4f}")


##################### To calculate the Gini impurity ############################

import pandas as pd
from sklearn.metrics import accuracy_score

# Load the datasets
df1 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\cor\Jun-03-2023labels_pose_30HzCorrelated c0 VMSxDMS.csv', header=0)
df2 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\cor\Jun-03-2023labels_pose_30HzCorrelated c0 VMSxDLS.csv', header=0)
df3 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\cor\Jun-03-2023labels_pose_30HzCorrelated c0 DMSxDLS.csv', header=0)
df4 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\after\Jun-03-2023labels_pose_30HzAfter Correlated c0 VMSxDMS.csv', header=0)
df5 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\after\Jun-03-2023labels_pose_30HzAfter Correlated c0 VMSxDLS.csv', header=0)
df6 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\after\Jun-03-2023labels_pose_30HzAfter Correlated c0 DMSxDLS.csv', header=0)

# Combine all datasets into a list
datasets = [df1, df2, df3, df4, df5, df6]

# Calculate Gini impurity for each dataset
gini_impurities = []
for dataset in datasets:
    syllable_labels = dataset['B-SOiD labels']
    class_counts = syllable_labels.value_counts()
    total_samples = len(syllable_labels)
    gini_impurity = 1 - sum((count / total_samples) ** 2 for count in class_counts)
    gini_impurities.append(gini_impurity)

# Print the Gini impurities
for i, gini_impurity in enumerate(gini_impurities):
    print(f"Gini impurity for dataset {i+1}: {gini_impurity}")







