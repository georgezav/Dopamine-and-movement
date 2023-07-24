# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 09:49:09 2023

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



def find_transition_indices(track_of_indeces):
    transition_indices = []
    prev_value = None

    for i, sublist in enumerate(track_of_indeces):
        current_value = sublist[0]

        if prev_value is not None and current_value != prev_value:
            transition_indices.append(i)

        prev_value = current_value

    return transition_indices

transition_indices = [0]
transition_indices.append(find_transition_indices(track_of_indeces)[0])
transition_indices.append(find_transition_indices(track_of_indeces)[1])
transition_indices.append(len(DATA_matrices))
print(transition_indices)


####################################################

### To find max values of each region ###

# Select the first 1030 lists from DATA_matrices
selected_lists = DATA_matrices[:1030]

# Create a new list to store the maximum values
max_values = []

# Iterate over each selected list
for sublist in selected_lists:
    # Get the sublist within the range of 100:180
    range_values = sublist[100:181]
    # Find the maximum value within the range
    max_value = max(range_values)
    # Append the max value to the new list
    max_values.append(max_value)

# Print the resulting list of maximum values
print(len(max_values))

print((track_of_indeces[82:131]))


rat_and_trial_VMS = []
for i in track_of_indeces[0:1030]:
    
    rat_and_trial_VMS.append([i[3],i[1]])
    


    
values_list = [16, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 58, 60]
### to save trials per rat, from left to right in the values_list parameter ###
    
    
trials_for_behavior_all = []
for q in range(0,len(values_list)):
    
    trials_for_behavior_loop = []
    for i in range(0,len(rat_and_trial_VMS)):
        
            
                
        if ((rat_and_trial_VMS)[i])[0] == values_list[q]:
            
                
                    
            trials_for_behavior_loop.append(((rat_and_trial_VMS)[i])[1])
                
    trials_for_behavior_all.append(trials_for_behavior_loop)   
    
    
print(len(trials_for_behavior_all))
print((rat_and_trial_VMS))










import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np


def process_data(DATA_matrices, transition_indices):
    # Divide DATA_matrices into subparameters based on transition_indices
    first_indices = transition_indices[0:1]
    second_indices = transition_indices[1:2]
    third_indices = transition_indices[2:3]
    fourth_indices = transition_indices[3:4]

    # Extract the first 100 values from each sublist in DATA_matrices
    first_group_1 = [sublist[75:95] for sublist in DATA_matrices[first_indices[0]:second_indices[0]]]
    first_group_2 = [sublist[75:95] for sublist in DATA_matrices[second_indices[0]:third_indices[0]]]
    first_group_3 = [sublist[75:95] for sublist in DATA_matrices[third_indices[0]:fourth_indices[0]]]

    # Calculate mean and standard deviation for each group
    first_group_1_mean = np.mean(first_group_1, axis=0)
    first_group_2_mean = np.mean(first_group_2, axis=0)
    first_group_3_mean = np.mean(first_group_3, axis=0)

    first_group_1_std = np.std(first_group_1, axis=0)
    first_group_2_std = np.std(first_group_2, axis=0)
    first_group_3_std = np.std(first_group_3, axis=0)

    # Calculate thresholds
    first_group_1_thresholds = (first_group_1_mean + 2 * first_group_1_std, first_group_1_mean - 2 * first_group_1_std)
    first_group_2_thresholds = (first_group_2_mean + 2 * first_group_2_std, first_group_2_mean - 2 * first_group_2_std)
    first_group_3_thresholds = (first_group_3_mean + 2 * first_group_3_std, first_group_3_mean - 2 * first_group_3_std)

    # Print threshold values
    print("Thresholds for 0:100 Range:")
    print("Group 1: +2 Std =", first_group_1_thresholds[0], ", -2 Std =", first_group_1_thresholds[1])
    print("Group 2: +2 Std =", first_group_2_thresholds[0], ", -2 Std =", first_group_2_thresholds[1])
    print("Group 3: +2 Std =", first_group_3_thresholds[0], ", -2 Std =", first_group_3_thresholds[1])

    # Calculate mean threshold for each group within the first 100 values
    first_group_1_mean_threshold = (np.mean(first_group_1_thresholds[0]), np.mean(first_group_1_thresholds[1]))
    first_group_2_mean_threshold = (np.mean(first_group_2_thresholds[0]), np.mean(first_group_2_thresholds[1]))
    first_group_3_mean_threshold = (np.mean(first_group_3_thresholds[0]), np.mean(first_group_3_thresholds[1]))

    # Print mean threshold values for each group
    print("\nMean of Thresholds for 0:100 Range:")
    print("Group 1: +2 Std Mean =", first_group_1_mean_threshold[0])
    print("Group 1: -2 Std Mean =", first_group_1_mean_threshold[1])
    print("Group 2: +2 Std Mean =", first_group_2_mean_threshold[0])
    print("Group 2: -2 Std Mean =", first_group_2_mean_threshold[1])
    print("Group 3: +2 Std Mean =", first_group_3_mean_threshold[0])
    print("Group 3: -2 Std Mean =", first_group_3_mean_threshold[1])

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.plot(first_group_1_mean, color='blue', label='VMS')
    plt.plot(first_group_2_mean, color='red', label='DMS')
    plt.plot(first_group_3_mean, color='green', label='DLS')

    plt.fill_between(range(len(first_group_1_mean)), first_group_1_thresholds[0], first_group_1_thresholds[1],
                     color='blue', alpha=0.3, label='2*std threshold VMS')
    plt.fill_between(range(len(first_group_2_mean)), first_group_2_thresholds[0], first_group_2_thresholds[1],
                     color='red', alpha=0.3, label='2*std threshold DMS')
    plt.fill_between(range(len(first_group_3_mean)), first_group_3_thresholds[0], first_group_3_thresholds[1],
                     color='green', alpha=0.3, label='2*std threshold DLS')

    plt.xlabel('Time(sec)',fontsize = 20)
    plt.ylabel('Δ[Dopamine](nM)',fontsize = 20)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend()
    plt.grid(True)

    plt.show()

process_data(DATA_matrices, transition_indices)



# Assuming DATA_matrices is your list of lists

# Convert DATA_matrices to a NumPy array for easier manipulation
data_array = np.array(DATA_matrices)

# Calculate the mean and standard deviation along the 0th axis (across all lists)
mean_values = np.mean(data_array, axis=0)
std_values = np.std(data_array, axis=0)

# Create an array of time points (assuming 350 points)
time_points = np.arange(350)

# Plot the mean values
plt.plot(time_points, mean_values, label='Mean')

# Plot the standard deviation as error bars
plt.errorbar(time_points, mean_values, yerr=std_values, fmt='.', label='Standard Deviation')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Mean and Standard Deviation')
plt.legend()

# Display the plot
plt.show()


# Convert DATA_matrices to a NumPy array for easier manipulation
data_array = np.array(DATA_matrices)

# Define the start and end indices for the desired range (point 50 to point 250)
start_index = 95
end_index = 250

# Extract the desired range of points from the data array
desired_range = data_array[:, start_index:end_index + 1]

# Calculate the mean and standard deviation across the desired range
mean_across_range = np.mean(desired_range, axis=1)
std_across_range = np.std(desired_range, axis=1)

# Calculate the overall mean and standard deviation across the desired range
mean_overall = np.mean(mean_across_range)
std_overall = np.mean(std_across_range)

# Print the overall mean and standard deviation
print("Overall Mean across the range:", mean_overall)
print("Overall Standard Deviation across the range:", std_overall)

import numpy as np
import matplotlib.pyplot as plt

def process_data(DATA_matrices, transition_indices):
    # Divide DATA_matrices into subparameters based on transition_indices
    first_indices = transition_indices[0:1]
    second_indices = transition_indices[1:2]
    third_indices = transition_indices[2:3]
    fourth_indices = transition_indices[3:4]

    # Extract the first 100 values from each sublist in DATA_matrices
    first_group_1 = [sublist[50:180] for sublist in DATA_matrices[first_indices[0]:second_indices[0]]]
    first_group_2 = [sublist[50:180] for sublist in DATA_matrices[second_indices[0]:third_indices[0]]]
    first_group_3 = [sublist[50:180] for sublist in DATA_matrices[third_indices[0]:fourth_indices[0]]]

    # Calculate mean and standard deviation for each group
    first_group_1_mean = np.mean(first_group_1, axis=0)
    first_group_2_mean = np.mean(first_group_2, axis=0)
    first_group_3_mean = np.mean(first_group_3, axis=0)

    first_group_1_std = np.std(first_group_1, axis=0)
    first_group_2_std = np.std(first_group_2, axis=0)
    first_group_3_std = np.std(first_group_3, axis=0)

    # Calculate thresholds
    first_group_1_thresholds = (first_group_1_mean + 2 * first_group_1_std, first_group_1_mean - 2 * first_group_1_std)
    first_group_2_thresholds = (first_group_2_mean + 2 * first_group_2_std, first_group_2_mean - 2 * first_group_2_std)
    first_group_3_thresholds = (first_group_3_mean + 2 * first_group_3_std, first_group_3_mean - 2 * first_group_3_std)

    # Print threshold values
    print("Thresholds for 0:100 Range:")
    print("Group 1: +2 Std =", first_group_1_thresholds[0], ", -2 Std =", first_group_1_thresholds[1])
    print("Group 2: +2 Std =", first_group_2_thresholds[0], ", -2 Std =", first_group_2_thresholds[1])
    print("Group 3: +2 Std =", first_group_3_thresholds[0], ", -2 Std =", first_group_3_thresholds[1])

    # Calculate mean threshold for each group within the first 100 values
    first_group_1_mean_threshold = (np.mean(first_group_1_thresholds[0]), np.mean(first_group_1_thresholds[1]))
    first_group_2_mean_threshold = (np.mean(first_group_2_thresholds[0]), np.mean(first_group_2_thresholds[1]))
    first_group_3_mean_threshold = (np.mean(first_group_3_thresholds[0]), np.mean(first_group_3_thresholds[1]))

    # Print mean threshold values for each group
    print("\nMean of Thresholds for 0:100 Range:")
    print("Group 1: +2 Std Mean =", first_group_1_mean_threshold[0])
    print("Group 1: -2 Std Mean =", first_group_1_mean_threshold[1])
    print("Group 2: +2 Std Mean =", first_group_2_mean_threshold[0])
    print("Group 2: -2 Std Mean =", first_group_2_mean_threshold[1])
    print("Group 3: +2 Std Mean =", first_group_3_mean_threshold[0])
    print("Group 3: -2 Std Mean =", first_group_3_mean_threshold[1])

    # Plotting
    plt.figure(figsize=(12, 6))
    x_values = np.arange(-5,8, .1)
    plt.plot(x_values, first_group_1_mean, color='blue', label='VMS')
    plt.plot(x_values, first_group_2_mean, color='green', label='DMS')
    plt.plot(x_values, first_group_3_mean, color='red', label='DLS')

    plt.fill_between(x_values, first_group_1_thresholds[0], first_group_1_thresholds[1],
                     color='blue', alpha=0.3, label='2*std threshold VMS')
    plt.fill_between(x_values, first_group_2_thresholds[0], first_group_2_thresholds[1],
                     color='green', alpha=0.3, label='2*std threshold DMS')
    plt.fill_between(x_values, first_group_3_thresholds[0], first_group_3_thresholds[1],
                     color='red', alpha=0.3, label='2*std threshold DLS')

    # Extract the first 350 values from each sublist in DATA_matrices
    second_group_1 = [sublist[:350] for sublist in DATA_matrices[first_indices[0]:second_indices[0]]]
    second_group_2 = [sublist[:350] for sublist in DATA_matrices[second_indices[0]:third_indices[0]]]
    second_group_3 = [sublist[:350] for sublist in DATA_matrices[third_indices[0]:fourth_indices[0]]]

    # Calculate mean and standard error of the mean for each group within the first 350 values
    second_group_1_mean = np.mean(second_group_1, axis=0)
    second_group_2_mean = np.mean(second_group_2, axis=0)
    second_group_3_mean = np.mean(second_group_3, axis=0)

    second_group_1_sem = np.std(second_group_1, axis=0) / np.sqrt(len(second_group_1))
    second_group_2_sem = np.std(second_group_2, axis=0) / np.sqrt(len(second_group_2))
    second_group_3_sem = np.std(second_group_3, axis=0) / np.sqrt(len(second_group_3))
    """
    x_values = np.arange(-9.5,25.5,0.1)
    # Plotting the second group with standard error of the mean and thresholds
    plt.plot(x_values,second_group_1_mean[:350], color='blue', linestyle='--', label='VMS (0:350)')
    plt.plot(x_values,second_group_2_mean[:350], color='green', linestyle='--', label='DMS (0:350)')
    plt.plot(x_values,second_group_3_mean[:350], color='red', linestyle='--', label='DLS (0:350)')

    plt.fill_between(x_values, second_group_1_mean[:350] + 2 * second_group_1_sem,
                     second_group_1_mean[:350] - 2 * second_group_1_sem, color='blue', alpha=0.3,
                     label='2*SEM threshold VMS (0:350)')
    plt.fill_between(x_values, second_group_2_mean[:350] + 2 * second_group_2_sem,
                     second_group_2_mean[:350] - 2 * second_group_2_sem, color='green', alpha=0.3,
                     label='2*SEM threshold DMS (0:350)')
    plt.fill_between(x_values, second_group_3_mean[:350] + 2 * second_group_3_sem,
                     second_group_3_mean[:350] - 2 * second_group_3_sem, color='red', alpha=0.3,
                     label='2*SEM threshold DLS (0:350)')
    
    """
    # Add threshold lines
    plt.axhline(first_group_1_mean_threshold[0], color='blue', linestyle='--', label='Group 1 +2 Std Mean')
    plt.axhline(first_group_1_mean_threshold[1], color='blue', linestyle='--', label='Group 1 -2 Std Mean')
    plt.axhline(first_group_2_mean_threshold[0], color='green', linestyle='--', label='Group 2 +2 Std Mean')
    plt.axhline(first_group_2_mean_threshold[1], color='green', linestyle='--', label='Group 2 -2 Std Mean')
    plt.axhline(first_group_3_mean_threshold[0], color='red', linestyle='--', label='Group 3 +2 Std Mean')
    plt.axhline(first_group_3_mean_threshold[1], color='red', linestyle='--', label='Group 3 -2 Std Mean')

    plt.xlabel('Time(sec)', fontsize=26)
    plt.ylabel('Δ[Dopamine](nM)', fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.grid(True)

    plt.show()

# Assuming you have defined DATA_matrices and transition_indices
process_data(DATA_matrices, transition_indices)























def classify_time_series(DATA_matrices, transition_indices):
    # Divide DATA_matrices into subparameters based on transition_indices
    first_indices = transition_indices[0:1]
    second_indices = transition_indices[1:2]
    third_indices = transition_indices[2:3]
    fourth_indices = transition_indices[3:4]

    # Extract the values from each sublist in DATA_matrices for the range 0:100
    first_group_1 = [sublist[5:95] for sublist in DATA_matrices[first_indices[0]:second_indices[0]]]
    first_group_2 = [sublist[5:95] for sublist in DATA_matrices[second_indices[0]:third_indices[0]]]
    first_group_3 = [sublist[5:95] for sublist in DATA_matrices[third_indices[0]:fourth_indices[0]]]

    # Calculate mean and standard deviation for each group in the range 0:100
    first_group_1_mean = np.mean(first_group_1)
    first_group_2_mean = np.mean(first_group_2)
    first_group_3_mean = np.mean(first_group_3)

    first_group_1_std = np.std(first_group_1)
    first_group_2_std = np.std(first_group_2)
    first_group_3_std = np.std(first_group_3)

    # Calculate thresholds for the range 0:100
    first_group_1_thresholds = (12,-8)
    first_group_2_thresholds = (first_group_2_mean + 4 * first_group_2_std, first_group_2_mean - 4 * first_group_2_std)
    first_group_3_thresholds = (first_group_3_mean + 4 * first_group_3_std, first_group_3_mean - 4 * first_group_3_std)
    print(first_group_1_thresholds)
    # Extract the values from each sublist in DATA_matrices for the range 100:180
    second_group_1 = [sublist[100:180] for sublist in DATA_matrices[first_indices[0]:second_indices[0]]]
    second_group_2 = [sublist[100:180] for sublist in DATA_matrices[second_indices[0]:third_indices[0]]]
    second_group_3 = [sublist[100:180] for sublist in DATA_matrices[third_indices[0]:fourth_indices[0]]]

    # Classify time series for each group based on the range 100:180 using the thresholds from 0:100
    second_group_1_classification = []
    second_group_2_classification = []
    second_group_3_classification = []

    for sublist in second_group_1:
        if any(value > first_group_1_thresholds[0] for value in sublist):
            second_group_1_classification.append("+")
        elif any(value < first_group_1_thresholds[1] for value in sublist):
            second_group_1_classification.append("-")
        else:
            second_group_1_classification.append("0")

    for sublist in second_group_2:
        if any(value > first_group_2_thresholds[0] for value in sublist):
            second_group_2_classification.append("+")
        elif any(value < first_group_2_thresholds[1] for value in sublist):
            second_group_2_classification.append("-")
        else:
            second_group_2_classification.append("0")

    for sublist in second_group_3:
        if any(value > first_group_3_thresholds[0] for value in sublist):
            second_group_3_classification.append("+")
        elif any(value < first_group_3_thresholds[1] for value in sublist):
            second_group_3_classification.append("-")
        else:
            second_group_3_classification.append("0")

    # Find the indices of positive, 0, and negative trials in each group
    first_group_1_positive_indices = [i for i, label in enumerate(second_group_1_classification) if label == "+"]
    first_group_1_zero_indices = [i for i, label in enumerate(second_group_1_classification) if label == "0"]
    first_group_1_negative_indices = [i for i, label in enumerate(second_group_1_classification) if label == "-"]

    first_group_2_positive_indices = [i for i, label in enumerate(second_group_2_classification) if label == "+"]
    first_group_2_zero_indices = [i for i, label in enumerate(second_group_2_classification) if label == "0"]
    first_group_2_negative_indices = [i for i, label in enumerate(second_group_2_classification) if label == "-"]

    first_group_3_positive_indices = [i for i, label in enumerate(second_group_3_classification) if label == "+"]
    first_group_3_zero_indices = [i for i, label in enumerate(second_group_3_classification) if label == "0"]
    first_group_3_negative_indices = [i for i, label in enumerate(second_group_3_classification) if label == "-"]
    
    value_to_add_2 = transition_indices[1]
    value_to_add_3 = transition_indices[2]
   
    first_group_2_positive_indices = [x + value_to_add_2 for x in first_group_2_positive_indices]
    first_group_2_zero_indices = [x + value_to_add_2 for x in first_group_2_zero_indices]
    first_group_2_negative_indices = [x + value_to_add_2 for x in first_group_2_negative_indices]

    first_group_3_positive_indices = [x + value_to_add_3 for x in first_group_3_positive_indices]
    first_group_3_zero_indices = [x + value_to_add_3 for x in first_group_3_zero_indices]
    first_group_3_negative_indices = [x + value_to_add_3 for x in first_group_3_negative_indices]
    
           
            
    
    # Step 1
    group_2_positive_values = []
    group_2_zero_values = []
    group_2_negative_values = []
    group_3_positive_values = []
    group_3_zero_values = []
    group_3_negative_values = []
        # Step 1
    group_1_positive_values = []
    group_1_zero_values = []
    group_1_negative_values = []
    
    # Step 2
    for index in first_group_1_positive_indices:
        group_1_positive_values.append(DATA_matrices[index])
    
    # Step 3
    for index in first_group_1_zero_indices:
        group_1_zero_values.append(DATA_matrices[index])
    
    # Step 4
    for index in first_group_1_negative_indices:
        group_1_negative_values.append(DATA_matrices[index])
    
    # Step 5
    for index in first_group_2_positive_indices:
        group_2_positive_values.append(DATA_matrices[index])
    
    # Step 6
    for index in first_group_2_zero_indices:
        group_2_zero_values.append(DATA_matrices[index])
    
    # Step 7
    for index in first_group_2_negative_indices:
        group_2_negative_values.append(DATA_matrices[index])
    
    # Step 8
    for index in first_group_3_positive_indices:
        group_3_positive_values.append(DATA_matrices[index])
    
    # Step 9
    for index in first_group_3_zero_indices:
        group_3_zero_values.append(DATA_matrices[index])
    
    # Step 10
    for index in first_group_3_negative_indices:
        group_3_negative_values.append(DATA_matrices[index])
    
    # Calculate mean along axis 0
    group_1_positive_mean = np.mean(group_1_positive_values, axis=0)
    group_1_zero_mean = np.mean(group_1_zero_values, axis=0)
    group_1_negative_mean = np.mean(group_1_negative_values, axis=0)
    group_2_positive_mean = np.mean(group_2_positive_values, axis=0)
    group_2_zero_mean = np.mean(group_2_zero_values, axis=0)
    group_2_negative_mean = np.mean(group_2_negative_values, axis=0)
    group_3_positive_mean = np.mean(group_3_positive_values, axis=0)
    group_3_zero_mean = np.mean(group_3_zero_values, axis=0)
    group_3_negative_mean = np.mean(group_3_negative_values, axis=0)
    
    # Calculate standard error of the mean along axis 0
    group_1_positive_sem = np.std(group_1_positive_values, axis=0) / np.sqrt(len(group_1_positive_values))
    group_1_zero_sem = np.std(group_1_zero_values, axis=0) / np.sqrt(len(group_1_zero_values))
    group_1_negative_sem = np.std(group_1_negative_values, axis=0) / np.sqrt(len(group_1_negative_values))
    group_2_positive_sem = np.std(group_2_positive_values, axis=0) / np.sqrt(len(group_2_positive_values))
    group_2_zero_sem = np.std(group_2_zero_values, axis=0) / np.sqrt(len(group_2_zero_values))
    group_2_negative_sem = np.std(group_2_negative_values, axis=0) / np.sqrt(len(group_2_negative_values))
    group_3_positive_sem = np.std(group_3_positive_values, axis=0) / np.sqrt(len(group_3_positive_values))
    group_3_zero_sem = np.std(group_3_zero_values, axis=0) / np.sqrt(len(group_3_zero_values))
    group_3_negative_sem = np.std(group_3_negative_values, axis=0) / np.sqrt(len(group_3_negative_values))



        
    # Create the figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(6, 10))
    
    # Plot Group 1
    axs[0].plot(group_1_positive_mean, color='red', label='Positive')
    axs[0].fill_between(range(len(group_1_positive_mean)),
                        group_1_positive_mean - group_1_positive_sem,
                        group_1_positive_mean + group_1_positive_sem,
                        color='red', alpha=0.2)
    axs[0].plot(group_1_zero_mean, color='grey', label='Zero')
    axs[0].fill_between(range(len(group_1_zero_mean)),
                        group_1_zero_mean - group_1_zero_sem,
                        group_1_zero_mean + group_1_zero_sem,
                        color='grey', alpha=0.2)
    axs[0].plot(group_1_negative_mean, color='blue', label='Negative')
    axs[0].fill_between(range(len(group_1_negative_mean)),
                        group_1_negative_mean - group_1_negative_sem,
                        group_1_negative_mean + group_1_negative_sem,
                        color='blue', alpha=0.2)
    axs[0].set_title('Group 1')
    axs[0].legend()
    
    # Plot Group 2
    axs[1].plot(group_2_positive_mean, color='red', label='Positive')
    axs[1].fill_between(range(len(group_2_positive_mean)),
                        group_2_positive_mean - group_2_positive_sem,
                        group_2_positive_mean + group_2_positive_sem,
                        color='red', alpha=0.2)
    axs[1].plot(group_2_zero_mean, color='grey', label='Zero')
    axs[1].fill_between(range(len(group_2_zero_mean)),
                        group_2_zero_mean - group_2_zero_sem,
                        group_2_zero_mean + group_2_zero_sem,
                        color='grey', alpha=0.2)
    axs[1].plot(group_2_negative_mean, color='blue', label='Negative')
    axs[1].fill_between(range(len(group_2_negative_mean)),
                        group_2_negative_mean - group_2_negative_sem,
                        group_2_negative_mean + group_2_negative_sem,
                        color='blue', alpha=0.2)
    axs[1].set_title('Group 2')
    axs[1].legend()
    
    # Plot Group 3
    axs[2].plot(group_3_positive_mean, color='red', label='Positive')
    axs[2].fill_between(range(len(group_3_positive_mean)),
                        group_3_positive_mean - group_3_positive_sem,
                        group_3_positive_mean + group_3_positive_sem,
                        color='red', alpha=0.2)
    axs[2].plot(group_3_zero_mean, color='grey', label='Zero')
    axs[2].fill_between(range(len(group_3_zero_mean)),
                        group_3_zero_mean - group_3_zero_sem,
                        group_3_zero_mean + group_3_zero_sem,
                        color='grey', alpha=0.2)
    axs[2].plot(group_3_negative_mean, color='blue', label='Negative')
    axs[2].fill_between(range(len(group_3_negative_mean)),
                        group_3_negative_mean - group_3_negative_sem,
                        group_3_negative_mean + group_3_negative_sem,
                        color='blue', alpha=0.2)
    axs[2].set_title('Group 3')
    axs[2].legend()
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    
    list_of_pospos_indices = first_group_1_positive_indices + first_group_2_positive_indices + first_group_3_positive_indices
    list_of_negneg_indices = first_group_1_negative_indices + first_group_2_negative_indices + first_group_3_negative_indices
    list_of_pos_indices_vms = first_group_1_positive_indices
    list_of_pos_indices_dms = first_group_2_positive_indices
    list_of_pos_indices_dls = first_group_3_positive_indices
    
    
    return list_of_pospos_indices , list_of_negneg_indices, list_of_pos_indices_vms, list_of_pos_indices_dms, list_of_pos_indices_dls



list_number = 0

print(len((classify_time_series(DATA_matrices, transition_indices))[2]))
if list_number == 0 or list_number == 1:
    l  =  classify_time_series(DATA_matrices, transition_indices)[list_number]

    rat_and_trial = []
    for i in range(0,len(l)):
        
        for k in range(i+1,len(l)):
            
            if (track_of_indeces[l[i]])[1] == (track_of_indeces[l[k]])[1] and (track_of_indeces[l[i]])[3] == (track_of_indeces[l[k]])[3]:
                
                rat_and_trial.append([(track_of_indeces[l[i]])[3],(track_of_indeces[l[i]])[1]])
    
    print(len(rat_and_trial))
    print(len((l)))
    
    
    
    values_list = [16, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 58, 60]
    ### to save trials per rat, from left to right in the values_list parameter ###
    
    
    trials_for_behavior_all = []
    for q in range(0,len(values_list)):
        trials_for_behavior_loop = []
        for i in range(0,len(rat_and_trial)):
            
                
            if ((rat_and_trial)[i])[0] == values_list[q]:
                
                    
                trials_for_behavior_loop.append(((rat_and_trial)[i])[1])
                    
        trials_for_behavior_all.append(trials_for_behavior_loop)   
    
    
    print(trials_for_behavior_all[5])
    print(len(trials_for_behavior_all[0]))
else:
    l  =  classify_time_series(DATA_matrices, transition_indices)[list_number]

    rat_and_trial = []
    for i in range(0,len(l)):
                
        rat_and_trial.append([(track_of_indeces[l[i]])[3],(track_of_indeces[l[i]])[1]])
    
    print(len(rat_and_trial))
    print(len((l)))
    
    
    
    values_list = [16, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 60]
    ### to save trials per rat, from left to right in the values_list parameter ###
    print(len(values_list))
    
    trials_for_behavior_all = []
    for q in range(0,len(values_list)):
        trials_for_behavior_loop = []
        for i in range(0,len(rat_and_trial)):
            
                
            if ((rat_and_trial)[i])[0] == values_list[q]:
                
                    
                trials_for_behavior_loop.append(((rat_and_trial)[i])[1])
                    
        trials_for_behavior_all.append(trials_for_behavior_loop)   
    

    
        
    



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
import pandas as pd
import glob
import numpy as np


folder_path = r"C:/Users/georg/Neuroscience/NiN internship/Behavioral analysis/Pavlovian"
values_list = [16, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 60]

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


for y in range(0,len(trials_for_behavior_all)-1):
    correlated_trials_index_for_csv_loop = []
        
    for p in (trials_for_behavior_all)[y]:
        print(p)
        if p >= len(trial_start_times[y]):
            continue
        else:
             
            correlated_trials_index_for_csv_loop.append((trial_start_times[y])[p])
                  
    correlated_trials_index_for_csv.append(correlated_trials_index_for_csv_loop)

   
non_correlated_trials_index_for_csv = []
import random

for y in range(len(trials_for_behavior_all)-1):
    non_correlated_trials_index_for_csv_loop = []

    available_indices = [i for i in range(len(trial_start_times[y])) if i not in trials_for_behavior_all[y]]
    
    for p in trials_for_behavior_all[y]:
        if p >= len(trial_start_times[y]):
            continue
        else:
            if not available_indices:
                break
            random_trial_index = random.choice(available_indices)
            available_indices.remove(random_trial_index)
            non_correlated_trials_index_for_csv_loop.append(trial_start_times[y][random_trial_index])

    non_correlated_trials_index_for_csv.append(non_correlated_trials_index_for_csv_loop)



# Convert the lists to sets
correlated_set = set(correlated_trials_index_for_csv[0])
non_correlated_set = set(non_correlated_trials_index_for_csv[0])

# Find the common values
common_values = correlated_set.intersection(non_correlated_set)

# Check if there are any common values
if common_values:
    print("The lists have common values:", common_values)
else:
    print("The lists do not have any common values.")





after_correlated_trials_index_for_csv = []

for y in range(len(trials_for_behavior_all)-1):
    after_correlated_trials_index_for_csv_loop = []

    for p in trials_for_behavior_all[y]:

        if p >= len(trial_start_times[y]):
            continue
        else:
            if (p+1) >= 49:
                continue
            after_correlated_trials_index_for_csv_loop.append(trial_start_times[y][p+1])

    after_correlated_trials_index_for_csv.append(after_correlated_trials_index_for_csv_loop)

after2_correlated_trials_index_for_csv = []

for y in range(len(trials_for_behavior_all)-1):
    after2_correlated_trials_index_for_csv_loop = []

    for p in trials_for_behavior_all[y]:

        if p >= len(trial_start_times[y]):
            continue
        else:
            if (p+1) >= 48:
                continue
            after2_correlated_trials_index_for_csv_loop.append(trial_start_times[y][p+2])

    after2_correlated_trials_index_for_csv.append(after2_correlated_trials_index_for_csv_loop)


empty_indices = []

for index, sublist in enumerate(correlated_trials_index_for_csv):
    if any(value < 0 for value in sublist):
        sublist.clear()
        empty_indices.append(index)

print("Indices of emptied lists:", empty_indices)


for sublist in after_correlated_trials_index_for_csv:

    if any(value < 0 for value in sublist):
        sublist.clear()
           

for sublist in non_correlated_trials_index_for_csv:

    if any(value < 0 for value in sublist):
        sublist.clear()
            
for sublist in after2_correlated_trials_index_for_csv:

    if any(value < 0 for value in sublist):
        sublist.clear()





def produce_data(cluster_num,selection, comparison,selection2, comparison2, time_before,time_after):


    
    import numpy as np
    
    
    
    middle_back_data_of_correlated_trials = []
    
    
        
    for index in range(0,len(((correlated_trials_index_for_csv)))):
        middle_back_data_of_correlated_trials_loop = []
             
        if len(((correlated_trials_index_for_csv))[index]) == 0:
            middle_back_data_of_correlated_trials.append([])
                
        else:
                    
            for trial_time in (((correlated_trials_index_for_csv))[index]):
                        
                time_window_data = ((csv_array[index])[int(trial_time - time_before):int(trial_time + time_after)])
                        
                x_selection = [sublist[0] for sublist in time_window_data]
                y_selection = [sublist[1] for sublist in time_window_data]
                                       
                middle_back_data_of_correlated_trials_loop.append([x_selection,y_selection])
            middle_back_data_of_correlated_trials.append(middle_back_data_of_correlated_trials_loop)
            
    
    
    
    after_middle_back_data_of_correlated_trials = []
    
    for index in range(len(after_correlated_trials_index_for_csv)):
        after_middle_back_data_of_correlated_trials_loop = []
    
        if len(after_correlated_trials_index_for_csv[index]) == 0:
            after_middle_back_data_of_correlated_trials.append([])
        else:
            for trial_time in after_correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[0] for sublist in time_window_data]
                y_selection = [sublist[1] for sublist in time_window_data]
                after_middle_back_data_of_correlated_trials_loop.append([x_selection, y_selection])
            after_middle_back_data_of_correlated_trials.append(after_middle_back_data_of_correlated_trials_loop)
    
    after2_middle_back_data_of_correlated_trials = []
    
    for index in range(len(after2_correlated_trials_index_for_csv)):
        after2_middle_back_data_of_correlated_trials_loop = []
    
        if len(after2_correlated_trials_index_for_csv[index]) == 0:
            after2_middle_back_data_of_correlated_trials.append([])
        else:
            for trial_time in after2_correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[0] for sublist in time_window_data]
                y_selection = [sublist[1] for sublist in time_window_data]
                after2_middle_back_data_of_correlated_trials_loop.append([x_selection, y_selection])
            after2_middle_back_data_of_correlated_trials.append(after2_middle_back_data_of_correlated_trials_loop)
    
    
    non_middle_back_data_of_correlated_trials = []
    
    for index in range(len(non_correlated_trials_index_for_csv)):
        non_middle_back_data_of_correlated_trials_loop = []
    
        if len(non_correlated_trials_index_for_csv[index]) == 0:
            non_middle_back_data_of_correlated_trials.append([])
        else:
            for trial_time in non_correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[0] for sublist in time_window_data]
                y_selection = [sublist[1] for sublist in time_window_data]
                non_middle_back_data_of_correlated_trials_loop.append([x_selection, y_selection])
            non_middle_back_data_of_correlated_trials.append(non_middle_back_data_of_correlated_trials_loop)

    ############# Tail base #################
    
    tail_data_of_correlated_trials = []
    
    for index in range(len(correlated_trials_index_for_csv)):
        tail_data_of_correlated_trials_loop = []
    
        if len(correlated_trials_index_for_csv[index]) == 0:
            tail_data_of_correlated_trials.append([])
        else:
            for trial_time in correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[2] for sublist in time_window_data]
                y_selection = [sublist[3] for sublist in time_window_data]
                tail_data_of_correlated_trials_loop.append([x_selection, y_selection])
            tail_data_of_correlated_trials.append(tail_data_of_correlated_trials_loop)
    
    
    after_tail_data_of_correlated_trials = []
    
    for index in range(len(after_correlated_trials_index_for_csv)):
        after_tail_data_of_correlated_trials_loop = []
    
        if len(after_correlated_trials_index_for_csv[index]) == 0:
            after_tail_data_of_correlated_trials.append([])
        else:
            for trial_time in after_correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[2] for sublist in time_window_data]
                y_selection = [sublist[3] for sublist in time_window_data]
                after_tail_data_of_correlated_trials_loop.append([x_selection, y_selection])
            after_tail_data_of_correlated_trials.append(after_tail_data_of_correlated_trials_loop)
    
    
    after2_tail_data_of_correlated_trials = []
    
    for index in range(len(after2_correlated_trials_index_for_csv)):
        after2_tail_data_of_correlated_trials_loop = []
    
        if len(after2_correlated_trials_index_for_csv[index]) == 0:
            after2_tail_data_of_correlated_trials.append([])
        else:
            for trial_time in after2_correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[2] for sublist in time_window_data]
                y_selection = [sublist[3] for sublist in time_window_data]
                after2_tail_data_of_correlated_trials_loop.append([x_selection, y_selection])
            after2_tail_data_of_correlated_trials.append(after2_tail_data_of_correlated_trials_loop)
    
    
    non_tail_data_of_correlated_trials = []
    
    for index in range(len(non_correlated_trials_index_for_csv)):
        non_tail_data_of_correlated_trials_loop = []
    
        if len(non_correlated_trials_index_for_csv[index]) == 0:
            non_tail_data_of_correlated_trials.append([])
        else:
            for trial_time in non_correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[2] for sublist in time_window_data]
                y_selection = [sublist[3] for sublist in time_window_data]
                non_tail_data_of_correlated_trials_loop.append([x_selection, y_selection])
            non_tail_data_of_correlated_trials.append(non_tail_data_of_correlated_trials_loop)

            
    ########## Snout ###############

    
    snoot_data_of_correlated_trials = []
    
    for index in range(0, len(correlated_trials_index_for_csv)):
        snoot_data_of_correlated_trials_loop = []
    
        if len(correlated_trials_index_for_csv[index]) == 0:
            snoot_data_of_correlated_trials.append([])
        else:
            for trial_time in correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[4] for sublist in time_window_data]
                y_selection = [sublist[5] for sublist in time_window_data]
                snoot_data_of_correlated_trials_loop.append([x_selection, y_selection])
            snoot_data_of_correlated_trials.append(snoot_data_of_correlated_trials_loop)
    
    
    after_snout_data_of_correlated_trials = []
    
    for index in range(0, len(after_correlated_trials_index_for_csv)):
        after_snout_data_of_correlated_trials_loop = []
    
        if len(after_correlated_trials_index_for_csv[index]) == 0:
            after_snout_data_of_correlated_trials.append([])
        else:
            for trial_time in after_correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[4] for sublist in time_window_data]
                y_selection = [sublist[5] for sublist in time_window_data]
                after_snout_data_of_correlated_trials_loop.append([x_selection, y_selection])
            after_snout_data_of_correlated_trials.append(after_snout_data_of_correlated_trials_loop)
    
    
    after2_snout_data_of_correlated_trials = []
    
    for index in range(0, len(after2_correlated_trials_index_for_csv)):
        after2_snout_data_of_correlated_trials_loop = []
    
        if len(after2_correlated_trials_index_for_csv[index]) == 0:
            after2_snout_data_of_correlated_trials.append([])
        else:
            for trial_time in after2_correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[4] for sublist in time_window_data]
                y_selection = [sublist[5] for sublist in time_window_data]
                after2_snout_data_of_correlated_trials_loop.append([x_selection, y_selection])
            after2_snout_data_of_correlated_trials.append(after2_snout_data_of_correlated_trials_loop)
    
    
    non_snout_data_of_correlated_trials = []
    
    for index in range(0, len(non_correlated_trials_index_for_csv)):
        non_snout_data_of_correlated_trials_loop = []
    
        if len(non_correlated_trials_index_for_csv[index]) == 0:
            non_snout_data_of_correlated_trials.append([])
        else:
            for trial_time in non_correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[4] for sublist in time_window_data]
                y_selection = [sublist[5] for sublist in time_window_data]
                non_snout_data_of_correlated_trials_loop.append([x_selection, y_selection])
            non_snout_data_of_correlated_trials.append(non_snout_data_of_correlated_trials_loop)

           
    ######## Neck ########
    ######## Neck ########
    neck_data_of_correlated_trials = []
    
    for index in range(0, len(correlated_trials_index_for_csv)):
        neck_data_of_correlated_trials_loop = []
    
        if len(correlated_trials_index_for_csv[index]) == 0:
            neck_data_of_correlated_trials.append([])
        else:
            for trial_time in correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[4] for sublist in time_window_data]
                y_selection = [sublist[5] for sublist in time_window_data]
                neck_data_of_correlated_trials_loop.append([x_selection, y_selection])
            neck_data_of_correlated_trials.append(neck_data_of_correlated_trials_loop)
    
    
    after_neck_data_of_correlated_trials = []
    
    for index in range(0, len(after_correlated_trials_index_for_csv)):
        after_neck_data_of_correlated_trials_loop = []
    
        if len(after_correlated_trials_index_for_csv[index]) == 0:
            after_neck_data_of_correlated_trials.append([])
        else:
            for trial_time in after_correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[4] for sublist in time_window_data]
                y_selection = [sublist[5] for sublist in time_window_data]
                after_neck_data_of_correlated_trials_loop.append([x_selection, y_selection])
            after_neck_data_of_correlated_trials.append(after_neck_data_of_correlated_trials_loop)
    
    
    after2_neck_data_of_correlated_trials = []
    
    for index in range(0, len(after2_correlated_trials_index_for_csv)):
        after2_neck_data_of_correlated_trials_loop = []
    
        if len(after2_correlated_trials_index_for_csv[index]) == 0:
            after2_neck_data_of_correlated_trials.append([])
        else:
            for trial_time in after2_correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[4] for sublist in time_window_data]
                y_selection = [sublist[5] for sublist in time_window_data]
                after2_neck_data_of_correlated_trials_loop.append([x_selection, y_selection])
            after2_neck_data_of_correlated_trials.append(after2_neck_data_of_correlated_trials_loop)
    
    
    non_neck_data_of_correlated_trials = []
    
    for index in range(0, len(non_correlated_trials_index_for_csv)):
        non_neck_data_of_correlated_trials_loop = []
    
        if len(non_correlated_trials_index_for_csv[index]) == 0:
            non_neck_data_of_correlated_trials.append([])
        else:
            for trial_time in non_correlated_trials_index_for_csv[index]:
                time_window_data = csv_array[index][int(trial_time - time_before):int(trial_time + time_after)]
                x_selection = [sublist[6] for sublist in time_window_data]
                y_selection = [sublist[7] for sublist in time_window_data]
                non_neck_data_of_correlated_trials_loop.append([x_selection, y_selection])
            non_neck_data_of_correlated_trials.append(non_neck_data_of_correlated_trials_loop)

            
            
    return non_tail_data_of_correlated_trials[selection:comparison] + non_tail_data_of_correlated_trials[selection2:comparison2], tail_data_of_correlated_trials[selection:comparison] + tail_data_of_correlated_trials[selection2:comparison2], after_tail_data_of_correlated_trials[selection:comparison] + after_tail_data_of_correlated_trials[selection2:comparison2], non_middle_back_data_of_correlated_trials[selection:comparison] + non_middle_back_data_of_correlated_trials[selection2:comparison2], middle_back_data_of_correlated_trials[selection:comparison] + middle_back_data_of_correlated_trials[selection2:comparison2], after_middle_back_data_of_correlated_trials[selection:comparison] + after_middle_back_data_of_correlated_trials[selection2:comparison2], non_snout_data_of_correlated_trials[selection:comparison] + non_snout_data_of_correlated_trials[selection2:comparison2], snoot_data_of_correlated_trials[selection:comparison] + snoot_data_of_correlated_trials[selection2:comparison2], after_snout_data_of_correlated_trials[selection:comparison] + after_snout_data_of_correlated_trials[selection2:comparison2], non_neck_data_of_correlated_trials[selection:comparison] +  non_neck_data_of_correlated_trials[selection2:comparison2], neck_data_of_correlated_trials[selection:comparison] + neck_data_of_correlated_trials[selection2:comparison2], after_neck_data_of_correlated_trials[selection:comparison] + after_neck_data_of_correlated_trials[selection2:comparison2], after2_tail_data_of_correlated_trials[selection:comparison] + after2_tail_data_of_correlated_trials[selection2:comparison2], after2_middle_back_data_of_correlated_trials[selection:comparison] + after2_middle_back_data_of_correlated_trials[selection2:comparison2],after2_snout_data_of_correlated_trials[selection:comparison] + after2_snout_data_of_correlated_trials[selection2:comparison2], after2_neck_data_of_correlated_trials[selection:comparison] + after2_neck_data_of_correlated_trials[selection2:comparison2]




#def zip_data(cluster_num,selection, comparison, time_before,time_after):
    
# VMS x DMS: 17,21,26,30
# DMS x DLS: 22,25,25,26

values_list = [16, 18, 19, 20, 21, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 38, 39, 41, 42, 43, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 60]

    
sim = (produce_data(0,17,21,26,30,0,600))

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

"""
print((classify_time_series(DATA_matrices, transition_indices)[2])[69:307])
print(rat_and_trial[69:307])
print(track_of_indeces[343])
"""
import pandas as pd

# Define the file paths
root_path = r'C:\Users\georg\Neuroscience\NiN internship\Behavioral analysis\Pavlovian\CSV Files from python code\Files to overwrite\\'
correlated_file = 'non pos VMS (higher thr).csv'

# Read the 'Correlated' CSV file with the first 3 rows as header
correlated_data = pd.read_csv(root_path + correlated_file, header=[0, 1, 2])

# Get the number of rows in the file
num_rows = correlated_data.shape[0]

# Determine the length of the data from array_1
array_1_length = array_2.shape[0]

# Determine the number of rows to overwrite
num_rows_to_overwrite = min(num_rows, array_1_length)

# Overwrite the specified columns with the data from array_1
correlated_data.iloc[:num_rows_to_overwrite, 1] = array_2[:num_rows_to_overwrite, 0]  # Overwrite 2nd column
correlated_data.iloc[:num_rows_to_overwrite, 2] = array_2[:num_rows_to_overwrite, 1]  # Overwrite 3rd column
correlated_data.iloc[:num_rows_to_overwrite, 4] = array_2[:num_rows_to_overwrite, 2]  # Overwrite 5th column
correlated_data.iloc[:num_rows_to_overwrite, 5] = array_2[:num_rows_to_overwrite, 3]  # Overwrite 6th column
correlated_data.iloc[:num_rows_to_overwrite, -12] = array_2[:num_rows_to_overwrite, 4]  # Overwrite -12th column
correlated_data.iloc[:num_rows_to_overwrite, -11] = array_2[:num_rows_to_overwrite, 5]  # Overwrite -11th column
correlated_data.iloc[:num_rows_to_overwrite, 13] = array_2[:num_rows_to_overwrite, 6]  # Overwrite 13th column
correlated_data.iloc[:num_rows_to_overwrite, 14] = array_2[:num_rows_to_overwrite, 7]  # Overwrite 14th column



# Create a list of column indexes to keep
columns_to_keep = [0]  # Keep the first column
columns_to_keep += [1, 2, 3, 4, 5, 6, -12, -11, -10, 13, 14, 15]  # Keep the specified columns

# Drop all other columns
correlated_data = correlated_data.iloc[:, columns_to_keep]

# Truncate the DataFrame to the desired length
correlated_data = correlated_data.iloc[:num_rows_to_overwrite]


# Save the updated 'Correlated' CSV file
correlated_data.to_csv(root_path + correlated_file, index=False)


print("Number of rows to overwrite:", num_rows_to_overwrite)













###############################################################



    
    ########### FOR VALUE 5 ##############
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    

def plot_occurancies(value):


    # Load the datasets
    df1 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos VMS (higher thr).csv', header=0)
    df2 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\nonplusplus\Jun-03-2023labels_pose_30Hznon pos VMS (higher thr).csv', header=0)
    df3 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\cor\Jun-03-2023labels_pose_30HzCorrelated c2 VMSxDLS.csv', header=0)
    df4 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\non\Jun-03-2023labels_pose_30HzNon Correlated c2 VMSxDLS.csv', header=0)
    df5 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\after\Jun-03-2023labels_pose_30HzAfter Correlated c0 VMSxDLS.csv', header=0)
    df6 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\after\Jun-03-2023labels_pose_30HzAfter Correlated c0 DMSxDLS.csv', header=0)

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
    plt.errorbar(x_range_df1, averaged_values_df1, yerr=standard_error_df1, fmt='o', color='blue', label='VMSxDLS c0')
    plt.errorbar(x_range_df2, averaged_values_df2, yerr=standard_error_df2, fmt='o', color='orange', label='VMSxDLS c1')
    #plt.errorbar(x_range_df2, averaged_values_df3, yerr=standard_error_df3, fmt='o', color='green', label='VMSxDLS c2')
    #plt.errorbar(x_range_df2, averaged_values_df4, yerr=standard_error_df4, fmt='o', color='black', label='non VMSxDLS')
    #plt.errorbar(x_range_df2, averaged_values_df5, yerr=standard_error_df5, fmt='o', color='darkviolet', label='After VMSxDLS')
    #plt.errorbar(x_range_df2, averaged_values_df6, yerr=standard_error_df6, fmt='o', color='grey', label='After DMSxDLS')
    
    # Connect points with lines
    plt.plot(x_range_df1, averaged_values_df1, color='blue', linestyle='-', linewidth=1)
    plt.plot(x_range_df2, averaged_values_df2, color='orange', linestyle='-', linewidth=1)
    #plt.plot(x_range_df2, averaged_values_df3, color='green', linestyle='-', linewidth=1)
    #plt.plot(x_range_df2, averaged_values_df4, color='black', linestyle='-', linewidth=1)
    #plt.plot(x_range_df2, averaged_values_df5, color='darkviolet', linestyle='-', linewidth=1)
    #plt.plot(x_range_df2, averaged_values_df6, color='grey', linestyle='-', linewidth=1)
 
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
    """
    normalized_values_df1 = [value / averaged_values_df4[idx] for idx, value in enumerate(averaged_values_df1)]
    normalized_values_df2 = [value / averaged_values_df5[idx] for idx, value in enumerate(averaged_values_df2)]
    normalized_values_df3 = [value / averaged_values_df6[idx] for idx, value in enumerate(averaged_values_df3)]
    normalized_values_df4 = [value / averaged_values_df4[idx] for idx, value in enumerate(averaged_values_df4)]
    normalized_values_df5 = [value / averaged_values_df5[idx] for idx, value in enumerate(averaged_values_df5)]
    normalized_values_df6 = [value / averaged_values_df6[idx] for idx, value in enumerate(averaged_values_df6)]

    # Scatter points with error bars
    plt.errorbar(x_range_df2, normalized_values_df1, yerr=standard_error_df1, fmt='o', color='blue', label='+ VMS')
    plt.errorbar(x_range_df2, normalized_values_df2, yerr=standard_error_df2, fmt='o', color='green', label='+ DMS')
    plt.errorbar(x_range_df2, normalized_values_df3, yerr=standard_error_df3, fmt='o', color='red', label='+ DLS')
    #plt.errorbar(x_range_df2, normalized_values_df4, yerr=standard_error_df4, fmt='o', color='violet', label='Non ++ VMSxDLS')
    #plt.errorbar(x_range_df2, normalized_values_df5, yerr=standard_error_df5, fmt='o', color='grey', label='Non ++ VMSxDMS')
    #plt.errorbar(x_range_df2, normalized_values_df6, yerr=standard_error_df6, fmt='o', color='black', label='Non ++ DMSxDLS')
    
    # Connect points with lines
    plt.plot(x_range_df2, normalized_values_df1, color='blue', linestyle='-', linewidth=1)
    plt.plot(x_range_df2, normalized_values_df2, color='green', linestyle='-', linewidth=1)
    plt.plot(x_range_df2, normalized_values_df3, color='red', linestyle='-', linewidth=1)
    #plt.plot(x_range_df2, normalized_values_df4, color='violet', linestyle='-', linewidth=1)
    #plt.plot(x_range_df2, normalized_values_df5, color='grey', linestyle='-', linewidth=1)
    #plt.plot(x_range_df2, normalized_values_df6, color='black', linestyle='-', linewidth=1)
    
    plt.axvline(x=0, color='black', linestyle='--', label='traylight')
    plt.axvline(x=5, color='black', linestyle='--', label='reward')
    
    legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time(sec)')
    plt.ylabel('Normalized Average Occurrence')
    plt.title('Occurrences of Syllable {}'.format(value))
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    
    plt.show()
    """
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

df1 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos VMS.csv', header=0)
df2 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos DLS.csv', header=0)
df3 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos DMS.csv', header=0)
df4 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzplusplus VMSxDLS.csv', header=0)
df5 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\nonplusplus\Jun-03-2023labels_pose_30HzNon plusplus VMSxDLS.csv', header=0)
df6 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\nonplusplus\Jun-03-2023labels_pose_30HzNon negneg VMSxDLS.csv', header=0)



import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

# Load the file and extract the column with syllable labels
df = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\nonplusplus\Jun-03-2023labels_pose_30HzNon plusplus VMSxDLS.csv', header=0)
syllable_labels = df['B-SOiD labels']

# Calculate the transition counts for each syllable
transition_counts = pd.DataFrame(columns=syllable_labels.unique(), index=syllable_labels.unique(), data=0)
for i in range(len(syllable_labels) - 1):
    current_syllable = syllable_labels.iloc[i]
    next_syllable = syllable_labels.iloc[i + 1]
    transition_counts.loc[current_syllable, next_syllable] += 1

# Calculate the transition probabilities
transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)

# Calculate the entropy for each syllable
entropy_per_syllable = -np.sum(transition_probs * np.log2(transition_probs), axis=1)

print("Entropy per syllable:")
print(entropy_per_syllable)






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the file and extract the column with syllable labels
df = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos VMS.csv', header=0)
syllable_labels = df['B-SOiD labels']

# Calculate the transition counts for each syllable
transition_counts = pd.DataFrame(columns=syllable_labels.unique(), index=syllable_labels.unique(), data=0)
for i in range(len(syllable_labels) - 1):
    current_syllable = syllable_labels.iloc[i]
    next_syllable = syllable_labels.iloc[i + 1]
    transition_counts.loc[current_syllable, next_syllable] += 1

# Calculate the transition probabilities
transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)

# Calculate the entropy for each syllable
entropy_per_syllable = -np.sum(transition_probs * np.log2(transition_probs), axis=1)

# Average entropy per syllable
average_entropy = np.mean(entropy_per_syllable)

# Standard error of the mean
sem = np.std(entropy_per_syllable) / np.sqrt(len(entropy_per_syllable))

# Plotting the bar graph with scatter points
plt.bar(0, average_entropy, alpha=0.5)
plt.errorbar(0, average_entropy, yerr=sem, fmt='none', color='gray', capsize=5, label='Standard Error of Mean')

plt.scatter(np.zeros_like(entropy_per_syllable), entropy_per_syllable, color='blue', marker='o')

plt.xlim(-1, 1)  # Adjust the x-axis limits to better fit the bar and scatter plot

plt.ylabel('Entropy')
plt.xlabel('Metric')
plt.title('Entropy of Syllables')
plt.legend()
plt.show()





import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_rel

# Load the datasets
# Load the datasets
df1 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos VMS.csv', header=0)
df2 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos DMS (2).csv', header=0)
df3 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos DLS.csv', header=0)
df4 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\nonplusplus\Jun-03-2023labels_pose_30Hznon pos VMS.csv', header=0)
df5 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\nonplusplus\Jun-03-2023labels_pose_30Hznon pos DMS (2).csv', header=0)
df6 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\nonplusplus\Jun-03-2023labels_pose_30Hznon pos DLS.csv', header=0)
# Calculate entropy for each dataset
datasets = [df1, df2, df3, df4, df5, df6]
dataset_names = ['+ VMS', '+ DMS', '+ DLS', 'Non + VMS','Non + DMS', 'Non + DLS']
colors = ['blue', 'green', 'red', 'darkblue','grey','black']
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
















################### To calculate Gini impurity ##################
# Load the datasets
df1 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos VMS.csv', header=0)
df2 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos DLS.csv', header=0)
df3 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos DMS.csv', header=0)
df4 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzplusplus VMSxDLS.csv', header=0)
df5 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\nonplusplus\Jun-03-2023labels_pose_30HzNon plusplus VMSxDLS.csv', header=0)
df6 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\nonplusplus\Jun-03-2023labels_pose_30HzNon negneg VMSxDLS.csv', header=0)

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




# Load the datasets
df1 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos VMS.csv', header=0)
df2 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos DLS.csv', header=0)
df3 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzpos DMS.csv', header=0)
df4 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30Hzplusplus VMSxDLS.csv', header=0)
df5 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\nonplusplus\Jun-03-2023labels_pose_30HzNon plusplus VMSxDLS.csv', header=0)
df6 = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\nonplusplus\Jun-03-2023labels_pose_30HzNon negneg VMSxDLS.csv', header=0)

# Combine all datasets into a list
datasets = [df1, df2, df3, df4, df5, df6]

# Calculate Gini impurity for each unique value in 'B-SOiD labels' column
gini_impurities = {}
for i, dataset in enumerate(datasets):
    unique_values = dataset['B-SOiD labels'].unique()
    gini_impurities[i+1] = {}
    for value in unique_values:
        subset = dataset[dataset['B-SOiD labels'] == value]
        class_counts = subset.shape[0]
        total_samples = dataset.shape[0]
        gini_impurity = 1 - (class_counts / total_samples) ** 2
        gini_impurities[i+1][value] = gini_impurity

# Print the Gini impurity for each unique value in each dataset
for dataset_num, impurity_dict in gini_impurities.items():
    print(f"Gini impurity for dataset {dataset_num}:")
    for value, impurity in impurity_dict.items():
        print(f"  - Value {value}: {impurity}")





































import pandas as pd
import numpy as np

# Load the dataset
df_new = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30HzVMS check for length.csv', header=0)

# Get the syllable labels
syllable_labels = df_new['B-SOiD labels']

# Divide the dataset into bins of 200 values
bin_size = 600
num_bins = len(df_new) // bin_size
print(num_bins)
mean_entropies = []
for i in range(num_bins):
    start_index = i * bin_size
    end_index = (i + 1) * bin_size
    bin_data = df_new[start_index:end_index]
    
    # Calculate the transition counts for each syllable in the current bin
    transition_counts = pd.DataFrame(columns=syllable_labels.unique(), index=syllable_labels.unique(), data=0)
    for j in range(start_index, end_index - 1):
        current_syllable = syllable_labels.iloc[j]
        next_syllable = syllable_labels.iloc[j + 1]
        transition_counts.loc[current_syllable, next_syllable] += 1
    
    # Calculate the transition probabilities in the current bin
    transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    
    # Calculate the entropy for each syllable in the current bin
    entropy_per_syllable = -np.sum(transition_probs * np.log2(transition_probs), axis=1)
    
    # Calculate the mean entropy of all syllables in the current bin
    mean_entropy = np.mean(entropy_per_syllable)
    
    # Save the mean entropy in the new variable
    mean_entropies.append(mean_entropy)

# Print the mean entropies in bins of 200 values
print("Mean Entropies (in bins of 200 values):")
print(mean_entropies)



print(len(mean_entropies))












import pandas as pd
import numpy as np
from scipy.stats import entropy

# Load the dataset
df_new = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30HzVMS check for length.csv', header=0)

# Get the syllable labels
syllable_labels = df_new['B-SOiD labels']

# Calculate entropy in bins of 600 values
num_bins = len(syllable_labels) // 600
entropy_values = []

for i in range(num_bins):
    start_idx = i * 600
    end_idx = start_idx + 600
    bin_labels = syllable_labels[start_idx:end_idx]
    unique_labels, counts = np.unique(bin_labels, return_counts=True)
    probabilities = counts / len(bin_labels)
    entropy_val = entropy(probabilities, base=2)
    entropy_values.append(entropy_val)

# Print the entropy values and length
print(entropy_values)
print("Length of entropy values:", len(entropy_values))
print(np.mean(entropy_values))














print(track_of_indeces[82:343])




index_for_dopamine = ((classify_time_series(DATA_matrices, transition_indices)[2])[69:307])
print((index_for_dopamine))
# Select the first 1030 lists from DATA_matrices
selected_lists = DATA_matrices
# Create a new list to store the maximum values
max_values = []

# Iterate over each selected list
for sublist in selected_lists:
    # Get the sublist within the range of 100:180
    range_values = sublist[100:301]
    # Find the maximum value within the range
    max_value = max(range_values)
    # Append the max value to the new list
    max_values.append(max_value)
print(len(max_values))


indexed_max_values = []
count = 0
for i in (sim[1]):
    
    for k in i:
        if len(k[0]) == 68:
            break
        
        if len(k[0]) < 599:
            count +=1
            continue
        
        else:
           
            indexed_max_values.append(max_values[(index_for_dopamine[count])])
            count += 1
        
    
print((indexed_max_values))
print(len(indexed_max_values))
























































print(len(mean_entropies))


print(rat_and_trial_VMS[0:230])


# Create a new list without sublists containing rat ID equal to 19 or 21
filtered_list = [sublist for sublist in rat_and_trial_VMS if sublist[0] not in [19, 21]]

# Print the filtered list
print(filtered_list[0:230])


plt.plot(filtered_list)



removed_indexes = []
for i, sublist in enumerate(rat_and_trial_VMS):
    rat_id = sublist[0]
    if rat_id == 19 or rat_id == 21:
        removed_indexes.append(i)

rat_and_trial_VMS_filtered = [sublist for i, sublist in enumerate(rat_and_trial_VMS) if i not in removed_indexes]
max_values_filtered = [value for i, value in enumerate(max_values) if i not in removed_indexes]

print("Filtered rat_and_trial_VMS:")
print(len(rat_and_trial_VMS_filtered))
print("Filtered max_values:")
print(len(max_values_filtered))























import pandas as pd
import numpy as np
from scipy.stats import entropy

# Load the dataset
df_new = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30HzVMS check for length.csv', header=0)

# Get the syllable labels
syllable_labels = df_new['B-SOiD labels']

# Calculate entropy of value 0 in bins of 600 values
num_bins = len(syllable_labels) // 600
entropy_values = []

for i in range(num_bins):
    start_idx = i * 600
    end_idx = start_idx + 600
    bin_labels = syllable_labels[start_idx:end_idx]
    count_zero = np.count_nonzero(bin_labels == 1)
    total_count = len(bin_labels)
    probability_zero = count_zero / total_count
    probability_one = 1 - probability_zero
    probabilities = [probability_zero, probability_one]
    entropy_val = entropy(probabilities, base=2)
    entropy_values.append(entropy_val)

# Print the entropy values and length
print(entropy_values)
print("Length of entropy values:", len(entropy_values))

print(np.mean(entropy_values))


################### To calculate the entropy in 10s period #####################
import pandas as pd
import numpy as np
from scipy.stats import entropy

# Load the dataset
df_new = pd.read_csv(r'C:\Users\georg\Neuroscience\bsoid\Final Train\check occurancies\plusplus analysis\plusplus\Jun-03-2023labels_pose_30HzVMS check for length.csv', header=0)

# Get the syllable labels
syllable_labels = df_new['B-SOiD labels']

# Calculate entropy of value 0 in bins of 600 values (considering only the first 300 values)
num_bins = len(syllable_labels) // 600
entropy_values = []

for i in range(num_bins):
    start_idx = i * 600
    end_idx = start_idx + 600
    bin_labels = syllable_labels[start_idx:end_idx]
    # Consider only the first 300 values in each bin
    bin_labels = bin_labels[:600]
    count_zero = np.count_nonzero(bin_labels == 6)
    total_count = len(bin_labels)
    probability_zero = count_zero / total_count
    probability_one = 1 - probability_zero
    probabilities = [probability_zero, probability_one]
    entropy_val = entropy(probabilities, base=2)
    entropy_values.append(entropy_val)

# Print the entropy values and length
print(entropy_values)
print("Length of entropy values:", len(entropy_values))

################# LEAVE IT FOR NOW ###############################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.utils import resample

# Assuming `max_values_filtered` and `mean_entropies` are NumPy arrays or lists
x = indexed_max_values
y = entropy_values

# Fit a line using polyfit
coefficients = np.polyfit(x, y, 1)
poly_line = np.poly1d(coefficients)

# Calculate correlation coefficient and p-value
correlation, p_value = pearsonr(x, y)

# Bootstrap resampling for confidence interval
bootstrap_iterations = 1000
bootstrap_coef = np.zeros((bootstrap_iterations, 2))
for i in range(bootstrap_iterations):
    resampled_data = resample(np.column_stack((x, y)))
    resampled_x, resampled_y = resampled_data[:, 0], resampled_data[:, 1]
    resampled_coefficients = np.polyfit(resampled_x, resampled_y, 1)
    bootstrap_coef[i] = resampled_coefficients

# Calculate confidence interval
confidence_interval = np.percentile(bootstrap_coef, [2.5, 97.5], axis=0)
lower_bound, upper_bound = confidence_interval[0], confidence_interval[1]

# Scatter plot with line of best fit and confidence interval shading
plt.scatter(x, y)
plt.plot(x, poly_line(x), color='red', label=f'Correlation: {correlation:.2f}, p-value: {p_value:.2f}')
plt.fill_between(x, np.polyval(upper_bound, x), np.polyval(lower_bound, x), color='red', alpha=0.2)
plt.xlabel('Dopamine Max Amplitude', fontsize=12)
plt.ylabel('Entropy', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Remove top and right ticks
plt.gca().tick_params(axis='x', top=False)
plt.gca().tick_params(axis='y', right=False)
plt.show()














print(track_of_indeces[82:319])
print((track_of_indeces[1]))


print((correlated_trials_index_for_csv[6]))


print(len(csv_array[6]))


adder = 0
adder2 = 0
for i in (sim[1]):

    for k in i:
        
        
    #print(len(i))

        adder += len(i)
        if len(k[0]) < 599:
            
            adder2 += 1
            
    print(adder2)








adder = 0
adder2 = 0 

for i in (sim[1]):

    for k in i:
        adder2 += 1
        
        if len(k[0]) == 68:
            break
        else:
            adder += 1
        
print(adder)


