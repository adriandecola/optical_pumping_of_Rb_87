# installing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression

##############################################################################
#                           Initial Model                                    #
##############################################################################

# First I will define a list that represents the different 
# F = 2 M_F={-2,-1,0,1,2} States
rb_atom_cloud = [0,0,0,0,0]
# These lists are the relative probabilities of states falling from F=3->F'=2
# M_F' state = [P(mf = mf'-1), P(mf=mf'), P(mf = mf'+1)]
prob_mf_prime3 = [1, 0, 0] # no other accessible states
prob_mf_prime2 = [ 2/3, 1/3, 0]
prob_mf_prime1 = [ 2/5, 8/15, 1/15]
prob_mf_prime0 = [ 1/5, 3/5, 1/5]
prob_mf_prime_neg_1 = [ 1/15, 8/15, 2/5]
prob_matrix = [prob_mf_prime_neg_1, prob_mf_prime0, prob_mf_prime1, 
               prob_mf_prime2, prob_mf_prime3]


def choose_atom_to_excite(rb_atom_cloud):
    """
    This fruitful function takes in the five element list representing a 
    Rubidium atom cloud and returns the index of the random atom chosen to 
    excite. It does this by randomly choosing a number from the number of atoms 
    in the cloud and then exciting the atom that many of atoms away from the 
    M_F = -2 substate.
    """
    num_of_atoms = sum(rb_atom_cloud)
    atom_to_excite = np.random.randint(low = 1, high = num_of_atoms)
    # now we must return the proper index refering to which MF substate 
    # we excite from
    atoms_counted_from_mf_neg_2 = rb_atom_cloud[0]
    i = 0 #index to excite from
    while atoms_counted_from_mf_neg_2 < atom_to_excite:
        atoms_counted_from_mf_neg_2 += rb_atom_cloud[i+1]
        i += 1
    return i

def excite_an_atom(rb_atom_cloud):
    """
    This fruitful funciton takes in the five element list representing a 
    Rubidium atom cloud and returns the same cloud with one of the atoms 
    having gone excitation and fallen back to the F=2 state. It does this 
    by choosing an atom at random. It then probabilistically chooses from the 
    allowed M_F substates the atom can fall to. As the atom is always excited 
    an MF substate from the positively polarized circular light and can only 
    fall down a state such that Delta MF = -1,0,1. The atom can either fall 
    back to the same MF state, one higher, or two higher: if these states are
    availible.
    """
    index_to_excite = choose_atom_to_excite(rb_atom_cloud)
    rb_atom_cloud[index_to_excite] += -1
    # a random number we will use alongside relative transition 
    #strengths to calculate which mf state the atom falls to
    probability_num = random.uniform(0,1) 
    
    ##deciding the partitions for probability atoms change in my state
    rel_prob_to_ground_state_ls = prob_matrix[index_to_excite]
    first_partition = rel_prob_to_ground_state_ls[0]
    second_partition = first_partition + rel_prob_to_ground_state_ls[1]

    # comparing the probability number to the partitions to see if the atom 
    # increase, decreases, or doesn't change mf state
    if probability_num <= first_partition:
        rb_atom_cloud[index_to_excite] += 1
    elif probability_num <= second_partition:
        rb_atom_cloud[index_to_excite + 1] += 1
    else:
        rb_atom_cloud[index_to_excite + 2] += 1

    return rb_atom_cloud

##############################################################################
#               Testing the Model on a Cloud of 100 Atoms                    #
##############################################################################


rb_atom_cloud = [20,20,20,20,20]
mf_states_to_plot = [rb_atom_cloud.copy()] #lists are mutable
mf_states_to_plot_lin = [rb_atom_cloud.copy()]
# creating a log spaced and linear list to plot the proportion across
p_absorbtions = np.logspace(start=1, stop = 3.5,num= 20)
p_absorbtions = p_absorbtions.astype(int)
p_absorbtions = np.insert(p_absorbtions, 0, 0)

p_absorbtions_lin = np.linspace(0, 1750, 100)
p_absorbtions_lin = p_absorbtions_lin.astype(int)

for i in range(p_absorbtions[-1]):
    excite_an_atom(rb_atom_cloud)
    if i+1 in p_absorbtions:
        mf_states_to_plot.append(rb_atom_cloud.copy())
    if i+1 in p_absorbtions_lin:
        mf_states_to_plot_lin.append(rb_atom_cloud.copy())


# gettting the proportion of each
mf2_states_to_plot = [mf_states_to_plot[i][4] for i in 
                      range(len(mf_states_to_plot))]
mf2_states_to_plot = np.array(mf2_states_to_plot)
mf2_states_proportion_to_plot = mf2_states_to_plot/100

mf2_states_to_plot_lin = [mf_states_to_plot_lin[i][4] for i in 
                          range(len(mf_states_to_plot_lin))]
mf2_states_to_plot_lin = np.array(mf2_states_to_plot_lin)
mf2_states_proportion_to_plot_lin = mf2_states_to_plot_lin/100

# plotting
plt.semilogx(p_absorbtions, mf2_states_proportion_to_plot)
plt.title("Proportion of the 100 Atoms in $M_F=2$ Substates")
plt.xlabel("Number of Absorbtions")
plt.ylabel("Proportions of Atoms in the $M_F=2$ Substate")
plt.show()

plt.plot(p_absorbtions_lin, mf2_states_proportion_to_plot_lin)
plt.title("Proportion of the 100 Atoms in $M_F=2$ Substates")
plt.xlabel("Number of Absorbtions")
plt.ylabel("Proportions of Atoms in the $M_F=2$ Substate")
plt.show()

##############################################################################
#             Gaining a More Accurate Picture Across Experiments             #
##############################################################################

# lets run the same experiment 100 times
mf2_states = [[20] for i in range(100)]
mf2_states_lin = [[20] for i in range(100)]
# creating a log spaced and linear list of the number of photon absorbtions for 
# which we will average the proportion of atoms in the M_F=2 substate
absorbtions = np.logspace(start=1, stop = 3.5,num= 20)
absorbtions = absorbtions.astype(int)
absorbtions = np.insert(absorbtions, 0, 0)

absorbtions_lin = np.linspace(0, 1750, 100)
absorbtions_lin = absorbtions_lin.astype(int)

# running the experiment 100 times
for i in range(100):
    rb_atom_cloud = [20,20,20,20,20]
    for j in range(absorbtions[-1]):
        excite_an_atom(rb_atom_cloud)
        if j+1 in absorbtions:
            mf2_states[i].append(rb_atom_cloud[4])
        if j+1 in absorbtions_lin:
            mf2_states_lin[i].append(rb_atom_cloud[4])

### gettting the average proportion across ###
mf2_average_prop = [0 for i in range(len(absorbtions))]
mf2_average_lin_prop = [0 for i in range(len(absorbtions_lin))]
# getting the total for each number of absorbtions we are interested in
for i in range(len(absorbtions)):
    for j in range(len(mf2_states)):
        mf2_average_prop[i] += mf2_states[j][i]
for i in range(len(absorbtions_lin)):
    for j in range(len(mf2_states_lin)):
        mf2_average_lin_prop[i] += mf2_states_lin[j][i]
# getting the average proportion for each number of absorbtions we are 
# intereseted in
for i in range(len(mf2_average_prop)):
    mf2_average_prop[i] = mf2_average_prop[i]/len(mf2_states)/100
for i in range(len(mf2_average_lin_prop)):
    mf2_average_lin_prop[i] = mf2_average_lin_prop[i]/len(mf2_states_lin)/100


# plotting
plt.semilogx(absorbtions, mf2_average_prop)
plt.title("Average Proportion of the 100 Atoms in $M_F=2$ Substates over "
          "100 Experiments")
plt.xlabel("Number of Absorbtions")
plt.ylabel("Proportions of Atoms in the $M_F=2$ Substate")
plt.show()

plt.plot(absorbtions_lin, mf2_average_lin_prop)
plt.title("Average Proportion of the 100 Atoms in $M_F=2$ Substates over "
          "100 Experiments")
plt.xlabel("Number of Absorbtions")
plt.ylabel("Proportions of Atoms in the $M_F=2$ Substate")
plt.show()

##############################################################################
#  Finding a Relation between Desired Purity and Number of Scatters Required #
##############################################################################

proportion_to_stop_array =[.8,.9,.95,.99]
num_scatters_necessary = [[], [], [], [], []]
num_atoms_array = np.linspace(100, 50000, 50)
num_scatters_per_atom_across_experiments = []
for t in range(len(proportion_to_stop_array)):

    # running the experiment
    for i in range(len(num_atoms_array)):
        num_atoms = num_atoms_array[i]
        # evenly splitting the atoms among the M_F states
        rb_cloud = [int(num_atoms/5) for j in range(5)] 
        scatters = 0 # initializing
        while rb_cloud[4] < num_atoms*proportion_to_stop_array[t]:
            excite_an_atom(rb_cloud)
            scatters += 1
        num_scatters_necessary[t].append(scatters)

    # Plotting the results
    plt.plot(num_atoms_array, num_scatters_necessary[t])
    plt.title(f"The Number of Scatters Required to get "
              f"{proportion_to_stop_array[t]*100}% of the Atoms in the $M_F=2$ "
              f"Substate")
    plt.xlabel("The Number of Rubidium Atoms")
    plt.ylabel("The Number of Photon Scatters")
    plt.show()

    ##############################################################################
    #                     Line of Best Fit Regression                            #
    ##############################################################################

    num_atoms_array_reg = num_atoms_array.reshape((-1, 1))
    num_scatters_necessary_reg = np.array(num_scatters_necessary[t])

    model = LinearRegression().fit(num_atoms_array_reg, num_scatters_necessary_reg)
    r_sq = model.score(num_atoms_array_reg, num_scatters_necessary_reg)

    print(f"Slope: {round(model.coef_[0],3)}")
    print(f"Coeffecient of Determination: {round(r_sq,4)}")

    est_num_scatters_nec = model.coef_[0] * num_atoms_array + model.intercept_
    num_scatters_per_atom_across_experiments.append(model.coef_[0])

    plt.plot(num_atoms_array, num_scatters_necessary[t], color = 'blue', 
             label='Number of Photon Scatters')
    plt.plot(num_atoms_array, est_num_scatters_nec, color = 'red', 
             label='Estimated Number of Photon Scatters')
    plt.title(f"The Number of Scatters Required to get "
              f"{proportion_to_stop_array[t]*100}% of the Atoms in the $M_F=2$ "
              f"Substate")
    plt.legend()
    plt.xlabel("The Number of Rubidium Atoms")
    plt.ylabel("The Number of Photon Scatters")
    plt.show()

    print(f"The estimated number of photon scatters per atom to get "
          f"{proportion_to_stop_array[t]*100}% of the atoms in the "
          f"$M_F=2$ Substate is "
          f"{round(num_scatters_per_atom_across_experiments[t],4)}.")

percent_purity = [80, 90, 95, 99]
###GRAPHING THE SCATTERS PER ATOM
plt.scatter(percent_purity, num_scatters_per_atom_across_experiments, color = "red")
plt.title(f"The Number of Scatters Required per Atom for Different Purities "
          f"in the $M_F=2$ Substate")
plt.legend()
plt.xlabel("Percent of atoms in the $M_F=2$ Substate")
plt.ylabel("The Number of Scatters Required per Atom")
plt.show()


##############################################################################
#            Looking at this Relation for more Final Purities                #
##############################################################################

proportion_to_stop_array = np.linspace(.8, .99, 20)
num_scatters_necessary = [[] for i in range(20)]
num_atoms_array = np.linspace(1000, 5000, 5)
num_scatters_per_atom_across_experiments = []
for t in range(len(proportion_to_stop_array)):
    print(f"Experiment {t+1} of 20.")

    # running the experiment
    for i in range(len(num_atoms_array)):
        num_atoms = num_atoms_array[i]
        # evenly splitting the atoms among the M_F states
        rb_cloud = [int(num_atoms/5) for j in range(5)] 
        scatters = 0 # initializing
        while rb_cloud[4] < num_atoms*proportion_to_stop_array[t]:
            excite_an_atom(rb_cloud)
            scatters += 1
        num_scatters_necessary[t].append(scatters)

    ##############################################################################
    #                     Line of Best Fit Regression                            #
    ##############################################################################

    num_atoms_array_reg = num_atoms_array.reshape((-1, 1))
    num_scatters_necessary_reg = np.array(num_scatters_necessary[t])

    model = LinearRegression().fit(num_atoms_array_reg, num_scatters_necessary_reg)

    num_scatters_per_atom_across_experiments.append(model.coef_[0])

percent_purity = proportion_to_stop_array* 100
# Graphing the scatters per atom
plt.plot(percent_purity, num_scatters_per_atom_across_experiments)
plt.title(f"The Number of Scatters Required per Atom for Different Purities in "
          f"the $M_F=2$ Substate")
plt.xlabel("Percent of Atoms in the $M_F=2$ Substate")
plt.ylabel("The Number of Scatters Required per Atom")
plt.show()