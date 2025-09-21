"""
Loop_visual.py
---------------
Author: Amir [Your Last Name]
Affiliation: [Your University / Research Group]
Year: 2025

Description:
This script analyzes loop structures in Calcium Aluminophosphate glasses 
using persistent homology. It computes persistence diagrams (PDs), 
accumulated persistence functions (APFs), and derives geometric 
descriptors such as roundness and eccentricity of loops. 
The script also connects PD features to the first sharp diffraction peak (FSDP) 
by estimating Q-distributions via kernel density estimation (KDE). 

Main Features:
- Load atomic structure from LAMMPS dump (via ASE).
- Build weighted alpha shapes using diode/dionysus.
- Generate persistence diagrams (dim=1 loops).
- Calculate APF and export to file.
- Compute loop size, roundness, eccentricity.
- Categorize loops by roundness/eccentricity bins.
- Plot density distributions of loops and compare with Sx(Q).
- Save outputs (CSV, PNG).

Requirements:
- numpy, pandas, matplotlib, seaborn
- ase, dionysus, diode
- scikit-learn
- colour
"""

import os
import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from sklearn.neighbors import KernelDensity
from ase import io
import dionysus as d
import diode
from colour import Color

wd = os.chdir('/path/')

xyz=io.read(f'/path/14.8Al2O3_20CaO_65.2P2O5_Grace.01610000.dump', index=":")

coord = xyz[-1].get_positions()
cell = xyz[-1].get_cell()

data = np.column_stack([xyz[-1].get_chemical_symbols(),coord])

dfpoints = pd.DataFrame(data, columns=["Atom", "x", "y", "z"])
radiusO  = 1.275 
radiusAl  = 0.535 
radiusP  = 0.225 
radiusCa = 1.035

conditions = [
(dfpoints["Atom"]=="H"),
(dfpoints["Atom"]=="He"),
(dfpoints["Atom"]=="Li"), 
(dfpoints["Atom"]=="Be"), 
]
choice_weight=[radiusO**2,radiusAl**2,radiusP**2,radiusCa**2]
dfpoints["w"]=np.select(conditions,choice_weight)

dfpoints["x"] = pd.to_numeric(dfpoints["x"])
dfpoints["y"] = pd.to_numeric(dfpoints["y"])
dfpoints["z"] = pd.to_numeric(dfpoints["z"])

points=dfpoints[["x","y","z","w"]].to_numpy()
simplices = diode.fill_weighted_alpha_shapes(points)
f = d.Filtration(simplices)
m = d.homology_persistence(f, progress=True)

dgms = d.init_diagrams(m, f)

# Export and plot the 1-dimensional persistence diagram
dim = 1
x = [p.birth for p in dgms[dim]]
y = [p.death for p in dgms[dim]]
pd_result = np.column_stack([x, y])
pd_result = np.delete(pd_result, np.where(np.isinf(pd_result))[0], axis=0)

fig, ax = plt.subplots(figsize=(4.5, 4.5))

plt.hist2d(pd_result[:, 0], pd_result[:, 1], bins=400, norm=LogNorm(), cmap='Spectral_r')
plt.axis([-0.25, 8, -0.25, 8])
plt.xlabel('Birth (Å$^2$)', size=14)
plt.ylabel('Death (Å$^2$)', size=14)
plt.xticks(np.arange(0, 8.001, 2), size=14)
plt.yticks(np.arange(0, 8.001, 2), size=14)
cbar=plt.colorbar()
cbar.set_ticks([])
cbar.set_label('Density (logarithmic)', fontsize=18)
plt.savefig('PH_dio_dim{}.png'.format(dim), format="png", dpi = 600, transparent=True, bbox_inches = 'tight')
np.savetxt('pd_dio_dim{}.csv'.format(dim), pd_result, fmt='%s', delimiter=',')  
plt.show()
data = list(zip(x,y))
output = 'birth_death_1.csv'

with open(output, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
#%% APF

birth_times = pd_result[:, 0]
death_times = pd_result[:, 1]
mean_age = (death_times + birth_times)/2
# Compute the ages of each point
life_time = death_times - birth_times
# Sorting
a = np.column_stack((mean_age, life_time))
a = a[a[:, 0].argsort()]

# calculate the APF
apf = np.cumsum(a[:, 1])

# plot it
plt.plot(a[:, 0], apf)
plt.xlabel('Mean Age')
plt.ylabel('Accumulated Persistence')
plt.show()

file = open(f"apf_1.txt", "w")

#file = open(f"apf_2.txt", "w")
for i in range(len(apf)):
    apf[i] = "{:.5f}".format(apf[i])
    a[i, 0] = "{:.5f}".format(a[i, 0])
    file.write(f"{a[i, 0]}\t{apf[i]}\n")
    
#%%

# Read data from file
birth_times = []
death_times = []
multiplicities = []

with open('birth_death_1.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if present
    for row in reader:
        birth_times.append(float(row[0]))
        death_times.append(float(row[1]))
       
# Constants
rO =  1.275 # Oxygen radius (Å)
# Calculate Q values
Q = []
sph_sum = 0

for death in range(len(death_times)):
    if death_times[death] > 0.5:
        l = 2 * np.sqrt(death_times[death] + rO**2)
        q = 2 * np.pi / l
        Q.append(q)

# Convert birth_times, death_times, and Q to NumPy arrays
Q = np.array(Q)
Q = Q[:,None]
# Perform kernel density estimation
kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Q)
x_range = np.linspace(np.min(Q)-0.4, np.max(Q)+1.4, num=1000)  # Adjust 'num' for desired density resolution


# Compute the log-density values for Q values
log_density = kde.score_samples(x_range[:,None])  # Reshape Q as a column vector
density = np.exp(log_density)

# Plot the density curve
plt.plot(x_range, density)
plt.xlabel('Q')
plt.ylabel('Density')
plt.title('Gaussian Kernel Density Estimation')

# Find the maximum point
max_index = np.argmax(log_density)
max_point = x_range[max_index]
max_density = np.exp(log_density[max_index])

#plt.hist(x_range,bins=10, normed=True)
plt.show()

print("Maximum Point: Q =", max_point)
print("Maximum Sph(Q):", max_density)

data = list(zip(x_range,density))
output = 'FSDP_PD_1.csv'

with open(output, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)


#%%

wd = os.chdir('/path/')

xyz=io.read(f'/path/14.8Al2O3_20CaO_65.2P2O5_Grace.01610000.dump', index=":")
#xyz[-1] = xyz[-1].repeat((2,2,2))        
coord = xyz[-1].get_positions()
cell = xyz[-1].get_cell()

data = np.column_stack([xyz[-1].get_chemical_symbols(),coord])

dfpoints = pd.DataFrame(data, columns=["Atom", "x", "y", "z"])
radiusO  = 1.275 
radiusAl  = 0.535
radiusP  = 0.225 
radiusCa = 1.035


conditions = [
(dfpoints["Atom"]=="H"),
(dfpoints["Atom"]=="He"),
(dfpoints["Atom"]=="Li"), 
(dfpoints["Atom"]=="Be"), 
]
choice_weight=[radiusO**2,radiusAl**2,radiusP**2,radiusCa**2]
dfpoints["w"]=np.select(conditions,choice_weight)

dfpoints["x"] = pd.to_numeric(dfpoints["x"])
dfpoints["y"] = pd.to_numeric(dfpoints["y"])
dfpoints["z"] = pd.to_numeric(dfpoints["z"])

points=dfpoints[["x","y","z","w"]].to_numpy()
simplices = diode.fill_weighted_alpha_shapes(points)
f = d.Filtration(simplices)
m = d.homology_persistence(f, progress=True)

dgms = d.init_diagrams(m, f)
birth_simpl=[] # Birth of the simplices
comp_cycl=[] # composition of the simplices
comp_repcycl=[] # representant of the cycles
for i,c in enumerate(m):
    if i % 1000 == 0:
        print(i)
    birth_simpl.append(f[i].data)
    comp_cycl.append([j for j in f[i]])
    a=[]
    for x in c: 
        a=a+[j for j in f[x.index]]
    a = list(set(a))
    comp_repcycl.append(a)
birth_simpl=np.array(birth_simpl)



# ## Representating cycle of points in the PD of loops
pd_cycle=[]
for ind, p in enumerate(dgms[1]):
    #print(ind)
    ## indices of the 2-simplices
    twocyc = np.where(np.fromiter(map(len,comp_cycl), dtype="int")==3)[0]
    ## indices of simplices born at the PD point death time
    diff = np.absolute(p.death-birth_simpl)
    idx_same_death = [i for i in np.where(diff==min(diff))[0]]
    ## intersection of the two sets
    idx_simpl_death=np.intersect1d(twocyc,idx_same_death)

    # test if m[idx_simpl_death] contains f[p.data].
    birth_cyc_in = [p.data in [j.index for j in m[i]] for i in idx_simpl_death]
    # not below that I arbitrary choose the first one
    pd_cycle.append(comp_repcycl[idx_simpl_death[birth_cyc_in][0]])

print("Done-1")

# Gather the PD of loop in a dataframe
dfPD = pd.DataFrame(data={
    "Dimension" : [1 for p in dgms[1]],
    "Birth" : [p.birth for p in dgms[1]],
    "Death" : [p.death for p in dgms[1]], 
    "idpoint": pd_cycle
    
}) 


print("Done-2")

# Add the size of the loop*
dfPD["Size"]=[i for i in map(len, dfPD["idpoint"])]
dfPD.head()
print("Done-3")
#Add number of atoms of a type in the loops
NbO=[]
NbAl=[]
NbP=[]
NbCa=[]


for i in range(dfPD.shape[0]):
    listemp=dfpoints.iloc[dfPD.iloc[i]["idpoint"]]["Atom"].tolist()
    NbO.append(listemp.count("H"))
    NbAl.append(listemp.count("He"))
    NbP.append(listemp.count("Li"))
    NbCa.append(listemp.count("Be"))


dfPD["NbO"]=NbO
dfPD["NbAl"]=NbAl
dfPD["NbP"]=NbP
dfPD["NbCa"]=NbCa

print("Done-4")

# Exclude 3-member loops that are O–P–O (2 O + 1 P)
dfPD = dfPD[~((dfPD["Size"] == 3) & (dfPD["NbP"] == 1) & (dfPD["NbO"] == 2))]
print("Done-5 (O–P–O 3-member loops excluded)")

#%%
# Calculate roundness for each loop
roundness = []

for ids in dfPD["idpoint"]:
    coords = dfpoints.iloc[ids][["x", "y", "z"]].to_numpy(dtype=float)
    centroid = np.mean(coords, axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)
    mean_r = np.mean(distances)
    std_r = np.std(distances)
    r = 1 - (std_r / mean_r if mean_r != 0 else 0)
    roundness.append(r)

dfPD["Roundness"] = roundness
print("Done-5 (Roundness added)")

# --- Categorize loops by roundness in bins of 0.1 ---
round_bins = np.arange(0.5, 1.1, 0.1)
birth_by_round = [[] for _ in range(len(round_bins)-1)]
death_by_round = [[] for _ in range(len(round_bins)-1)]
loop_counts_by_round = [0 for _ in range(len(round_bins)-1)]  # count of loops per bin

for _, row in dfPD.iterrows():
    r = row["Roundness"]
    for i in range(len(round_bins) - 1):
        if round_bins[i] <= r < round_bins[i + 1]:
            birth_by_round[i].append(row["Birth"])
            death_by_round[i].append(row["Death"])
            loop_counts_by_round[i] += 1
            break
        elif r == 1.0 and i == len(round_bins) - 2:
            birth_by_round[i].append(row["Birth"])
            death_by_round[i].append(row["Death"])
            loop_counts_by_round[i] += 1

# --- Print loop counts per roundness group ---
print("\nLoop counts per roundness group:")
for i in range(len(loop_counts_by_round)):
    print(f"{round_bins[i]:.1f} ≤ R < {round_bins[i+1]:.1f}: {loop_counts_by_round[i]} loops")

from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

from colour import Color
    
colors = []
     
steps = len(range(300, 2100, 360))
start_color = np.array([2/255, 175/255, 232/255])
end_color  = np.array([210/255, 0/255, 71/255])
     
for i in range(steps):
        red = start_color[0] - (start_color[0] - end_color[0])*i/steps
        green =start_color[1] - (start_color[1] - end_color[1])*i/steps
        blue = start_color[2] - (start_color[2] - end_color[2])*i/steps
        c = Color(rgb=(red, green, blue))
        colors.append(c.hex)
        

plt.figure()
for i in range(steps):
    plt.scatter(birth_by_round[i], death_by_round[i], color=colors[i], s=0.1)

# Create custom legend handles with larger marker size
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
           markersize=10, label=f'{round_bins[i]:.1f} ≤ R < {round_bins[i+1]:.1f}')
    for i in range(steps)
]

fig = plt.gcf()                 # get current figure
fig.set_size_inches(6.5,6.5) 
plt.axis([-0.25, 10, 0, 10])
plt.xticks(np.arange(0, 10, 2), size=30,fontname='Arial')
plt.yticks(np.arange(0, 10, 2), size=30,fontname='Arial')
plt.xticks([])
plt.yticks([])

plt.savefig('PH_roundness_bins.png', format="png", dpi=600, transparent=True, bbox_inches='tight')
plt.show()

#%%
# --- Extract birth and death for loops with roundness in [0.8, 0.9) and size 6 ---
birth_6atom_08to09 = []
death_6atom_08to09 = []

for idx, row in dfPD.iterrows():
    r = row["Roundness"]
    size = row["Size"]
    if 0.8 <= r < 0.9 and size == 6:
        birth_6atom_08to09.append(row["Birth"])
        death_6atom_08to09.append(row["Death"])

plt.scatter(birth_6atom_08to09, death_6atom_08to09, color='black', s=0.1)
plt.show()

from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

rO = 1.3  # oxygen radius (Å)

# Skip if empty
if len(death_6atom_08to09) > 0:

    # Calculate Q for each death time
    Q = []
    for death in death_6atom_08to09:
        l = 2 * np.sqrt(death + rO**2)
        q = 2 * np.pi / l
        Q.append(q)

    Q = np.array(Q)[:, None]

    # KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Q)
    x_range = np.linspace(np.min(Q) - 0.4, np.max(Q) + 1.4, num=1000)
    log_density = kde.score_samples(x_range[:, None])
    density = np.exp(log_density)
    # Print peak information
    max_index = np.argmax(density)
    max_Q = x_range[max_index]
    max_density_val = density[max_index]
    print(f"Group {i+1}: Max Q = {max_Q:.3f}, Max Density = {max_density_val:.3f}")

    # Plot
    plt.plot(x_range, density, label='Roundness [0.8–0.9), size=6')
    plt.xlabel('Q (Å⁻¹)')
    plt.ylabel('Density')
    plt.title('KDE of Q from Death Times (6-atom loops, Roundness 0.8–0.9)')
    plt.legend()
    plt.grid(True)
    plt.show()


#%%

from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
import csv

rO = 1.3  # oxygen radius (Å)
wd = os.chdir('/path/file')

# Loop over each roundness bin group
for i, (birth_list, death_list) in enumerate(zip(birth_by_round, death_by_round)):

    # Skip empty groups
    if len(death_list) == 0:
        continue

    # Calculate Q for each death time
    Q = []
    for death in death_list:
        l = 2 * np.sqrt(death + rO**2)
        q = 2 * np.pi / l
        Q.append(q)

    Q = np.array(Q)[:, None]

    # KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Q)
    x_range = np.linspace(np.min(Q) - 0.4, np.max(Q) + 1.4, num=1000)
    log_density = kde.score_samples(x_range[:, None])
    density = np.exp(log_density)

    # Plot density curve for this roundness group
    plt.plot(x_range, density,color=colors[i], label=f'{round_bins[i]:.1f} ≤ R < {round_bins[i+1]:.1f}', linewidth=3.5)

    # Export to CSV
    csv_filename = f'FSDP_PD_roundness_group_{i+1}.csv'
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Q', 'Density'])
        writer.writerows(zip(x_range, density))

    # Print peak information
    max_index = np.argmax(density)
    max_Q = x_range[max_index]
    max_density_val = density[max_index]
    print(f"Group {i+1}: Max Q = {max_Q:.3f}, Max Density = {max_density_val:.3f}")

# Total density distribution

# Read data from file
birth_times = []
death_times = []
multiplicities = []

with open('birth_death_1.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if present
    for row in reader:
        birth_times.append(float(row[0]))
        death_times.append(float(row[1]))
       
       # multiplicities.append(int(row[2]))

# Constants
rO =  1.3 # Oxygen radius (Å)
# Calculate Q values
Q = []
sph_sum = 0

for death in range(len(death_times)):
#    if death_times[death] > 0.5:
        l = 2 * np.sqrt(death_times[death] + rO**2)
        q = 2 * np.pi / l
        Q.append(q)

# Convert birth_times, death_times, and Q to NumPy arrays
Q = np.array(Q)
Q = Q[:,None]
# Perform kernel density estimation
kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Q)
#kde.fit(Q[:, np.newaxis])  # Reshape Q as a column vector
x_range_tot = np.linspace(np.min(Q)-0.4, np.max(Q)+1.4, num=1000)  # Adjust 'num' for desired density resolution


# Compute the log-density values for Q values
log_density_tot = kde.score_samples(x_range_tot[:,None])  # Reshape Q as a column vector
density_tot = np.exp(log_density_tot)

# Plot the density curve
plt.plot(x_range_tot, density_tot,color='black', linestyle= '--',label=f'Total density distribution', linewidth=3.5)

# Sx(Q) calculated from MD

md_Al15 = pd.read_csv(f'/path/structure_factor_x_ray_Al15.txt',
                           sep=" ",  skipinitialspace=True, header=None)
md_Al15 = md_Al15.to_numpy()
x_md_Al15= []
y_md_Al15 =[]
for i in range(len(md_Al15)):
    x_md_Al15.append(md_Al15[i][0]+0.05)
    y_md_Al15.append(md_Al15[i][1])

plt.plot(x_md_Al15, y_md_Al15, color='Green', label=f"S$_X$(Q)", linestyle="solid", linewidth=3.5)
fig = plt.gcf()                 # get current figure

fig.set_size_inches(6, 5)  
# Final plot
plt.xlabel('Q (Å$^-$$^1$)', size=30,fontname='Arial', labelpad=20)
plt.ylabel('Density', size=30,fontname='Arial', labelpad=20)
plt.axis([0, 3, 0, 3.8])
plt.xticks(np.arange(0, 3.001, 1), size=30,fontname='Arial')
plt.yticks(np.arange(0, 3.501, 1), size=30,fontname='Arial')
plt.grid(False)
plt.tight_layout()
plt.savefig("density_distribution_roundness_groups_all_death.png", dpi=300, transparent=True, bbox_inches='tight')
plt.show()

#%%
# Calculate eccentricity for each loop
eccentricity = []

for ids in dfPD["idpoint"]:
    coords = dfpoints.iloc[ids][["x", "y", "z"]].to_numpy(dtype=float)
    centroid = np.mean(coords, axis=0)
    coords_centered = coords - centroid

    # Best-fit plane using SVD
    u, s, vh = np.linalg.svd(coords_centered)
    normal = vh[2]  # normal to the plane

    # Project coordinates to 2D
    plane_coords = coords_centered - np.outer(np.dot(coords_centered, normal), normal)
    x = np.dot(plane_coords, vh[0])
    y = np.dot(plane_coords, vh[1])
    data_2d = np.vstack([x, y]).T

    cov = np.cov(data_2d, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    a, b = np.sqrt(sorted(eigvals, reverse=True))  # a: major, b: minor
    e = np.sqrt(1 - (b / a)**2) if a != 0 else 0
    eccentricity.append(e)

dfPD["Eccentricity"] = eccentricity
print("Done-6 (Eccentricity added)")

# --- Categorize by eccentricity bins ---
ecc_bins = np.arange(0.0, 1.01, 0.2)
birth_by_ecc = [[] for _ in range(len(ecc_bins)-1)]
death_by_ecc = [[] for _ in range(len(ecc_bins)-1)]
loop_counts_by_ecc = [0 for _ in range(len(ecc_bins)-1)]  # <-- new counter list

for _, row in dfPD.iterrows():
    e = row["Eccentricity"]
    for i in range(len(ecc_bins) - 1):
        if ecc_bins[i] <= e < ecc_bins[i + 1]:
            birth_by_ecc[i].append(row["Birth"])
            death_by_ecc[i].append(row["Death"])
            loop_counts_by_ecc[i] += 1
            break
        elif e == 1.0 and i == len(ecc_bins) - 2:
            birth_by_ecc[i].append(row["Birth"])
            death_by_ecc[i].append(row["Death"])
            loop_counts_by_ecc[i] += 1

# --- Print loop counts per eccentricity group ---
print("\nLoop counts per eccentricity group:")
for i in range(len(loop_counts_by_ecc)):
    print(f"{ecc_bins[i]:.1f} ≤ e < {ecc_bins[i+1]:.1f}: {loop_counts_by_ecc[i]} loops")


from colour import Color
    
colors = []
     
steps = len(range(300, 2100, 360))
start_color = np.array([2/255, 175/255, 232/255])
end_color  = np.array([210/255, 0/255, 71/255])
     
for i in range(steps):
        red = start_color[0] - (start_color[0] - end_color[0])*i/steps
        green =start_color[1] - (start_color[1] - end_color[1])*i/steps
        blue = start_color[2] - (start_color[2] - end_color[2])*i/steps
        c = Color(rgb=(red, green, blue))
        colors.append(c.hex)
        

plt.figure()
for i in range(steps):
    label = f'{ecc_bins[i]:.1f} ≤ e < {ecc_bins[i+1]:.1f}'
    plt.scatter(birth_by_ecc[i], death_by_ecc[i], color=colors[i], s=1, label=label)


# Create custom legend handles with larger marker size
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
           markersize=10, label=f'{ecc_bins[i]:.1f} ≤ E < {ecc_bins[i+1]:.1f}')
    for i in range(steps)
]
fig = plt.gcf()                 # get current figure
fig.set_size_inches(6.5,6.5) 
plt.axis([-0.25, 10, 0, 10])
plt.xticks(np.arange(0, 10, 2), size=30,fontname='Arial')
plt.yticks(np.arange(0, 10, 2), size=30,fontname='Arial')
plt.xticks([])
plt.yticks([])
plt.savefig('PH_eccentricity_bins.png', format="png", dpi=600, transparent=True, bbox_inches='tight')
plt.show()

#%%

# Define the colormap
cmap = get_cmap('nipy_spectral')  

# Define the number of steps you want in the color map
steps = len(range(300, 2100, 200))

# Generate colors from the colormap
colors = [cmap(i / steps) for i in range(steps)]
# Define bins and labels
bins = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, np.inf]
group_labels = ['L$_3$', 'L$_4$', 'L$_5$', 'L$_6$', 'L$_7$', 'L$_8$', 'L$_9$', 'L$_1$$_0$', 'L$_n$$_>$$_1$$_0$']

# Categorize into size groups
dfPD["SizeGroup"] = pd.cut(dfPD["Size"], bins=bins, labels=group_labels)
import seaborn as sns

# Set color palette
palette = sns.color_palette("Set2", len(group_labels))

# Plot
plt.figure(figsize=(8, 6))
for i, label in enumerate(group_labels):
    subset = dfPD[dfPD["SizeGroup"] == label]
    plt.scatter(
        subset["Eccentricity"],
        subset["Roundness"],
        label=label,
        color=colors[i],
        alpha=1,
        s=15
    )
fig = plt.gcf()                             
fig.set_size_inches(8,5.0) 
plt.axis([0, 1.05, 0.5, 1.05])

# Labels and legend
plt.xticks(np.arange(0.2, 1.001, 0.2), size=30)
plt.yticks(np.arange(0.6, 1.001, 0.2), size=30)
plt.xticks([])

#plt.xlabel("Eccentricity", size=30,fontname='Arial', labelpad=20)
plt.ylabel("Roundness", size=30,fontname='Arial', labelpad=20)
plt.tight_layout()

plt.savefig("roundness_vs_eccentricity_by_sizegroup.png", dpi=300, transparent=True)
plt.show()
#%%

# Group by SizeGroup and calculate mean and variance
stats = dfPD.groupby("SizeGroup")[["Roundness", "Eccentricity"]].agg(['mean', 'var'])
stats.to_csv("roundness_eccentricity_stats_by_sizegroup_Al15.csv")

# Display the result
print(stats)

# Calculate Pearson correlation between Roundness, Eccentricity, and Size for each SizeGroup
correlation_results = {}
for group, group_df in dfPD.groupby('SizeGroup'):
    corr = group_df[['Roundness', 'Eccentricity', 'Size']].corr(method='pearson')
    correlation_results[group] = corr

# Extract and format Roundness vs. Eccentricity correlation for each group
print("Ring Size\tRoundness vs. Eccentricity")
for group in sorted(correlation_results.keys()):
    corr_val = correlation_results[group].loc["Roundness", "Eccentricity"]
    sign = "+" if corr_val >= 0 else "−"
    print(f"{group}\t\t{sign}{abs(corr_val):.2f}")
#%%
# --- Extract birth and death for loops with roundness in [0.8, 0.9) and size 6 ---
birth_6atom_06to08 = []
death_6atom_06to08 = []

for idx, row in dfPD.iterrows():
    e = row["Eccentricity"]
    size = row["Size"]
    if 0.6 <= e < 0.8 and size == 6:
        birth_6atom_06to08.append(row["Birth"])
        death_6atom_06to08.append(row["Death"])

plt.scatter(birth_6atom_06to08, death_6atom_06to08, color='black', s=0.1)
plt.show()

rO = 1.3  # oxygen radius (Å)

# Skip if empty
if len(death_6atom_06to08) > 0:

    # Calculate Q for each death time
    Q = []
    for death in death_6atom_06to08:
        l = 2 * np.sqrt(death + rO**2)
        q = 2 * np.pi / l
        Q.append(q)

    Q = np.array(Q)[:, None]

    # KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Q)
    x_range = np.linspace(np.min(Q) - 0.4, np.max(Q) + 1.4, num=1000)
    log_density = kde.score_samples(x_range[:, None])
    density = np.exp(log_density)
    # Print peak information
    max_index = np.argmax(density)
    max_Q = x_range[max_index]
    max_density_val = density[max_index]
    print(f"Group {i+1}: Max Q = {max_Q:.3f}, Max Density = {max_density_val:.3f}")

    # Plot
    plt.plot(x_range, density, label='Eccentricity [0.6–0.8), size=6')
    plt.xlabel('Q (Å⁻¹)')
    plt.ylabel('Density')
    plt.title('KDE of Q from Death Times (6-atom loops, Eccentricity 0.6–0.8)')
    plt.legend()
    plt.grid(True)
    plt.show()

#%%
    
colors = []
     
steps = len(range(300, 2100, 360))
start_color = np.array([2/255, 175/255, 232/255])
end_color  = np.array([210/255, 0/255, 71/255])
     
for i in range(steps):
        red = start_color[0] - (start_color[0] - end_color[0])*i/steps
        green =start_color[1] - (start_color[1] - end_color[1])*i/steps
        blue = start_color[2] - (start_color[2] - end_color[2])*i/steps
        c = Color(rgb=(red, green, blue))
        colors.append(c.hex)
        
# Loop over each roundness bin group
for i, (birth_list, death_list) in enumerate(zip(birth_by_ecc, death_by_ecc)):

    # Skip empty groups
    if len(death_list) == 0:
        continue

    # Calculate Q for each death time
    Q = []
    for death in death_list:
        l = 2 * np.sqrt(death + rO**2)
        q = 2 * np.pi / l
        Q.append(q)

    Q = np.array(Q)[:, None]

    # KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Q)
    x_range = np.linspace(np.min(Q) - 0.4, np.max(Q) + 1.4, num=1000)
    log_density = kde.score_samples(x_range[:, None])
    density = np.exp(log_density)

    # Plot density curve for this roundness group
    plt.plot(x_range, density, color=colors[i],label=f'{ecc_bins[i]:.1f} ≤ E < {ecc_bins[i+1]:.1f}', linewidth=3.5)

    # Export to CSV
    csv_filename = f'FSDP_PD_eccentiric_group_{i+1}.csv'
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Q', 'Density'])
        writer.writerows(zip(x_range, density))

    # Print peak information
    max_index = np.argmax(density)
    max_Q = x_range[max_index]
    max_density_val = density[max_index]
    print(f"Group {i+1}: Max Q = {max_Q:.3f}, Max Density = {max_density_val:.3f}")

# Total density distribution

# Read data from file
birth_times = []
death_times = []
multiplicities = []

with open('birth_death_1.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if present
    for row in reader:
        birth_times.append(float(row[0]))
        death_times.append(float(row[1]))
       
       # multiplicities.append(int(row[2]))

# Constants
rO =  1.3 # Oxygen radius (Å)
# Calculate Q values
Q = []
sph_sum = 0

for death in range(len(death_times)):
#    if death_times[death] > 0.5:
        l = 2 * np.sqrt(death_times[death] + rO**2)
        q = 2 * np.pi / l
        Q.append(q)

# Convert birth_times, death_times, and Q to NumPy arrays
Q = np.array(Q)
Q = Q[:,None]
# Perform kernel density estimation
kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Q)
#kde.fit(Q[:, np.newaxis])  # Reshape Q as a column vector
x_range_tot = np.linspace(np.min(Q)-0.4, np.max(Q)+1.4, num=1000)  # Adjust 'num' for desired density resolution


# Compute the log-density values for Q values
log_density_tot = kde.score_samples(x_range_tot[:,None])  # Reshape Q as a column vector
density_tot = np.exp(log_density_tot)

# Plot the density curve
plt.plot(x_range_tot, density_tot,color='black', linestyle= '--',label=f'Total density distribution', linewidth=3.5 )

# Sx(Q) calculated from MD

md_Al15 = pd.read_csv(f'/path/S_X_(Q)_Exp/structure_factor_x_ray_Al15.txt',
                           sep=" ",  skipinitialspace=True, header=None)
md_Al15 = md_Al15.to_numpy()
x_md_Al15= []
y_md_Al15 =[]
for i in range(len(md_Al15)):
    x_md_Al15.append(md_Al15[i][0]+0.05)
    y_md_Al15.append(md_Al15[i][1])

plt.plot(x_md_Al15, y_md_Al15, color='Green', label=f"S$_X$(Q)", linestyle="solid", linewidth=3.5)
fig = plt.gcf()

fig.set_size_inches(6, 5) 
# Final plot
plt.xlabel('Q (Å$^-$$^1$)', size=30,fontname='Arial', labelpad=20)
plt.ylabel('Density', size=30,fontname='Arial', labelpad=20)
plt.axis([0, 3, 0, 4])
plt.xticks(np.arange(0, 3.001, 1), size=30,fontname='Arial')
plt.yticks(np.arange(0, 3.501, 1), size=30,fontname='Arial')
plt.grid(False)
plt.tight_layout()
plt.savefig("density_distribution_eccentiric_groups_all_death.png", dpi=300, transparent=True, bbox_inches='tight')
plt.show()


#%%

array = dfPD.to_numpy()
count = 0
rings = []
birth_time_4_si_o = []
death_time_4_si_o = []
birth_time_4_6 = []
death_time_4_6 = []
birth_time_7_10 = []
death_time_7_10 = []
birth_time_3 = []
death_time_3 = []
birth_time_4 = []
death_time_4 = []
birth_time_5 = []
death_time_5 = []
birth_time_6 = []
death_time_6 = []
birth_time_7 = []
death_time_7 = []
birth_time_8 = []
death_time_8 = []
birth_time_9 = []
death_time_9 = []
birth_time_10 = []
death_time_10 = []
birth_time_11 = []
death_time_11 = []
size=[]
for i in range(0,len(array)):
    if array[i][4] == 4:
        birth_time_4.append(array[i][1])
        death_time_4.append(array[i][2])
    elif array[i][4] == 3:
        birth_time_3.append(array[i][1])
        death_time_3.append(array[i][2])
    elif array[i][4] == 5:
        birth_time_5.append(array[i][1])
        death_time_5.append(array[i][2])
    elif array[i][4] == 6:
        birth_time_6.append(array[i][1])
        death_time_6.append(array[i][2])
    elif array[i][4] == 7:
        birth_time_7.append(array[i][1])
        death_time_7.append(array[i][2])
    elif array[i][4] == 8:
        birth_time_8.append(array[i][1])
        death_time_8.append(array[i][2])
    elif array[i][4] == 9:
        birth_time_9.append(array[i][1])
        death_time_9.append(array[i][2])
    elif array[i][4] == 10:
        birth_time_10.append(array[i][1])
        death_time_10.append(array[i][2])
for i in range(0,len(array)):        
    if array[i][4] >= 11: 
        birth_time_11.append(array[i][1])
        death_time_11.append(array[i][2])
        size.append(array[i][4])


# Define the colormap
cmap = get_cmap('nipy_spectral')  # 'nipy_spectral', 'rainbow', 'jet', or any other full-spectrum colormap

# Define the number of steps you want in the color map
steps = len(range(300, 2100, 200))

# Generate colors from the colormap
colors = [cmap(i / steps) for i in range(steps)]

    
#plt.hist2d(birth_time, death_time, bins=400, norm=LogNorm(), cmap='Spectral_r')
plt.scatter(birth_time_3, death_time_3, color=colors[0],s=0.1,label='L$_3$')
plt.scatter(birth_time_4, death_time_4, color=colors[1],s=0.1,label='L$_4$')
plt.scatter(birth_time_5, death_time_5, color=colors[2],s=0.1,label='L$_5$')
plt.scatter(birth_time_6, death_time_6, color=colors[3],s=0.1,label='L$_6$')
plt.scatter(birth_time_7, death_time_7, color=colors[4],s=0.1,label='L$_7$')
plt.scatter(birth_time_8, death_time_8, color=colors[5],s=0.1,label='L$_8$')
plt.scatter(birth_time_9, death_time_9, color=colors[6],s=0.1,label='L$_9$')
plt.scatter(birth_time_10, death_time_10, color=colors[7],s=0.1,label='L$_1$$_0$')
plt.scatter(birth_time_11, death_time_11, color=colors[8],s=0.01,label='$L_{m} > L_{10}$')
#plt.scatter(birth_time_4_si_o, death_time_4_si_o, color=colors[0],s=0.5, label='Si-O-Si-O')
plt.axis([-0.25,10, 0, 10])
fig = plt.gcf()                 # get current figure
fig.set_size_inches(6.5,6.5) 
plt.axis([-0.25, 10, 0, 10])
plt.xticks(np.arange(0, 10, 2), size=30,fontname='Arial')
plt.yticks(np.arange(0, 10, 2), size=30,fontname='Arial')
plt.xticks([])
plt.yticks([])

plt.savefig('PH_loop_size_with_L3.png', format="png", dpi = 600, transparent=True, bbox_inches = 'tight')
plt.show()
#%%

rO = 1.3  # Oxygen radius (Å)
group_labels = ['L$_3$','L$_4$', 'L$_5$', 'L$_6$', 'L$_7$', 'L$_8$', 'L$_9$', 'L$_1$$_0$', '$L_{m} > L_{10}$']
group_deaths = [
    death_time_3,
    death_time_4,
    death_time_5,
    death_time_6,
    death_time_7,
    death_time_8,
    death_time_9,
    death_time_10,
    death_time_11
]

from matplotlib.cm import get_cmap

# Define the colormap
cmap = get_cmap('nipy_spectral')  # 'nipy_spectral', 'rainbow', 'jet', or any other full-spectrum colormap

# Define the number of steps you want in the color map
steps = len(range(300, 2100, 200))

# Generate colors from the colormap
colors = [cmap(i / steps) for i in range(steps)]

plt.figure()

for i, deaths in enumerate(group_deaths):
    if i == 0:
        continue
    count = len(deaths)
    print(f"{group_labels[i]}: {count} loops")

    if count == 0:
        continue

    # Calculate Q
    Q = [2 * np.pi / (2 * np.sqrt(death + rO**2)) for death in deaths]
    Q = np.array(Q)[:, None]

    # KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Q)
    x_range = np.linspace(np.min(Q) - 0.4, np.max(Q) + 1.4, num=1000)
    log_density = kde.score_samples(x_range[:, None])
    density = np.exp(log_density)

    plt.plot(x_range, density, label=group_labels[i], color=colors[i], linewidth=2.5)

    # Save to CSV
    with open(f'FSDP_PD_{group_labels[i]}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Q", "Density"])
        writer.writerows(zip(x_range, density))

    # Optional: print peak info
    max_idx = np.argmax(density)
    print(f"    Max Q = {x_range[max_idx]:.3f}, Max Density = {density[max_idx]:.3f}")

# Total density distribution

# Read data from file
birth_times = []
death_times = []
multiplicities = []

with open('birth_death_1.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header if present
    for row in reader:
        birth_times.append(float(row[0]))
        death_times.append(float(row[1]))
       
       # multiplicities.append(int(row[2]))

# Constants
rO =  1.3 # Oxygen radius (Å)
# Calculate Q values
Q = []
sph_sum = 0

for death in range(len(death_times)):
#    if death_times[death] > 0.5:
        l = 2 * np.sqrt(death_times[death] + rO**2)
        q = 2 * np.pi / l
        Q.append(q)

# Convert birth_times, death_times, and Q to NumPy arrays
Q = np.array(Q)
Q = Q[:,None]
# Perform kernel density estimation
kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Q)
x_range_tot = np.linspace(np.min(Q)-0.4, np.max(Q)+1.4, num=1000)  # Adjust 'num' for desired density resolution


# Compute the log-density values for Q values
log_density_tot = kde.score_samples(x_range_tot[:,None])  # Reshape Q as a column vector
density_tot = np.exp(log_density_tot)

# Plot the density curve
plt.plot(x_range_tot, density_tot,color='black', linestyle= '--',label=f'S$_P$$_H$(Q)' , linewidth=2.5)

# Sx(Q) calculated from MD

md_60P = pd.read_csv(f'/path/structure_factor_x_ray_P60.txt',
                           sep=" ",  skipinitialspace=True, header=None)
md_60P = md_60P.to_numpy()
x_md_60P= []
y_md_60P =[]
for i in range(len(md_60P)):
    x_md_60P.append(md_60P[i][0]+0.05)
    y_md_60P.append(md_60P[i][1])

plt.plot(x_md_60P, y_md_60P, color='black', label=f"S$_X$(Q)", linestyle="solid", linewidth=2.5)
# Final plot
plt.xlabel('Q (Å$^-$$^1$)', size=30,fontname='Arial', labelpad=20)
plt.ylabel('Density', size=30,fontname='Arial', labelpad=20)
plt.axis([0, 3, 0, 4])
plt.xticks(np.arange(0, 3.001, 1), size=30)#,fontname='Arial')
#plt.xticks([])
plt.yticks(np.arange(0, 3.501, 1), size=30,fontname='Arial')
# plt.legend(edgecolor='black', bbox_to_anchor=(.01, 1.0), loc='upper left', frameon=False, fontsize=18)

fig = plt.gcf()                 # get current figure

fig.set_size_inches(6, 5.5)          # resize it


plt.grid(False)
plt.savefig('FSDP_PD_loop_sizes_legends.png', dpi=600, transparent=True, bbox_inches='tight')
plt.show()

#%%
from colour import Color
    
colors = []
     
steps = len(range(300, 2100, 600))
start_color = np.array([2/255, 175/255, 232/255])
end_color  = np.array([210/255, 0/255, 71/255])
     
for i in range(steps):
        red = start_color[0] - (start_color[0] - end_color[0])*i/steps
        green =start_color[1] - (start_color[1] - end_color[1])*i/steps
        blue = start_color[2] - (start_color[2] - end_color[2])*i/steps
        c = Color(rgb=(red, green, blue))
        colors.append(c.hex)
rO = 1.3  # Oxygen radius (Å)
group_labels = [ 'Total L$_6$', 'L$_6$ with 0.8 ≤ R < 0.9', 'L$_6$ with 0.6 ≤ E < 0.8' ]
group_deaths = [
    death_time_6,
    death_6atom_08to09,
    death_6atom_06to08,
]

#colors = [cmap(i / len(group_deaths)) for i in range(len(group_deaths))]

plt.figure()

for i, deaths in enumerate(group_deaths):
    count = len(deaths)
    print(f"{group_labels[i]}: {count} loops")

    if count == 0:
        continue

    # Calculate Q
    Q = [2 * np.pi / (2 * np.sqrt(death + rO**2)) for death in deaths]
    Q = np.array(Q)[:, None]

    # KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Q)
    x_range = np.linspace(np.min(Q) - 0.4, np.max(Q) + 1.4, num=1000)
    log_density = kde.score_samples(x_range[:, None])
    density = np.exp(log_density)

    plt.plot(x_range, density, label=group_labels[i], color=colors[i], linewidth=3.5)
fig = plt.gcf()                 # get current figure

fig.set_size_inches(6, 5) 
# Final plot
plt.ylabel('Density', size=30,fontname='Arial', labelpad=20)
plt.axis([1, 2.5, 0, 3])
plt.xticks(np.arange(1, 2.501, 0.5), size=30,fontname='Arial')
plt.yticks(np.arange(1, 3.001, 1), size=30,fontname='Arial')
plt.xticks([])
plt.grid(False)
plt.tight_layout()
plt.savefig('FSDP_PD_loop_6_member.png', dpi=600, transparent=True, bbox_inches='tight')
plt.show()

#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data (including Birth, Death, and Eccentricity)
df6 = dfPD[dfPD["Size"] == 6].copy()
df6["Eccentricity"] = 1 - df6["Roundness"]

# Define eccentricity bins
ecc_bins = np.arange(0.0, 1.1, 0.1)
bin_labels = [f'{ecc_bins[i]:.1f}–{ecc_bins[i+1]:.1f}' for i in range(len(ecc_bins)-1)]

# Prepare colormap

from colour import Color
    
colors = []
     
steps = len(range(300, 2100, 180))
start_color = np.array([2/255, 175/255, 232/255])
end_color  = np.array([210/255, 0/255, 71/255])
     
for i in range(steps):
        red = start_color[0] - (start_color[0] - end_color[0])*i/steps
        green =start_color[1] - (start_color[1] - end_color[1])*i/steps
        blue = start_color[2] - (start_color[2] - end_color[2])*i/steps
        c = Color(rgb=(red, green, blue))
        colors.append(c.hex)
        

# Prepare data bins
birth_by_ecc = [[] for _ in range(len(ecc_bins)-1)]
death_by_ecc = [[] for _ in range(len(ecc_bins)-1)]

# Bin loops based on eccentricity
for _, row in df6.iterrows():
    ecc = row["Eccentricity"]
    for i in range(len(ecc_bins) - 1):
        if ecc_bins[i] <= ecc < ecc_bins[i + 1] or (ecc == 1.0 and i == len(ecc_bins) - 2):
            birth_by_ecc[i].append(row["Birth"])
            death_by_ecc[i].append(row["Death"])
            break

# Plot
plt.figure()
for i in range(len(ecc_bins) - 1):
    label = f'{ecc_bins[i]:.1f} ≤ E < {ecc_bins[i+1]:.1f}'
    plt.scatter(birth_by_ecc[i], death_by_ecc[i], color=colors[i], s=5, label=label)

# --- Count and display number of loops per eccentricity group ---
print("Loop counts per eccentricity bin:")
for i in range(len(ecc_bins) - 1):
    count = len(birth_by_ecc[i])
    label = f'{ecc_bins[i]:.1f} ≤ E < {ecc_bins[i+1]:.1f}' if i < len(ecc_bins) - 2 else f'{ecc_bins[i]:.1f} ≤ E ≤ 1.0'
    print(f"{label}: {count} loops")

plt.xlabel("Birth (Å$^2$)", fontsize=14)
plt.ylabel("Death (Å$^2$)", fontsize=14)
plt.title("Persistence Diagram of Size-6 Loops Categorized by Eccentricity", fontsize=16)
plt.axis([-0.25, 10, 0, 10])
plt.xticks(np.arange(0, 10.1, 2), fontsize=12)
plt.yticks(np.arange(0, 10.1, 2), fontsize=12)
plt.legend(edgecolor='black', title='Eccentricity Bins', loc='upper left', fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.savefig("PD_L6_Eccentricity_Categorized.png", dpi=600, transparent=True, bbox_inches='tight')
plt.show()


