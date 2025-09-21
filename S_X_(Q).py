"""
PDF and Structure Factor Analysis
---------------------------------
Author: Amir [Your Last Name]
Affiliation: [Your University / Research Group]
Year: 2025

Description:
------------
This script analyzes molecular dynamics trajectories of oxide glasses 
from LAMMPS dump files. It calculates:

- Partial pair distribution functions (PDFs)
- Total PDF
- Partial and total structure factors S(Q)
- Outputs results as text files for reproducibility and further analysis
- Plots form factors, S(Q), and PDF

Usage:
------
1. Place the trajectory dump file in the working directory.
2. Update `workdir` and `dump_filename` below.
3. Run the script:
   $ python PDF_analysis.py

Dependencies:
-------------
- numpy
- pandas
- matplotlib
- scipy
"""
import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


wd = os.chdir('/path/to/workdir')

with open('14.8Al2O3_20CaO_65.2P2O5_Grace.01610000.dump','r') as datafile:
    data_list = []
    for d in datafile:
        d = d.strip('\n')
        a = d.split(' ')
        data_list.append(a)
        
num_lines = len(data_list) #Count number of rows in trajectory file
num_atoms = int(data_list[3][0]) #number of Na in box
num_steps = num_lines/(num_atoms+9) #Count number of timesteps in trajectory file

nspecies = 4

rrangemax = 1 #if 0 specify rrange below in rmax value. if 1 the max integer rrange will be found automatically
rmax = 10 #in Ångstrøm
sq = 1 #add Structure factor (1) or not (0)
GRTOT = 1
TR = 1

n = 1 #Number of trajectory inputs to average over in output
nbin = 1600 #Number of bins in g(r)'
qrange = 12 #range of Q (Å-1) in the experimental Sq. Used in broadening!!
dq = 0.02 #Discretization of s(q)

t1 = (num_lines/(num_atoms+9)) - (n - 1)
t2 = (num_lines/(num_atoms+9))

# Constants for elements
elements = {
    "O": {
        "a": [3.0485, 2.2868, 1.5463, 0.867],
        "b": [13.2771, 5.7011, 0.3239, 32.9089],
        "c": 0.2508
    },
    "Al": {
        "a": [6.4202, 1.9002, 1.5936, 1.9646],
        "b": [3.0387, 0.7426, 31.5472, 85.0886],
        "c": 1.1151
    },
    "P": {
        "a": [6.4345, 4.1791, 1.78, 1.4908],
        "b": [1.9067, 27.157, 0.526, 68.1645],
        "c": 1.1149
    },
    "Ca": {
        "a": [8.6266, 7.3873, 1.5899, 1.0211],
        "b": [10.4421, 0.6599, 85.7484, 178.437],
        "c": 1.3751
    }
}

# Range of q values (scattering vector) from 0 to 25 Å^-1
q_values = np.linspace(0, 25, 1600)

# Function to calculate f(q)
def calculate_fq(q, a, b, c):
    fq = sum(ai * np.exp(-bi * (q / (4 * np.pi)) ** 2) for ai, bi in zip(a, b)) + c
    return fq

# Calculate f(q) for each element and save as numpy arrays in fq
fq = []
plt.figure(figsize=(10, 6))
for element, params in elements.items():
    f_q_values = np.array([calculate_fq(q, params['a'], params['b'], params['c']) for q in q_values])
    fq.append(f_q_values)
    plt.plot(q_values, f_q_values, label=f"{element}")


# Plot settings
plt.xlabel("q (Å⁻¹)")
plt.ylabel("f(q)")
plt.title("Atomic Form Factor f(q) for O, Al, P, and Ca")
plt.legend()
plt.grid(True)
plt.show()

data_array = np.zeros((n,num_atoms,4))
index = np.zeros((n,num_atoms,1))
for i in range(n):
    for j in range(num_atoms):
        line = data_list[9+i*(num_atoms+9)+j]
        data_array[i,j,:] = [float(g) for g in line[1:5]]
        index[i,j,:] = line[1]
box_size = np.zeros((n,3,2))

for i in range(n):
    for j in range(3):
        line = data_list[5+i*(num_atoms+9)+j]
        box_size[i,:] = [float(g) for g in line]

del data_list

xlo = box_size[0,0,0]   
xhi = box_size[0,0,1]
ylo = box_size[0,1,0]
yhi = box_size[0,1,1]
zlo = box_size[0,2,0]
zhi = box_size[0,2,1]  
        
if rrangemax==1:
    if xhi-xlo<=yhi-ylo and xhi-xlo<=zhi-zlo: #Determine radial range to count in
        rrange=round((xhi-xlo)/2)
    elif yhi-ylo<=xhi-xlo and yhi-ylo<=zhi-zlo:
        rrange=round((yhi-ylo)/2);
    elif zhi-zlo<=xhi-xlo and zhi-zlo<=yhi-ylo:
        rrange=round((zhi-zlo)/2);
else:
    rrange=rmax;

distances = np.zeros((n,num_atoms,num_atoms))


def distance_table_vector(liste,dimensions):
    dim = np.array([dimensions[0,1]-dimensions[0,0], dimensions[1,1]-dimensions[1,0], dimensions[2,1]-dimensions[2,0]])
    
    x_dif = np.abs(liste[:,1][np.newaxis, :] - liste[:,1][:, np.newaxis])
    y_dif = np.abs(liste[:,2][np.newaxis, :] - liste[:,2][:, np.newaxis])
    z_dif = np.abs(liste[:,3][np.newaxis, :] - liste[:,3][:, np.newaxis])
    x_dif = np.where(x_dif > 0.5 * dim[0], np.abs(x_dif - dim[0]), x_dif)
    y_dif = np.where(y_dif > 0.5 * dim[1], np.abs(y_dif - dim[1]), y_dif)
    z_dif = np.where(z_dif > 0.5 * dim[2], np.abs(z_dif - dim[2]), z_dif)
    i_i = np.sqrt(x_dif ** 2 + y_dif ** 2 + z_dif ** 2 )
    return i_i


for i in range(n):
    distances[i,:,:] = distance_table_vector(data_array[i,:,:],box_size[i,:,:])


 
def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() )
    return list_of_objects

species_index = []

for i in range(n):
    species_i = init_list_of_objects(nspecies)
    for ind, j in enumerate(data_array[i,:,0]):
        if j == 1:
            species_i[0].append(ind)
        if j == 2:
            species_i[1].append(ind)
        if j == 3:
            species_i[2].append(ind)
        if j == 4:
            species_i[3].append(ind)
    species_index.append(species_i)

numofeach = [np.count_nonzero(data_array[0,:,0] == i+1) for i in range(nspecies)]
c = [numofeach[i]/sum(numofeach) for i in range(nspecies)]

del data_array


dist_1_1 = np.zeros((n, numofeach[0],numofeach[0]))
dist_1_2 = np.zeros((n, numofeach[0],numofeach[1]))
dist_1_3 = np.zeros((n, numofeach[0],numofeach[2]))
dist_1_4 = np.zeros((n, numofeach[0],numofeach[3]))
dist_2_1 = np.zeros((n, numofeach[1],numofeach[0])) 
dist_2_2 = np.zeros((n, numofeach[1],numofeach[1]))
dist_2_3 = np.zeros((n, numofeach[1],numofeach[2])) 
dist_2_4 = np.zeros((n, numofeach[1],numofeach[3])) 
dist_3_1 = np.zeros((n, numofeach[2],numofeach[0])) 
dist_3_2 = np.zeros((n, numofeach[2],numofeach[1]))
dist_3_3 = np.zeros((n, numofeach[2],numofeach[2])) 
dist_3_4 = np.zeros((n, numofeach[2],numofeach[3])) 
dist_4_1 = np.zeros((n, numofeach[3],numofeach[0])) 
dist_4_2 = np.zeros((n, numofeach[3],numofeach[1]))
dist_4_3 = np.zeros((n, numofeach[3],numofeach[2])) 
dist_4_4 = np.zeros((n, numofeach[3],numofeach[3])) 



for i in range(n):
    dist_1_1[i,:,:] = distances[i,:,:][np.ix_(species_index[i][0],species_index[i][0])]
    dist_1_2[i,:,:] = distances[i,:,:][np.ix_(species_index[i][0],species_index[i][1])]
    dist_1_3[i,:,:] = distances[i,:,:][np.ix_(species_index[i][0],species_index[i][2])]
    dist_1_4[i,:,:] = distances[i,:,:][np.ix_(species_index[i][0],species_index[i][3])]
    dist_2_1[i,:,:] = distances[i,:,:][np.ix_(species_index[i][1],species_index[i][0])]
    dist_2_2[i,:,:] = distances[i,:,:][np.ix_(species_index[i][1],species_index[i][1])]
    dist_2_3[i,:,:] = distances[i,:,:][np.ix_(species_index[i][1],species_index[i][2])]
    dist_2_4[i,:,:] = distances[i,:,:][np.ix_(species_index[i][1],species_index[i][3])]
    dist_3_1[i,:,:] = distances[i,:,:][np.ix_(species_index[i][2],species_index[i][0])]
    dist_3_2[i,:,:] = distances[i,:,:][np.ix_(species_index[i][2],species_index[i][1])]
    dist_3_3[i,:,:] = distances[i,:,:][np.ix_(species_index[i][2],species_index[i][2])]
    dist_3_4[i,:,:] = distances[i,:,:][np.ix_(species_index[i][2],species_index[i][3])]
    dist_4_1[i,:,:] = distances[i,:,:][np.ix_(species_index[i][3],species_index[i][0])]
    dist_4_2[i,:,:] = distances[i,:,:][np.ix_(species_index[i][3],species_index[i][1])]
    dist_4_3[i,:,:] = distances[i,:,:][np.ix_(species_index[i][3],species_index[i][2])]
    dist_4_4[i,:,:] = distances[i,:,:][np.ix_(species_index[i][3],species_index[i][3])]


pdist = [dist_1_1, dist_1_2, dist_1_3, dist_1_4, dist_2_1, dist_2_2, dist_2_3, dist_2_4, dist_3_1, dist_3_2, dist_3_3, dist_3_4, dist_4_1, dist_4_2, dist_4_3, dist_4_4]

del distances
del dist_1_1
del dist_1_2
del dist_1_3
del dist_1_4
del dist_2_1
del dist_2_2
del dist_2_3
del dist_2_4
del dist_3_1
del dist_3_2
del dist_3_3
del dist_3_4
del dist_4_1
del dist_4_2
del dist_4_3
del dist_4_4

dr=rrange/nbin
edges = np.linspace(0,rrange,nbin+1)
xval=edges[1:]-0.5*(rrange/nbin)
volbin = []

for i in range(nbin):
    vol = ((4/3)*math.pi*(edges[i+1])**3)-((4/3)*math.pi*(edges[i])**3)
    volbin.append(vol)
   
g_r_all = np.zeros([16,nbin,n])
    
for i in range(n):
    for j in range(16):
        avenumden = (np.shape(pdist[j])[1]*np.shape(pdist[j])[2])/((box_size[i,0,1] - box_size[i,0,0])*(box_size[i,1,1] - box_size[i,1,0])*(box_size[i,2,1] - box_size[i,2,0]))
        g_r_his = np.histogram(pdist[j][i,:,:], bins=nbin, range = (0, rrange))[0]
        g_r = (g_r_his/volbin)/avenumden
        g_r[0] = 0
        g_r_all[j,:,i] = g_r

del g_r        

g_r_avg = np.zeros([16,nbin])
for i in range(16):
    g_r_avg[i,:] = np.average(g_r_all[i,:,:], axis = 1)

del g_r_all
def broad(dist, x, qrange):
    
    #Broad by using a convolution using a Gaussian function
    sub_xval = x[np.newaxis, :] - x[:, np.newaxis]
    add_xval = x[np.newaxis, :] + x[:, np.newaxis]
    
    FWHM = 5.437/qrange
    sigma = FWHM/2.355
    foubroad = np.zeros([1600,1600])
    foubroad =dist*(norm.pdf(sub_xval,0,sigma)-norm.pdf(add_xval,0,sigma))
    dist_broad=np.trapz(foubroad,x)
    return dist_broad

aveden = np.zeros(n)
for i in range(n):
    aveden[i] = (num_atoms)/((box_size[i,0,1] - box_size[i,0,0])*(box_size[i,1,1] - box_size[i,1,0])*(box_size[i,2,1] - box_size[i,2,0]))
aveden = np.average(aveden)

c_fq_product = sum([c[i] * fq[i] for i in range(nspecies)])
c_b = (np.sum(c_fq_product)**2) / 100

dividetot = 0
timesby = []
for i in range(nspecies):
    for j in range(nspecies):
        dividetot = dividetot + c[i]*c[j]*fq[i]*fq[j]
        timesby.append(c[i]*c[j]*fq[i]*fq[j])
    

if GRTOT == 1:
    
    gr_tot = np.zeros([nbin])
    for i in range(16):
        gr_tot = gr_tot + (timesby[i]*g_r_avg[i,:])/dividetot
    
    gr_tot_broad = broad(gr_tot, xval, 50)
    
if TR == 1:
    
    T_r = 4*math.pi*xval*aveden*gr_tot_broad*c_b


g_r_keen=(gr_tot_broad-1)*c_b
D_r = 4*math.pi*xval*aveden*g_r_keen


g_r_1_1_broad = broad(g_r_avg[0,:],xval, 12)
g_r_1_2_broad = broad(g_r_avg[1,:],xval, 12)
g_r_1_3_broad = broad(g_r_avg[2,:],xval, 12)
g_r_1_4_broad = broad(g_r_avg[3,:],xval, 12)
g_r_2_1_broad = broad(g_r_avg[4,:],xval, 12)
g_r_2_2_broad = broad(g_r_avg[5,:],xval, 12)
g_r_2_3_broad = broad(g_r_avg[6,:],xval, 12)
g_r_2_4_broad = broad(g_r_avg[7,:],xval, 12)
g_r_3_1_broad = broad(g_r_avg[8,:],xval, 12)
g_r_3_2_broad = broad(g_r_avg[9,:],xval, 12)
g_r_3_3_broad = broad(g_r_avg[10,:],xval, 12)
g_r_3_4_broad = broad(g_r_avg[11,:],xval, 12)
g_r_4_1_broad = broad(g_r_avg[12,:],xval, 12)
g_r_4_2_broad = broad(g_r_avg[13,:],xval, 12)
g_r_4_3_broad = broad(g_r_avg[14,:],xval, 12)
g_r_4_4_broad = broad(g_r_avg[15,:],xval, 12)


d_r_1_1_broad = 4*math.pi*xval*aveden*(g_r_1_1_broad-1)
d_r_1_2_broad = 4*math.pi*xval*aveden*(g_r_1_2_broad-1)
d_r_1_3_broad = 4*math.pi*xval*aveden*(g_r_1_3_broad-1)
d_r_1_4_broad = 4*math.pi*xval*aveden*(g_r_1_4_broad-1)
d_r_2_1_broad = 4*math.pi*xval*aveden*(g_r_2_1_broad-1)
d_r_2_2_broad = 4*math.pi*xval*aveden*(g_r_2_2_broad-1)
d_r_2_3_broad = 4*math.pi*xval*aveden*(g_r_2_3_broad-1)
d_r_2_4_broad = 4*math.pi*xval*aveden*(g_r_2_4_broad-1)
d_r_3_1_broad = 4*math.pi*xval*aveden*(g_r_3_1_broad-1)
d_r_3_2_broad = 4*math.pi*xval*aveden*(g_r_3_2_broad-1)
d_r_3_3_broad = 4*math.pi*xval*aveden*(g_r_3_3_broad-1)
d_r_3_4_broad = 4*math.pi*xval*aveden*(g_r_3_4_broad-1)
d_r_4_1_broad = 4*math.pi*xval*aveden*(g_r_4_1_broad-1)
d_r_4_2_broad = 4*math.pi*xval*aveden*(g_r_4_2_broad-1)
d_r_4_3_broad = 4*math.pi*xval*aveden*(g_r_4_3_broad-1)
d_r_4_4_broad = 4*math.pi*xval*aveden*(g_r_4_4_broad-1)


#SQ

if sq ==1:
    qrange = 25
    dq =0.025
        qval = np.linspace(0.0,qrange,1600)
    q_r = np.ones((np.shape(qval)[0],np.shape(xval)[0]))
    q_r = np.transpose(q_r*xval)*qval
    q_r = np.sin(q_r)/q_r
    
    A_q = np.ones(( np.shape(qval)[0],16,np.shape(xval)[0]))
    A_q = A_q * 4*math.pi*xval**2*(g_r_avg-1)*(np.sin(math.pi*xval/rrange)/(math.pi*xval/rrange))
    A_q = np.moveaxis(A_q,0,-1)*q_r
    
    S_q_ij= np.zeros((16,np.shape(qval)[0]))
    for i in range(16):
        S_q_ij[i,:] = 1 + aveden*np.trapz(np.transpose(A_q[i,:,:]),xval)
        
    S_q_tot = np.zeros(np.shape(qval)[0])
    for i in range(nspecies**2):
        S_q_tot = S_q_tot + (timesby[i]*S_q_ij[i,:])/dividetot
        if i == 0:
            S_q_O_O = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_O_O.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_O_O[i]}\n")
        elif i == 1:
            S_q_O_Al = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_O_Al.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_O_Al[i]}\n")        
        elif i == 2:
            S_q_O_P = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_O_P.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_O_P[i]}\n")           
        elif i == 3:
            S_q_O_Ca = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_O_Ca.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_O_Ca[i]}\n")      
        elif i == 4:
            S_q_Al_O = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_Al_O.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_Al_O[i]}\n")    
        elif i == 5:
            S_q_Al_Al = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_Al_Al.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_Al_Al[i]}\n")            
        elif i == 6:
            S_q_Al_P = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_Al_P.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_Al_P[i]}\n")
        elif i == 7:
            S_q_Al_Ca = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_Al_Ca.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_Al_Ca[i]}\n")  
        elif i == 8:
            S_q_P_O = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_P_O.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_P_O[i]}\n")  
        elif i == 9:
            S_q_P_Al = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_P_Al.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_P_Al[i]}\n")  
        elif i == 10:
            S_q_P_P = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_P_P.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_P_P[i]}\n")  
        elif i == 11:
            S_q_P_Ca = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_P_Ca.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_P_Ca[i]}\n")  
        elif i == 12:
            S_q_Ca_O = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_Ca_O.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_Ca_O[i]}\n")  
        elif i == 13:
            S_q_Ca_Al = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_Ca_Al.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_Ca_Al[i]}\n") 
        elif i == 14:
            S_q_Ca_P = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_Ca_P.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_Ca_P[i]}\n") 
        elif i == 15:
            S_q_Ca_Ca = timesby[i]*S_q_ij[i,:]/dividetot
            file = open(f"structure_factor_Ca_Ca.txt", "w")
            for i in range(len(qval)):
                qval[i] = "{:.2f}".format(qval[i])
                file.write(f"{qval[i]} {S_q_Ca_Ca[i]}\n") 
                
Print = np.transpose(np.vstack((S_q_tot,S_q_ij)))
np.savetxt('SQ.dat',Print)        
Print = np.transpose(np.vstack((S_q_tot,S_q_ij)))
np.savetxt('SQ_x_ray.dat',Print)


    
#%%




plt.plot(qval, S_q_tot,"r")
plt.xlim(0,25)

plt.xlabel("Q (Å$^-$$^1$))", size= 14)
plt.ylabel("S(Q)", size= 14)
#plt.legend(loc="upper right")
plt.show()
lenght_qval = len(qval)
file = open(f"structure_factor_x_ray.txt", "w")
for i in range(len(qval)):
    qval[i] = "{:.2f}".format(qval[i])
    file.write(f"{qval[i]} {S_q_tot[i]}\n")

#%%

shorten_xval = []
shorten_x_value = []
for i in xval:
    if i < 10:
        shorten_xval.append(i)
        



shorten_gr_tot_exp = []
shorten_gr_tot_broad = []
for index in range(len(gr_tot_broad)):
    if index < len(shorten_xval):
        shorten_gr_tot_broad.append(gr_tot_broad[index])

line1 = plt.plot(shorten_xval, shorten_gr_tot_broad, "r", label="Simulation")
plt.title("PDF Comparison between Experimental Data and VASP Simulation Result",
          fontdict = {'family':'serif','color':'blue','size':10})
plt.xlabel("r (Å)", fontdict = {'family':'serif','color':'blue','size':10})
plt.ylabel("G(r)", fontdict = {'family':'serif','color':'blue','size':10})
plt.legend(loc="upper right")


plt.show()

file = open(f"PDF_x_ray.txt", "w")
for i in range(len(shorten_xval)):
    shorten_xval[i] = "{:.5f}".format(shorten_xval[i])
    file.write(f"{shorten_xval[i]}\t{shorten_gr_tot_broad[i]}\n")
    

#%%

partials = {
    "O_O": g_r_1_1_broad,
    "O_Al": g_r_1_2_broad,
    "O_P": g_r_1_3_broad,
    "O_Ca": g_r_1_4_broad,
    "Al_O": g_r_2_1_broad,
    "Al_Al": g_r_2_2_broad,
    "Al_P": g_r_2_3_broad,
    "Al_Ca": g_r_2_4_broad,
    "P_O": g_r_3_1_broad,
    "P_Al": g_r_3_2_broad,
    "P_P": g_r_3_3_broad,
    "P_Ca": g_r_3_4_broad,
    "Ca_O": g_r_4_1_broad,
    "Ca_Al": g_r_4_2_broad,
    "Ca_P": g_r_4_3_broad,
    "Ca_Ca": g_r_4_4_broad,
}

for name, gr in partials.items():
    with open(f"PDF_{name}.txt", "w") as f:
        for i in range(len(xval)):
            f.write(f"{xval[i]:.5f}\t{gr[i]:.5f}\n")

