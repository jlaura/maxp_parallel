#MaxP Regions Multi

import pysal
import copy
import random
import numpy as np
#from pysal.common import *
from pysal.region import randomregion as RR

#New imports
import multiprocessing as mp
from multiprocessing.sharedctypes import Array
import ctypes

#Testing
import time

def initialize(job,z,w,neighborsdict,floor,floor_variable,numP,cores,maxattempts=100,threshold=0,suboptimal=None):
    
    def iterate_to_p(job,z,w,neighborsdict,floor,floor_variable,numP,cores, maxattempts, threshold,suboptimal):
        solving = True 
        attempts = 0
        while solving and attempts <= maxattempts:
            regions = []
            enclaves = []
            seeds = [] #What are seeds doing?
            if not seeds:
                #If seeds is empty we need to generate a random weights list
                candidates = copy.copy(w.id_order)
                candidates = np.random.permutation(candidates)
                candidates = candidates.tolist()  
            else:
                seeds = copy.copy(seeds)
                nonseeds = [i for i in w.id_order if i not in seeds]
                candidates = seeds
                candidates.extend(nonseeds)
            while candidates:#Here we pick a random starting seed from all available candidates (not already in a region)
                seed = candidates.pop(0)
                region = [seed]
                building_region = True
                while building_region:
                    #check if floor is satisfied
                    if check_floor(region,floor_variable,w): #If this returns true, the region satisifies the floor constraint is 'completed'.  
                        regions.append(region)
                        building_region = False 
                    else: #Else - we are under the floor and must add the 'best' region.
                        potential = []
                        for area in region: #what are the neighbors of the current region
                            neighbors = neighborsdict[area]
                            neighbors = [neigh for neigh in neighbors if neigh in candidates]
                            neighbors = [neigh for neigh in neighbors if neigh not in region]
                            neighbors = [neigh for neigh in neighbors if neigh not in potential]
                            potential.extend(neighbors)
                        if potential:
                            # add a random neighbor
                            neigID = random.randint(0, len(potential) - 1)
                            neigAdd = potential.pop(neigID)
                            region.append(neigAdd)
                            # remove it from candidates
                            candidates.remove(neigAdd)
                        else:
                            #print 'enclave'
                            enclaves.extend(region)
                            building_region = False
                            #check to see if any regions were made before going to enclave stage
            if threshold == 0:
                if len(regions) >= threshold:
                    attempts += 1
                    yield regions
                else:
                    attempts += 1
            else:#Here we standardize the answers without limit to number of iterations.  Will that work?
                if len(suboptimal) == 0:
                    break
                if len(regions) == threshold:
                    suboptimal.pop()
                    yield regions              

    for regions in iterate_to_p(job,z,w,neighborsdict,floor,floor_variable,numP,cores, maxattempts, threshold,suboptimal): 
        check_soln(regions, numP,cores,w,z)
        
    
def assign_enclaves(column, z, neighbordict):
    #Remember - the soln space stores membership by index.
    enclaves = np.where(sharedSoln[1:,column] == -1)#Returns a tuple of unassigned enclaves
    for enclave in enclaves[0]:#Iterate over the enclaves
        neighbors = neighbordict[enclave]
        #Iterate over the neighbors to the enclaves
        wss=float('inf')
        for neighbor in neighbors:
            #Now I need to know what groups the neighbor is in.
            group = sharedSoln[1:,column][neighbor]
            if group == -1: #Because we could assign an enclave to another enclave, fail the floor test that we do not perform again, and have a low variance...pain to debug this guy!
                break
            #Then add the enclave to that neighbor and test the variance
            sharedSoln[1:,column][enclave] = group
            new_wss = objective_function_vec(column, z)
            if new_wss < wss:
                wss = new_wss
    #Replace the p count with the wss, we can get at p whenever later with np.unique(p)
    sharedSoln[:,column][0] = wss
 
def check_soln(regions,numP,cores,w,z): #check the solution for min z
    '''This function queries the current IFS space to see if the currently computed soln is better than all other solns.'''
    
    def _regions_to_array(regions, newSoln):
        regionid = 0
        for region in regions:
            for member in region:
                newSoln[member] = regionid
            regionid += 1    
    
    sharedSoln = np.frombuffer(cSoln.get_obj())
    sharedSoln.shape = (numP,cores)
    if len(regions) >= sharedSoln[0].min(): #If any of the indices are less than p
        cSoln.get_lock()#Lock the entire shared memory array while we alter it
        column = np.argmin(sharedSoln[0]) #Get the index of the min value
        sharedSoln[0][column] = len(regions)#Write p to index 0
        newSoln = sharedSoln[1:,column] #Get a slice of the array,skipping index 0 that is the p counter
        newSoln[:] = -1 #Empty the column to be written
        _regions_to_array(regions, newSoln) #Iterate over the regions and assign their membership into the soln space               
                
def check_floor(region,floor_variable,w):
    selectionIDs = [w.id_order.index(i) for i in region]
    cv = np.sum(floor_variable[selectionIDs]) #TODO: FloorVariable needs to be defined.
    if cv >= floor:
        return True
    else:
        return False 
    
def objective_function_vec(column,attribute_vector):
    '''
    This is an objection function checker designed to access the 
    shared memory space.  It is suggested that this is faster than vectorization
    because we do not have to initialize additional in memory temp arrays.
    
    Parameters
    ----------
    z        :ndarray
              An array of attributes for each polygon
              
    Returns
    -------
    None     :-
              This writes the objective function to the 0 index of the sharedmem
              space, and overwrites sum(p) from the initialization phase.
    '''
    groups = sharedSoln[1:,column]
    wss = 0
    for group in np.unique(groups):
        #print group, attribute[groups==group]
        wss+= np.sum(np.var(attribute_vector[groups == group]))
    return wss
    #sharedSoln[0:,column][0] = wss

def standardize(current_p):
    feasible = []
    for column in sharedSoln.T:
        if column[0] == current_p:
            feasible.append(column)
    print feasible
        
                                      
#Setup the test data:
w = pysal.lat2W(10, 10)
z = np.random.random_sample((w.n, 2))
p = np.ones((w.n, 1), float) 
floor_variable = p
floor = 3

#Multiprocessing and shared memory initialization
cores = mp.cpu_count()
cores = cores * 2 #Hyperthreading - Can we check?
numP = len(p)+1
lock = mp.Lock()
cSoln = Array(ctypes.c_double, numP*cores, lock=lock)
numSoln = np.frombuffer(cSoln.get_obj())
numSoln.shape = (numP,cores)
numSoln[:] = -1

neighbordict = dict(w.neighbors)#We have to pass something pickable, not a class instance of W

#Phase Ia - Initialize a nubmer of IFS equal to the number of cores
jobs = []
for core in range(0,cores):
    proc = mp.Process(target=initialize, args=(core,z,w,neighbordict,floor,floor_variable,numP, cores))
    jobs.append(proc)
for job in jobs:
    job.start()
for job in jobs:
    job.join()
del jobs[:], proc, job

sharedSoln = np.frombuffer(cSoln.get_obj())
sharedSoln.shape = (numP, cores)
if sharedSoln.all() == -1: 
    print "No initial feasible solutions found. Perhaps increase the number of iterations?"
    sys.exit(0)
#print sharedSoln

#Phase Ib - Standardize the values.
current_max_p = sharedSoln[0].max()
suboptimal = np.where(sharedSoln[0] < current_max_p)[0]
if suboptimal.size == 0:
    print "Solutions standardized, assigning enclaves"
else:
    manager = mp.Manager() #Create a manager to manage the coutdown
    suboptimal_countdown = manager.list(suboptimal)
    print "IFS with vaired p generated.  Standardizing to p=%i." %current_max_p
    jobs = []
    for core in range(0,cores):
        proc = mp.Process(target=initialize, args=(core,z,w,neighbordict,floor,floor_variable,numP,cores, 1, current_max_p,suboptimal_countdown))
        
        jobs.append(proc)
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()
    del jobs[:], proc, job        
#print sharedSoln
    
#Phase Ic - Assign enclaves
jobs = []
for column_num in range(sharedSoln.shape[1]):
    proc = mp.Process(target=assign_enclaves, args=(column_num, z[:,0], neighbordict))
    jobs.append(proc)
for job in jobs:
    job.start()
for job in jobs:
    job.join()
del jobs[:], proc, job
#print sharedSoln

#Phase Id - Set 50% soln to best current soln to favor current best at initialization of Phase II.
num_top_half = (cores // 2) - 1 #Get the whole number of cores 
current_best = np.argmin(sharedSoln[0]) ; current_best_value = np.min(sharedSoln[0])
print current_best, current_best_value
for soln in range(num_top_half):
    replace = np.argmax(sharedSoln[0])
    sharedSoln[:,replace] = sharedSoln[:,current_best]
    
print sharedSoln
