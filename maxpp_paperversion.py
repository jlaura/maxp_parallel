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

def initialize(job,z,w,neighborsdict,floor,floor_variable,numP,cores,maxattempts=100,threshold=0):
    #Get the identity of the current process (core).
    #current = mp.current_process()
    #print "Current process ID: ", current._identity[0]
    #This should be a low overhead for loop.
    
    def iterate_to_p(job,z,w,neighborsdict,floor,floor_variable,numP,cores,maxattempts, threshold):
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
            if len(regions) >= threshold:
                attempts += 1
                yield regions
            else:
                attempts += 1

    for regions in iterate_to_p(job,z,w,neighborsdict,floor,floor_variable,numP,cores, 100, 0): 
        p = len(regions)
        #Here we check the solution against the current solution space.
        check_soln(regions, numP,cores,w,z)
     

    #Now we need to standardize the max number of regions between all IFS.
    sharedSoln = np.frombuffer(cSoln.get_obj())
    current_max_p = sharedSoln[0].max()
    print "IFS with vaired p generated.  Attempting to standardize to p=%i" %current_max_p  
    for regions in iterate_to_p(job,z,w,neighborsdict,floor,floor_variable,numP,cores,100,current_max_p):
        p=len(regions)
        check_soln(regions, numP, cores, w,z, unique=True)
    
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
    print column, wss
                
    #print column, wss, len(enclaves[0])
        #Then move the enclave to the next neighbor's group and test variance
        #If the variance is lower in the second group, keep that group and test the next
        #else, retain the first group and test the next
            
            #Now that I know the neighbors to the enclave I want to test each possability
            # and assign to the one with the lowest variance. Punking out with a for loop
            # for today
            
            
            
            #neighbors = [neighbor for neighbor in neighbors if neighbor not in enclaves]
            #candidates = []
            #for neighbor in neighbors:
                #region = a2r[neighbor]
                #if region not in candidates:
                    #candidates.append(region)
            #if candidates:
                ## add enclave to random region
                #regID = random.randint(0, len(candidates) - 1)
                #rid = candidates[regID]
                #regions[rid].append(enclave)
                #a2r[enclave] = rid
                ## structure to loop over enclaves until no more joining is possible
                #encCount = len(enclaves)
                #encAttempts = 0
                #feasible = True
            #else:
                ## put back on que, no contiguous regions yet
                #enclaves.append(enclave)
                #encAttempts += 1
                #feasible = False
 
def check_soln(regions,numP,cores,w,z, unique=False): #check the solution for min z
    '''This function queries the current IFS space to see if the currently computed soln is better than all other solns.'''
    
    def _regions_to_array(regions, newSoln):
        regionid = 0
        for region in regions:
            for member in region:
                newSoln[member] = regionid
            regionid += 1    
    
    sharedSoln = np.frombuffer(cSoln.get_obj())
    sharedSoln.shape = (numP,cores)
    if unique == False:
        if len(regions) >= sharedSoln[0].min(): #If any of the indices are less than p
            cSoln.get_lock()#Lock the entire shared memory array while we alter it
            column = np.argmin(sharedSoln[0]) #Get the index of the min value
            sharedSoln[0][column] = len(regions)#Write p to index 0
            newSoln = sharedSoln[1:,column] #Get a slice of the array,skipping index 0 that is the p counter
            newSoln[:] = -1 #Empty the column to be written
            _regions_to_array(regions, newSoln) #Iterate over the regions and assign their membership into the soln space 
    else:
        #Check that the soln is not identical to an existing soln
        temp_array = np.zeros(numP-1)
        temp_array[:] =-1
        regionid = 0
        for region in regions:
            for member in region:
                temp_array[member] = regionid
            regionid += 1            
        cSoln.get_lock()
        for column in range(sharedSoln.shape[1]):
            if not np.array_equal(temp_array, sharedSoln[1:,column]):
                sharedSoln[1:,column] = temp_array
                sharedSoln[0:,column][0] = len(regions)
                break                
                
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

#Phase Ib - Assign enclaves
jobs = []
for column_num in range(sharedSoln.shape[1]):
    proc = mp.Process(target=assign_enclaves, args=(column_num, z[:,0], neighbordict))
    jobs.append(proc)
for job in jobs:
    job.start()
for job in jobs:
    job.join()
del jobs[:], proc, job
print sharedSoln

#Phase II
#Step I - Compute the current value for each soln.
#jobs = []
#for column in range(sharedSoln.shape[1]):
    #p = mp.Process(target=objective_function_vec, args=(column,z[:,0]))
    #jobs.append(p)

#for job in jobs:
    #job.start()
#for job in jobs:
    #job.join()
#del jobs[:], p, job
#print sharedSoln

#Step II - Initiate Swapping
#Each core is going to get one answer and start to iterate.


