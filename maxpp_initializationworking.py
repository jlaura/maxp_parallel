#MaxP Regions Multi

import pysal
import copy
import random
import numpy as np
#from pysal.common import *
from pysal.region import randomregion as RR

#New imports
import multiprocessing as mp
import ctypes

#Testing imports
import time

LARGE = 10 ** 6

def initialize(job,z,w,neighborsdict,floor,floor_variable,maxattempts=100):
    #Get the identity of the current process (core).
    current = mp.current_process()
    #print "Current process ID: ", current._identity[0]
    #This should be a low overhead for loop.
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
                        #print region
                        enclaves.extend(region)
                        building_region = False
                        #check to see if any regions were made before going to enclave stage
                    
        if regions:
            feasible = True
        else:
            attempts += 1
            break

        enclaves = enclaves[:]
        a2r = {}
        for r, region in enumerate(regions):
            for area in region:
                a2r[area] = r
        encCount = len(enclaves)
        encAttempts = 0
        while enclaves and encAttempts != encCount:
            enclave = enclaves.pop(0)
            neighbors = neighborsdict[enclave]
            neighbors = [neighbor for neighbor in neighbors if neighbor not in enclaves]
            candidates = []
            for neighbor in neighbors:
                region = a2r[neighbor]
                if region not in candidates:
                    candidates.append(region)
            if candidates:
                # add enclave to random region
                regID = random.randint(0, len(candidates) - 1)
                rid = candidates[regID]
                regions[rid].append(enclave)
                a2r[enclave] = rid
                # structure to loop over enclaves until no more joining is possible
                encCount = len(enclaves)
                encAttempts = 0
                feasible = True
            else:
                # put back on que, no contiguous regions yet
                enclaves.append(enclave)
                encAttempts += 1
                feasible = False
        if feasible:
            regions = regions
            area2region = a2r
            p = len(regions)
            print "Feasible soln found with %i regions by core %i on attempt %i" %(p, current._identity[0], attempts)
            attempts += 1
        else:
            if attempts == MAX_ATTEMPTS:
                print 'No initial solution found'
                p = 0
            attempts += 1

def check_floor(region,floor_variable,w):
    selectionIDs = [w.id_order.index(i) for i in region]
    cv = sum(floor_variable[selectionIDs]) #TODO: FloorVariable needs to be defined.
    if cv >= floor:
        return True
    else:
        return False


def initSolnSpace(numSoln_):
    '''Initialize the ctypes arrays as global variables for multiprocessing.

    This is the shared soln space.
    '''
    global sharedSoln
    sharedSoln = numSoln_    

'''This is a multi-phase algorithm.  Step 1: We need to generate IFS  We are going to attempt to generate 1 IFS for each core and add the to a shared memory space.  Each core has MAX_ATTEMPTS to compute an IFS.  At complettion we load n IFS into a shared memory space. '''


#Setup the test data:
w = pysal.lat2W(10, 10) #A contiguity weights object
z = np.random.random_sample((w.n, 2)) #Each local is assigned two attributes
p = np.ones((w.n, 1), float) #The region that each location belongs to.
floor = 3 #The minimum bound or value for each region

#1. Compute an initial feasible solution

'''Setup the shared memory space.  For each IFS I need to store w,p
z remains static
p is easy, as the length is static and the integers change with membership in a region
    p should be n-dimensional, where each dimension is an IFS.  We can use the core's
    pid to assign a core to a dimension
w is easy, in that the contiquity of individual polygons does not change; what changes is
    the contiguity of a region.'''

#Grab the number of available core
cores = mp.cpu_count()
cores = cores *2 #Hyperthreading and testing on a dual core MBP
#Grab the length of p
numP = len(p)
#Setup a shared mem array for solutions with dim = numP * cores
cSoln = mp.RawArray(ctypes.c_double, numP*cores)
numSoln = np.frombuffer(cSoln)
numSoln.shape = (numP,cores)
numSoln[:] = 1
initSolnSpace(numSoln) #initialize the solution space as a shared memory array
'''The soln space is an array that holds node id as the index and membership as the attribute.'''

#We have 4 slots now for IFS.  What I want to do is write IFS in order of 'goodness'.  When a core returns an IFS we have a check function that the core runs.  The IFS is then entered into the appropriate dimension of the solution space.

neighbordict = dict(w.neighbors) #This is interesting - we can not pass a class instance through apply_async and need to conver to a dict.
pool = mp.Pool(processes=cores) #Create a pool of workers, one for each core
for job in range(cores): #Prep to assign each core a job
    pool.apply_async(initialize, args=(job,z,w,neighbordict,floor,p,)) #Async apply each job
pool.close()
pool.join()