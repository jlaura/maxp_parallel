import time
import numpy as np
from random import randint, uniform
from numpy.random import RandomState
import pysal

'''Test Data Generation a la PySAL tests.'''                                      
#Setup the test data:
w = pysal.lat2W(10, 10)
random_init = RandomState(123456789)
z = random_init.random_sample((w.n, 1))
#print z.max(), z.min(), z.std()
p = np.ones((w.n, 1), float) 
floor_variable = p
floor = 3

'''START TIMING HERE - AFTER TEST DATA GENERATED'''
time0 = time.time()

solution = pysal.region.Maxp(w, z, floor, floor_variable, initial=100)
time1 = time.time()

wss = solution.objective_function(solution.regions)

print "Regions: ", solution.p
print "Total processing time: ", time1 - time0
print "Solution: ", wss