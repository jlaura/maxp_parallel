maxp_parallel
=============
1.12.13

1. python maxpp.py to test the implementation.  This is now populating the shared solution space with locking working.  Great to learn more about ctypes, multiprocessing.Array, and how to get locks working.  All the FJ Algorithm slicing work is coming in handy...lot less time messing with syntax.

2. Multicore initialization appears to be working.  We are currently printing the number of regions found, the core that found the region, and the iteration number.  It appears that the algorithm is finding solutions on all attempts.  Does this make sense, in the context of an 10x10 weights array where the constrains are (1) contiguity and (2) minimum membership (floor) of 3.  I think so.
	
3. Some error is occurring that is causing one or more of the processes to return an error.  The child process does not return a meaningful stacktrace.  In fact, the only reason that I am asserting an error is occurring is because of the extra printed material at the start of processing.  To duplicate, start a new tab, run the script with at least 1500 lines of scroll back and observe the dict at the top.

Dummy edit to test.
