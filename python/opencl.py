#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pyopencl as cl
import raw_code as rc
import tool

import pyopencl.array as cl_array
import numpy as np
import numpy.linalg as la
import os
import time
import thread
import examples

SA = 5000
SB = 5000

#uncomment this line to debug opencl compilation
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

	#run a matrix multiplication on a gpu
print(examples.Mat_mult_gpu(
	np.random.rand(3000*2000).astype(np.float32), 
	np.random.rand(2000*3000).astype(np.float32), 
	[3000,2000]))
	#run a matrix multiplication on a core with double precision
print(examples.Mat_mult_double());
	#will crash because there is no gpu with double precision on that machine
print(examples.Mat_mult_gpu_double());
