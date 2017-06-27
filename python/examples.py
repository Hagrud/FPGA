import pyopencl as cl
import pyopencl.array as cl_array
import raw_code as rc
import tool
import numpy
import time

def get_double_precision_gpu():
	return get_double_precision(cl.device_type.GPU)

def get_double_precision(dtype = cl.device_type.ALL):
	return tool.get_devices_where_one(["cl_khr_fp64", "cl_amd_fp64"], dtype)

def get_gpu():
	return tool.get_devices_where_all([], cl.device_type.GPU)

def get_one_context(dic):
	for key in dic.keys():
		if len(dic[key]) != 0:
			return cl.Context(dic[key]), dic[key]
	raise NameError('No compute unit availables.')	

def Mat_mult_gpu(a = numpy.random.rand(5000*5000).astype(numpy.float32),
		 b = numpy.random.rand(5000*5000).astype(numpy.float32), shape = [5000, 5000]):
	ctx, devices = get_one_context(get_gpu())
	kernel = rc.cl_mult
	return Mat_mult(ctx, devices[0], kernel, a, b, shape)

def Mat_mult_gpu_double(a = numpy.random.rand(5000*5000).astype(numpy.double),
			b = numpy.random.rand(5000*5000).astype(numpy.double), shape = [5000, 5000]):
	ctx, devices = get_one_context(get_double_precision_gpu())
	kernel = rc.cl_type_double + rc.cl_mult
	return Mat_mult(ctx, devices[0], kernel, a, b, shape)

def Mat_mult_double(	a = numpy.random.rand(5000*5000).astype(numpy.double),
			b = numpy.random.rand(5000*5000).astype(numpy.double), shape = [5000, 5000]):
	ctx, devices = get_one_context(get_double_precision())
	kernel = rc.cl_type_double + rc.cl_mult
	return Mat_mult(ctx, devices[0], kernel, a, b, shape)

def Vec_add_gpu(	a = numpy.random.rand(5000*5000).astype(numpy.float32),
			b = numpy.random.rand(5000*5000).astype(numpy.float32), shape = [5000, 5000]):
	ctx, devices = get_one_context(get_gpu())
	kernel = rc.cl_add
	return Vec_add(ctx, devices[0], kernel, a, b)

def Vec_add_gpu_double(	a = numpy.random.rand(5000*5000).astype(numpy.float32),
			b = numpy.random.rand(5000*5000).astype(numpy.float32), shape = [5000, 5000]):
	ctx, devices = get_one_context(get_double_precision_gpu())
	kernel = rc.cl_type_double + rc.cl_add
	return Vec_add(ctx, devices[0], kernel, a, b)

def Vec_add_double( 	a = numpy.random.rand(5000*5000).astype(numpy.float32),
			b = numpy.random.rand(5000*5000).astype(numpy.float32), shape = [5000, 5000]):
	ctx, devices = get_one_context(get_double_precision())
	kernel = rc.cl_type_double + rc.cl_add
	return Vec_add(ctx, devices[0], kernel, a, b)
	
	None
	
def Vec_add(ctx, device, kernel, a, b):
		#create the queue
	q = cl.CommandQueue(ctx, device)

		#send data to the device.
	a_dev, b_dev = tool.send_arrays_to_queue(a, (a, b))
	c_dev = cl_array.empty_like(a_dev)

		#Build the kernel
	prg = cl.Program(ctx, kernel).build()
	
		#Launch the computatiom
	t = time.time()
	prg.add(q, a.shape, None, a_dev.data, b_dev.data, c_dev.data)

        return a_dev, b_dev, c_dev, t

def Mat_mult(ctx, device, kernel, a, b, shape):

		#create the queue
	q = cl.CommandQueue(ctx, device)

		#send data to the device.
	a_dev, b_dev, s_1, s_2 = tool.send_arrays_to_queue(q, (a,b, numpy.array(shape[0], dtype=int),
			    					    numpy.array(shape[1], dtype=int)))
	c_dev = cl_array.zeros(q, (shape[0]*shape[0]), a_dev.dtype)

		#Build the kernel
	prg = cl.Program(ctx, kernel).build()
	
		#Launch the computation
	t = time.time()
	prg.mult(q, [shape[0]*shape[0]], None, a_dev.data, b_dev.data, c_dev.data, s_1.data, s_2.data)
	return a_dev, b_dev, c_dev, t

