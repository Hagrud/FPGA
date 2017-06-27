#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pyopencl as cl
import os
import sys
import pyopencl.array as cl_array

def send_arrays_to_queue(q, arrays):
	ret = []
	for array in arrays:
		ret.append(cl_array.to_device(q, array))
	return ret

def get_devices_where_one(exts, dtype = cl.device_type.ALL):
	final = {}
	for p in cl.get_platforms():
		ds = []
		for d in p.get_devices(dtype):
			if(not d.get_info(cl.device_info.AVAILABLE)):
				continue
			extensions = d.get_info(cl.device_info.EXTENSIONS)
			for ext in exts:
				if ext in extensions:
					ds.append(d)
					break
		final[p] = ds
	return final

def get_devices_where_all(exts, dtype = cl.device_type.ALL):
	final = {}
	for p in cl.get_platforms():
		ds = []
		for d in p.get_devices(dtype):
			if(not d.get_info(cl.device_info.AVAILABLE)):
				continue
			b = True
			extensions = d.get_info(cl.device_info.EXTENSIONS)
			for ext in exts:
				if not(ext in extensions):
					b = False
					break
			if b:
				ds.append(d)
		final[p] = ds
	return final

def scan_platforms(sP, sD, sE):
	i = 0
	for p in cl.get_platforms():
		if(sP):
			print("[" + str(i)  + "] " + str(p))
		for d in p.get_devices():
			if(sD and sE):
				print("[" + str(i) + "] " + str(d) + 
					d.get_info(cl.device_info.EXTENSIONS))
			elif(sD):
				print("[" + str(i) + "]" + str(d))
		i+=1

def main():
	while(read(sys.argv)):
		None

def read(options):
	if(len(options) <= 1):
		return False
	opt = options.pop(1)
	if(opt[:5] == "-scan"):
		scan(opt[5:])
	elif(opt == "-f" or opt == "-fand"):
		None
	return True

def scan(option):
	sD = "d" in option
	sP = "p" in option
	sE = "e" in option
	scan_platforms(sP, sD, sE)	

if __name__ == "__main__":
	main()
