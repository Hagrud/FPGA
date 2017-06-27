#AMD gpu does not handle double precision.
cl_type_double="""
#define _TYPE_ double
"""

cl_type="""
#ifndef _TYPE_
#define _TYPE_ float
#endif
"""

cl_add=cl_type + """
kernel void add(global const _TYPE_ *a, 
		global const _TYPE_ *b,
		global _TYPE_ *c){
	size_t i = get_global_id(0);
	c[i] = a[i] + b[i];
}
"""

cl_mult=cl_type + """
kernel void mult(global const _TYPE_ *a,
		 global const _TYPE_ *b,
		 global _TYPE_ *c,
	         global const int *sX, global const int *sY){
	size_t id = get_global_id(0);
	size_t i = id%sX[0];
	size_t j = id/sX[0];
	float l = 0;
	for(int k = 0; k < sY[0]; k++){
		float e1 = a[j*sY[0] + k];
		float e2 = b[i + k*sX[0]];
		l += e1*e2;
	}
	
	c[id] = l;
}
"""

cl_tran=cl_type + """
kernel void tran(global const _TYPE_ *in,
		 global _TYPE_ *out){
}
"""

cl_mandel=cl_type + """
kernel void mandel(global const _TYPE_ q*,
	           global const int max_iter*,
		   global int *out){
	size_t id = get_global_id(0);

	_TYPE_ nreal, real, imag = 0;

	for(int iter = 0; iter < max_iter; iter++){
		nreal = real*real - imag*imag + q[2*id];
		imag = real*imag + q[2*id+1];
		real = nreal;

		if(real*real + imag*imag > 4){
			out[id] = iter;
		} 
	}
}
"""
