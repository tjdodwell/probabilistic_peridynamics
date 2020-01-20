////////////////////////////////////////////////////////////////////////////////
//
// opencl_enable_fp64.cl
//
// Safe method for enabling 64 bit floating point OpenCL extension (doubles)
//
// Copyright (c) Farshid Mossaiby, 2016, 2017
//
////////////////////////////////////////////////////////////////////////////////


// Try enabling the Khronos version of FP64

#ifdef cl_khr_fp64

	#pragma OPENCL EXTENSION cl_khr_fp64: enable

// Extension not available, try AMD specific FP64

#else

	#ifdef cl_amd_fp64

		#pragma OPENCL EXTENSION cl_amd_fp64: enable
	
	#else

// None of the extensions were available
	
		#error "opencl_enable_fp64.cl: cannot enable cl_amd_fp64 or cl_khr_fp64. FP64 is not available on this hardware."
		
	#endif
	
#endif
