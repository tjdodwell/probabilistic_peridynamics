// This is a naive subtract kernel. 
//TODO investigate using a single iterator. Then each iteration has a separate thread.
__kernel void subtract(
	const int N,
	__global float* A,
	__global float* B,
	__global float* C)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    float tmp = 0;
    if ((i < N) && (j < N))
    {
         C[i*N +j] = A[i*N + j] - B[i*N + j];
             
    }
}


// Tiles a vector of length N into a square matrix of size (N, N)
// TODO to be completed
__kernel void tile(
	const int N,
	__global float* A,
	__global float* B,)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    float tmp = 0;
    if ((i < N) && (j < N))
    {
         C[i*N +j] = A[i*N + j] - B[i*N + j];
             
    }
}

// This naive transpose kernel suffers from completely non-coalesced writes.
// It can be up to 10x slower than the kernel above for large matrices.
__kernel void transpose_naive(__global float *odata, __global float* idata, int offset, int width, int height)
{
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);
    
    if (xIndex + offset < width && yIndex < height)
    {
        unsigned int index_in  = xIndex + offset + width * yIndex;
        unsigned int index_out = yIndex + height * xIndex;
        odata[index_out] = idata[index_in]; 
    }
}

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.

#define BLOCK_DIM 16

__kernel void transpose(
		__global float *odata, 
		__global float *idata, 
		int offset, 
		int width, 
		int height, 
		__local float* block)
{
	// read the matrix tile into shared memory
	unsigned int xIndex = get_global_id(0);
	unsigned int yIndex = get_global_id(1);

	if((xIndex + offset < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex + offset;
		block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)] = idata[index_in];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// write the transposed matrix tile to global memory
	xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
	yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
	if((xIndex < height) && (yIndex + offset < width))
    {
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[get_local_id(0)*(BLOCK_DIM+1)+get_local_id(1)];
	}