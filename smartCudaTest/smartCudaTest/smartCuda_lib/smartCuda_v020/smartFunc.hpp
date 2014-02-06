/*
 *  SMART CUDA 0.2.0 (18th Jan, 2014) - Initial Release
 *
 *
 *  Copyright 2014 by Mark Amo-Boateng (smartcuda@outlook.com)
 *
 *  For tutorials and more information, go to:
 *  http://markamo.github.io/Smart-Cuda/
 *
 *
 */


#pragma once 

#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "..\smartCuda_020.h"

////CORE FUNCTION POINTERS /////
////simple functions for reduction /////
//template <typename T> inline
//__host__ __device__ T smartPlus ( T lhs, T rhs ) { return lhs + rhs; }
//
//template <typename T> inline
//__host__ __device__ T smartMinus ( T lhs, T rhs ) { return lhs - rhs; }
//
//template <typename T> inline
//__host__ __device__ T smartMax ( T lhs, T rhs ) { return (lhs > rhs) ? lhs : rhs; }
//
//template <typename T> inline
//__host__ __device__ T smartMin ( T lhs, T rhs ) { return (lhs < rhs) ? lhs : rhs; }
//
//template <typename T> inline
//__host__ __device__ T smartAve ( T lhs, T rhs ) { return (lhs + rhs) / (T) 2; }

#define REDUCTION_INIT_NONE -1
#define REDUCTION_INIT_ALL_ZEROS 0
#define REDUCTION_INIT_ALL_ONES 1
#define REDUCTION_INIT_FIRST 10
#define REDUCTION_INIT_FIRST_ALL 11
#define REDUCTION_INIT_ALL 22

////#define REDUCTION_INIT_CUSTOM 2

template<typename T>
class smartOperator
{
protected:
	int init_type;

public:
	__host__ __device__
	smartOperator(int _init_type = REDUCTION_INIT_NONE)
	{
		init_type = _init_type;
	}

	__host__ __device__
	int reduction_init_type()
	{
		return init_type;
	}

	__host__ __device__ T operator()();

	//__host__ __device__ T post_op(T results, int size){return results;};
};

template <typename T> 
class smartPlus: public smartOperator<T>
{
public: 
	__host__ __device__
		smartPlus() : smartOperator(REDUCTION_INIT_ALL_ZEROS){;};
	
	__host__ __device__ T operator() ( T lhs, T rhs )  { return lhs + rhs; }

};

template <typename T> 
class smartMultiply: public smartOperator<T>
{
public: 
	__host__ __device__
	smartMultiply() : smartOperator(REDUCTION_INIT_ALL_ONES){;};
	
	__host__ __device__ T operator() ( T lhs, T rhs )  { return lhs * rhs; }

};


template <typename T> 
class smartMax: public smartOperator<T>
{
public: 
	__host__ __device__
	smartMax() : smartOperator(REDUCTION_INIT_FIRST_ALL){;};
	
	__host__ __device__ T operator() ( T lhs, T rhs )  { return (lhs > rhs) ? lhs : rhs; }
};

template <typename T> 
class smartMin: public smartOperator<T>
{
public: 
	__host__ __device__
	smartMin() : smartOperator(REDUCTION_INIT_FIRST_ALL){;};
	
	__host__ __device__ T operator() ( T lhs, T rhs ) { return (lhs < rhs) ? lhs : rhs; }
};

template <typename T> 
class smartOR: public smartOperator<T>
{
public: 
	__host__ __device__
	smartOR() : smartOperator(REDUCTION_INIT_FIRST){;};
	
	__host__ __device__ T operator() ( T lhs, T rhs ) { return lhs || rhs; }
};


template <typename T> 
class smartAND: public smartOperator<T>
{
public: 
	__host__ __device__
	smartAND() : smartOperator(REDUCTION_INIT_FIRST){;};
	
	__host__ __device__ T operator() ( T lhs, T rhs ) { return lhs || rhs; }
};



//////other operators: not to be used for reduction 'cos not associative  and mutual ///////
template <typename T> 
class smartMinus: public smartOperator<T>
{
public: 
	__host__ __device__
		smartMinus() : smartOperator(REDUCTION_INIT_ALL_ZEROS){;};
	
	__host__ __device__ T operator() ( T lhs, T rhs )  { return lhs - rhs; }

};

/////Beta test /////
template <typename T> 
class smartAverage: public smartOperator<T>
{
private: 
	size_t size;
public: 
	__host__ __device__
	smartAverage() : smartOperator(REDUCTION_INIT_FIRST){;};

	//__host__ __device__ T operator() ( T lhs, T rhs ) { return ((lhs + rhs) ); }  ///// total/2 = total>>1;

	//__host__ __device__ T operator() (T results, bool post = false, T _size = 0){return (!post) ? results / (T)_size : results; };

	__host__ __device__ T operator() ( T lhs, T rhs ) { return ((lhs + rhs) /(T) 2 ); }  ///// total/2 = total>>1;

	//__host__ __device__
	//smartAverage(size_t _size) : smartOperator(REDUCTION_INIT_FIRST){size = _size;};
	//__host__ __device__ T operator() ( T lhs, T rhs ) { return ((lhs + rhs) /(T) _size ); }  ///// total/2 = total>>1;

	//__host__ __device__ T post_op(T results, int _size){return results / (T)(_size); };
};


////future implementations ////
//scan, sort, count, etc
//template <typename T> inline
//__host__ __device__ T smartCnt ( T lhs, T rhs ) {  return (lhs == rhs) ? lhs : rhs; }


////BASIC FUNCTION APPLICATION////
/////function pointer version////
template <typename T> __global__
 void appy_func_core(T* dev_Array, const int size, T fn ( T x ) )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		dev_Array[i] = fn(dev_Array[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T> __global__
 void appy_func_core(T* dest, T* src, const int size, T fn ( T x ) )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		dest[i] = fn(src[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}


template <typename T> __global__
 void appy_func_core(T* dest, T* src1, T* src2, const int size, T fn ( T x1, T x2 ) )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		dest[i] = fn(src1[i], src2[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

////operator overload version///

template <typename T, class Op> __global__
 void appy_func_core(T* dev_Array, const int size, Op fn)
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		dev_Array[i] = fn(dev_Array[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T, class Op> __global__
 void appy_func_core(T* dest, T* src, const int size, Op fn)
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		dest[i] = fn(src[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}


template <typename T, class Op> __global__
 void appy_func_core(T* dest, T* src1, T* src2, const int size, Op fn)
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		dest[i] = fn(src1[i], src2[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}


////TRANSFORMATIONS ////////
////typed transformations
template <typename T, class Op> __global__
 void transfrom_core(T* arr, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T, class Op> __global__
 void transfrom_core(T* arr, T* arr1, T* arr2, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T, class Op> __global__
 void transfrom_core(T* arr, T* arr1, T* arr2, T* arr3, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T, class Op> __global__
 void transfrom_core(T* arr, T* arr1, T* arr2, T* arr3, T* arr4, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i], arr4[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T, class Op> __global__
 void transfrom_core(T* arr, T* arr1, T* arr2, T* arr3, T* arr4,  T* arr5, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i], arr4[i], arr5[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T, class Op> __global__
 void transfrom_core(T* arr, T* arr1, T* arr2, T* arr3, T* arr4,  T* arr5,  T* arr6, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T, class Op> __global__
 void transfrom_core(T* arr, T* arr1, T* arr2, T* arr3, T* arr4,  T* arr5,  T* arr6, T* arr7, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i], arr7[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}


template <typename T, class Op> __global__
 void transfrom_core(T* arr, T* arr1, T* arr2, T* arr3, T* arr4,  T* arr5,  T* arr6, T* arr7, T* arr8, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i], arr7[i], arr8[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}


template <typename T, class Op> __global__
 void transfrom_core(T* arr, T* arr1, T* arr2, T* arr3, T* arr4,  T* arr5,  T* arr6, T* arr7, T* arr8, T* arr9, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i], arr7[i], arr8[i], arr9[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}


////typed transformations
template <typename T, class Op> __global__
 void transfrom_core_t(T* arr, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T, typename T1, typename T2, class Op> __global__
 void transfrom_core_t(T* arr, T1* arr1, T2* arr2, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T, typename T1, typename T2,  typename T3, class Op> __global__
 void transfrom_core_t(T* arr, T1* arr1, T2* arr2, T3* arr3, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T, typename T1, typename T2,  typename T3, typename T4, class Op> __global__
 void transfrom_core_t(T* arr, T1* arr1, T2* arr2, T3* arr3, T4* arr4, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i], arr4[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T, typename T1, typename T2,  typename T3, typename T4, typename T5, class Op> __global__
 void transfrom_core_t(T* arr, T1* arr1, T2* arr2, T3* arr3, T4* arr4,  T5* arr5, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i], arr4[i], arr5[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T, typename T1, typename T2,  typename T3, typename T4, typename T5, typename T6, class Op> __global__
 void transfrom_core_t(T* arr, T1* arr1, T2* arr2, T3* arr3, T4* arr4,  T5* arr5,  T6* arr6, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

template <typename T, typename T1, typename T2,  typename T3, typename T4, typename T5, typename T6, typename T7, class Op> __global__
 void transfrom_core_t(T* arr, T1* arr1, T2* arr2, T3* arr3, T4* arr4,  T5* arr5,  T6* arr6, T7* arr7, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i], arr7[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}


template <typename T, typename T1, typename T2,  typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, class Op> __global__
 void transfrom_core_t(T* arr, T1* arr1, T2* arr2, T3* arr3, T4* arr4,  T5* arr5,  T6* arr6, T7* arr7, T8* arr8, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i], arr7[i], arr8[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}


template <typename T, typename T1, typename T2,  typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, class Op> __global__
 void transfrom_core_t(T* arr, T1* arr1, T2* arr2, T3* arr3, T4* arr4,  T5* arr5,  T6* arr6, T7* arr7, T8* arr8, T9* arr9, const int size, Op fn )
{	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		arr[i] = fn(arr1[i], arr2[i], arr3[i], arr4[i], arr5[i], arr6[i], arr7[i], arr8[i], arr9[i]);
		i += blockDim.x * gridDim.x;
	}  	
	
}

/////REDUCTION FUNCTIONS////////

template<typename T, class Op>
__KERNEL__ void
smartReduce_core2( T *out, const T *in, size_t N,  Op fn )
{
    extern __shared__ T sPartials[];  
	int tid = threadIdx.x;
	int gid = threadIdx.x +blockIdx.x * blockDim.x;
	T sum =  0; ////in[0];
	volatile int num_elements = 0;
	//if(gid < N)
	{
		int init_all = fn.reduction_init_type();
		//if (init_all == 1) {sum = in[0]; sPartials[tid] =  in[0];}
		
		switch (init_all)
		{
		case REDUCTION_INIT_NONE:
			break;
		case REDUCTION_INIT_ALL_ZEROS:
			{sum = 0; sPartials[tid] = 0;}
			break;
		case REDUCTION_INIT_ALL_ONES:
			{sum = 1; sPartials[tid] =  1;}
			break;
		case REDUCTION_INIT_FIRST:
			{sum = in[blockIdx.x*blockDim.x + tid]; /*if (tid == 0)*/ {sPartials[0] =  in[blockIdx.x*blockDim.x + tid];}}
			break;
		case REDUCTION_INIT_FIRST_ALL:
			{sum = in[blockIdx.x*blockDim.x]; sPartials[tid] =  in[blockIdx.x*blockDim.x];}
			break;
		case REDUCTION_INIT_ALL:
			{sum = in[blockIdx.x*blockDim.x + tid]; sPartials[tid] =  in[blockIdx.x*blockDim.x + tid];}
			break;
		default:
			break;
		}
		
		__syncthreads();

		smartArrayWrapper<T,smartInlineArray> wHa(in,N,scopeGlobal);	
		T* host_iter;		

		////for (host_iter = wHa.begin(); host_iter != wHa.end(); host_iter++)
		//{
		//	//printf("global index is %d \tIterator index is %d: \t Value is = %f\n",__global_index(0), host_iter - wHa.begin(), *host_iter );
		//	printf("local index is %d. \Global index Y is %d \t Global Unique index is = %d\n",__local_index(0), __global_index(0), __global_unique_index(0) );
		//	//wHa.at(host_iter - wHa.begin()+1) = wHa.at(host_iter - wHa.begin()) + 3;
		//}

		for (size_t i = blockIdx.x*blockDim.x + tid;  i < N; i += blockDim.x*gridDim.x ) 
		{		
			{sum = fn (sum, in[i]);num_elements++;}      
		}

		sPartials[tid] = sum;
		//sPartials[tid] = fn(sPartials[tid], sum);
		__syncthreads();

		for ( int activeThreads = blockDim.x>>1; activeThreads; activeThreads >>= 1 ) 
		{
			if ( tid < activeThreads ) 
			{
				//sPartials[tid] = sum;
				sPartials[tid] = fn( sPartials[tid], sPartials[tid+activeThreads]);    
				num_elements++;
			}
			__syncthreads();
		}
	

		if ( tid == 0 ) 
		{
			out[blockIdx.x] = sPartials[0]; ////fn(sPartials[0],true, num_elements);
			//printf("last sum %e\t%e\t%d\n",  sPartials[0], out[blockIdx.x] , num_elements);
		}
	}
}

template<typename T, class Op>
void smartReduce_kl(T *answer, T *partial, const T *in, size_t N, int numBlocks, int numThreads, Op fn ) ////kl - kernel lancher
{
    if ( N < numBlocks*numThreads ) 
	{
        numBlocks = (N+numThreads-1)/numThreads;
    }

	//printf ("Launch %i\n", 1);
    smartReduce_core2<T,Op><<< numBlocks, numThreads, numThreads * sizeof(T)>>>(partial, in, N, fn );
	//cudaDeviceSynchronize();
	//printf ("Launch %i\n", 2);
    smartReduce_core2<T,Op><<< 1, numThreads, numThreads * sizeof(T)>>>(answer, partial, numBlocks, fn );
	cudaDeviceSynchronize();
}


/////BETA REDUCTION FUNCTIONS ////

///////smart Reduce Function //////
//
//template <typename T, unsigned int blockSize>
//__device__ void warpReduce(volatile T *sdata, unsigned int tid, T fn ( T x1, T x2 ) ) 
//{
//	if (blockSize >= 64) sdata[tid] = fn (sdata[tid], sdata[tid + 32]);
//	if (blockSize >= 32) sdata[tid] = fn (sdata[tid], sdata[tid + 16]);
//	if (blockSize >= 16) sdata[tid] = fn (sdata[tid], sdata[tid +  8]);
//	if (blockSize >=  8) sdata[tid] = fn (sdata[tid], sdata[tid +  4]);
//	if (blockSize >=  4) sdata[tid] = fn (sdata[tid], sdata[tid +  2]);
//	if (blockSize >=  2) sdata[tid] = fn (sdata[tid], sdata[tid +  1]);
//}
//
//template <typename T, unsigned int blockSize>
//__global__ void smartReduce_core(T *g_idata, T *g_odata, unsigned int n, T fn ( T x1, T x2 ) ) 
//{
//	extern __shared__ T sdata[];
//	unsigned int tid = threadIdx.x;
//	unsigned int i = blockIdx.x*(blockSize*2) + tid;
//	unsigned int gridSize = blockSize*2*gridDim.x;
//	sdata[tid] = 0;
//
//	while (i < n) { sdata[tid] = fn (g_idata[i], g_idata[i+blockSize]); i += gridSize; } 
//	__syncthreads();
//
//	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = fn (sdata[tid], sdata[tid + 256]); } __syncthreads(); }
//	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = fn (sdata[tid], sdata[tid + 128]); } __syncthreads(); }
//	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = fn (sdata[tid], sdata[tid +  64]); } __syncthreads(); }
//	if (tid < 32) warpReduce<T, blockSize>(sdata, tid, fn);__syncthreads();
//
//	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
//}
//
//
//template <typename T>
//T smartReduceHostOnDevice(T* in_arr, int num_elements, const int block_size,  T fn ( T x1, T x2 ) )
//{
//	cudaError_t cudaStatus;
//
//	const size_t num_blocks = (num_elements/block_size) + ((num_elements%block_size) ? 1 : 0);
//	
//	T* dev_arr = smartArray<T, smartDevice>(num_elements);
//
//	T* h_partial_sums = smartArray<T,smartHost> (num_blocks + 1);
//	T* d_partial_sums = smartArray<T,smartDevice> (num_blocks + 1);
//	
//	smartArrayWrapper<T,  smartHost> H_SUMS(h_partial_sums, num_blocks + 1, scopeLocal);
//	smartArrayWrapper<T,smartDevice> D_SUMS(d_partial_sums, num_blocks + 1, scopeLocal);
//	
//	smartArrayWrapper<T,smartDevice> DEV_ARR(dev_arr, num_elements, scopeLocal);
//	DEV_ARR.copy(in_arr,num_elements,smartHost);
//
//
//	// launch one kernel to compute, per-block, a partial sum
//	//smartReduce_core<T,blocksize><<<num_blocks,block_size,block_size * sizeof(T)>>>(dev_arr, d_partial_sums, num_elements);
//
//	switch (block_size)
//	{
//	case 1024:
//		smartReduce_core<T,1024><<<num_blocks,block_size,block_size * sizeof(T)>>>(dev_arr, d_partial_sums, num_elements,fn);break;
//	case 512:
//		smartReduce_core<T,512><<<num_blocks,block_size,block_size * sizeof(T)>>>(dev_arr, d_partial_sums, num_elements,fn);break;
//	case 256:
//		smartReduce_core<T,256><<<num_blocks,block_size,block_size* sizeof(T)>>>(dev_arr, d_partial_sums, num_elements,fn);break;
//	case 128:
//		smartReduce_core<T,128><<<num_blocks,block_size,block_size* sizeof(T)>>>(dev_arr, d_partial_sums, num_elements,fn);break;
//	case 64:
//		smartReduce_core<T, 64><<<num_blocks,block_size,block_size* sizeof(T)>>>(dev_arr, d_partial_sums, num_elements,fn);break;
//	case 32:
//		smartReduce_core<T, 32><<<num_blocks,block_size,block_size* sizeof(T)>>>(dev_arr, d_partial_sums, num_elements,fn);break;
//	case 16:
//		smartReduce_core<T, 16><<<num_blocks,block_size,block_size* sizeof(T)>>>(dev_arr, d_partial_sums, num_elements,fn);break;
//	case 8:
//		smartReduce_core<T,  8><<<num_blocks,block_size,block_size* sizeof(T)>>>(dev_arr, d_partial_sums, num_elements,fn);break;
//	case 4:
//		smartReduce_core<T,  4><<<num_blocks,block_size,block_size* sizeof(T)>>>(dev_arr, d_partial_sums, num_elements,fn);break;
//	case 2:
//		smartReduce_core<T,  2><<<num_blocks,block_size,block_size* sizeof(T)>>>(dev_arr, d_partial_sums, num_elements,fn);break;
//	case 1:
//		smartReduce_core<T,  1><<<num_blocks,block_size,block_size* sizeof(T)>>>(dev_arr, d_partial_sums, num_elements,fn);break;
//	}
//
//	cudaDeviceSynchronize();
//
//	// launch a single block to compute the sum of the partial sums
//	smartReduce_core<T, 1><<<1,num_blocks,num_blocks * sizeof(T)>>>(d_partial_sums, d_partial_sums, num_blocks, fn);
//
//	cudaDeviceSynchronize();
//	
//	///P_SUMS.copy(d_partial_sums,num_blocks,smartDevice);
//	H_SUMS = D_SUMS;
//
//	T final_sum = H_SUMS(0);
//	return final_sum;
//}

/////

template <typename T, class Op, unsigned int blockSize>
__device__ void warpReduce(volatile T *sdata, unsigned int tid, Op fn ) 
{
	if (blockSize >= 64) sdata[tid] = fn (sdata[tid], sdata[tid + 32]);
	if (blockSize >= 32) sdata[tid] = fn (sdata[tid], sdata[tid + 16]);
	if (blockSize >= 16) sdata[tid] = fn (sdata[tid], sdata[tid +  8]);
	if (blockSize >=  8) sdata[tid] = fn (sdata[tid], sdata[tid +  4]);
	if (blockSize >=  4) sdata[tid] = fn (sdata[tid], sdata[tid +  2]);
	if (blockSize >=  2) sdata[tid] = fn (sdata[tid], sdata[tid +  1]);
}

template <typename T, class Op, unsigned int blockSize>
__global__ void smartReduce_core(T *g_idata, T *g_odata, unsigned int n, Op fn ) 
{
	extern __shared__ T sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	sdata[tid] = 0;

	while (i < n) { sdata[tid] = fn (g_idata[i], g_idata[i+blockSize]); i += gridSize; } 
	__syncthreads();

	//unsigned int tidx = threadIdx.x;
	//while (tidx < n) 
	//{ 
	//	//sdata[tidx] = fn (g_idata[i], g_idata[i+blockSize]); 
	//	//if (blockSize >= 1024) { if (tidx < 512) { sdata[tidx] = fn (sdata[tidx], sdata[tidx + 1024]); } __syncthreads(); }
	//	if (blockSize >=  512) { if (tidx < 256) { sdata[tidx] = fn (sdata[tidx], sdata[tidx + 256]); } __syncthreads(); }
	//	if (blockSize >=  256) { if (tidx < 128) { sdata[tidx] = fn (sdata[tidx], sdata[tidx + 128]); } __syncthreads(); }
	//	if (blockSize >=  128) { if (tidx <  64) { sdata[tidx] = fn (sdata[tidx], sdata[tidx +  64]); } __syncthreads(); }
	//	if (tidx < 32) warpReduce<T, Op,  blockSize>(sdata, tidx, fn);__syncthreads();
	//	tidx += gridSize; 
	//} 
	//__syncthreads();

	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = fn (sdata[tid], sdata[tid + 1024]); } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = fn (sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = fn (sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = fn (sdata[tid], sdata[tid +  64]); } __syncthreads(); }

	if (tid < 32) warpReduce<T, Op,  blockSize>(sdata, tid, fn);

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <typename T, class Op>
T smartReduceHostOnDevice_beta(T* in_arr, int num_elements, int block_size,  Op fn )
{
	cudaError_t cudaStatus;

	if (block_size > 1024) {block_size = 1024;}
	size_t num_blocks = (num_elements/block_size) + ((num_elements%block_size) ? 1 : 0);	
	if (num_blocks > 65535) {num_blocks = 65535;}

	//num_blocks = 64;

	T* dev_arr = smartArray<T, smartDevice>(num_elements + 1);

	T* h_partial_sums = smartArray<T,smartHost> (num_blocks + 1);
	T* d_partial_sums = smartArray<T,smartDevice> (num_blocks + 1);
	
	

	smartArrayWrapper<T,  smartHost> H_SUMS(h_partial_sums, num_blocks + 1, scopeLocal);
	smartArrayWrapper<T,smartDevice> D_SUMS(d_partial_sums, num_blocks + 1, scopeLocal);	
	smartArrayWrapper<T,smartDevice> DEV_ARR(dev_arr, num_elements, scopeLocal);

	DEV_ARR.copy(in_arr,num_elements,smartHost);


	// launch one kernel to compute, per-block, a partial sum
	//smartReduce_core<T,blocksize><<<num_blocks,block_size,block_size * sizeof(T)>>>(dev_arr, d_partial_sums, num_elements);

	size_t shared_mem = block_size * sizeof(T);

	switch (block_size)
	{
	case 1024:
		smartReduce_core<T, Op,1024><<<num_blocks,block_size,shared_mem>>>(dev_arr, d_partial_sums, num_elements,fn);break;
	case 512:
		smartReduce_core<T, Op, 512><<<num_blocks,block_size,shared_mem>>>(dev_arr, d_partial_sums, num_elements,fn);break;
	case 256:
		smartReduce_core<T, Op, 256><<<num_blocks,block_size,shared_mem>>>(dev_arr, d_partial_sums, num_elements,fn);break;
	case 128:
		smartReduce_core<T, Op, 128><<<num_blocks,block_size,shared_mem>>>(dev_arr, d_partial_sums, num_elements,fn);break;
	case 64:
		smartReduce_core<T, Op,  64><<<num_blocks,block_size,shared_mem>>>(dev_arr, d_partial_sums, num_elements,fn);break;
	case 32:
		smartReduce_core<T, Op,  32><<<num_blocks,block_size,shared_mem>>>(dev_arr, d_partial_sums, num_elements,fn);break;
	case 16:
		smartReduce_core<T, Op,  16><<<num_blocks,block_size,shared_mem>>>(dev_arr, d_partial_sums, num_elements,fn);break;
	case 8:
		smartReduce_core<T, Op,   8><<<num_blocks,block_size,shared_mem>>>(dev_arr, d_partial_sums, num_elements,fn);break;
	case 4:
		smartReduce_core<T, Op,   4><<<num_blocks,block_size,shared_mem>>>(dev_arr, d_partial_sums, num_elements,fn);break;
	case 2:
		smartReduce_core<T, Op,   2><<<num_blocks,block_size,shared_mem>>>(dev_arr, d_partial_sums, num_elements,fn);break;
	case 1:
		smartReduce_core<T, Op,   1><<<num_blocks,block_size,shared_mem>>>(dev_arr, d_partial_sums, num_elements,fn);break;
	}

	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
       fprintf(stderr, "Memory1 Set launch failed: %s\n", cudaGetErrorString(cudaStatus));        
    }
	cudaDeviceSynchronize();

	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
       fprintf(stderr, "Memory2 Set launch failed: %s\n", cudaGetErrorString(cudaStatus));        
    }

	// launch a single block to compute the sum of the partial sums
	smartReduce_core<T, Op,  1><<<1,num_blocks,shared_mem>>>(d_partial_sums, d_partial_sums, num_blocks, fn);
	cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
       fprintf(stderr, "Memory3 Set launch failed: %s\n", cudaGetErrorString(cudaStatus));        
    }
	cudaDeviceSynchronize();
	
	///P_SUMS.copy(d_partial_sums,num_blocks,smartDevice);
	H_SUMS = D_SUMS;

	//printf("GPU Kernel Reduce value is %e\n",H_SUMS[0]);
	T final_sum = H_SUMS[0];
	return final_sum;
}

