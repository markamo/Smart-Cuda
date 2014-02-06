/*
 *  SMART CUDA 0.2.0 (18th Jan, 2014) - Initial Release
 *
 *
 *  Copyright 2013 by Mark Amo-Boateng (smartcuda@outlook.com)
 *
 *  For tutorials and more information, go to:
 *  http://markamo.github.io/Smart-Cuda/
 *
 *
 */

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//    #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
//#endif

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__); \
return EXIT_FAILURE;}} while(0)

////memory allocations
#define smartHost 0
#define smartDevice 1
#define smartPinnedHost 2
#define smartInlineArray 11 //// or smartInlineArray
#define smartFunction 11 ///// for future purposes


////memory lifespan
#define scopeLocal false
#define scopeGlobal true

////deallocating global wrapper memories////
#define ON true;
#define OFF false;

////memory initialization /////
#define _MEMSET_INIT_VAL_ 0
#define _OUT_OF_BOUNDS_VAL_ -99999

 __device__
extern bool __clean_globals = false;
//bool smartGlobalFree = false;

 __device__
extern int __device_id = 0;

#define CUDA_ERR(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__); \
return EXIT_FAILURE;}} while(0)
 
#define smartKernel(name)              __global__ void name ## _kernel
#define smartKernelName(name)          name ## _kernel

#define launch_smartKernel(name, gridDim, blockDim, sharedBytes, streamId, ...) \
    name ## _kernel<<< (gridDim) , (blockDim), (sharedBytes), (streamId) >>>(__VA_ARGS__)


//#define parfor(int e_idx = start_idx; e_idx < end_index; e_idx++ ) \



template <typename T> 
__host__ __device__ void smartFill(T *dev_Array, const int size, const T init)
{
    int i = 0;
	for (i = 0; i < size; i++)
	{
		dev_Array[i] = init;
		//i += 1;
	}    
	
}

template <typename T> 
__host__ __device__ void smartSequence(T *dev_Array, const int size, const T init, T step = 1)
{
    int i = 0;
	for (i = 0; i < size; i++)
	{
		dev_Array[i] = init + i * step;
		//i += 1;
	}    	
}

template <typename T> 
__global__ void smartFillAsync_core(T *dev_Array, const int size, const T init)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		dev_Array[i] = init;
		i += blockDim.x * gridDim.x;
	}  
	
}

template <typename T> 
__global__ void smartSequenceAsync_core(T *dev_Array, const int size, const T init, T step = 1)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		dev_Array[i] = init + i * step;
		i += blockDim.x * gridDim.x;
	}    	
}


template <typename T> 
__host__ __device__ cudaError_t smartFillAsync35(T *dev_Array, const int size, const T init)
{
	cudaError_t cudaStatus;
    smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}
	return cudaStatus;
}

template <typename T> 
__host__ __device__ cudaError_t smartSequenceAsync35(T *dev_Array, const int size, const T init, T step = 1)
{
	cudaError_t cudaStatus;
    smartSequenceAsync_core<<<128, 128>>>(dev_Array, size, init,step);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}
	return cudaStatus;
}

template <typename T> 
__host__ cudaError_t smartFillAsync(T *dev_Array, const int size, const T init)
{
	cudaError_t cudaStatus;
    smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}
	return cudaStatus;
}

template <typename T> 
__host__ cudaError_t smartSequenceAsync(T *dev_Array, const int size, const T init, T step = 1)
{
	cudaError_t cudaStatus;
    smartSequenceAsync_core<<<128, 128>>>(dev_Array, size, init,step);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}
	return cudaStatus;
}

template <typename T> 
	inline T* seq_allocSmartDeviceArrayAsync( int size, T init = 0, bool setINI = false, T step=1) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(1D) failed!\n");
        //////goto Error;
    }

	/*if (setINI)
	{
		smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		smartSequenceAsync_core<<<128, 128>>>(dev_Array, size, init,step);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			////goto Error;
		}
	}
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* allocSmartDeviceArrayAsync( int size, T init = 0, bool setINI = false)////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(1D) failed\n!");
        //////goto Error;
    }

	/*if (setINI)
	{
		smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			////goto Error;
		}
	}
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* seq_allocSmartDeviceArray( int size, T init = 0, bool setINI = false, T step=1) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(1D) failed!\n");
        //////goto Error;
    }

	if (setINI)
	{
		smartSequence(dev_Array, size, init,step);		
	}
	
//////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* allocSmartDeviceArray( int size, T init = 0, bool setINI = false) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(1D) failed\n!");
        //////goto Error;
    }

	if (setINI)
	{
		smartFill(dev_Array, size, init);		
	}
	
//////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}
	
template <typename T> 
	inline T* allocSmartDeviceArrayAsync( int sizeX, int sizeY, T init = 0, bool setINI = false) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(2D) failed!\n");
        //////goto Error;
    }

	/*if (setINI)
	{
		smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			////goto Error;
		}
	}
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* seq_allocSmartDeviceArrayAsync( int sizeX, int sizeY, T init = 0, bool setINI = false, T step = 1) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(2D) failed!\n");
        //////goto Error;
    }

	/*if (setINI)
	{
		smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		smartSequenceAsync_core<<<128, 128>>>(dev_Array, size, init);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			////goto Error;
		}
	}
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}
	
template <typename T> 
	inline T* allocSmartDeviceArray( int sizeX, int sizeY, T init = 0, bool setINI = false) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(2D) failed!\n");
        //////goto Error;
    }

	if (setINI)
	{
		smartFill(dev_Array, size, init);		
	}
	
//////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* seq_allocSmartDeviceArray( int sizeX, int sizeY, T init = 0, bool setINI = false, T step =1) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(2D) failed!\n");
        //////goto Error;
    }

	if (setINI)
	{
		smartFill(dev_Array, size, init,step);
		
	}
	
//////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}
	
template <typename T> 
	inline T* allocSmartDeviceArrayAsync( int sizeX, int sizeY, int sizeZ, T init = 0, bool setINI = false) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(3D) failed!\n");
        //////goto Error;
    }

	/*if (setINI)
	{
		smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			////goto Error;
		}
	}
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

	template <typename T> 
	inline T* seq_allocSmartDeviceArrayAsync( int sizeX, int sizeY, int sizeZ, T init = 0, bool setINI = false, T step =1) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(3D) failed!\n");
        //////goto Error;
    }

	/*if (setINI)
	{
		smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		smartSequenceAsync_core<<<128, 128>>>(dev_Array, size, init, step);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			////goto Error;
		}
	}
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* allocSmartDeviceArray( int sizeX, int sizeY, int sizeZ, T init = 0, bool setINI = false) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(3D) failed!\n");
        //////goto Error;
    }

	if (setINI)
	{
		smartFill(dev_Array, size, init);		
	}
	
//////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* seq_allocSmartDeviceArray( int sizeX, int sizeY, int sizeZ,T init = 0, bool setINI = false, T step =1) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(3D) failed!\n");
        //////goto Error;
    }

	if (setINI)
	{
		smartFill(dev_Array, size, init,step);
		
	}
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}
	
template <typename T> 
	inline T* allocSmartDeviceArrayAsync( int sizeX, int sizeY, int sizeZ, int sizeW, T init = 0, bool setINI = false) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ * sizeW;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(4D) failed!\n");
        //////goto Error;
    }

	/*if (setINI)
	{
		smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			////goto Error;
		}
	}
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* seq_allocSmartDeviceArrayAsync( int sizeX, int sizeY, int sizeZ, int sizeW, T init = 0, T step=1, bool setINI = false) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ * sizeW;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(4D) failed!\n");
        //////goto Error;
    }

	/*if (setINI)
	{
		smartFillAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		smartSequenceAsync_core<<<128, 128>>>(dev_Array, size, init, step);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			////goto Error;
		}
	}
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* allocSmartDeviceArray( int sizeX, int sizeY, int sizeZ, int sizeW, T init = 0, bool setINI = false) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ * sizeW;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(4D) failed!\n");
        //////goto Error;
    }

	if (setINI)
	{
		smartFill(dev_Array, size, init,step);
		
	}
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* seq_allocSmartDeviceArray( int sizeX, int sizeY, int sizeZ, int sizeW, T init = 0, T step=1, bool setINI = false) ////, cudaError_t cudaStatus = cudaSuccess)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ * sizeW;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("CUDA Memory Allocation(4D) failed!\n");
        //////goto Error;
    }

	if (setINI)
	{
		smartFill(dev_Array, size, init,step);
		
	}
	
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline cudaError_t smartDeviceFree(T* dev_mem)
	{
		cudaError_t cudaStatus;
		cudaFree(dev_mem);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Cuda Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}
		return cudaStatus;
	}

template <typename T> 
	inline cudaError_t smartDeviceFree(T* dev_mem, cudaError_t &cudaStatus)
	{
		////cudaError_t cudaStatus;
		cudaFree(dev_mem);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Cuda Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}
		return cudaStatus;
	}

template <typename T> 
	inline cudaError_t smartHostFree(T* dev_mem)
	{
		cudaError_t cudaStatus;
		free(dev_mem);
		dev_mem = NULL;
			
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Cuda Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}
		return cudaStatus;
	}

template <typename T> 
	inline cudaError_t smartHostFree(T* dev_mem, cudaError_t &cudaStatus)
	{
		////cudaError_t cudaStatus;
		free(dev_mem);
		dev_mem = NULL;
			
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Cuda Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}
		return cudaStatus;
	}

template <typename T> 
	inline cudaError_t smartPinnedHostFree(T* dev_mem)
	{
		cudaError_t cudaStatus;
		cudaFreeHost(dev_mem);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Cuda Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}
		return cudaStatus;
	}

template <typename T> 
	inline cudaError_t smartPinnedHostFree(T* dev_mem, cudaError_t &cudaStatus)
	{
		////cudaError_t cudaStatus;
		cudaFreeHost(dev_mem);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Cuda Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			//////goto Error;
		}
		return cudaStatus;
	}


template <typename T> 
__host__ __device__
inline cudaError_t smartInlineArrayFree(T* dev_mem)
{
	cudaError_t cudaStatus;
	free(dev_mem);
	if(dev_mem != NULL) {cudaStatus = cudaErrorMemoryAllocation;}
	dev_mem = NULL;
	return cudaStatus;
}

template <typename T> 
__host__ __device__
inline cudaError_t smartInlineArrayFree(T* dev_mem, cudaError_t &cudaStatus)
{
	////cudaError_t cudaStatus;
	free(dev_mem);
	if(dev_mem != NULL) {cudaStatus = cudaErrorMemoryAllocation;}
	dev_mem = NULL;

	return cudaStatus;
}

template <typename T> 
	inline cudaError_t freeSmartDeviceArray(T* dev_mem, int array_type)
	{
		cudaError_t cudaStatus;
		if (array_type == smartDevice)
		{
			cudaFree(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Device Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartInlineArray)
		{
			free(dev_mem);
			if(dev_mem != NULL) {cudaStatus = cudaErrorMemoryAllocation;}
			dev_mem = NULL;
			return cudaStatus;
		}
		else
		{
			cudaStatus = cudaErrorMemoryAllocation;
			printf("Cuda Device Memory Release failed: Unknown Device memory\n");
			return cudaStatus;
		}
		//return cudaStatus;
	}

template <typename T> 
	inline cudaError_t freeSmartDeviceArray(T* dev_mem, int array_type, cudaError_t &cudaStatus)
	{
		////cudaError_t cudaStatus;
		if (array_type == smartDevice)
		{
			cudaFree(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Device Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartInlineArray)
		{
			free(dev_mem);
			if(dev_mem != NULL) {cudaStatus = cudaErrorMemoryAllocation;}
			dev_mem = NULL;
			return cudaStatus;
		}
		else
		{
			cudaStatus = cudaErrorMemoryAllocation;
			printf("Cuda Device Memory Release failed: Unknown Device memory\n");
			return cudaStatus;
		}
	}

	template <typename T>
cudaError_t smartCopyHost2Device(T* dev_mem, const T* host_mem, unsigned int size, cudaError_t cudaStatus = cudaSuccess)
{
	cudaStatus = cudaMemcpy(dev_mem, host_mem, size * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy Host to Device failed!\n");
        //////goto Error;
    }
	return cudaStatus;
}

template <typename T>
cudaError_t smartCopyDevice2Host(T* host_mem, T* dev_mem, unsigned int size, cudaError_t cudaStatus = cudaSuccess)
{
	cudaStatus = cudaMemcpy(host_mem, dev_mem, size * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy Device to Host failed!\n");
        //////goto Error;
    }
	return cudaStatus;
}

template <typename T>
cudaError_t smartCopyDevice2DevicePeer(T* host_mem, int host_id, T* dev_mem, int src_id, unsigned int size, cudaError_t cudaStatus = cudaSuccess)
{
	cudaStatus = cudaMemcpyPeer(host_mem, host_id, dev_mem, src_id, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy Device to Device failed!\n");
        //////goto Error;
    }
	return cudaStatus;
}

template <typename T>
cudaError_t smartCopyDevice2Device(T* host_mem, T* dev_mem, unsigned int size, cudaError_t cudaStatus = cudaSuccess)
{
	cudaStatus = cudaMemcpy(host_mem, dev_mem, size * sizeof(T), cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy Device to Device failed!\n");
        //////goto Error;
    }
	return cudaStatus;
}


template <typename T>
cudaError_t smartCopyHost2Host(T* host_mem, T* dev_mem, unsigned int size, cudaError_t cudaStatus = cudaSuccess)
{
	cudaStatus = cudaMemcpy(host_mem, dev_mem, size * sizeof(T), cudaMemcpyHostToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy Host to Host failed!\n");
        //////goto Error;
    }
	return cudaStatus;
}

/////BETA Code/////// 
//template<int X, int Y, int Z, int W>
class smartIndex
{
private: 
	int dimX; int dimY; int dimZ; int dimW;
public:
	__device__  __host__ smartIndex() 
	{
		
	}
	__device__  __host__  smartIndex(int i,int j=1, int k=1, int w=1) 
	{
		if((i < 0 || j < 0 || k < 0) || (w < 0)) 
		{
			// Take appropriate action here. This is just a placeholder response.
			printf("Array Dimensions [%d][%d][%d][%d] of smart Array is out of bounds\n",i,j,k,w);			
			//////exit(1);
		}
		dimX = i;dimY = j; dimZ = k; dimW = w;
	}
	
	__device__  __host__ int operator()(int i=1,int j=1, int k=1, int w=1) 
	{
		if((i < 0 || j < 0 || k < 0) || (w < 0)) 
		{
			// Take appropriate action here. This is just a placeholder response.
			printf("Array Dimensions [%d][%d][%d][%d] of smart Array is out of bounds\n",i,j,k,w);			
			//////exit(1);
		}
		dimX = i;dimY = j; dimZ = k; dimW = w;
		return dimX * dimY * dimZ * dimW;
	}

	__device__  __host__
	int getDimX() { return dimX; }
	__device__  __host__
	int getDimY() { return dimY; }
	__device__  __host__
	int getDimZ() { return dimZ; }
	__device__  __host__
	int getDimW() { return dimW; }
};
/////End BETA Code/////// 

template <typename T, int mem_loc> 
	inline __host__ __device__ T* smartArray( int sizeX) ////, cudaError_t &cudaStatus)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX;

	if (mem_loc == smartHost)
	{
		////allocate pageable host memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			printf("Dynamic Memory Allocation on Device failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
		memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else if (mem_loc == smartDevice)
	{
		cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			printf("Dynamic Memory Allocation on Device failed!\n");
			//////goto Error;
		}
		cudaMemset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else if (mem_loc == smartPinnedHost) {		

		cudaStatus = cudaMallocHost((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			printf("Dynamic Pinned Memory Allocation on Host failed!\n");
			//////goto Error;
		}
		memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else if (mem_loc == smartInlineArray)
	{
		////allocate pageable host memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			printf( "Dynamic Memory Allocation in Global Kernel failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
		memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else
	{
		printf("Cannot allocate unsupported memory\n");	
		cudaStatus  = cudaErrorMemoryAllocation;
		//return cudaStatus;
	}
	
//////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T, int mem_loc> 
	inline  __host__ __device__ T* smartArray( int sizeX, int sizeY) ////, cudaError_t &cudaStatus)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY;

	if (mem_loc == smartHost)
	{
		////allocate pageable host memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			printf("Dynamic Memory Allocation on Device failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
		memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else if (mem_loc == smartDevice)
	{
		cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			printf("Dynamic Memory Allocation on Device failed!\n");
			//////goto Error;
		}
		cudaMemset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else if (mem_loc == smartPinnedHost) {		

		cudaStatus = cudaMallocHost((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			printf("Dynamic Pinned Memory Allocation on Host failed!\n");
			//////goto Error;
		}
		memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else if (mem_loc == smartInlineArray)
	{
		////allocate pageable host memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			printf( "Dynamic Memory Allocation in Global Kernel failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
		memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else
	{
		printf("Cannot allocate unsupported memory\n");	
		cudaStatus  = cudaErrorMemoryAllocation;
		//return cudaStatus;
	}
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T, int mem_loc> 
	inline  __host__ __device__ T* smartArray( int sizeX, int sizeY, int sizeZ) ////, cudaError_t &cudaStatus)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ;

	if (mem_loc == smartHost)
	{
		////allocate pageable host memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			printf("Dynamic Memory Allocation on Device failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
		memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else if (mem_loc == smartDevice)
	{
		cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			printf("Dynamic Memory Allocation on Device failed!\n");
			//////goto Error;
		}
		cudaMemset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else if (mem_loc == smartPinnedHost) {		

		cudaStatus = cudaMallocHost((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			printf("Dynamic Pinned Memory Allocation on Host failed!\n");
			//////goto Error;
		}
		memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else if (mem_loc == smartInlineArray)
	{
		////allocate pageable host memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			printf( "Dynamic Memory Allocation in Global Kernel failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
		memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else
	{
		printf("Cannot allocate unsupported memory\n");	
		cudaStatus  = cudaErrorMemoryAllocation;
		//return cudaStatus;
	}
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T, int mem_loc> 
	inline  __host__ __device__ T* smartArray( int sizeX, int sizeY, int sizeZ, int sizeW) ////, cudaError_t &cudaStatus)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ * sizeW;

	if (mem_loc == smartHost)
	{
		////allocate pageable host memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			printf("Dynamic Memory Allocation on Device failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
		memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else if (mem_loc == smartDevice)
	{
		cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			printf("Dynamic Memory Allocation on Device failed!\n");
			//////goto Error;
		}
		cudaMemset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else if (mem_loc == smartPinnedHost) {		

		cudaStatus = cudaMallocHost((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			printf("Dynamic Pinned Memory Allocation on Host failed!\n");
			//////goto Error;
		}
		memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else if (mem_loc == smartInlineArray)
	{
		////allocate pageable host memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			printf( "Dynamic Memory Allocation in Global Kernel failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
		memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));
	}
	else
	{
		printf("Cannot allocate unsupported memory\n");	
		cudaStatus  = cudaErrorMemoryAllocation;
		//return cudaStatus;
	}
	
////Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}


template <typename T> 
	inline __device__ T* smartInlineArrayArray(int sizeX) 
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX;

	{
		////allocatekernel global memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			printf( "Dynamic Memory Allocation in Global Kernel failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
	}
	
	memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));

	return dev_Array;
}

template <typename T> 
	inline __device__ T* smartInlineArrayArray( int sizeX, int sizeY) ////, cudaError_t &cudaStatus)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY;

	{
		////allocatekernel global memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			printf( "Dynamic Memory Allocation in Global Kernel failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
	}
	
	memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));


	return dev_Array;
}

template <typename T>
	inline __device__ T* smartInlineArrayArray( int sizeX, int sizeY, int sizeZ) ////, cudaError_t &cudaStatus)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ;

	{
		////allocatekernel global memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			printf( "Dynamic Memory Allocation in Global Kernel failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
	}
	
	memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));

	return dev_Array;
}

template <typename T>
	inline __device__ T* smartInlineArrayArray( int sizeX, int sizeY, int sizeZ, int sizeW) ////, cudaError_t &cudaStatus)
{
	cudaError_t cudaStatus;
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ * sizeW;

	{
		////allocatekernel global memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			printf( "Dynamic Memory Allocation in Global Kernel failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
	}
	
	memset(dev_Array, _MEMSET_INIT_VAL_, size * sizeof(T));

	return dev_Array;
}


template <typename T> 
	inline __host__ __device__ cudaError_t smartFree(T* dev_mem, int array_type)
	{
		cudaError_t cudaStatus;
		if (array_type == smartHost)
		{
			free(dev_mem);
			dev_mem = NULL;
			cudaStatus = cudaSuccess;
			if (dev_mem != NULL) {
				cudaStatus = cudaErrorMemoryAllocation;
				printf("Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
		}
		else if (array_type == smartDevice)
		{
			cudaFree(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Device Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartPinnedHost)
		{
			cudaFreeHost(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartInlineArray)
		{
			free(dev_mem);
			if(dev_mem != NULL) {cudaStatus = cudaErrorMemoryAllocation;}
			dev_mem = NULL;
			return cudaStatus;
		}
		else
		{
			cudaStatus = cudaErrorMemoryAllocation;
			printf("Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		return cudaStatus;
	}

template <typename T> 
	inline cudaError_t  __host__ __device__ smartFree(T* dev_mem, int array_type, cudaError_t &cudaStatus)
	{
		//cudaError_t cudaStatus;
		if (array_type == smartHost)
		{
			free(dev_mem);
			dev_mem = NULL;
			cudaStatus = cudaSuccess;
			if (dev_mem != NULL) {
				cudaStatus = cudaErrorMemoryAllocation;
				printf("Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
		}
		else if (array_type == smartDevice)
		{
			cudaFree(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Device Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartPinnedHost)
		{
			cudaFreeHost(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartInlineArray)
		{
			free(dev_mem);
			if(dev_mem != NULL) {cudaStatus = cudaErrorMemoryAllocation; }
			dev_mem = NULL;
			return cudaStatus;
		}
		else
		{
			cudaStatus = cudaErrorMemoryAllocation;
			printf("Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		return cudaStatus;
	}

template <typename T, int mem_loc> 
//__host__ __device__
class smartArrayWrapper
{
	//friend class smartNavigator< T,  mem_loc>;
private: 
	////pointer to array
	T* dev_arr;
	//const T* dev_arr;

	////array dimensions
	int lengthX;
	int lengthY;
	int lengthZ;
	int lengthW;
	size_t array_size;

	////specialize views
	int vX,vY,vZ,vW; 

	////array wrapper type ////0= host, 1 = device
	////useful for copies
	int array_type;

	////destructor behaviour //// 0 leave array, 1 destroy on scope exit
	bool _global_scope; ////destroy_array_on_scope_exit;
	//bool _is_copy; /////used to control destructor on copy assignment///
	bool _is_cleaned;

	//////increment and decrement operators for navigation of the data array
	//unsigned int idx; /////current location
	//unsigned int ix,iy,iz,iw; ////current index access
	//unsigned int vidx; /////current specialized view location
	//unsigned int vix,viy,viz,viw; ////current index access for views

	//// keeping track of device id for peer to peer memory copy
	int device_id; 
	
	__device__  __host__
	void initializeArray(T* dev=NULL, int lenX=1,int lenY=1,int lenZ=1,int lenW=1, bool global_scope = true ) 
	{
		dev_arr = dev;
		lengthX = lenX;
		lengthY = lenY;
		lengthZ = lenZ;
		lengthW = lenW;
		array_size = lengthX * lengthY * lengthZ * lengthW;

		////initialize views to default dimensions
		vX = 1; vY = 1; vZ = 1; vW = 1;

		////set location type
		array_type = mem_loc;

		////destructor behaviour //// 0 leave array, 1 destroy on scope exit
		_global_scope = global_scope;
		_is_cleaned = false;
		//_is_copy = true;

		////increment operators for navigation
		idx = 0; vidx = 0;
		ix = 0; iy =0; iz = 0; iw = 0;

		////device id 
		device_id = __device_id;

	}
	
	__device__  __host__
	int getIndex(int i) 
	{	
		ix = i; ////iy = j; iz = k; iw = w;
		int j = 0; int k = 0; int w = 0;
		
		int _idx = i + j * lengthX + k * lengthX * lengthY + w * lengthX * lengthY * lengthZ;		
		return _idx;
	}

	__device__  __host__
	int getIndex(int i, int j) 
	{	
		ix = i; iy = j; ////iz = k; iw = w;
		int k = 0; int w = 0;
		
		int _idx = i + j * lengthX + k * lengthX * lengthY + w * lengthX * lengthY * lengthZ;		
		return _idx;
	}

	__device__  __host__
	int getIndex(int i, int j, int k) 
	{	
		ix = i; iy = j; iz = k; ////iw = w;
		int w = 0;
		
		int _idx = i + j * lengthX + k * lengthX * lengthY + w * lengthX * lengthY * lengthZ;		
		return _idx;
	}

	__device__  __host__
	int getIndex(int i, int j, int k, int w) 
	{		
		ix = i; iy = j; iz = k; iw = w;
		int _idx = i + j * lengthX + k * lengthX * lengthY + w * lengthX * lengthY * lengthZ;		
		return _idx;
	}

	__device__  __host__
	int getViewIndex(int i, int j=0, int k=0, int w=0) 
	{		
		vix = i; viy = j; viz = k; viw = w;
		int _idx = i + j * vX + k * vX * vY + w * vX * vY * vZ;
		return dev_arr[_idx];
	}

	__device__  __host__
	cudaError_t inner_copy(T* src_array, int src_type, int src_size, int src_device_id)
	{
		cudaError_t cudaStatus;
		
		//// size of data to copy, using the least of the 2 to avoid errors
		unsigned int copySize = (getlen() > src_size) ? getlen() : src_size;

		if(array_type == smartHost || array_type == smartPinnedHost) ////host -> host/device copies
		{
			if(src_type == smartHost || src_type == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Host<T>(dev_arr,src_array,copySize);
			}
			else if (src_type == smartDevice)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,src_array,copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice || array_type == smartInlineArray) ////device -> host/device copies
		{
			if(src_type == smartHost || src_type == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,src_array,copySize);
			}
			else if (src_type == smartDevice|| src_type == smartInlineArray)
			{
				if (device_id == src_device_id)
				{
					cudaStatus = smartCopyDevice2Device<T>(dev_arr,src_array,copySize);
				}
				else
				{
					cudaStatus = smartCopyDevice2DevicePeer<T>(dev_arr,device_id, src_array,scr_device_id, copySize);
				}
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			printf("Cannot copy unsupported type\n");	
			cudaStatus = cudaErrorMemoryAllocation;
			return cudaStatus;
		}		
		return cudaStatus;
	}

	

public: 

	////increment and decrement operators for navigation of the data array
	unsigned int idx; /////current location
	unsigned int ix,iy,iz,iw; ////current index access
	unsigned int vidx; /////current specialized view location
	unsigned int vix,viy,viz,viw; ////current index access for views


	__device__  __host__
	smartArrayWrapper() 
	{
		initializeArray();		
	}

	__device__  __host__
	smartArrayWrapper(bool global_scope) 
	{
		initializeArray(NULL,1,1,1,1,global_scope);

		//dev_arr = NULL;
		//lengthX = 1;
		//lengthY = 1;
		//lengthZ = 1;
		//lengthW = 1;
		//////initialize views to default dimensions
		//vX = 1; vY = 1; vZ = 1; vW = 1;

		//////set location type
		//array_type = mem_loc;

		//////destructor behaviour //// 0 leave array, 1 destroy on scope exit
		//_global_scope = global_scope;
		//_is_copy = true;
	}


	__device__  __host__
	smartArrayWrapper(T* dev, int lenX, bool global_scope = true ) 
	{
		initializeArray(dev,lenX,1,1,1,global_scope);

		//dev_arr = dev;
		//lengthX = lenX;
		//lengthY = 1;
		//lengthZ = 1;
		//lengthW = 1;
		//////initialize views to default dimensions
		//vX = 1; vY = 1; vZ = 1; vW = 1;

		//////set location type
		//array_type = mem_loc;

		//////destructor behaviour //// 0 leave array, 1 destroy on scope exit
		//_global_scope = global_scope;
		//_is_copy = true;
	}

	__device__  __host__
	smartArrayWrapper(T* dev, int lenX,int lenY, bool global_scope = true ) 
	{
		initializeArray(dev,lenX,lenY,1,1,global_scope);

		//dev_arr = dev;
		//lengthX = lenX;
		//lengthY = lenY;
		//lengthZ = 1;
		//lengthW = 1;
		//////initialize views to default dimensions
		//vX = 1; vY = 1; vZ = 1; vW = 1;

		//////set location type
		//array_type = mem_loc;

		//////destructor behaviour //// 0 leave array, 1 destroy on scope exit
		//_global_scope = global_scope;
		//_is_copy = true;
	}

	__device__  __host__
	smartArrayWrapper(T* dev, int lenX,int lenY,int lenZ, bool global_scope = true ) 
	{
		initializeArray(dev,lenX,lenY,lenZ,1,global_scope);

		//dev_arr = dev;
		//lengthX = lenX;
		//lengthY = lenY;
		//lengthZ = lenZ;
		//lengthW = 1;
		//////initialize views to default dimensions
		//vX = 1; vY = 1; vZ = 1; vW = 1;

		//////set location type
		//array_type = mem_loc;

		//////destructor behaviour //// 0 leave array, 1 destroy on scope exit
		//_global_scope = global_scope;
		//_is_copy = true;
	}

	__device__  __host__
	smartArrayWrapper(T* dev, int lenX,int lenY,int lenZ,int lenW, bool global_scope = true ) 
	{
		initializeArray(dev,lenX,lenY,lenZ,lenW,global_scope);

		//dev_arr = dev;
		//lengthX = lenX;
		//lengthY = lenY;
		//lengthZ = lenZ;
		//lengthW = lenW;
		//////initialize views to default dimensions
		//vX = 1; vY = 1; vZ = 1; vW = 1;

		//////set location type
		//array_type = mem_loc;

		//////destructor behaviour //// 0 leave array, 1 destroy on scope exit
		//_global_scope = global_scope;
		//_is_copy = true;
	}	

	////constant version initializers///////
	__device__  __host__
	smartArrayWrapper(const T* dev, int lenX, bool global_scope = true ) 
	{
		initializeArray(const_cast<T*>(dev),lenX,1,1,1,scopeGlobal);
		const_cast<const T*>(dev);
	}

	__device__  __host__
	smartArrayWrapper(const T* dev, int lenX,int lenY, bool global_scope = true ) 
	{
		initializeArray(const_cast<T*>(dev),lenX,lenY,1,1,scopeGlobal);
		const_cast<const T*>(dev);
	}

	__device__  __host__
	smartArrayWrapper(const T* dev, int lenX,int lenY,int lenZ, bool global_scope = true ) 
	{
		initializeArray(const_cast<T*>(dev),lenX,lenY,lenZ,1,scopeGlobal);
		const_cast<const T*>(dev);
	}

	__device__  __host__
	smartArrayWrapper(const T* dev, int lenX,int lenY,int lenZ,int lenW, bool global_scope = true ) 
	{
		initializeArray(const_cast<T*>(dev),lenX,lenY,lenZ,lenW,scopeGlobal);
		const_cast<const T*>(dev);
	}	
	/////end of const versions //////

	__device__  __host__
	~smartArrayWrapper()
	{
		if(!_is_cleaned)
		{
			if(!_global_scope )
			{
				_is_cleaned = true;
				//smartFree<T>(*this);
				smartFree<T>(inner_ptr(),array_type);	
				
			}
			else if(_global_scope && __clean_globals)
			{
				_is_cleaned = true;
				//smartFree<T>(*this);
				smartFree<T>(inner_ptr(),array_type);	
				
			}
			
			//_is_copy = false;
		}
	}

	template <typename T>
	__device__  __host__
	void wrap(smartArrayWrapper<T,mem_loc> dev ) 
	{
		initializeArray(dev.inner_ptr(),dev.getlenX(),dev.getlenY(),dev.getlenZ(),dev.getlenW(),mem_loc);
	}

	template <typename T>
	__device__  __host__
	void wrapView(smartArrayWrapper<T,mem_loc> dev ) 
	{
		initializeArray(dev.inner_ptr(),dev.getViewDimX(),dev.getViewDimX(),dev.getViewDimX(),dev.getViewDimX(),mem_loc);
	}

	/*

	__device__  __host__
	void wrap(smartArrayWrapper<T,smartHost> dev ) 
	{
		initializeArray(dev.inner_ptr(),dev.getlenX(),.getlenX(),.getlenX(),.getlenX(),dev.getType());
	}

	__device__  __host__
	void wrap(smartArrayWrapper<T,smartPinnedHost> dev ) 
	{
		initializeArray(dev.inner_ptr(),dev.getlenX(),.getlenX(),.getlenX(),.getlenX(),dev.getType());
	}

	__device__  __host__
	void wrap(smartArrayWrapper<T,smartDevice> dev ) 
	{
		initializeArray(dev.inner_ptr(),dev.getlenX(),.getlenX(),.getlenX(),.getlenX(),dev.getType());
	}
	*/

	template <typename T>
	__device__  __host__
	void wrap(T* dev, int lenX, bool global_scope = true ) 
	{
		initializeArray(dev,lenX,1,1,1,global_scope);
	}

	template <typename T>
	__device__  __host__
	void wrap(T* dev, int lenX,int lenY, bool global_scope = true ) 
	{
		initializeArray(dev,lenX,lenY,1,1,global_scope);
	}

	template <typename T>
	__device__  __host__
	void wrap(T* dev, int lenX,int lenY,int lenZ, bool global_scope = true ) 
	{
		initializeArray(dev,lenX,lenY,lenZ,1,global_scope);
	}

	template <typename T>
	__device__  __host__
	void wrap(T* dev, int lenX,int lenY,int lenZ,int lenW, bool global_scope = true ) 
	{
		initializeArray(dev,lenX,lenY,lenZ,lenW,global_scope);
	}

	
	template<typename custom>
	__device__  __host__
	void wrapCustom( custom other, bool global_scope = true ) 
	{
		T* dev = other.inner_ptr();
		int lenX = other.size();
		initializeArray(dev,lenX,1,1,1,global_scope);
	}

	template<typename custom>
	__device__  __host__
	void wrapSTL( custom other, bool global_scope = true ) 
	{
		T* dev = &other[0];
		int lenX = other.size();
		initializeArray(dev,lenX,1,1,1,global_scope);
	}

	/////const versions of the wrap function /////
	template <typename T>
	__device__  __host__
	void wrap(const T* dev, int lenX, bool global_scope = true ) 
	{
		initializeArray(const_cast<T*>(dev),lenX,1,1,1,scopeGlobal);
		const_cast<const T*>(dev);
	}

	template <typename T>
	__device__  __host__
	void wrap(const T* dev, int lenX,int lenY, bool global_scope = true ) 
	{
		initializeArray(const_cast<T*>(dev),lenX,lenY,1,1,scopeGlobal);
		const_cast<const T*>(dev);
	}

	template <typename T>
	__device__  __host__
	void wrap(const T* dev, int lenX,int lenY,int lenZ, bool global_scope = true ) 
	{
		initializeArray(const_cast<T*>(dev),lenX,lenY,lenZ,1,scopeGlobal);
		const_cast<const T*>(dev);
	}

	template <typename T>
	__device__  __host__
	void wrap(const T* dev, int lenX,int lenY,int lenZ,int lenW, bool global_scope = true ) 
	{
		initializeArray(const_cast<T*>(dev),lenX,lenY,lenZ,lenW,scopeGlobal);
		const_cast<const T*>(dev);
	}

	//
	//template<typename custom>
	//__device__  __host__
	//void wrapCustom(const custom other, bool global_scope = true ) 
	//{
	//	T* dev = other.inner_ptr();
	//	int lenX = other.size();
	//	initializeArray(const_cast<T*>(dev),lenX,1,1,1,scopeGlobal);
	//	const_cast<const T*>(dev);
	//}

	//template<typename custom>
	//__device__  __host__
	//void wrapSTL(const custom other, bool global_scope = true ) 
	//{
	//	T* dev = &other[0];
	//	int lenX = other.size();
	//	initializeArray(const_cast<T*>(dev),lenX,1,1,1,scopeGlobal);
	//	const_cast<const T*>(dev);
	//}
	//
	/////end of const versions ///////


	__device__  __host__
	T &operator()(int i, bool check_bounds = true) 
	{
		if(((i < 0 || i > lengthX-1) || dev_arr==NULL) && check_bounds)
		{
			// Take appropriate action here. This is just a placeholder response.
			printf("Array not INITIALIZED or Index [%d] of smart Array is out of bounds\n",i);			
			////exit(1);
		}
		//T temp = dev_arr[i];
		idx = i;
		return dev_arr[i];
	}

	__device__  __host__
	T &operator()(int i, int j, bool check_bounds = true)
	{
		if((((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1))|| dev_arr==NULL) && check_bounds)
		{
			// Take appropriate action here. This is just a placeholder response.
			printf("Array not INITIALIZED or Index [%d][%d] of smart Array is out of bounds\n",i,j);			
			////exit(1);
		}
		int _idx = i + j * lengthX;
		idx = _idx;
		return dev_arr[_idx];
	}

	__device__  __host__
	T &operator()(int i, int j, int k, bool check_bounds = true)
	{
		if((((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1))|| dev_arr==NULL)&& check_bounds)
		{
			// Take appropriate action here. This is just a placeholder response.
			printf("Array not INITIALIZED or Index [%d][%d][%d] of smart Array is out of bounds\n",i,j,k);			
			////exit(1);
		}
		int _idx = i + j * lengthX + k * lengthX * lengthY;
		idx = _idx;
		return dev_arr[_idx];
	}

	__device__  __host__
	T &operator()(int i, int j, int k, int w, bool check_bounds = true)
	{
		if(((((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1)) ||(w < 0 || w > lengthW-1))|| dev_arr==NULL) && check_bounds)
		{
			// Take appropriate action here. This is just a placeholder response.
			printf("Array not INITIALIZED or Index [%d][%d][%d][%d] of smart Array is out of bounds\n",i,j,k,w);			
			////exit(1);
		}
		int _idx = i + j * lengthX + k * lengthX * lengthY + w * lengthX * lengthY * lengthZ;
		idx = _idx;
		return dev_arr[_idx];
	}

	__device__  __host__
	T &operator[](int _idx)
	{
		if((idx < 0 || idx > lengthX * lengthY * lengthZ * lengthW -1) || dev_arr==NULL)
		{
			// Take appropriate action here. This is just a placeholder response.
			printf("Array not INITIALIZED or Index [%d] of smart Array is out of bounds\n",idx);			
			////exit(1);
		}
		idx = _idx;
		return dev_arr[_idx];
	}

//	__device__  __host__
//		T getVal(int _idx) {idx = _idx; return dev_arr[_idx];}
//
//	__device__  __host__
//const T &operator[](int _idx) const
//	{
//		if((idx < 0 || idx > lengthX * lengthY * lengthZ * lengthW -1) || dev_arr==NULL)
//		{
//			// Take appropriate action here. This is just a placeholder response.
//			printf("Array not INITIALIZED or Index [%d] of smart Array is out of bounds\n",idx);			
//			////exit(1);
//		}
//		//idx = _idx;
//		//return dev_arr[_idx];
//		return getVal(_idx);
//	}
//
//	
	__device__  __host__
	cudaError_t &operator=(smartArrayWrapper<T,smartDevice> &other)
	{
		cudaError_t cudaStatus;
		//other._is_copy = true; ////prevent destructor from deleting array pointer

		//if (this == &other)////check for self assignment
		//{
		//	printf("Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//////exit(1);			
		//}

		//if(dev_arr == NULL) ////allocate memory if it does not exist
		//{
		//	dev_arr = smartArray<T,mem_loc>(other.getlen(), cudaStatus);
		//	lengthX = other.getlenX();
		//	lengthY = other.getlenY();
		//	lengthZ = other.getlenZ();
		//	lengthW = other.getlenW();
		//}
		
		//// size of data to copy, using the least of the 2 to avoid errors
		unsigned int copySize = (getlen() > other.getlen()) ? getlen() : other.getlen();

		if(array_type == smartHost || array_type == smartPinnedHost) ////host -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice || array_type == smartInlineArray) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				if (device_id == other.deviceID())
				{
					cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
				}
				else
				{
					cudaStatus = smartCopyDevice2DevicePeer<T>(dev_arr,device_id, other.inner_ptr(),other.deviceID(), copySize);
				}
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			printf("Cannot copy unsupported type\n");	
			cudaStatus = cudaErrorMemoryAllocation;
			return cudaStatus;
		}		
		return cudaStatus;
	}

	__device__  __host__
	cudaError_t &operator=(smartArrayWrapper<T,smartHost> &other)
	{
		cudaError_t cudaStatus;
		//other._is_copy = true; ////prevent destructor from deleting array pointer

		//if (this == &other)////check for self assignment
		//{
		//	printf("Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//////exit(1);			
		//}
		
		
		//if(dev_arr == NULL) ////allocate memory if it does not exist
		//{
		//	dev_arr = smartArray<T,mem_loc>(other.getlen(), cudaStatus);
		//	lengthX = other.getlenX();
		//	lengthY = other.getlenY();
		//	lengthZ = other.getlenZ();
		//	lengthW = other.getlenW();
		//}

		//// size of data to copy, using the least of the 2 to avoid errors
		unsigned int copySize = (getlen() > other.getlen()) ? getlen() : other.getlen();

		if(array_type == smartHost || array_type == smartPinnedHost) ////host -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice || array_type == smartInlineArray) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				if (device_id == other.deviceID())
				{
					cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
				}
				else
				{
					cudaStatus = smartCopyDevice2DevicePeer<T>(dev_arr,device_id, other.inner_ptr(),other.deviceID(), copySize);
				}
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			printf("Cannot copy unsupported type\n");	
			cudaStatus = cudaErrorMemoryAllocation;
			return cudaStatus;
		}		
		
		return cudaStatus;
	}

	__device__  __host__
	cudaError_t &operator=(smartArrayWrapper<T,smartPinnedHost> &other)
	{
		cudaError_t cudaStatus;
		//other._is_copy = true; ////prevent destructor from deleting array pointer

		//if (this == &other)////check for self assignment
		//{
		//	printf("Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//////exit(1);			
		//}
		
		
		//if(dev_arr == NULL) ////allocate memory if it does not exist
		//{
		//	dev_arr = smartArray<T,mem_loc>(other.getlen(), cudaStatus);
		//	lengthX = other.getlenX();
		//	lengthY = other.getlenY();
		//	lengthZ = other.getlenZ();
		//	lengthW = other.getlenW();
		//}

		//// size of data to copy, using the least of the 2 to avoid errors
		unsigned int copySize = (getlen() > other.getlen()) ? getlen() : other.getlen();

		if(array_type == smartHost || array_type == smartPinnedHost) ////host -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice || array_type == smartInlineArray) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				if (device_id == other.deviceID())
				{
					cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
				}
				else
				{
					cudaStatus = smartCopyDevice2DevicePeer<T>(dev_arr,device_id, other.inner_ptr(),other.deviceID(), copySize);
				}
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			printf("Cannot copy unsupported type\n");	
			cudaStatus = cudaErrorMemoryAllocation;
			return cudaStatus;
		}		
		
		return cudaStatus;
	}

	__device__  __host__
	cudaError_t &operator=(smartArrayWrapper<T,smartInlineArray> &other)
	{
		cudaError_t cudaStatus;
		//other._is_copy = true; ////prevent destructor from deleting array pointer

		//if (this == &other)////check for self assignment
		//{
		//	printf("Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//////exit(1);			
		//}
		
		
		//if(dev_arr == NULL) ////allocate memory if it does not exist
		//{
		//	dev_arr = smartArray<T,mem_loc>(other.getlen(), cudaStatus);
		//	lengthX = other.getlenX();
		//	lengthY = other.getlenY();
		//	lengthZ = other.getlenZ();
		//	lengthW = other.getlenW();
		//}

		//// size of data to copy, using the least of the 2 to avoid errors
		unsigned int copySize = (getlen() > other.getlen()) ? getlen() : other.getlen();

		if(array_type == smartHost || array_type == smartPinnedHost) ////host -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice || array_type == smartInlineArray) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				if (device_id == other.deviceID())
				{
					cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
				}
				else
				{
					cudaStatus = smartCopyDevice2DevicePeer<T>(dev_arr,device_id, other.inner_ptr(),other.deviceID(), copySize);
				}
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			printf("Cannot copy unsupported type\n");	
			cudaStatus = cudaErrorMemoryAllocation;
			return cudaStatus;
		}		
		
		return cudaStatus;
	}


template <typename custom>
	//__device__  __host__
	cudaError_t &operator=(custom other)
	{
		cudaError_t cudaStatus;				
		//// size of data to copy, using the least of the 2 to avoid errors
		T* arr = &other[0];
		int size = other.size();
		int type = smartHost;
		int dev_id = 0; //// __device_id; //// global device id

		unsigned int copySize = (getlen() > size) ? getlen() : size;

		if(array_type == smartHost || array_type == smartPinnedHost) ////host -> host/device copies
		{
			if(type == smartHost || array_type == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Host<T>(dev_arr,arr,copySize);
			}
			else if (type == smartDevice || type == smartInlineArray)
			{
				if (device_id == dev_id)
				{
					cudaStatus = smartCopyDevice2Device<T>(dev_arr,arr,copySize);
				}
				else
				{
					cudaStatus = smartCopyDevice2DevicePeer<T>(dev_arr,device_id, arr,dev_id, copySize);
				}
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice || array_type == smartInlineArray) ////device -> host/device copies
		{
			if(type == smartHost || array_type == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,arr,copySize);
			}
			else if (type == smartDevice || type == smartInlineArray)
			{
				if (device_id == dev_id)
				{
					cudaStatus = smartCopyDevice2Device<T>(dev_arr,arr,copySize);
				}
				else
				{
					cudaStatus = smartCopyDevice2DevicePeer<T>(dev_arr,device_id, arr, dev_id, copySize);
				}
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			printf("Cannot copy unsupported type\n");	
			cudaStatus = cudaErrorMemoryAllocation;
			return cudaStatus;
		}		
		return cudaStatus;
	}

	
	//////NAVIGATION ///////
	////increment and decrement operators
	__device__  __host__
	T &operator++()
	{
		idx++;
		////if (idx > getlen() -1) {idx = getlen() -1;}
		return dev_arr[idx];		
	}

	__device__  __host__
	T operator++(int)
	{
		idx++;
		int temp = dev_arr[idx];
		return temp;			
	}

	__device__  __host__
	T &operator--()
	{
		idx--;
		////if (idx > getlen() -1) {idx = getlen() -1;}
		return dev_arr[idx];		
	}

	__device__  __host__
	T operator--(int)
	{
		idx--;
		int temp = dev_arr[idx];
		return temp;				
	}

	//////smartArrayWrapper.At() Implementations //////
	__device__  __host__
	T &at(int i) 
	{		
		return operator()(i);
	}

	__device__  __host__
	T &at(int i, int j)
	{
		return operator()(i,j);
	}

	__device__  __host__
	T &at(int i, int j, int k)
	{
		return operator()(i,j,k);
	}

	__device__  __host__
	T &at(int i, int j, int k, int w)
	{
		return operator()(i,j,k,w);
	}



	////PEEK, ADVANCE, SEEK
	__device__  __host__
	T peek(int adv = 1) ////view data from advance without moving current index position
	{
		unsigned int temp = idx + adv;
		return dev_arr[temp];			
	}

	__device__  __host__
	T adv(int adv = 1) //// advance adv steps from current index position////
	{
		idx = idx + adv;
		return dev_arr[idx];			
	}

	__device__  __host__
	T seek(int adv = 1) //// move current index position////
	{
		idx = adv;
		return dev_arr[idx];			
	}

	/////4 DIMENSIONAL SEEK,PEEK, ADVANCE
	__device__  __host__
	T seek4(int i) //// move current index position////
	{
		idx = getIndex(i);
		return dev_arr[idx];			
	}

	__device__  __host__
	T seek4(int i, int j) //// move current index position////
	{
		idx = getIndex(i, j);
		return dev_arr[idx];			
	}

	__device__  __host__
	T seek4(int i, int j, int k) //// move current index position////
	{
		idx = getIndex(i, j, k);
		return dev_arr[idx];			
	}

	__device__  __host__
	T seek4(int i, int j, int k, int w) //// move current index position////
	{
		idx = getIndex(i, j, k, w);
		return dev_arr[idx];			
	}

	__device__  __host__
	T* seek4_ref(int i, int j, int k, int w) //// reference to move current index position////
	{
		idx = getIndex(i, j, k, w);
		return &dev_arr[idx];			
	}

	////4D advance
	__device__  __host__
	T adv4(int i) //// move current index position////
	{
		idx += getIndex(i);
		return dev_arr[idx];			
	}

	__device__  __host__
	T adv4(int i, int j) //// move current index position////
	{
		idx += getIndex(i, j);
		return dev_arr[idx];			
	}

	__device__  __host__
	T adv4(int i, int j, int k) //// move current index position////
	{
		idx += getIndex(i, j, k);
		return dev_arr[idx];			
	}

	__device__  __host__
	T adv4(int i, int j, int k, int w) //// move current index position////
	{
		idx += getIndex(i, j, k, w);
		return dev_arr[idx];			
	}

	__device__  __host__
	T* adv4_ref(int i, int j, int k, int w) //// move current index position////
	{
		idx += getIndex(i, j, k, w);
		return &dev_arr[idx];			
	}

	////4D Peek
	__device__  __host__
	T peek4(int i) //// move current index position////
	{
		int _idx = idx +  getIndex(i);
		return dev_arr[_idx];			
	}

	__device__  __host__
	T peek4(int i, int j) //// move current index position////
	{
		int _idx = idx +  getIndex(i, j);
		return dev_arr[_idx];			
	}

	__device__  __host__
	T peek4(int i, int j, int k) //// move current index position////
	{
		int _idx = idx +  getIndex(i, j, k);
		return dev_arr[_idx];			
	}

	__device__  __host__
	T peek4(int i, int j, int k, int w) //// move current index position////
	{
		int _idx = idx +  getIndex(i, j, k, w);
		return dev_arr[_idx];			
	}


	////return current index position
	__device__  __host__
	int pos()
	{		
		return idx;			
	}

	/////specialized views section for navigation
	__device__  __host__
	int vpos()
	{		
		return vidx;			
	}

	__device__  __host__
	T vpeek(int adv = 1) ////view data from advance without moving current index position
	{
		unsigned int temp = vidx + adv;
		return dev_arr[temp];			
	}

	__device__  __host__
	T vadv(int adv = 1) //// advance adv steps from current index position////
	{
		vidx = idx + adv;
		return dev_arr[vidx];			
	}

	__device__  __host__
	T vseek(int adv = 1) //// move current index position////
	{
		vidx = adv;
		return dev_arr[vidx];			
	}

	__device__  __host__
	T vadv4(int i, int j=0, int k=0, int w=0) //// move current index position////
	{
		vidx += getViewIndex(i, j, k, w);
		return dev_arr[vidx];			
	}

	__device__  __host__
	T vseek4(int i, int j, int k, int w) //// move current index position////
	{
		vidx = getViewIndex(i, j, k, w);
		return dev_arr[vidx];			
	}

	__device__  __host__
	T vpeek4(int i, int j=0, int k=0, int w=0) //// move current index position////
	{
		int _idx = vidx +  getViewIndex(i, j, k, w);
		return dev_arr[_idx];			
	}


	//__device__  __host__ ////original copy engine
	//cudaError_t copy(smartArrayWrapper other)
	//{
	//	cudaError_t cudaStatus;
	//	if (this == &other) ////check for self assignment
	//	{
	//		printf("Cannot self assign\n");	
	//		cudaStatus  = cudaErrorMemoryAllocation;
	//		return cudaStatus;
	//		//////exit(1);			
	//	}
	//	
	//	//// size of data to copy, using the least of the 2 to avoid errors
	//	unsigned int copySize = (getlen() > other.getlen()) ? getlen() : other.getlen();
	//	if(array_type == smartHost) ////host -> host/device copies
	//	{
	//		if(other.getType() == smartHost)
	//		{
	//			cudaStatus = smartCopyHost2Host<T>(dev_arr,other.inner_ptr(),copySize);
	//		}
	//		else if (other.getType() == smartDevice)
	//		{
	//			cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
	//		}
	//		else
	//		{
	//			////unsupported copy types
	//			printf("Cannot copy unsupported type\n");	
	//			cudaStatus  = cudaErrorMemoryAllocation;
	//			return cudaStatus;
	//		}
	//	}
	//	else if (array_type == smartDevice) ////device -> host/device copies
	//	{
	//		if(other.getType() == smartHost)
	//		{
	//			cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
	//		}
	//		else if (other.getType() == smartDevice)
	//		{
	//			cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
	//		}
	//		else
	//		{
	//			////unsupported copy types
	//			printf("Cannot copy unsupported type\n");	
	//			cudaStatus  = cudaErrorMemoryAllocation;
	//			return cudaStatus;
	//		}
	//	}
	//	else ////unsupported copy types
	//	{
	//		printf("Cannot copy unsupported type\n");	
	//		cudaStatus  = cudaErrorMemoryAllocation;
	//		return cudaStatus;
	//	}
	//	
	//	
	//	return cudaStatus;
	//}

	__device__  __host__
	cudaError_t copy(smartArrayWrapper<T,smartDevice> &other)
	{
		cudaError_t cudaStatus;
		//other._is_copy = true; ////prevent destructor from deleting array pointer

		//if (this == &other)////check for self assignment
		//{
		//	printf("Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//////exit(1);			
		//}

		//if(dev_arr == NULL) ////allocate memory if it does not exist
		//{
		//	dev_arr = smartArray<T,mem_loc>(other.getlen(), cudaStatus);
		//	lengthX = other.getlenX();
		//	lengthY = other.getlenY();
		//	lengthZ = other.getlenZ();
		//	lengthW = other.getlenW();
		//}
		
		//// size of data to copy, using the least of the 2 to avoid errors
		unsigned int copySize = (getlen() > other.getlen()) ? getlen() : other.getlen();

		if(array_type == smartHost || array_type == smartPinnedHost) ////host -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice || array_type == smartInlineArray) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			printf("Cannot copy unsupported type\n");	
			cudaStatus = cudaErrorMemoryAllocation;
			return cudaStatus;
		}		
		return cudaStatus;
	}

	__device__  __host__
	cudaError_t copy(smartArrayWrapper<T,smartHost> &other)
	{
		cudaError_t cudaStatus;
		//other._is_copy = true; ////prevent destructor from deleting array pointer

		//if (this == &other)////check for self assignment
		//{
		//	printf("Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//////exit(1);			
		//}
		
		
		//if(dev_arr == NULL) ////allocate memory if it does not exist
		//{
		//	dev_arr = smartArray<T,mem_loc>(other.getlen(), cudaStatus);
		//	lengthX = other.getlenX();
		//	lengthY = other.getlenY();
		//	lengthZ = other.getlenZ();
		//	lengthW = other.getlenW();
		//}

		//// size of data to copy, using the least of the 2 to avoid errors
		unsigned int copySize = (getlen() > other.getlen()) ? getlen() : other.getlen();

		if(array_type == smartHost || array_type == smartPinnedHost) ////host -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice || array_type == smartInlineArray) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			printf("Cannot copy unsupported type\n");	
			cudaStatus = cudaErrorMemoryAllocation;
			return cudaStatus;
		}		
		
		return cudaStatus;
	}

	__device__  __host__
	cudaError_t copy(smartArrayWrapper<T,smartPinnedHost> &other)
	{
		cudaError_t cudaStatus;
		//other._is_copy = true; ////prevent destructor from deleting array pointer

		//if (this == &other)////check for self assignment
		//{
		//	printf("Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//////exit(1);			
		//}
		
		
		//if(dev_arr == NULL) ////allocate memory if it does not exist
		//{
		//	dev_arr = smartArray<T,mem_loc>(other.getlen(), cudaStatus);
		//	lengthX = other.getlenX();
		//	lengthY = other.getlenY();
		//	lengthZ = other.getlenZ();
		//	lengthW = other.getlenW();
		//}

		//// size of data to copy, using the least of the 2 to avoid errors
		unsigned int copySize = (getlen() > other.getlen()) ? getlen() : other.getlen();

		if(array_type == smartHost || array_type == smartPinnedHost) ////host -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice || array_type == smartInlineArray) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			printf("Cannot copy unsupported type\n");	
			cudaStatus = cudaErrorMemoryAllocation;
			return cudaStatus;
		}		
		
		
		return cudaStatus;
	}

		__device__  __host__
	cudaError_t copy(smartArrayWrapper<T,smartInlineArray> &other)
	{
		cudaError_t cudaStatus;
		//other._is_copy = true; ////prevent destructor from deleting array pointer

		//if (this == &other)////check for self assignment
		//{
		//	printf("Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//////exit(1);			
		//}
		
		
		//if(dev_arr == NULL) ////allocate memory if it does not exist
		//{
		//	dev_arr = smartArray<T,mem_loc>(other.getlen(), cudaStatus);
		//	lengthX = other.getlenX();
		//	lengthY = other.getlenY();
		//	lengthZ = other.getlenZ();
		//	lengthW = other.getlenW();
		//}

		//// size of data to copy, using the least of the 2 to avoid errors
		unsigned int copySize = (getlen() > other.getlen()) ? getlen() : other.getlen();

		if(array_type == smartHost || array_type == smartPinnedHost) ////host -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice || array_type == smartInlineArray) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice || other.getType() == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			printf("Cannot copy unsupported type\n");	
			cudaStatus = cudaErrorMemoryAllocation;
			return cudaStatus;
		}		
		
		
		return cudaStatus;
	}


	__device__  __host__
	cudaError_t copy(T* other, unsigned int other_size, int other_array_type = 1)
	{
		cudaError_t cudaStatus;
		//if (this != &other)
		//{
		//	printf("Cannot copy unsupported type\n");	
		//	cudaStatus != cudaSuccess;
		//	return cudaStatus;
		//	//////exit(1);			
		//}
		
		//// size of data to copy, using the least of the 2 to avoid errors
		int copySize = (getlen() > other_size) ? getlen() : other_size;

		if(array_type == smartHost || array_type == smartPinnedHost) ////host -> host/device copies
		{
			if(other_array_type == smartHost || other_array_type == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Host<T>(dev_arr,other,copySize);
			}
			else if (other_array_type == smartDevice || other_array_type == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other,copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice || array_type == smartInlineArray) ////device -> host/device copies
		{
			if(other_array_type == smartHost || other_array_type == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other,copySize);
			}
			else if (other_array_type == smartDevice || other_array_type == smartInlineArray)
			{
				cudaStatus = smartCopyDevice2Device<T>(dev_arr,other,copySize);
			}
			else
			{
				////unsupported copy types
				printf("Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			printf("Cannot copy unsupported type\n");	
			cudaStatus = cudaErrorMemoryAllocation;
			return cudaStatus;
		}		
		
		
		return cudaStatus;
	}

	__device__  __host__
	cudaError_t destroy()
	{
		cudaError_t cudaStatus;
		//cudaStatus = smartFree<T>(*this);	
		cudaStatus = smartFree<T>(inner_ptr(),array_type);	
		initializeArray();
		return cudaStatus;
	}
	
	// Return the pointer to the array.
	__device__  __host__
	T* inner_ptr() { return dev_arr; }
	__device__  __host__
	T* &inner_ptr_ref() { return dev_arr; }

	// Return the location of the array.
	__device__  __host__
	int getType() { return array_type; }

	// Return the length of the array.
	//__device__  __host__
	//size_t size() { return lengthX * lengthY * lengthZ * lengthW; }

	__device__  __host__
	size_t size() { return lengthX * lengthY * lengthZ * lengthW; }
	
	__device__  __host__
	int getlen(int _index = -1) 
	{
		int len = 0;
		switch (_index)
		{
		case 0:
			len = getlenX();
			break;
		case 1:
			len = getlenY();
			break;
		case 2:
			len = getlenZ();
			break;
		case 3:
			len = getlenW();
			break;

		default:
			len = lengthX * lengthY * lengthZ * lengthW; 
			break;
		}
		
		return len;
	}


	__device__  __host__
	int getlenX() { return lengthX; }
	__device__  __host__
	int getlenY() { return lengthY; }
	__device__  __host__
	int getlenZ() { return lengthZ; }
	__device__  __host__
	int getlenW() { return lengthW; }

	__device__  __host__
	int deviceID() { return device_id; }

	// Specialized views of the array.
	__device__  __host__
	int getViewLen(int _index = -1) 
	{
		int len = 0;
		switch (_index)
		{
		case 0:
			len = getViewLenX();
			break;
		case 1:
			len = getViewLenY();
			break;
		case 2:
			len = getViewLenZ();
			break;
		case 3:
			len = getViewLenW();
			break;

		default:
			len = getViewLenX() * getViewLenY() * getViewLenZ() * getViewLenW(); 
			break;
		}
		
		return len;
	}

	__device__  __host__
	int getViewLenX() { return vX; }
	__device__  __host__
	int getViewLenY() { return vY; }
	__device__  __host__
	int getViewLenZ() { return vZ; }
	__device__  __host__
	int getViewLenW() { return vW; }

	__device__  __host__
	bool setViewDim(int X = 1, int Y = 1, int Z = 1, int W = 1 ) 
	{ 
		bool setSuccess = true;
		if (X < 0) {setSuccess = false; return setSuccess;}
		if (Y < 0) {setSuccess = false; return setSuccess;}
		if (Z < 0) {setSuccess = false; return setSuccess;}
		if (W < 0) {setSuccess = false; return setSuccess;}

		if((X * Y * Z * W >= 0) && (X * Y * Z * W <= lengthX * lengthY * lengthZ * lengthW))
		{
			vX = X; vY = Y; vZ = Z; vW = W;
			setSuccess = true;
		}
		else
		{
			setSuccess = false; ////set view failed
		}

		return setSuccess;
	}

	__device__  __host__
	T viewAt_1D(int i)
	{
		if(i < 0 || i > vX * vY * vZ * vW -1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			printf("View Index [%d] of smart Array is out of bounds\n",i);			
			////exit(1);
		}
		vidx = 1;
		return dev_arr[i];
	}

	__device__  __host__
	T viewAt_2D(int i, int j)
	{
		if((i < 0 || i > vX-1) || (j < 0 || j > vY-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			printf("View Index [%d][%d] of smart Array is out of bounds\n",i,j);			
			////exit(1);
		}
		int _idx = i + j * vX;
		vidx = _idx;
		return dev_arr[_idx];
	}

	__device__  __host__
	T viewAt_3D(int i, int j, int k)
	{
		if((i < 0 || i > vX-1) || (j < 0 || j > vY-1) || (k < 0 || k > vZ-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			printf("View Index [%d][%d][%d] of smart Array is out of bounds\n",i,j,k);			
			////exit(1);
		}
		int _idx = i + j * vX + k * vX * vY;
		vidx = _idx;
		return dev_arr[_idx];
	}

	__device__  __host__
	T viewAt_4D(int i, int j, int k, int w)
	{
		if(((i < 0 || i > vX-1) || (j < 0 || j > vY-1) || (k < 0 || k > vZ-1)) ||(w < 0 || w > vW-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			printf("View Index [%d][%d][%d][%d] of smart Array is out of bounds\n",i,j,k,w);			
			////exit(1);
		}
		int _idx = i + j * vX + k * vX * vY + w * vX * vY * vZ;
		vidx = _idx;
		return dev_arr[_idx];
	}

	__device__  __host__
	T viewAt(int i, int j=0, int k=0, int w=0)
	{
		if(((i < 0 || i > vX-1) || (j < 0 || j > vY-1) || (k < 0 || k > vZ-1)) ||(w < 0 || w > vW-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			printf("View Index [%d][%d][%d][%d] of smart Array is out of bounds\n",i,j,k,w);			
			////exit(1);
		}
		int _idx = i + j * vX + k * vX * vY + w * vX * vY * vZ;
		vidx = _idx;
		return dev_arr[_idx];
	}

	//////////////////////ITERATORS FOR STL LIKE ALGORITHMS /////////////////////////////
	//__device__  __host__
    typedef T* iterator;
	//__device__  __host__
    typedef const T* const_iterator;
	
	__device__  __host__
	iterator begin() { return &dev_arr[0]; }
	__device__  __host__
    const_iterator begin() const { return &dev_arr[0]; }
	__device__  __host__
    iterator end() { return &dev_arr[array_size]; }
	__device__  __host__
    const_iterator end() const { return &dev_arr[array_size]; }

	////Beta Navigation along Dimensions//////////////
	__device__  __host__
	T nextX(int step = 1)
	{		
		return adv4(step,0,0,0);
	}

	__device__  __host__
	T prevX(int step = 1)
	{		
		return adv4(-step,0,0,0);
	}

	__device__  __host__
	T nextY(int step = 1)
	{		
		return adv4(0,step,0,0);
	}

	__device__  __host__
	T prevY(int step = 1)
	{		
		return adv4(0,-step,0,0);
	}

	__device__  __host__
	T nextZ(int step = 1)
	{		
		return adv4(0,0,step,0);
	}

	__device__  __host__
	T prevZ(int step = 1)
	{		
		return adv4(0,0,-step,0);
	}

	__device__  __host__
	T nextW(int step = 1)
	{		
		return adv4(0,0,0,step);
	}

	__device__  __host__
	T prevW(int step = 1)
	{		
		return adv4(0,0,0,-step);
	}

//////////////////////////////////////////////////////////////
	/////BETA CLASSES FOR DIMENSIONAL NAVIGATION////
//////////////////////////////////////////////////////////////
	//template<typename T> 
	//class dimNav
	//{
	//private: 
	//	int dimIdx; ////= 0; ////0,1,2,3,4 -> x,y,z,w
	//	
	//public:
	//	//__device__  __host__
	//	//dimNav()
	//	//{
	//	//	/*int _dimIdx = 0
	//	//	dimIdx = dimIdx;*/
	//	//}

	//	__device__  __host__
	//	dimNav(int _dimIdx = 0): dimIdx(_dimIdx)
	//	{
	//		//dimIdx = dimIdx;
	//	}

	//	__device__  __host__
	//	T &operator[](int _idx)
	//	{		
	//		switch (_idx)
	//		{
	//		case 0:////x navigation
	//			return adv4(_idx,0,0,0);
	//			break;
	//		case 1: /////y navigation
	//			return adv4(0,_idx,0,0);
	//			break;
	//		case 2://///z navigation
	//			return adv4(0,0,_idx,0);
	//			break;
	//		case 3://///w navigation
	//			return adv4(0,0,0,_idx);
	//			break;
	//			
	//		default://// default is x navigation
	//			return adv4(_idx,0,0,0);
	//			break;
	//		}		
	//		
	//	}

	//	__device__  __host__
	//	void setLocation (int locX = 0, int locY = 0, int locZ = 0, int locW = 0)
	//	{
	//		adv4(locX, locY, locZ, locW);
	//	}

	//	
	//	__device__  __host__
	//	T* begin() 
	//	{ 
	//		int _idx = 0;
	//		switch (_idx)
	//		{
	//		case 0:////x navigation
	//			return &seek4(_idx,iy,iz,iw);
	//			break;
	//		case 1: /////y navigation
	//			return &seek4(ix,_idx,iz,iw);
	//			break;
	//		case 2://///z navigation
	//			return &seek4(ix,iy,_idx,iw);
	//			break;
	//		case 3://///w navigation
	//			return &seek4(ix,iy,iz,_idx);
	//			break;
	//			
	//		default://// default is x navigation
	//			return &seek4(_idx,iy,iz,iw);
	//			break;
	//		}		
	//	}

	//	__device__  __host__
	//	const T* begin() const 
	//	{
	//		return const_cast<const T*>(begin());
	//		//int _idx = 0;
	//		//switch (_idx)
	//		//{
	//		//case 0:////x navigation

	//		//	return &seek4(_idx,iy,iz,iw);
	//		//	break;
	//		//case 1: /////y navigation
	//		//	return &seek4(ix,_idx,iz,iw);
	//		//	break;
	//		//case 2://///z navigation
	//		//	return &seek4(ix,iy,_idx,iw);
	//		//	break;
	//		//case 3://///w navigation
	//		//	return &seek4(ix,iy,iz,_idx);
	//		//	break;
	//		//	
	//		//default://// default is x navigation
	//		//	return &seek4(_idx,iy,iz,iw);
	//		//	break;
	//		//}		
	//	
	//	}

	//	__device__  __host__
	//	T* end() 
	//	{ 
	//		int _idx = getlen(dimIdx);
	//		switch (_idx)
	//		{
	//		case 0:////x navigation
	//			return &seek4(_idx,iy,iz,iw);
	//			break;
	//		case 1: /////y navigation
	//			return &seek4(ix,_idx,iz,iw);
	//			break;
	//		case 2://///z navigation
	//			return &seek4(ix,iy,_idx,iw);
	//			break;
	//		case 3://///w navigation
	//			return &seek4(ix,iy,iz,_idx);
	//			break;
	//			
	//		default://// default is x navigation
	//			return &seek4(_idx,iy,iz,iw);
	//			break;
	//		}		
	//			
	//	}


	//	__device__  __host__
	//	const T* end() const 
	//	{ 
	//		return const_cast<const T*>(end());
	//		//int _idx = getlen(dimIdx);
	//		//switch (_idx)
	//		//{
	//		//case 0:////x navigation
	//		//	return &seek4(_idx,iy,iz,iw);
	//		//	break;
	//		//case 1: /////y navigation
	//		//	return &seek4(ix,_idx,iz,iw);
	//		//	break;
	//		//case 2://///z navigation
	//		//	return &seek4(ix,iy,_idx,iw);
	//		//	break;
	//		//case 3://///w navigation
	//		//	return &seek4(ix,iy,iz,_idx);
	//		//	break;
	//		//	
	//		//default://// default is x navigation
	//		//	return &seek4(_idx,iy,iz,iw);
	//		//	break;
	//		//}		
	//	
	//	}


	//};

	//template<typename T> 
	//class viewNav
	//{
	//private: 
	//	int dimIdx; ////= 0; ////0,1,2,3,4 -> x,y,z,w
	//	
	//public:
	//	__device__  __host__
	//	viewNav(int _dimIdx = 0)
	//	{
	//		dimIdx = dimIdx;
	//	}

	//	__device__  __host__
	//	T &operator[](int _idx)
	//	{		
	//		switch (_idx)
	//		{
	//		case 0:////x navigation
	//			return vadv4(_idx,0,0,0);
	//			break;
	//		case 1: /////y navigation
	//			return vadv4(0,_idx,0,0);
	//			break;
	//		case 2://///z navigation
	//			return vadv4(0,0,_idx,0);
	//			break;
	//		case 3://///w navigation
	//			return vadv4(0,0,0,_idx);
	//			break;
	//			
	//		default://// default is x navigation
	//			return vadv4(_idx,0,0,0);
	//			break;
	//		}		
	//		
	//	}

	//	__device__  __host__
	//	void setLocation (int locX = 0, int locY = 0, int locZ = 0, int locW = 0)
	//	{
	//		vadv4(locX, locY, locZ, locW);
	//	}

	//	
	//	__device__  __host__
	//	T* begin() 
	//	{ 
	//		int _idx = 0;
	//		switch (_idx)
	//		{
	//		case 0:////x navigation
	//			return &vseek4(_idx,viy,viz,viw);
	//			break;
	//		case 1: /////y navigation
	//			return &vseek4(vix,_idx,viz,viw);
	//			break;
	//		case 2://///z navigation
	//			return &vseek4(vix,viy,_idx,viw);
	//			break;
	//		case 3://///w navigation
	//			return &vseek4(vix,viy,viz,_idx);
	//			break;
	//			
	//		default://// default is x navigation
	//			return &vseek4(_idx,viy,viz,viw);
	//			break;
	//		}		
	//	}

	//	__device__  __host__
	//	const T* begin() const 
	//	{
	//		return const_cast<const T*>(begin());
	//		
	//	}

	//	__device__  __host__
	//	T* end() 
	//	{ 
	//		int _idx = getViewLen(dimIdx);
	//		switch (_idx)
	//		{
	//		case 0:////x navigation
	//			return &vseek4(_idx,viy,viz,viw);
	//			break;
	//		case 1: /////y navigation
	//			return &vseek4(vix,_idx,viz,viw);
	//			break;
	//		case 2://///z navigation
	//			return &vseek4(vix,viy,_idx,viw);
	//			break;
	//		case 3://///w navigation
	//			return &vseek4(vix,viy,viz,_idx);
	//			break;
	//			
	//		default://// default is x navigation
	//			return &vseek4(_idx,viy,viz,viw);
	//			break;
	//		}		
	//			
	//	}


	//	__device__  __host__
	//	const T* end() const 
	//	{ 
	//		return const_cast<const T*>(end());	
	//	
	//	}


	//};

	//typedef dimNav<T> dimXarray;
	/////dimXarray(0); /// xarr;


};


/////END OF SMART ARRAY WRAPPER//////////////////////
//
//template<typename T, int mem_loc> 
//class smartNavigator
//{
//	//friend class smartArrayWrapper<T,mem_loc>;
//private: 
//	int dimIdx; ////= 0; ////0,1,2,3,4 -> x,y,z,w
//	smartArrayWrapper<T,mem_loc> arr;
//
//public:
//	//__device__  __host__
//	//dimNav()
//	//{
//	//	/*int _dimIdx = 0
//	//	dimIdx = dimIdx;*/
//	//}
//
//	__device__  __host__
//	smartNavigator(smartArrayWrapper<T,mem_loc> &_arr, int _dimIdx = 0)////: dimIdx(_dimIdx)
//	{
//		dimIdx = dimIdx;
//		arr = &_arr;
//	}
//
//
//	__device__  __host__
//	T &operator[](int _idx)
//	{		
//		switch (dimIdx)
//		{
//		case 0:////x navigation
//			return arr.adv4(_idx,0,0,0);
//			break;
//		case 1: /////y navigation
//			return arr.adv4(0,_idx,0,0);
//			break;
//		case 2://///z navigation
//			return arr.adv4(0,0,_idx,0);
//			break;
//		case 3://///w navigation
//			return arr.adv4(0,0,0,_idx);
//			break;
//				
//		default://// default is x navigation
//			return arr.adv4(_idx,0,0,0);
//			break;
//		}		
//			
//	}
//
//	__device__  __host__
//	void setLocation (int locX = 0, int locY = 0, int locZ = 0, int locW = 0)
//	{
//		arr.adv4(locX, locY, locZ, locW);
//	}
//
//
//	__device__  __host__
//	void setIndex (int _dimIdx = 0)
//	{
//		dimIdx = _dimIdx;
//	}
//
//		
//	__device__  __host__
//	T* begin(int _dimIdx = 0) 
//	{ 
//		int _idx = 0;
//
//		dimIdx = _dimIdx;
//
//		int ix = arr.ix, iy = arr.iy, iz = arr.iz, iw = arr.iw;
//
//		switch (_dimIdx)
//		{
//		case 0:////x navigation
//			return  arr.seek4_ref(_idx,iy,iz,iw);
//			break;
//		case 1: /////y navigation
//			return  arr.seek4_ref(ix,_idx,iz,iw);
//			break;
//		case 2://///z navigation
//			return  arr.seek4_ref(ix,iy,_idx,iw);
//			break;
//		case 3://///w navigation
//			return  arr.seek4_ref(ix,iy,iz,_idx);
//			break;
//				
//		default://// default is x navigation
//			return  arr.seek4_ref(_idx,iy,iz,iw);
//			break;
//		}		
//	}
//
//	__device__  __host__
//	const T* begin(int _dimIdx = 0) const 
//	{
//		return const_cast<const T*>(begin(_dimIdx));
//		
//	}
//
//	__device__  __host__
//	T* end(int _dimIdx = 0) 
//	{ 
//		dimIdx = _dimIdx;
//		int _idx = arr.getlen(dimIdx);
//		int ix = arr.ix, iy = arr.iy, iz = arr.iz, iw = arr.iw;
//
//		switch (_dimIdx)
//		{
//		case 0:////x navigation
//			return  arr.seek4_ref(_idx,iy,iz,iw);
//			break;
//		case 1: /////y navigation
//			return  arr.seek4_ref(ix,_idx,iz,iw);
//			break;
//		case 2://///z navigation
//			return  arr.seek4_ref(ix,iy,_idx,iw);
//			break;
//		case 3://///w navigation
//			return  arr.seek4_ref(ix,iy,iz,_idx);
//			break;
//				
//		default://// default is x navigation
//			return  arr.seek4_ref(_idx,iy,iz,iw);
//			break;
//		}		
//				
//	}
//
//
//	__device__  __host__
//	const T* end(int _dimIdx = 0) const 
//	{ 
//		return const_cast<const T*>(end(_dimIdx));
//		
//	}
//	
//};
//
//template<typename T, int mem_loc> 
//class smartViewNavigator
//{
//private: 
//	int dimIdx; ////= 0; ////0,1,2,3,4 -> x,y,z,w
//	smartArrayWrapper<T,mem_loc> arr;
//
//public:
//	//__device__  __host__
//	//dimNav()
//	//{
//	//	/*int _dimIdx = 0
//	//	dimIdx = dimIdx;*/
//	//}
//
//	__device__  __host__
//	smartViewNavigator(smartArrayWrapper<T,mem_loc> _arr, int _dimIdx = 0)////: dimIdx(_dimIdx)
//	{
//		dimIdx = dimIdx;
//		arr = _arr;
//	}
//
//
//	__device__  __host__
//	T &operator[](int _idx)
//	{		
//		switch (dimIdx)
//		{
//		case 0:////x navigation
//			return arr.vadv4(_idx,0,0,0);
//			break;
//		case 1: /////y navigation
//			return arr.vadv4(0,_idx,0,0);
//			break;
//		case 2://///z navigation
//			return arr.vadv4(0,0,_idx,0);
//			break;
//		case 3://///w navigation
//			return arr.vadv4(0,0,0,_idx);
//			break;
//				
//		default://// default is x navigation
//			return arr.vadv4(_idx,0,0,0);
//			break;
//		}		
//			
//	}
//
//	__device__  __host__
//	void setLocation (int locX = 0, int locY = 0, int locZ = 0, int locW = 0)
//	{
//		arr.vadv4(locX, locY, locZ, locW);
//	}
//
//
//	__device__  __host__
//	void setIndex (int _dimIdx = 0)
//	{
//		dimIdx = _dimIdx;
//	}
//
//		
//	__device__  __host__
//	T* begin(int _dimIdx = 0) 
//	{ 
//		int _idx = 0;
//
//		dimIdx = _dimIdx;
//
//		switch (_dimIdx)
//		{
//		case 0:////x navigation
//			return &arr.vseek4(_idx,iy,iz,iw);
//			break;
//		case 1: /////y navigation
//			return &arr.vseek4(ix,_idx,iz,iw);
//			break;
//		case 2://///z navigation
//			return &arr.vseek4(ix,iy,_idx,iw);
//			break;
//		case 3://///w navigation
//			return &arr.vseek4(ix,iy,iz,_idx);
//			break;
//				
//		default://// default is x navigation
//			return &arr.vseek4(_idx,iy,iz,iw);
//			break;
//		}		
//	}
//
//	__device__  __host__
//	const T* begin(int _dimIdx = 0) const 
//	{
//		return const_cast<const T*>(begin(_dimIdx));
//		
//	}
//
//	__device__  __host__
//	T* end(int _dimIdx = 0) 
//	{ 
//		dimIdx = _dimIdx;
//		int _idx = arr.getViewLen(dimIdx);
//
//		switch (_dimIdx)
//		{
//		case 0:////x navigation
//			return &arr.vseek4(_idx,iy,iz,iw);
//			break;
//		case 1: /////y navigation
//			return &arr.vseek4(ix,_idx,iz,iw);
//			break;
//		case 2://///z navigation
//			return &arr.vseek4(ix,iy,_idx,iw);
//			break;
//		case 3://///w navigation
//			return &arr.vseek4(ix,iy,iz,_idx);
//			break;
//				
//		default://// default is x navigation
//			return &arr.vseek4(_idx,iy,iz,iw);
//			break;
//		}		
//				
//	}
//
//
//	__device__  __host__
//	const T* end(int _dimIdx = 0) const 
//	{ 
//		return const_cast<const T*>(end(_dimIdx));
//		
//	}
//	
//};
//


template <typename T> 
	inline __device__  __host__
		cudaError_t smartFree(smartArrayWrapper<T,smartHost> other)
	{
		cudaError_t cudaStatus;
		int array_type = other.getType();
		T* dev_mem = other.inner_ptr();

		////cudaError_t cudaStatus;
		if (array_type == smartHost)
		{
			free(dev_mem);
			dev_mem = NULL;
			cudaStatus = cudaSuccess;
			if (dev_mem != NULL) {
				cudaStatus = cudaErrorMemoryAllocation;
				printf("Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
		}
		else if (array_type == smartDevice)
		{
			cudaFree(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Device Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartPinnedHost)
		{
			cudaFreeHost(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartInlineArray)
		{
			free(dev_mem);
			if(dev_mem != NULL) {cudaStatus = cudaErrorMemoryAllocation; }
			dev_mem = NULL;
			return cudaStatus;
		}
		else
		{
			cudaStatus = cudaErrorMemoryAllocation;
			printf("Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		return cudaStatus;
	}

template <typename T> 
	inline __device__  __host__ cudaError_t smartFree(smartArrayWrapper<T,smartPinnedHost> other)
	{
		cudaError_t cudaStatus;
		int array_type = other.getType();
		T* dev_mem = other.inner_ptr();
		
		////cudaError_t cudaStatus;
		if (array_type == smartHost)
		{
			free(dev_mem);
			dev_mem = NULL;
			cudaStatus = cudaSuccess;
			if (dev_mem != NULL) {
				cudaStatus = cudaErrorMemoryAllocation;
				printf("Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
		}
		else if (array_type == smartDevice)
		{
			cudaFree(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Device Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartPinnedHost)
		{
			cudaFreeHost(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartInlineArray)
		{
			free(dev_mem);
			if(dev_mem != NULL) {cudaStatus = cudaErrorMemoryAllocation; }
			dev_mem = NULL;
			return cudaStatus;
		}
		else
		{
			cudaStatus = cudaErrorMemoryAllocation;
			printf("Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		return cudaStatus;
	}

template <typename T> 
	inline __device__  __host__ cudaError_t smartFree(smartArrayWrapper<T,smartDevice> other)
	{
		cudaError_t cudaStatus;
		int array_type = other.getType();
		T* dev_mem = other.inner_ptr();
		
		////cudaError_t cudaStatus;
		if (array_type == smartHost)
		{
			free(dev_mem);
			dev_mem = NULL;
			cudaStatus = cudaSuccess;
			if (dev_mem != NULL) {
				cudaStatus = cudaErrorMemoryAllocation;
				printf("Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
		}
		else if (array_type == smartDevice)
		{
			cudaFree(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Device Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartPinnedHost)
		{
			cudaFreeHost(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartInlineArray)
		{
			free(dev_mem);
			if(dev_mem != NULL) {cudaStatus = cudaErrorMemoryAllocation; }
			dev_mem = NULL;
			return cudaStatus;
		}
		else
		{
			cudaStatus = cudaErrorMemoryAllocation;
			printf("Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		return cudaStatus;
	}

template <typename T> 
	inline __device__  __host__
		cudaError_t smartFree(smartArrayWrapper<T,smartInlineArray> other)
	{
		cudaError_t cudaStatus;
		int array_type = other.getType();
		T* dev_mem = other.inner_ptr();

		////cudaError_t cudaStatus;
		if (array_type == smartHost)
		{
			free(dev_mem);
			dev_mem = NULL;
			cudaStatus = cudaSuccess;
			if (dev_mem != NULL) {
				cudaStatus = cudaErrorMemoryAllocation;
				printf("Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
		}
		else if (array_type == smartDevice)
		{
			cudaFree(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Device Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartPinnedHost)
		{
			cudaFreeHost(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				printf("Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//////goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartInlineArray)
		{
			free(dev_mem);
			if(dev_mem != NULL) {cudaStatus = cudaErrorMemoryAllocation; }
			dev_mem = NULL;
			return cudaStatus;
		}
		else
		{
			cudaStatus = cudaErrorMemoryAllocation;
			printf("Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		return cudaStatus;
	}

////miscellanous wrappers
class smartDeviceManager
{
	////beta////
private:

	int device_id; //// device id
	bool reset_on_exit; ///used to check if automatic reset of device is called
	bool _is_mapped_device; ///// check if a device has been wrapped

public:

	smartDeviceManager()
	{
		device_id = -1;
		reset_on_exit = false;
		_is_mapped_device = false;
	}

	smartDeviceManager(int dev_id)
	{
		device_id = dev_id;
		reset_on_exit = false;
		_is_mapped_device = true;
	}

	smartDeviceManager(int dev_id, bool _reset_on_exit)
	{
		device_id = dev_id;
		reset_on_exit = _reset_on_exit;
		_is_mapped_device = true;
	}

	~smartDeviceManager()
	{
		if(reset_on_exit)
		{
			//initializeDevice();
			if (_is_mapped_device)
			{
				resetDevice();
			}
		}
	}

	cudaError_t chooseDevice(cudaDeviceProp deviceProp)
	{
		int device;
		cudaError_t cudaStatus;
		cudaStatus = cudaChooseDevice(&device, &deviceProp);
		if (cudaStatus != cudaSuccess) {
			printf("Choose Device failed!  Do you have a CUDA-capable GPU installed?");        
		}
		device_id = device;
		return cudaStatus;
	}

	void mapDevice(int dev_id = 0)
	{
		device_id = dev_id;
		_is_mapped_device = true;
	}

	void removeDeviceMap()
	{
		device_id = -1;
		_is_mapped_device = false;
	}

	cudaError_t setDevice()
	{
		///keeping track of device ////
		//int t_device_id = __device_id;
		//cudaStatus = cudaSetDevice(__device_id); ////retore original device
		cudaError_t cudaStatus;
		cudaStatus = cudaSetDevice(device_id);		
		//cudaStatus = cudaSetDevice(__device_id); ////retore original device
		if (cudaStatus != cudaSuccess) {
			printf("Set Device failed!  Do you have a CUDA-capable GPU installed?\n");        
		}

		return cudaStatus;
	}

	cudaError_t initializeDevice(int dev_id=0)
	{
		device_id = dev_id;
		cudaError_t cudaStatus;
		cudaStatus = cudaSetDevice(device_id);
		if (cudaStatus != cudaSuccess) {
			printf("Initialization failed!  Do you have a CUDA-capable GPU installed?\n"); 
			return cudaStatus;
		}
		_is_mapped_device = true;
		return cudaStatus;
	}

	cudaError_t resetDevice(bool clean_global_state = false)
	{
		cudaError_t cudaStatus;
		int t_device_id = __device_id; //// store current device
		
		setDevice();
		cudaStatus = cudaDeviceReset();

		cudaStatus = cudaSetDevice(t_device_id); ////retore original device	

		if (cudaStatus != cudaSuccess) {
			printf("Device Reset failed!\n");	
			return cudaStatus;
		}

		__clean_globals = clean_global_state;
		return cudaStatus;
	}

	cudaError_t  synchronize()
	{
		cudaError_t cudaStatus;
		int t_device_id = __device_id; //// store current device
		
		if (device_id != __device_id)
		{
			setDevice();
			cudaStatus = cudaDeviceSynchronize();		
			cudaStatus = cudaSetDevice(t_device_id); ////retore original device	
		}
		else
		{
			cudaStatus = cudaDeviceSynchronize();
		}

		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
			
		}
		return cudaStatus;
	}

	////set limits - default in bytes
	cudaError_t  setDeviceLimit(int memory_size)
	{
		cudaError_t cudaStatus;
		int t_device_id = __device_id; //// store current device
		
		setDevice();
		cudaStatus = cudaDeviceSetLimit(cudaLimitMallocHeapSize, memory_size);
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSetLimit returned error code %d!\n", cudaStatus);
			
		}
		cudaStatus = cudaSetDevice(t_device_id); ////retore original device	

		
		return cudaStatus;
	}

	////set limits - default in KB
	cudaError_t  setDeviceLimitKB(int memory_size)
	{
		cudaError_t cudaStatus;
		int t_device_id = __device_id; //// store current device
		
		setDevice();
		cudaStatus = cudaDeviceSetLimit(cudaLimitMallocHeapSize, memory_size*1024);
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSetLimit returned error code %d!\n", cudaStatus);
			
		}
		cudaStatus = cudaSetDevice(t_device_id); ////retore original device	

		
		return cudaStatus;
	}

	////set limits - default in MB
	cudaError_t  setDeviceLimitMB(int memory_size)
	{
		cudaError_t cudaStatus;
		int t_device_id = __device_id; //// store current device
		
		setDevice();
		cudaStatus = cudaDeviceSetLimit(cudaLimitMallocHeapSize, memory_size*1024*1024);
		if (cudaStatus != cudaSuccess) {
			printf("cudaDeviceSetLimit returned error code %d!\n", cudaStatus);
			
		}

		cudaStatus = cudaSetDevice(t_device_id); ////retore original device	

		
		return cudaStatus;
	}

	int  getDeviceID()
	{
		return device_id;
	}

	cudaDeviceProp  getDeviceProperties()
	{
		cudaError_t cudaStatus;
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device_id);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Device selection and properties failed!\n", cudaStatus);			
		}

		return deviceProp;
	}

	cudaDeviceProp  printDeviceProperties()
	{
		cudaError_t cudaStatus;
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device_id);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("Device selection and properties failed!\n", cudaStatus);			
		}

		printf("Device %d has compute capability %d.%d.\n",
				device_id, deviceProp.major, deviceProp.minor);

		return deviceProp;
	}
};

class smartEvent
{
private: 
	cudaEvent_t start, stop;
	bool _stoped;

public:

	smartEvent()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		initialize();
	}

	~smartEvent()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void initialize()
	{
		cudaEventRecord(start, 0);
		cudaEventRecord(stop, 0);
		_stoped = false;
	}

	inline
	void recStart()
	{
		initialize();
		cudaEventRecord(start, 0);
	}	

	inline
	void recStop()
	{
		cudaEventRecord(stop, 0);
		_stoped = true;
	}


	float elapsedTime()
	{
		float elapsedTime = 0;
		if(_stoped)
		{
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsedTime, start, stop);
		}
		return elapsedTime;
	}

	float printElapsedTime()
	{
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("Elapsed time is %fms\n",elapsedTime);
		return elapsedTime;
	}
};


/////BETA testing of code: Smart Config //////////////////////
class smartConfig
{
////beta testing/////
private:
	int device_id;
	int problem_size;
	cudaDeviceProp deviceProp;
	int configRank; //// rank of the config //// 1, 2, 3 for 1,2,3 dimensions. like x,y,z dimensions

	///return params 
	int threads_per_block;
	int num_blocks;


	void _initialize(int data_size, int dev_id, int threads, int blocks)
	{
		device_id = dev_id;
		problem_size = data_size;
		cudaGetDeviceProperties(&deviceProp, device_id);
		threads_per_block = threads;
		num_blocks = blocks;
	}

	void _calcConfig(int dim_size)
	{
		int launch_thread_size = problem_size + problem_size % 32;
		int launch_block_size = (launch_thread_size / deviceProp.maxThreadsPerBlock) + ((launch_thread_size%deviceProp.maxThreadsPerBlock)?1:0);
		num_blocks = (launch_block_size % 128) + launch_block_size;
		int maxBlockDimX = deviceProp.maxThreadsDim[0];
		if (num_blocks > maxBlockDimX ) num_blocks = maxBlockDimX;
	}

public:

	smartConfig()
	{
		_initialize(0,0,-1,-1);		
	}

	smartConfig(int data_size)
	{
		_initialize(data_size,0,-1,-1);
	}

	smartConfig(int data_size, int dev_id)
	{
		_initialize(data_size,dev_id,-1,-1);
	}
	
	smartConfig(int data_size, int dev_id, int threads, int blocks)
	{
		_initialize(data_size,dev_id, threads, blocks);
	}

	~smartConfig()
	{

	}

	void setDataSize(int data_size)
	{
		problem_size = data_size;
	}

	void config(int threads, int blocks)
	{
		threads_per_block = threads;
		num_blocks = blocks;
	}

	void setDeviceId(int dev_id)
	{
		device_id = dev_id;
	}

	int getDataSize()
	{
		return problem_size;
	}

	int getDeviceId()
	{
		return device_id;
	}

	int getBlockSize()
	{
		if (num_blocks <= 0)
		{
			_calcConfig(problem_size);
		}
		return num_blocks;
	}

	int getThreadSize()
	{
		if (threads_per_block <=0)
		{
			return deviceProp.maxThreadsPerBlock;
		}
		return threads_per_block;
	}

};
///// end BETA testing of code //////////////////////

//////////////////////KERNEL THREAD, BLOCK AND GRID INDEX AND SIZE SUB FUNCTIONS //////////////////////
#define __KERNEL__ __global__
//
//Notes: all num and size have default value of -1 to calculate the total number/size across all dimensions,
//		   use 0,1,2 to find the number or size for a specific dimension (x,y,z). 
//		   index and offset have default value of 0, and return the default for the X dimension
//

////local index in 1D
__device__ inline int __local_index(int dim = 0)
{
	int tid = 0;
	switch(dim)
	{
	case 0:
		tid = threadIdx.x;
		break;
	case 1:
		tid = threadIdx.y;
		break;
	case 2:
		tid = threadIdx.z;
		break;
	default:
		tid = threadIdx.x;
	}
	
	return tid;
}

////global index along one dimension
__device__ inline int __global_index(int dim = 0)
{
	int tid = 0;
	switch(dim)
	{
	case 0:
		tid =  blockIdx.x * blockDim.x + threadIdx.x;
		break;
	case 1:
		tid =  blockIdx.y * blockDim.y + threadIdx.y;
		break;
	case 2:
		tid =  blockIdx.z * blockDim.z + threadIdx.z;
		break;
	default:
		tid =  blockIdx.x * blockDim.x + threadIdx.x;
	}
	
	return tid;
}

////number of blocks
__device__ inline int __num_blocks(int dim = -1 )
{
	int tid = 0;
	switch(dim)
	{
	case 0:
		tid =  gridDim.x;
		break;
	case 1:
		tid =  gridDim.y;
		break;
	case 2:
		tid =  gridDim.z;
		break;
	default:
		tid =  gridDim.x * gridDim.y * gridDim.z;
	}
	
	return tid;
}

/////alias for num_blocks
__device__ inline int __num_groups(int dim = -1)
{		
	return __num_blocks(dim);
}


__device__ inline int __block_index(int dim = 0)
{
	int tid = 0;
	switch(dim)
	{
	case 0:
		tid =  blockIdx.x;
		break;
	case 1:
		tid =  blockIdx.y;
		break;
	case 2:
		tid =  blockIdx.z;
		break;
	default:
		tid =  blockIdx.x;
	}
	
	return tid;
}

/////alias for block_index
__device__ inline int __group_index(int dim = 0)
{		
	return __block_index(dim);
}

////number of threads in a block
__device__ inline int __block_size(int dim = -1)
{
	int tid = 0;
	switch(dim)
	{
	case 0:
		tid =  blockDim.x;
		break;
	case 1:
		tid =  blockDim.y;
		break;
	case 2:
		tid =  blockDim.z;
		break;
	/*case 3:
		tid =  blockDim.x * blockDim.y * blockDim.z ;
		break;*/
	default:
		//tid =  blockDim.x;
		tid =  blockDim.x * blockDim.y * blockDim.z ;
	}
	
	return tid;
}

/////alias for __block_size
__device__ inline int __group_size(int dim = -1)
{		
	return __block_size(dim);
}

/////alias for __block_size
__device__ inline int __local_size(int dim = -1)
{		
	return __block_size(dim);
}

////total number of threads in a grid
__device__ inline int __launch_size(int dim = -1)
{
	int tid = 0;
	switch(dim)
	{
	case 0:
		tid =  gridDim.x * blockDim.x;
		break;
	case 1:
		tid =  gridDim.y * blockDim.y;
		break;
	case 2:
		tid =  gridDim.z * blockDim.z;
		break;
	default:
		tid =  (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y) * (gridDim.z * blockDim.z);
	}
	
	return tid;
}

/////alias for __launch_size
__device__ inline int __global_size(int dim = -1)
{		
	return __launch_size(dim);
}

/////alias for __launch_size
__device__ inline int __stride(int dim = 0)
{		
	return __launch_size(dim);
}


///// BETA --linear element position in 3D  grid and block launches (for reference linear access is x) ///////

////unique  block global index along one dimension in a 3D space
__device__ inline int __unique_block_index(int dim = 0)
{
	int tid = 0;
	switch(dim)
	{
	case 0:
		tid = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; 
		break;
	case 1:
		tid = blockIdx.y + blockIdx.z * gridDim.y + gridDim.y * gridDim.z * blockIdx.x; 
		break;
	case 2:
		tid = blockIdx.z + blockIdx.x * gridDim.z + gridDim.z * gridDim.x * blockIdx.y; 
		break;
	default:
		tid = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; 
	}
	
	return tid;
}

/////alias for unique block_index
__device__ inline int __unique_group_index(int dim = 0)
{		
	return __block_index(dim);
}


////local offset for indexing in 3D space
__device__ inline int __unique_local_offset(int dim = 0)
{
	int tid = 0;
	switch(dim)
	{
	case 0:
		tid = (threadIdx.y * (blockDim.x)) + (threadIdx.z * (blockDim.x * blockDim.y));
		break;
	case 1:
		tid = (threadIdx.z * (blockDim.y)) + (threadIdx.x * (blockDim.y * blockDim.z)); 
		break;
	case 2:
		tid = (threadIdx.x * (blockDim.z)) + (threadIdx.y * (blockDim.z * blockDim.x)); 
		break;
	default:
		tid = (threadIdx.y * (blockDim.x)) + (threadIdx.z * (blockDim.x * blockDim.y));
	}
	
	return tid;
}



////global block offset to get to the local block along one dimension
__device__ inline int __global_unique_index(int dim = 0)
{
	int tid = 0;
	switch(dim)
	{
	case 0:
		tid = __unique_block_index(dim) * __block_size() //// (blockDim.x * blockDim.y * blockDim.z)   
			+ __unique_local_offset(dim) ////(threadIdx.y * (blockDim.x)) + (threadIdx.z * (blockDim.x * blockDim.y)) 
			+ __local_index(dim); ////(threadIdx.x);
		break;
	case 1:
		tid = __unique_block_index(dim) * __block_size() //// (blockDim.x * blockDim.y * blockDim.z)   
			+ __unique_local_offset(dim) ////(threadIdx.z * (blockDim.y)) + (threadIdx.x * (blockDim.y * blockDim.z)) 
			+ __local_index(dim); ////(threadIdx.y);
		break;
	case 2:
		tid = __unique_block_index(dim) * __block_size() //// (blockDim.x * blockDim.y * blockDim.z)   
			+ __unique_local_offset(dim) ////(threadIdx.x * (blockDim.z)) + (threadIdx.y * (blockDim.z * blockDim.x)) 
			+ __local_index(dim); ////(threadIdx.z);
		break;
	default:
		tid = __unique_block_index(dim) * __block_size() //// (blockDim.x * blockDim.y * blockDim.z)   
			+ __unique_local_offset(dim) ////(threadIdx.y * (blockDim.x)) + (threadIdx.z * (blockDim.x * blockDim.y)) 
			+ __local_index(dim); ////(threadIdx.x);
	}
	
	return tid;
}


/////alias for unique block_index
__device__ inline int __global_index_unique(int dim = 0)
{		
	return __global_unique_index(dim);
}
