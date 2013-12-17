/*
 *  SMART CUDA 0.1.0 (17th December, 2013) - Initial Release
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

#define smartHost 0
#define smartDevice 1
#define smartPinnedHost 2
#define scopeLocal false
#define scopeGlobal true

////deallocating global wrapper memories////
#define ON true;
#define OFF false;
 __device__
extern bool __clean_globals = false;
//bool smartGlobalFree = false;

template <typename T> 
__host__ __device__ void initializeSmartArray(T *dev_Array, const int size, const T init)
{
    int i = 0;
	while(i < size)
	{
		dev_Array[i] = init;
		i += 1;
	}    
	
}

template <typename T> 
__host__ __device__ void idx_initializeSmartArray(T *dev_Array, const int size, const T init, T step = 1)
{
    int i = 0;
	while(i < size)
	{
		dev_Array[i] = init + i * step;
		i += 1;
	}    	
}



template <typename T> 
__global__ void initializeSmartArrayAsync_core(T *dev_Array, const int size, const T init)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		dev_Array[i] = init;
		i += blockDim.x * gridDim.x;
	}    	
}

template <typename T> 
__global__ void idx_initializeSmartArrayAsync_core(T *dev_Array, const int size, const T init, T step = 1)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < size)
	{
		dev_Array[i] = init + i * step;
		i += blockDim.x * gridDim.x;
	}    	
}


template <typename T> 
__host__ __device__ cudaError_t initializeSmartArrayAsync(T *dev_Array, const int size, const T init)
{
	cudaError_t cudaStatus;
    initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
	}
	return cudaStatus;
}

template <typename T> 
__host__ __device__ cudaError_t idx_initializeSmartArrayAsync(T *dev_Array, const int size, const T init, T step = 1)
{
	cudaError_t cudaStatus;
    idx_initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init,step);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
	}
	return cudaStatus;
}

template <typename T> 
	inline T* idx_allocSmartGPUArrayAsync( int size, T init = 0, bool setINI = false, T step=1, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(1D) failed!\n");
        //goto Error;
    }

	/*if (setINI)
	{
		initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		idx_initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init,step);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			goto Error;
		}
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* allocSmartGPUArrayAsync( int size, T init = 0, bool setINI = false, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(1D) failed\n!");
        //goto Error;
    }

	/*if (setINI)
	{
		initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			goto Error;
		}
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* idx_allocSmartGPUArray( int size, T init = 0, bool setINI = false, T step=1, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(1D) failed!\n");
        //goto Error;
    }

	if (setINI)
	{
		idx_initializeSmartArray(dev_Array, size, init,step);		
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* allocSmartGPUArray( int size, T init = 0, bool setINI = false, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(1D) failed\n!");
        //goto Error;
    }

	if (setINI)
	{
		initializeSmartArray(dev_Array, size, init);		
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}
	
template <typename T> 
	inline T* allocSmartGPUArrayAsync( int sizeX, int sizeY, T init = 0, bool setINI = false, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(2D) failed!\n");
        //goto Error;
    }

	/*if (setINI)
	{
		initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			goto Error;
		}
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* idx_allocSmartGPUArrayAsync( int sizeX, int sizeY, T init = 0, bool setINI = false, T step = 1, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(2D) failed!\n");
        //goto Error;
    }

	/*if (setINI)
	{
		initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		idx_initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			goto Error;
		}
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}
	
template <typename T> 
	inline T* allocSmartGPUArray( int sizeX, int sizeY, T init = 0, bool setINI = false, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(2D) failed!\n");
        //goto Error;
    }

	if (setINI)
	{
		initializeSmartArray(dev_Array, size, init);		
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* idx_allocSmartGPUArray( int sizeX, int sizeY, T init = 0, bool setINI = false, T step =1, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(2D) failed!\n");
        //goto Error;
    }

	if (setINI)
	{
		initializeSmartArray(dev_Array, size, init,step);
		
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}
	
template <typename T> 
	inline T* allocSmartGPUArrayAsync( int sizeX, int sizeY, int sizeZ, T init = 0, bool setINI = false, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(3D) failed!\n");
        //goto Error;
    }

	/*if (setINI)
	{
		initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			goto Error;
		}
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

	template <typename T> 
	inline T* idx_allocSmartGPUArrayAsync( int sizeX, int sizeY, int sizeZ, T init = 0, bool setINI = false, T step =1, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(3D) failed!\n");
        //goto Error;
    }

	/*if (setINI)
	{
		initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		idx_initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init, step);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			goto Error;
		}
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* allocSmartGPUArray( int sizeX, int sizeY, int sizeZ, T init = 0, bool setINI = false, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(3D) failed!\n");
        //goto Error;
    }

	if (setINI)
	{
		initializeSmartArray(dev_Array, size, init);		
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* idx_allocSmartGPUArray( int sizeX, int sizeY, int sizeZ,T init = 0, bool setINI = false, T step =1, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(3D) failed!\n");
        //goto Error;
    }

	if (setINI)
	{
		initializeSmartArray(dev_Array, size, init,step);
		
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}
	
template <typename T> 
	inline T* allocSmartGPUArrayAsync( int sizeX, int sizeY, int sizeZ, int sizeW, T init = 0, bool setINI = false, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ * sizeW;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(4D) failed!\n");
        //goto Error;
    }

	/*if (setINI)
	{
		initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			goto Error;
		}
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* idx_allocSmartGPUArrayAsync( int sizeX, int sizeY, int sizeZ, int sizeW, T init = 0, T step=1, bool setINI = false, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ * sizeW;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(4D) failed!\n");
        //goto Error;
    }

	/*if (setINI)
	{
		initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init);
		cudaDeviceSynchronize();
	}*/

	if (setINI)
	{
		idx_initializeSmartArrayAsync_core<<<128, 128>>>(dev_Array, size, init, step);
		////check for launch errors
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}

		cudaDeviceSynchronize();
		////check for sync errors
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Memory Initialization!\n", cudaStatus);
			goto Error;
		}
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* allocSmartGPUArray( int sizeX, int sizeY, int sizeZ, int sizeW, T init = 0, bool setINI = false, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ * sizeW;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(4D) failed!\n");
        //goto Error;
    }

	if (setINI)
	{
		initializeSmartArray(dev_Array, size, init,step);
		
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline T* idx_allocSmartGPUArray( int sizeX, int sizeY, int sizeZ, int sizeW, T init = 0, T step=1, bool setINI = false, cudaError_t cudaStatus = cudaSuccess)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ * sizeW;
	 cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation(4D) failed!\n");
        //goto Error;
    }

	if (setINI)
	{
		initializeSmartArray(dev_Array, size, init,step);
		
	}
	
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T> 
	inline cudaError_t freeSmartGPUArray(T* dev_mem)
	{
		cudaError_t cudaStatus;
		cudaFree(dev_mem);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Cuda Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}
		return cudaStatus;
	}

template <typename T> 
	inline cudaError_t freeSmartGPUArray(T* dev_mem, cudaError_t &cudaStatus)
	{
		////cudaError_t cudaStatus;
		cudaFree(dev_mem);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Cuda Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}
		return cudaStatus;
	}

template <typename T> 
	inline cudaError_t freeSmartGPUArray(T* dev_mem, int array_type)
	{
		cudaError_t cudaStatus;
		if (array_type == 0)
		{
			free(dev_mem);
			dev_mem = NULL;
			cudaStatus = cudaSuccess;
		}
		else if (array_type == 1)
		{
			cudaFree(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Cuda Device Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == 2)
		{
			cudaFreeHost(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//goto Error;
			}
			return cudaStatus;
		}
		else
		{
			cudaStatus = cudaErrorMemoryAllocation;
			fprintf(stderr, "Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		
	}

template <typename T> 
	inline cudaError_t freeSmartGPUArray(T* dev_mem, int array_type, cudaError_t &cudaStatus)
	{
		////cudaError_t cudaStatus;
		if (array_type == smartHost)
		{
			free(dev_mem);
			dev_mem = NULL;
			cudaStatus = cudaSuccess;
		}
		else if (array_type == smartDevice)
		{
			cudaFree(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Cuda Device Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartPinnedHost)
		{
			cudaFreeHost(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//goto Error;
			}
			return cudaStatus;
		}
		else
		{
			cudaStatus = cudaErrorMemoryAllocation;
			fprintf(stderr, "Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		
	}

	template <typename T>
cudaError_t smartCopyHost2Device(T* dev_mem, const T* host_mem, unsigned int size, cudaError_t cudaStatus = cudaSuccess)
{
	cudaStatus = cudaMemcpy(dev_mem, host_mem, size * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy Host to Device failed!\n");
        //goto Error;
    }
	return cudaStatus;
}

template <typename T>
cudaError_t smartCopyDevice2Host(T* host_mem, T* dev_mem, unsigned int size, cudaError_t cudaStatus = cudaSuccess)
{
	cudaStatus = cudaMemcpy(host_mem, dev_mem, size * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy Device to Host failed!\n");
        //goto Error;
    }
	return cudaStatus;
}

template <typename T>
cudaError_t smartCopyDevice2Device(T* host_mem, T* dev_mem, unsigned int size, cudaError_t cudaStatus = cudaSuccess)
{
	cudaStatus = cudaMemcpy(host_mem, dev_mem, size * sizeof(T), cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy Device to Device failed!\n");
        //goto Error;
    }
	return cudaStatus;
}

template <typename T>
cudaError_t smartCopyHost2Host(T* host_mem, T* dev_mem, unsigned int size, cudaError_t cudaStatus = cudaSuccess)
{
	cudaStatus = cudaMemcpy(host_mem, dev_mem, size * sizeof(T), cudaMemcpyHostToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy Host to Host failed!\n");
        //goto Error;
    }
	return cudaStatus;
}

//template<int X, int Y, int Z, int W>
class arrayDim
{
private: 
	int dimX; int dimY; int dimZ; int dimW;
public:
	__device__  __host__ arrayDim() 
	{
		
	}
	__device__  __host__ arrayDim(int i=1,int j=1, int k=1, int w=1) 
	{
		if((i < 0 || j < 0 || k < 0) || (w < 0)) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Array Dimensions [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
			exit(1);
		}
		dimX = i;dimY = j; dimZ = k; dimW = w;
	}
	
	__device__  __host__ int &operator()(int i=1,int j=1, int k=1, int w=1) 
	{
		if((i < 0 || j < 0 || k < 0) || (w < 0)) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Array Dimensions [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
			exit(1);
		}
		dimX = i;dimY = j; dimZ = k; dimW = w;
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

template <typename T, int mem_loc> 
	inline T* smartArray( int sizeX, cudaError_t &cudaStatus)
{
	T* dev_Array = 0;	
	int size = sizeX;

	if (mem_loc == 0)
	{
		////allocate pageable host memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			fprintf(stderr, "Dynamic Memory Allocation on Device failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
	}
	else if (mem_loc == 1)
	{
		cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Dynamic Memory Allocation on Device failed!\n");
			//goto Error;
		}
	}
	else if (mem_loc == 2) {		

		cudaStatus = cudaMallocHost((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Dynamic Pinned Memory Allocation on Host failed!\n");
			//goto Error;
		}
	}
	else
	{
		fprintf(stderr, "Cannot allocate unsupported memory\n");	
		cudaStatus  = cudaErrorMemoryAllocation;
		//return cudaStatus;
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T, int mem_loc> 
	inline T* smartArray( int sizeX, int sizeY, cudaError_t &cudaStatus)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY;

	if (mem_loc == 0)
	{
		////allocate pageable host memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			fprintf(stderr, "Dynamic Memory Allocation on Device failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
	}
	else if (mem_loc == 1)
	{
		cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Dynamic Memory Allocation on Device failed!\n");
			//goto Error;
		}
	}
	else if (mem_loc == 2) {		

		cudaStatus = cudaMallocHost((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Dynamic Pinned Memory Allocation on Host failed!\n");
			//goto Error;
		}
	}
	else
	{
		fprintf(stderr, "Cannot allocate unsupported memory\n");	
		cudaStatus  = cudaErrorMemoryAllocation;
		//return cudaStatus;
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T, int mem_loc> 
	inline T* smartArray( int sizeX, int sizeY, int sizeZ, cudaError_t &cudaStatus)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ;

	if (mem_loc == 0)
	{
		////allocate pageable host memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			fprintf(stderr, "Dynamic Memory Allocation on Device failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
	}
	else if (mem_loc == 1)
	{
		cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Dynamic Memory Allocation on Device failed!\n");
			//goto Error;
		}
	}
	else if (mem_loc == 2) {		

		cudaStatus = cudaMallocHost((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Dynamic Pinned Memory Allocation on Host failed!\n");
			//goto Error;
		}
	}
	else
	{
		fprintf(stderr, "Cannot allocate unsupported memory\n");	
		cudaStatus  = cudaErrorMemoryAllocation;
		//return cudaStatus;
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}

template <typename T, int mem_loc> 
	inline T* smartArray( int sizeX, int sizeY, int sizeZ, int sizeW, cudaError_t &cudaStatus)
{
	T* dev_Array = 0;	
	int size = sizeX * sizeY * sizeZ * sizeW;

	if (mem_loc == 0)
	{
		////allocate pageable host memory
		dev_Array = (T*)malloc(size * sizeof(T));
		if (NULL == dev_Array) {
			/* Handle error… */
			fprintf(stderr, "Dynamic Memory Allocation on Device failed!\n");
			cudaStatus = cudaErrorMemoryAllocation;
		}
	}
	else if (mem_loc == 1)
	{
		cudaStatus = cudaMalloc((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Dynamic Memory Allocation on Device failed!\n");
			//goto Error;
		}
	}
	else if (mem_loc == 2) {		

		cudaStatus = cudaMallocHost((void**)&dev_Array, size * sizeof(T));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Dynamic Pinned Memory Allocation on Host failed!\n");
			//goto Error;
		}
	}
	else
	{
		fprintf(stderr, "Cannot allocate unsupported memory\n");	
		cudaStatus  = cudaErrorMemoryAllocation;
		//return cudaStatus;
	}
	
Error:
   // cudaFree(dev_Array);
	
	return dev_Array;
}


template <typename T, int array_type> 
	inline cudaError_t freeGPUArray(T* dev_mem)
	{
		cudaError_t cudaStatus;
		if (array_type == smartHost)
		{
			free(dev_mem);
			dev_mem = NULL;
			cudaStatus = cudaSuccess;
		}
		else if (array_type == smartDevice)
		{
			cudaFree(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Cuda Device Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartPinnedHost)
		{
			cudaFreeHost(dev_mem);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//goto Error;
			}
			return cudaStatus;
		}
		else
		{
			cudaStatus = cudaErrorMemoryAllocation;
			fprintf(stderr, "Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		
	}


template <typename T, int array_type> 
	inline cudaError_t freeGPUArray(T* dev_mem, cudaError_t &cudaStatus)
	{
		if (array_type == smartHost)
		{
			free(dev_mem);
			dev_mem = NULL;
			cudaStatus = cudaSuccess;
		}
		else if (array_type == smartDevice)
		{
			cudaFree(dev_mem);
			dev_mem = NULL;
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Cuda Device Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//goto Error;
			}
			return cudaStatus;
		}
		else if (array_type == smartPinnedHost)
		{
			cudaFreeHost(dev_mem);
			dev_mem = NULL;
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
				//goto Error;
			}
			return cudaStatus;
		}
		else
		{
			cudaStatus = cudaErrorMemoryAllocation;
			fprintf(stderr, "Cuda Host Memory Release failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}
		
	}


template <typename T, int mem_loc> 
	class smartArrayWrapper
{
private: 
	////pointer to array
	T* dev_arr;

	////array dimensions
	int lengthX;
	int lengthY;
	int lengthZ;
	int lengthW;

	////specialize views
	int vX,vY,vZ,vW; 

	////array wrapper type ////0= host, 1 = device
	////useful for copies
	int array_type;

	////destructor behaviour //// 0 leave array, 1 destroy on scope exit
	bool _global_scope; ////destroy_array_on_scope_exit;
	//bool _is_copy; /////used to control destructor on copy assignment///

	////increment and decrement operators for navigation of the data array
	unsigned int idx; /////current location
	unsigned int vidx; /////current specialized view location
	unsigned int ix,iy,iz,iw;


	__device__  __host__
	void initializeArray(T* dev=NULL, int lenX=1,int lenY=1,int lenZ=1,int lenW=1, bool global_scope = true ) 
	{
		dev_arr = dev;
		lengthX = lenX;
		lengthY = lenY;
		lengthZ = lenZ;
		lengthW = lenW;
		////initialize views to default dimensions
		vX = 1; vY = 1; vZ = 1; vW = 1;

		////set location type
		array_type = mem_loc;

		////destructor behaviour //// 0 leave array, 1 destroy on scope exit
		_global_scope = global_scope;
		//_is_copy = true;

		////increment operators for navigation
		idx = 0; vidx = 0;
		ix = 0; iy =0; iz = 0; iw = 0;
	}

	__device__  __host__
	int getIndex(int i) 
	{	
		int j = 0; int k = 0; int w = 0;
		//if((((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1)) ||(w < 0 || w > lengthW-1))|| dev_arr==NULL)
		//{
		//	// Take appropriate action here. This is just a placeholder response.
		//	fprintf(stderr, "Array not INITIALIZED or Index [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
		//	exit(1);
		//}

		int _idx = i + j * lengthX + k * lengthX * lengthY + w * lengthX * lengthY * lengthZ;		
		return _idx;
	}

	__device__  __host__
	int getIndex(int i, int j) 
	{	
		int k = 0; int w = 0;
		//if((((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1)) ||(w < 0 || w > lengthW-1))|| dev_arr==NULL)
		//{
		//	// Take appropriate action here. This is just a placeholder response.
		//	fprintf(stderr, "Array not INITIALIZED or Index [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
		//	exit(1);
		//}

		int _idx = i + j * lengthX + k * lengthX * lengthY + w * lengthX * lengthY * lengthZ;		
		return _idx;
	}

	__device__  __host__
	int getIndex(int i, int j, int k) 
	{	
		int w = 0;
		//if((((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1)) ||(w < 0 || w > lengthW-1))|| dev_arr==NULL)
		//{
		//	// Take appropriate action here. This is just a placeholder response.
		//	fprintf(stderr, "Array not INITIALIZED or Index [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
		//	exit(1);
		//}

		int _idx = i + j * lengthX + k * lengthX * lengthY + w * lengthX * lengthY * lengthZ;		
		return _idx;
	}

	__device__  __host__
	int getIndex(int i, int j, int k, int w) 
	{		
		//if((((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1)) ||(w < 0 || w > lengthW-1))|| dev_arr==NULL)
		//{
		//	// Take appropriate action here. This is just a placeholder response.
		//	fprintf(stderr, "Array not INITIALIZED or Index [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
		//	exit(1);
		//}

		int _idx = i + j * lengthX + k * lengthX * lengthY + w * lengthX * lengthY * lengthZ;		
		return _idx;
	}

	__device__  __host__
	int getViewIndex(int i, int j=0, int k=0, int w=0) 
	{		
		//if(((i < 0 || i > vX-1) || (j < 0 || j > vY-1) || (k < 0 || k > vZ-1)) ||(w < 0 || w > vW-1))
		//{
		//	// Take appropriate action here. This is just a placeholder response.
		//	fprintf(stderr, "View Index [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
		//	exit(1);
		//}

		int _idx = i + j * vX + k * vX * vY + w * vX * vY * vZ;
		return dev_arr[_idx];
	}

public: 
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

	__device__  __host__
	~smartArrayWrapper()
	{
		if(!_global_scope )
		{
			freeGPUArray<T,mem_loc>(dev_arr);
		}
		else if(_global_scope && __clean_globals)
		{
			freeGPUArray<T,mem_loc>(dev_arr);
		}
		//_is_copy = false;
	}

	__device__  __host__
	void wrap(T* dev, int lenX, bool global_scope = true ) 
	{
		initializeArray(dev,lenX,1,1,1,global_scope);
	}

	__device__  __host__
	void wrap(T* dev, int lenX,int lenY, bool global_scope = true ) 
	{
		initializeArray(dev,lenX,lenY,1,1,global_scope);
	}

	__device__  __host__
	void wrap(T* dev, int lenX,int lenY,int lenZ, bool global_scope = true ) 
	{
		initializeArray(dev,lenX,lenY,lenZ,1,global_scope);
	}

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


	__device__  __host__  T &operator()(int i) 
	{
		if((i < 0 || i > lengthX-1) || dev_arr==NULL)
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Array not INITIALIZED or Index [%d] of GPU array is out of bounds\n",i);			
			exit(1);
		}
		//T temp = dev_arr[i];
		idx = i;
		return dev_arr[i];
	}

	__device__  __host__
	T &operator()(int i, int j)
	{
		if(((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1))|| dev_arr==NULL)
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Array not INITIALIZED or Index [%d][%d] of GPU array is out of bounds\n",i,j);			
			exit(1);
		}
		int _idx = i + j * lengthX;
		idx = _idx;
		return dev_arr[_idx];
	}

	__device__  __host__
	T &operator()(int i, int j, int k)
	{
		if(((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1))|| dev_arr==NULL)
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Array not INITIALIZED or Index [%d][%d][%d] of GPU array is out of bounds\n",i,j,k);			
			exit(1);
		}
		int _idx = i + j * lengthX + k * lengthX * lengthY;
		idx = _idx;
		return dev_arr[_idx];
	}

	__device__  __host__
	T &operator()(int i, int j, int k, int w)
	{
		if((((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1)) ||(w < 0 || w > lengthW-1))|| dev_arr==NULL)
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Array not INITIALIZED or Index [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
			exit(1);
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
			fprintf(stderr, "Array not INITIALIZED or Index [%d] of GPU array is out of bounds\n",idx);			
			exit(1);
		}
		idx = _idx;
		return dev_arr[_idx];
	}

	
	__device__  __host__
	cudaError_t &operator=(smartArrayWrapper<T,smartDevice> &other)
	{
		cudaError_t cudaStatus;
		//other._is_copy = true; ////prevent destructor from deleting array pointer

		//if (this == &other)////check for self assignment
		//{
		//	fprintf(stderr, "Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//exit(1);			
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
			else if (other.getType() == smartDevice)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice)
			{
				cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			fprintf(stderr, "Cannot copy unsupported type\n");	
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
		//	fprintf(stderr, "Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//exit(1);			
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
			else if (other.getType() == smartDevice)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice)
			{
				cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
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
		//	fprintf(stderr, "Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//exit(1);			
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
			else if (other.getType() == smartDevice)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice)
			{
				cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		
		
		return cudaStatus;
	}

	template <typename custom>
	__device__  __host__
	cudaError_t &operator=(custom other)
	{
		cudaError_t cudaStatus;				
		//// size of data to copy, using the least of the 2 to avoid errors
		T* arr = &other[0];
		int size = other.size();
		int type = smartHost;

		unsigned int copySize = (getlen() > size) ? getlen() : size;

		if(array_type == smartHost || array_type == smartPinnedHost) ////host -> host/device copies
		{
			if(type == smartHost || array_type == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Host<T>(dev_arr,arr,copySize);
			}
			else if (type == smartDevice)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,arr,copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice) ////device -> host/device copies
		{
			if(type == smartHost || array_type == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,arr,copySize);
			}
			else if (type == smartDevice)
			{
				cudaStatus = smartCopyDevice2Device<T>(dev_arr,arr,copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			fprintf(stderr, "Cannot copy unsupported type\n");	
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
	//		fprintf(stderr, "Cannot self assign\n");	
	//		cudaStatus  = cudaErrorMemoryAllocation;
	//		return cudaStatus;
	//		//exit(1);			
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
	//			fprintf(stderr, "Cannot copy unsupported type\n");	
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
	//			fprintf(stderr, "Cannot copy unsupported type\n");	
	//			cudaStatus  = cudaErrorMemoryAllocation;
	//			return cudaStatus;
	//		}
	//	}
	//	else ////unsupported copy types
	//	{
	//		fprintf(stderr, "Cannot copy unsupported type\n");	
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
		//	fprintf(stderr, "Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//exit(1);			
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
			else if (other.getType() == smartDevice)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice)
			{
				cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			fprintf(stderr, "Cannot copy unsupported type\n");	
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
		//	fprintf(stderr, "Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//exit(1);			
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
			else if (other.getType() == smartDevice)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice)
			{
				cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
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
		//	fprintf(stderr, "Cannot self assign\n");	
		//	cudaStatus  = cudaErrorMemoryAllocation;
		//	return cudaStatus;
		//	//exit(1);			
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
			else if (other.getType() == smartDevice)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == smartDevice) ////device -> host/device copies
		{
			if(other.getType() == smartHost || other.getType() == smartPinnedHost)
			{
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else if (other.getType() == smartDevice)
			{
				cudaStatus = smartCopyDevice2Device<T>(dev_arr,other.inner_ptr(),copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		
		
		return cudaStatus;
	}

	__device__  __host__
	cudaError_t copy(T* other, unsigned int other_size, int other_array_type = 1)
	{
		cudaError_t cudaStatus;
		//if (this != &other)
		//{
		//	fprintf(stderr, "Cannot copy unsupported type\n");	
		//	cudaStatus != cudaSuccess;
		//	return cudaStatus;
		//	//exit(1);			
		//}
		
		//// size of data to copy, using the least of the 2 to avoid errors
		unsigned int copySize = (getlen() > other_size) ? getlen() : other_size;

		if(array_type == 0) ////host -> host/device copies
		{
			if(other_array_type == 0)
			{
				cudaStatus = smartCopyHost2Host<T>(dev_arr,other,copySize);
			}
			else if (other_array_type == 1)
			{
				cudaStatus = smartCopyDevice2Host<T>(dev_arr,other,copySize);				
				
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus  = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else if (array_type == 1) ////device -> host/device copies
		{
			if(other_array_type == 0)
			{				
				cudaStatus = smartCopyHost2Device<T>(dev_arr,other,copySize);
			}
			else if (other_array_type == 1)
			{
				cudaStatus = smartCopyDevice2Device<T>(dev_arr,other,copySize);
			}
			else
			{
				////unsupported copy types
				fprintf(stderr, "Cannot copy unsupported type\n");	
				cudaStatus  = cudaErrorMemoryAllocation;
				return cudaStatus;
			}
		}
		else ////unsupported copy types
		{
			fprintf(stderr, "Cannot copy unsupported type\n");	
			cudaStatus  = cudaErrorMemoryAllocation;
			return cudaStatus;
		}
		
		
		return cudaStatus;
	}

	__device__  __host__
	cudaError_t destroy()
	{
		cudaError_t cudaStatus;
		cudaStatus = freeGPUArray<T,mem_loc>(inner_ptr());	
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
	__device__  __host__
	int size() { return lengthX * lengthY * lengthZ * lengthW; }
	
	__device__  __host__
	int getlen() { return lengthX * lengthY * lengthZ * lengthW; }
	__device__  __host__
	int getlenX() { return lengthX; }
	__device__  __host__
	int getlenY() { return lengthY; }
	__device__  __host__
	int getlenZ() { return lengthZ; }
	__device__  __host__
	int getlenW() { return lengthW; }

	// Specialized views of the array.
	__device__  __host__
	int getViewDimX() { return vX; }
	__device__  __host__
	int getViewDimY() { return vY; }
	__device__  __host__
	int getViewDimZ() { return vZ; }
	__device__  __host__
	int getViewDimW() { return vW; }

	__device__  __host__
	bool setViewDim(int X = 1, int Y = 1, int Z = 1, int W = 1, bool setSuccess = true) 
	{ 
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
			fprintf(stderr, "View Index [%d] of GPU array is out of bounds\n",i);			
			exit(1);
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
			fprintf(stderr, "View Index [%d][%d] of GPU array is out of bounds\n",i,j);			
			exit(1);
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
			fprintf(stderr, "View Index [%d][%d][%d] of GPU array is out of bounds\n",i,j,k);			
			exit(1);
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
			fprintf(stderr, "View Index [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
			exit(1);
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
			fprintf(stderr, "View Index [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
			exit(1);
		}
		int _idx = i + j * vX + k * vX * vY + w * vX * vY * vZ;
		vidx = _idx;
		return dev_arr[_idx];
	}


};


template <typename T,  int len> 
class smartDeviceArrayWrapper1D
{
private: 
	T* dev_arr;
	int length;

public: 
	__device__  __host__
	smartDeviceArrayWrapper1D(T* dev)
	{
		dev_arr = dev;
		length = len;
	}

	__device__  __host__  T &operator()(int i)
	{
		if(i < 0 || i > length-1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",i);			
			exit(1);
		}
		return dev_arr[i];
	}

	__device__  __host__ T &operator[](int i)
	{
		if(i < 0 || i > length-1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",i);			
			exit(1);
		}
		return dev_arr[i];
	}

	// Return the length of the array.
	__device__  __host__
	int getlen() { return length; }
};

template <typename T, int len> 
 class const_smartDeviceArrayWrapper1D
{
private: 
	const T* dev_arr;
	int length;
	
	////specialize views
	//int vX,vY,vZ,vW; 

public: 
	__device__  __host__
	 const_smartDeviceArrayWrapper1D(const T* dev) 
	{
		dev_arr = dev;
		length = len;
	}

	__device__  __host__  T &operator()(int i) 
	{
		if(i < 0 || i > length-1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",i);			
			exit(1);
		}
		T temp = dev_arr[i];
		return temp; //dev_arr[i];
	}

	__device__  __host__ T &operator[](int i) 
	{
		if(i < 0 || i > length-1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",i);			
			exit(1);
		}
		T temp = dev_arr[i];
		return temp; //dev_arr[i];
	}

	// Return the length of the array.
	__device__  __host__
	int getlen() { return length; }

	//// Specialized views of the array.
	//__device__  __host__
	//int getViewDimX() { return vX; }
	//__device__  __host__
	//int getViewDimY() { return vY; }
	//__device__  __host__
	//int getViewDimZ() { return vZ; }
	//__device__  __host__
	//int getViewDimW() { return vW; }

	//__device__  __host__
	//bool setViewDim(int X = 1, int Y = 1, int Z = 1, int W = 1, bool setSuccess = true) 
	//{ 
	//	if (X < 0) {setSuccess = false; return setSuccess;}
	//	if (Y < 0) {setSuccess = false; return setSuccess;}
	//	if (Z < 0) {setSuccess = false; return setSuccess;}
	//	if (W < 0) {setSuccess = false; return setSuccess;}

	//	if((X * Y * Z * K >= 0) && (X * Y * Z * W <= length))
	//	{
	//		vX = X; vY = Y; vZ = Z; vW = W;
	//		setSuccess = true;
	//	}
	//	else
	//	{
	//		setSuccess = false; ////set view failed
	//	}

	//	return setSuccess;
	//}

	//__device__  __host__
	//T viewAt_1D(int i)
	//{
	//	if(i < 0 || i > vX * vY * vZ * vW -1) 
	//	{
	//		// Take appropriate action here. This is just a placeholder response.
	//		fprintf(stderr, "View Index [%d] of GPU array is out of bounds\n",i);			
	//		exit(1);
	//	}
	//	return dev_arr[i];
	//}

	//__device__  __host__
	//T viewAt_2D(int i, int j)
	//{
	//	if((i < 0 || i > vX-1) || (j < 0 || j > vY-1))
	//	{
	//		// Take appropriate action here. This is just a placeholder response.
	//		fprintf(stderr, "View Index [%d][%d] of GPU array is out of bounds\n",i,j);			
	//		exit(1);
	//	}
	//	int idx = i + j * vX;
	//	return dev_arr[idx];
	//}

	//__device__  __host__
	//T viewAt_3D(int i, int j, int k)
	//{
	//	if((i < 0 || i > vX-1) || (j < 0 || j > vY-1) || (k < 0 || k > vZ-1))
	//	{
	//		// Take appropriate action here. This is just a placeholder response.
	//		fprintf(stderr, "View Index [%d][%d][%d] of GPU array is out of bounds\n",i,j,k);			
	//		exit(1);
	//	}
	//	int idx = i + j * vX + k * vX * vY;
	//	return dev_arr[idx];
	//}

	//__device__  __host__
	//T viewAt_3D(int i, int j, int k, int w)
	//{
	//	if(((i < 0 || i > vX-1) || (j < 0 || j > vY-1) || (k < 0 || k > vZ-1)) ||(w < 0 || w > vW-1))
	//	{
	//		// Take appropriate action here. This is just a placeholder response.
	//		fprintf(stderr, "View Index [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
	//		exit(1);
	//	}
	//	int idx = i + j * vX + k * vX * vY + w * vX * vY * vZ;
	//	return dev_arr[idx];
	//}

	//__device__  __host__
	//T viewAt(int i, int j=0, int k=0, int w=0)
	//{
	//	if(((i < 0 || i > vX-1) || (j < 0 || j > vY-1) || (k < 0 || k > vZ-1)) ||(w < 0 || w > vW-1))
	//	{
	//		// Take appropriate action here. This is just a placeholder response.
	//		fprintf(stderr, "View Index [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
	//		exit(1);
	//	}
	//	int idx = i + j * vX + k * vX * vY + w * vX * vY * vZ;
	//	return dev_arr[idx];
	//}
};

template <typename T, int lenX, int lenY>
class smartDeviceArrayWrapper2D
{
private: 
	T* dev_arr;
	int lengthX, lengthY;

public: 
	__device__  __host__
	smartDeviceArrayWrapper2D(T* dev)
	{
		dev_arr = dev;
		lengthX = lenX;
		lengthY = lenY;
	}

	__device__  __host__
	T &operator()(int idx)
	{
		if(idx < 0 || idx > lengthX * lengthY -1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",idx);			
			exit(1);
		}
		return dev_arr[idx];
	}

	__device__  __host__
	T &operator()(int i, int j)
	{
		if((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d][%d] of GPU array is out of bounds\n",i,j);			
			exit(1);
		}
		int idx = i + j * lengthX;
		return dev_arr[idx];
	}

	__device__  __host__
	T &operator[](int idx)
	{
		if(idx < 0 || idx > lengthX * lengthY -1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",idx);			
			exit(1);
		}
		return dev_arr[idx];
	}

	//int i;

	//template <typename T>
	//class proxy
	//{
	//private: 
	////	T* _array;
	////	int i;

	//public: 
	////	proxy(T* _array): _array(_array){}
	////	proxy(int i): i(i){}
	//	__device__  __host__
	//	T &operator[](int i)
	//	{	
	//		i = i;
	//		//return i;
	//	//	return _array[i];
	//	}		
	//};

	//__device__  __host__
	//proxy<T>& operator[] (int j) 
	//{
	//	int idx = i + j * lengthX;
	//	return dev_arr[idx];
	//}


	// Return the length of the array.
	__device__  __host__
	int getlenX() { return lengthX; }
	__device__  __host__
	int getlenY() { return lengthY; }
};

template <typename T, int lenX, int lenY>
class const_smartDeviceArrayWrapper2D
{
private: 
	const T* dev_arr;
	int lengthX, lengthY;

public: 
	__device__  __host__
	const_smartDeviceArrayWrapper2D(const T* dev)
	{
		dev_arr = dev;
		lengthX = lenX;
		lengthY = lenY;
	}

	__device__  __host__
	T &operator()(int idx)
	{
		if(idx < 0 || idx > lengthX * lengthY -1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",idx);			
			exit(1);
		}
		T temp = dev_arr[idx];
		return temp; //dev_arr[idx];
	}

	__device__  __host__
	T &operator()(int i, int j)
	{
		if((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d][%d] of GPU array is out of bounds\n",i,j);			
			exit(1);
		}
		int idx = i + j * lengthX;
		T temp = dev_arr[idx];
		return temp; //dev_arr[idx];
	}

	__device__  __host__
	T &operator[](int idx)
	{
		if(idx < 0 || idx > lengthX * lengthY -1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",idx);			
			exit(1);
		}
		T temp = dev_arr[idx];
		return temp; //dev_arr[idx];
	}

	//int i;

	//template <typename T>
	//class proxy
	//{
	//private: 
	////	T* _array;
	////	int i;

	//public: 
	////	proxy(T* _array): _array(_array){}
	////	proxy(int i): i(i){}
	//	__device__  __host__
	//	T &operator[](int i)
	//	{	
	//		i = i;
	//		//return i;
	//	//	return _array[i];
	//	}		
	//};

	//__device__  __host__
	//const proxy<T>& operator[] (int j) 
	//{
	//	int idx = i + j * lengthX;
	//	T temp = dev_arr[idx];
	//	return temp; //dev_arr[idx];
	//}


	// Return the length of the array.
	__device__  __host__
	int getlenX() { return lengthX; }
	__device__  __host__
	int getlenY() { return lengthY; }
};

template <typename T, int lenX, int lenY, int lenZ>
class smartDeviceArrayWrapper3D
{
private: 
	T* dev_arr;
	int lengthX, lengthY, lengthZ;

public: 
	__device__  __host__
	smartDeviceArrayWrapper3D(T* dev)
	{
		dev_arr = dev;
		lengthX = lenX;
		lengthY = lenY;
		lengthZ = lenZ;
	}

	__device__  __host__
	T &operator()(int idx)
	{
		if(idx < 0 || idx > lengthX * lengthY * lengthZ -1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",idx);			
			exit(1);
		}
		return dev_arr[idx];
	}

	__device__  __host__
	T &operator()(int i, int j)
	{
		if((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d][%d] of GPU array is out of bounds\n",i,j);			
			exit(1);
		}
		int idx = i + j * lengthX + 1 * lengthX * lengthY;
		return dev_arr[idx];
	}	

	__device__  __host__
	T &operator()(int i, int j, int k)
	{
		if((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d][%d][%d] of GPU array is out of bounds\n",i,j,k);			
			exit(1);
		}
		int idx = i + j * lengthX + k * lengthX * lengthY;
		return dev_arr[idx];
	}

	__device__  __host__
	T &operator[](int idx)
	{
		if(idx < 0 || idx > lengthX * lengthY * lengthZ -1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",idx);			
			exit(1);
		}
		return dev_arr[idx];
	}

	//int i;

	//template <typename T>
	//class proxy
	//{
	//private: 
	////	T* _array;
	////	int i;

	//public: 
	////	proxy(T* _array): _array(_array){}
	////	proxy(int i): i(i){}
	//	__device__  __host__
	//	T &operator[](int i)
	//	{	
	//		i = i;
	//		//return i;
	//	//	return _array[i];
	//	}		
	//};

	//__device__  __host__
	//proxy<T>& operator[] (int j) 
	//{
	//	int idx = i + j * lengthX;
	//	return dev_arr[idx];
	//}


	// Return the length of the array.
	__device__  __host__
	int getlenX() { return lengthX; }
	__device__  __host__
	int getlenY() { return lengthY; }
	__device__  __host__
	int getlenZ() { return lengthZ; }

};

template <typename T, int lenX, int lenY, int lenZ>
class const_smartDeviceArrayWrapper3D
{
private: 
	const T* dev_arr;
	int lengthX, lengthY, lengthZ;

public: 
	__device__  __host__
	const_smartDeviceArrayWrapper3D(const T* dev)
	{
		dev_arr = dev;
		lengthX = lenX;
		lengthY = lenY;
		lengthZ = lenZ;
	}

	__device__  __host__
	T &operator()(int idx)
	{
		if(idx < 0 || idx > lengthX * lengthY * lengthZ -1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",idx);			
			exit(1);
		}
		T temp = dev_arr[idx];
		return temp; //dev_arr[idx];
	}

	__device__  __host__
	T &operator()(int i, int j)
	{
		if((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d][%d] of GPU array is out of bounds\n",i,j);			
			exit(1);
		}
		int idx = i + j * lengthX + 1 * lengthX * lengthY;
		T temp = dev_arr[idx];
		return temp; //dev_arr[idx];
	}

	__device__  __host__
	T &operator()(int i, int j, int k)
	{
		if((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d][%d][%d] of GPU array is out of bounds\n",i,j,k);			
			exit(1);
		}
		int idx = i + j * lengthX + k * lengthX * lengthY;
		T temp = dev_arr[idx];
		return temp; //dev_arr[idx];
	}

	__device__  __host__
	T &operator[](int idx)
	{
		if(idx < 0 || idx > lengthX * lengthY * lengthZ -1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",idx);			
			exit(1);
		}
		T temp = dev_arr[idx];
		return temp; //dev_arr[idx];
	}

	//int i;

	//template <typename T>
	//class proxy
	//{
	//private: 
	////	T* _array;
	////	int i;

	//public: 
	////	proxy(T* _array): _array(_array){}
	////	proxy(int i): i(i){}
	//	__device__  __host__
	//	T &operator[](int i)
	//	{	
	//		i = i;
	//		//return i;
	//	//	return _array[i];
	//	}		
	//};

	//__device__  __host__
	//proxy<T>& operator[] (int j) 
	//{
	//	int idx = i + j * lengthX;
	//	return dev_arr[idx];
	//}


	// Return the length of the array.
	__device__  __host__
	int getlenX() { return lengthX; }
	__device__  __host__
	int getlenY() { return lengthY; }
	__device__  __host__
	int getlenZ() { return lengthZ; }

};

template <typename T, int lenX, int lenY, int lenZ, int lenW>
class smartDeviceArrayWrapper4D
{
private: 
	T* dev_arr;
	int lengthX, lengthY, lengthZ, lengthW;

public: 
	__device__  __host__
	smartDeviceArrayWrapper4D(T* dev)
	{
		dev_arr = dev;
		lengthX = lenX;
		lengthY = lenY;
		lengthZ = lenZ;
		lengthW = lenW;
	}

	__device__  __host__
	T &operator()(int idx)
	{
		if(idx < 0 || idx > lengthX * lengthY * lengthZ -1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",idx);			
			exit(1);
		}
		return dev_arr[idx];
	}

	__device__  __host__
	T &operator()(int i, int j)
	{
		if((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d][%d] of GPU array is out of bounds\n",i,j);			
			exit(1);
		}
		int idx = i + j * lengthX + 0 * lengthX * lengthY + 0 * lengthX * lengthY * lengthZ;
		return dev_arr[idx];
	}	

	__device__  __host__
	T &operator()(int i, int j, int k)
	{
		if((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d][%d][%d] of GPU array is out of bounds\n",i,j,k);			
			exit(1);
		}
		int idx = i + j * lengthX + k * lengthX * lengthY + 0 * lengthX * lengthY * lengthZ;
		return dev_arr[idx];
	}

	__device__  __host__
	T &operator()(int i, int j, int k, int w)
	{
		if(((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1)) ||(w < 0 || w > lengthW-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
			exit(1);
		}
		int idx = i + j * lengthX + k * lengthX * lengthY + w * lengthX * lengthY * lengthZ;
		return dev_arr[idx];
	}

	__device__  __host__
	T &operator[](int idx)
	{
		if(idx < 0 || idx > lengthX * lengthY * lengthZ * lengthW -1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",idx);			
			exit(1);
		}
		return dev_arr[idx];
	}

	// Return the length of the array.
	__device__  __host__
	int getlenX() { return lengthX; }
	__device__  __host__
	int getlenY() { return lengthY; }
	__device__  __host__
	int getlenZ() { return lengthZ; }
	__device__  __host__
	int getlenW() { return lengthW; }

};

template <typename T, int lenX, int lenY, int lenZ, int lenW>
class const_smartDeviceArrayWrapper4D
{
private: 
	const T* dev_arr;
	int lengthX, lengthY, lengthZ, lengthW;

public: 
	__device__  __host__
	const_smartDeviceArrayWrapper4D(const T* dev)
	{
		dev_arr = dev;
		lengthX = lenX;
		lengthY = lenY;
		lengthZ = lenZ;
		lengthW = lenW;
	}

	__device__  __host__
	T &operator()(int idx)
	{
		if(idx < 0 || idx > lengthX * lengthY * lengthZ -1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",idx);			
			exit(1);
		}
		T temp = dev_arr[idx];
		return temp; //dev_arr[idx];
	}

	__device__  __host__
	T &operator()(int i, int j)
	{
		if((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d][%d] of GPU array is out of bounds\n",i,j);			
			exit(1);
		}
		int idx = i + j * lengthX + 1 * lengthX * lengthY + 1 * lengthX * lengthY * lengthZ;
		T temp = dev_arr[idx];
		return temp; //dev_arr[idx];
	}	

	__device__  __host__
	T &operator()(int i, int j, int k)
	{
		if((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d][%d][%d] of GPU array is out of bounds\n",i,j,k);			
			exit(1);
		}
		int idx = i + j * lengthX + k * lengthX * lengthY + 1 * lengthX * lengthY * lengthZ;
		T temp = dev_arr[idx];
		return temp; //dev_arr[idx];
	}

	__device__  __host__
	T &operator()(int i, int j, int k, int w)
	{
		if(((i < 0 || i > lengthX-1) || (j < 0 || j > lengthY-1) || (k < 0 || k > lengthZ-1)) ||(w < 0 || w > lengthW-1))
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d][%d][%d][%d] of GPU array is out of bounds\n",i,j,k,w);			
			exit(1);
		}
		int idx = i + j * lengthX + k * lengthX * lengthY + w * lengthX * lengthY * lengthZ;
		T temp = dev_arr[idx];
		return temp; //dev_arr[idx];
	}

	__device__  __host__
	T &operator[](int idx)
	{
		if(idx < 0 || idx > lengthX * lengthY * lengthZ * lengthW -1) 
		{
			// Take appropriate action here. This is just a placeholder response.
			fprintf(stderr, "Index [%d] of GPU array is out of bounds\n",idx);			
			exit(1);
		}
		T temp = dev_arr[idx];
		return temp; //dev_arr[idx];
	}

	// Return the length of the array.
	__device__  __host__
	int getlenX() { return lengthX; }
	__device__  __host__
	int getlenY() { return lengthY; }
	__device__  __host__
	int getlenZ() { return lengthZ; }
	__device__  __host__
	int getlenW() { return lengthW; }

};
