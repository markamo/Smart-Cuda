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

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "..\smartCuda_020.h"


 __device__
curandState *defaultStates;
bool _default_rand_initialized = false;
int rand_size = 0;


__global__ void setup_rand_kernel(curandState *state, unsigned int size, unsigned int seed)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	//int id = threadIdx.x + blockIdx.x * 64;
	/* Each thread gets same seed, a different sequence 
	number, no offset */
	//curand_init(1234, id, 0, &state[id]);

	//curandState localState = state[id];
	while (id < size)
	{
		curand_init(seed, id, 0, &state[id]);
		//curand_init(seed, id, 0, &localState);
		id += blockDim.x * gridDim.x;
	}
	//state[id] = localState;
}

template <typename T>
__global__ void generate_uniform_kernel(T* result, int size, curandState *state )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	/* Copy state to local memory for efficiency */
	curandState localState = state[id];
	int state_id = id;

	/* Generate pseudo-random uniforms */
	while (id < size)
	{
		curandState localState = state[id];
		result[id] = (T) (curand_uniform(&localState));	
		id += blockDim.x * gridDim.x;
		////state[id] = localState;	
	}

	/* Copy state back to global memory */
	state[state_id] = localState;	
}

template <typename T>
__global__ void generate_uniform_range_kernel(T* result, T lower, T upper, int size, curandState *state )
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	/* Copy state to local memory for efficiency */
	curandState localState = state[id];
	int state_id = id;

	/* Generate pseudo-random uniforms */
	while (id < size)
	{
		//curandState localState = state[id];
		result[id] = (T) (lower + (curand_uniform(&localState) * (upper - lower)));	
		id += blockDim.x * gridDim.x;
		////state[id] = localState;	
	}

	/* Copy state back to global memory */
	state[state_id] = localState;	
}


template <typename T>
__device__ 
inline T unif_grand(curandState *state,  int id)
{
	curandState local_state;
	T rnd = curand_uniform(&state[id]);
	return rnd;
}


template <typename T>
__device__ 
inline T unif_grand(curandState &state)
{
	//curandState local_state;
	T rnd = curand_uniform(&state);
	return rnd;
}



template <typename T> 
__host__ cudaError_t smartRandu(T *dev_Array, const int size, curandState *state = defaultStates)
{
	cudaError_t cudaStatus;
    generate_uniform_kernel<<<64, 64>>>(dev_Array, size, state);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}
	return cudaStatus;
}


template <typename T> 
__host__ cudaError_t smartRandu(T *dev_Array, const int sizeX, const int sizeY, curandState *state = defaultStates)
{
	int size = sizeX * sizeY;
	cudaError_t cudaStatus;
    generate_uniform_kernel<<<64, 64>>>(dev_Array, size, state);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}
	return cudaStatus;
}


template <typename T> 
__host__ cudaError_t smartRandu(T *dev_Array, const int sizeX, const int sizeY, const int sizeZ, curandState *state = defaultStates)
{
	int size = sizeX * sizeY * sizeZ;
	cudaError_t cudaStatus;
    generate_uniform_kernel<<<64, 64>>>(dev_Array, size, state);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}
	return cudaStatus;
}


template <typename T> 
__host__ cudaError_t smartRandu(T *dev_Array, const int sizeX, const int sizeY, const int sizeZ, const int sizeW, curandState *state = defaultStates)
{
	int size = sizeX * sizeY * sizeZ * sizeW;
	cudaError_t cudaStatus;
    generate_uniform_kernel<<<64, 64>>>(dev_Array, size, state);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}
	return cudaStatus;
}



template <typename T> 
__host__ cudaError_t smartRandr(T *dev_Array, T min, T max, const int size, curandState *state = defaultStates)
{
	cudaError_t cudaStatus;
    generate_uniform_range_kernel<<<64, 64>>>(dev_Array, min, max, size, state);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}
	return cudaStatus;
}


template <typename T> 
__host__ cudaError_t smartRandr(T *dev_Array, T min, T max, const int sizeX, const int sizeY, curandState *state = defaultStates)
{
	int size = sizeX * sizeY;
	cudaError_t cudaStatus;
    generate_uniform_range_kernel<<<64, 64>>>(dev_Array, min, max, size, state);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}
	return cudaStatus;
}


template <typename T> 
__host__ cudaError_t smartRandr(T *dev_Array, T min, T max, const int sizeX, const int sizeY, const int sizeZ, curandState *state = defaultStates)
{
	int size = sizeX * sizeY * sizeZ;
	cudaError_t cudaStatus;
    generate_uniform_range_kernel<<<64, 64>>>(dev_Array, min, max, size, state);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}
	return cudaStatus;
}


template <typename T> 
__host__ cudaError_t smartRandr(T *dev_Array, T min, T max, const int sizeX, const int sizeY, const int sizeZ, const int sizeW, curandState *state = defaultStates)
{
	int size = sizeX * sizeY * sizeZ * sizeW;
	cudaError_t cudaStatus;
    generate_uniform_range_kernel<<<64, 64>>>(dev_Array, min, max, size, state);
	////check for launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Memory initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}
	return cudaStatus;
}






__host__ cudaError_t initDefaultRand(int size = 64 * 64, int seed = time(NULL))
{
	cudaError_t cudaStatus;

	if (!_default_rand_initialized)
	{
		defaultStates = smartArray<curandState,smartDevice>(size);
		setup_rand_kernel<<<64, 64>>>(defaultStates,size,seed);
		rand_size = size;
		_default_rand_initialized = true;
	}
		
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Random initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}

	cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("Random initialization launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}

	return cudaStatus;

}

__host__ cudaError_t releaseDefaultRand()
{
	cudaError_t cudaStatus;

	if (_default_rand_initialized)
	{
		smartFree<curandState>(defaultStates,smartDevice);
		rand_size = 0;
		_default_rand_initialized = false;
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Random initialization release failed: %s\n", cudaGetErrorString(cudaStatus));
		//////goto Error;
	}
	return cudaStatus;

}

