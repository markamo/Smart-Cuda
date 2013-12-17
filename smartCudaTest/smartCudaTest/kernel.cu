
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#include "smartCuda_lib\smartCuda_010.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] + b[i];
	i += blockDim.x * gridDim.x;
}

int main()
{

cudaError_t cudaStatus = cudaSuccess;
 cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        //goto Error;
    }

	const int arraySize = 100;
	int* h_a = smartArray<int,smartHost>(arraySize, cudaStatus);
	int* hp_a = smartArray<int,smartPinnedHost>(arraySize, cudaStatus);
	int* h_b = smartArray<int,smartHost>(arraySize, cudaStatus);
	int* h_c = smartArray<int,smartHost>(arraySize, cudaStatus);

	const int lenX = 10;
	const int lenY = 20;
	const int lenZ = 5;
	const int lenW = 3;

	int* h_1D = smartArray<int,smartHost>(lenX, cudaStatus);
	int* h_2D = smartArray<int,smartHost>(lenX, lenY, cudaStatus);
	int* h_3D = smartArray<int,smartHost>(lenX, lenY, lenZ, cudaStatus);
	int* h_4D = smartArray<int,smartHost>(lenX, lenY, lenZ, lenW, cudaStatus);
	

	{
		idx_initializeSmartArray<int>(h_a,arraySize,0,1);
		idx_initializeSmartArray<int>(h_b,arraySize,-0,1);
		initializeSmartArray<int>(h_c,arraySize,0);
		idx_initializeSmartArray<int>(hp_a,arraySize,0);

		int* d_a = smartArray<int,smartDevice>(arraySize, cudaStatus);
		int* d_a1 = smartArray<int,smartDevice>(arraySize, cudaStatus);
		int* d_b = smartArray<int,smartDevice>(arraySize, cudaStatus);
		int* d_c = smartArray<int,smartDevice>(arraySize, cudaStatus);

		initializeSmartArrayAsync_core<int><<<128, 128>>>(d_c,arraySize,-100);

		smartArrayWrapper<int,smartHost> wHa(h_a,arraySize,scopeGlobal);
		smartArrayWrapper<int,smartHost> wHb(h_b,arraySize,scopeGlobal);
		smartArrayWrapper<int,smartHost> wHc(h_c,arraySize,scopeGlobal);

		smartArrayWrapper<int,smartDevice> wDa(d_a,arraySize,scopeGlobal);
		smartArrayWrapper<int,smartDevice> wDb(d_b,arraySize,scopeGlobal);
		smartArrayWrapper<int,smartDevice> wDc(d_c,arraySize,scopeGlobal);

		//smartArrayWrapper<int,smartPinnedHost> wHa1;
		//wHa1.wrap(h_a,arraySize,scopeGlobal);

		wDa = wHa;
		wDb = wHb;

		//smartArrayWrapper<int,smartHost> wHa1;
		//wHa1.wrap(h_a,arraySize,scopeGlobal);
		////wHa1 = wHa;


		wDa.copy(wHa.inner_ptr(),wHa.getlen(),wHa.getType());
		wDb.copy(wHb.inner_ptr(),wHb.getlen(),wHb.getType());

		addKernel<<<192, 192>>>(wDc.inner_ptr(), wDa.inner_ptr(), wDb.inner_ptr());


		 cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		  //  goto Error;
		}
	
		cudaStatus = wHc = wDc;
		

		for (int i = 0; i < arraySize; i++)
		{
			 printf("index %d: %d + %d = %d\n", i, wHa[i],wHb[i], wHc[i]);
		}

		
	}	 

	//tuple<int*,int,int> tA (h_b, arraySize,smartHost);

	smartArrayWrapper<int,smartHost> wHa(h_a,arraySize,scopeLocal);
	smartArrayWrapper<int,smartPinnedHost> wHa1;
	wHa1.wrap(hp_a,arraySize,scopeGlobal);

	//wHa.inner_ptr_ref() = smartArray<int,smartHost>(arraySize, cudaStatus);	
	//wHa = wHa1;

	for (int i = 0; i < arraySize / 10; i++)
	{
			printf("data \t%d:\t%d = %d\n", i, wHa++,++wHa1);
			printf("pos \t%d: \t%d = %d\n", i, wHa.pos(),wHa1.pos());
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	__clean_globals = ON;
    return 0;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
