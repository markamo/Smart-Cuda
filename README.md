![Smart CUDA](logo.png)
# Welcome to Smart CUDA Library Project Page

***

Smart CUDA library is a lightweight C/C++ wrapper of the CUDA runtime API for productive natural-like CUDA programming. Smart CUDA follows the natural CUDA programming style; however, it provides low level abstraction to allow for a more convenient programming. In this way, Smart CUDA enhances developer productivity while attaining very high performance. Smart CUDA integrates seamlessly with C/C++ arrays and pointers, STL vector, and Thrust arrays. Smart CUDA only wraps the data on the device and host, therefore data on the host and device can have several wrappers and different views. This makes it very flexible, and allows easy integration with other platform and libraries.

# Why Smart CUDA library? 
Even though I am relatively new to C/C++ and CUDA programming, I realized that a lightweight wrapper could boost gpu programming productivity. Thus, I developed several wrappers that preserved the natural programming style as I went through the CUDA Programming Guide. Smart CUDA is the compilation of the basic wrappers I have currently developed. Smart CUDA library is meant to complement the efforts of other libraries such as [Thrust](http://thrust.github.io/) and [VexCL](http://ddemidov.github.io/vexcl) and help boost gpu programming productivity.

***

# FEATURES

## Header-only library
```C++
#include "smartCuda_lib\smartCuda.h" ////include current version of smartCuda

```
## Minimal Learning
**Only two things to learn: Smart Array and Smart Array Wrapper.** The Smart Array is for data allocation and Smart Array Wrapper is for management of allocated data.

```C++
//// memory and data allocation 
//// smartArray has overloads for allocating up to 4D data
template <typename T, int mem_loc> 
	inline T* smartArray(int sizeX);
	
template <typename T, int mem_loc> 
	inline T* smartArray( int sizeX, cudaError_t &cudaStatus);

//// smartArray wrapper wraps arrays on cpu and gpu memories for convenient access and management
template <typename T, int mem_loc> 
	class smartArrayWrapper{};
```
***
# Smart Array
## Easy Memory Allocation
```C++
cudaError_t cudaStatus = cudaSuccess;
const int arraySize = 100000;
int* h_a = smartArray<int,smartHost>(arraySize); ////pageable host memory
int* hp_a = smartArray<int,smartPinnedHost>(arraySize); ////pinned host memory
int* d_a = smartArray<int,smartDevice>(arraySize); ////device memory 
```

## Multidimensional array allocation (up to 4D)
```C++
const int lenX = 10;
const int lenY = 20;
const int lenZ = 5;
const int lenW = 3;

////allocation on CPU
int* h_1D = smartArray<int,smartHost>(lenX);
int* h_2D = smartArray<int,smartHost>(lenX, lenY);
int* h_3D = smartArray<int,smartHost>(lenX, lenY, lenZ);
int* h_4D = smartArray<int,smartHost>(lenX, lenY, lenZ, lenW);

////allocation on GPU
int* d_1D = smartArray<int,smartDevice>(lenX);
int* d_2D = smartArray<int,smartDevice>(lenX, lenY);
int* d_3D = smartArray<int,smartDevice>(lenX, lenY, lenZ);
int* d_4D = smartArray<int,smartDevice>(lenX, lenY, lenZ, lenW);

////use of inline array to allocate memory in kernels
int* arr = smartArray<int,smartInlineArray>(arr_size); //create an inline array of size arr_size;
...
__global__ void testKernel(int arr_size)
{
    int* arr = smartArray<int,smartInlineArray>(arr_size); //create an inline array of size arr_size;
    ...
   smartInlineArrayFree(arr);

}
...
```
***

# Smart Array Wrapper
## Convenient wrapper for data management
```C++
...
////wrap array on host
smartArrayWrapper<int,smartHost> wHa(h_a,arraySize,scopeLocal);  
smartArrayWrapper<int,smartHost> wHb(h_b,arraySize,scopeLocal);
smartArrayWrapper<int,smartHost> wHc(h_c,arraySize,scopeLocal);

////wrap array on device
smartArrayWrapper<int,smartDevice> wDa(d_a,arraySize,scopeLocal);
smartArrayWrapper<int,smartDevice> wDb(d_b,arraySize,scopeLocal);
smartArrayWrapper<int,smartDevice> wDc(d_c,arraySize,scopeLocal);

/////alternative wrap method
smartArrayWrapper<int,smartHost> wHa1;
wHa1.wrap(h_a,arraySize,scopeGlobal);
...

////access the underlying array using the object.inner_ptr()

......
__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] + b[i];
	i += blockDim.x * gridDim.x;
}

....
////access the underlying array using the object.inner_ptr()
addKernel<<<16, 16>>>(wDc.inner_ptr(), wDa.inner_ptr(), wDb.inner_ptr());

```


## Seamless data transfers between CPU and GPU
```C++
...
////wrap array on host
smartArrayWrapper<int,smartHost> wHa(h_a,arraySize,scopeLocal);  
smartArrayWrapper<int,smartHost> wHb(h_b,arraySize,scopeLocal);
smartArrayWrapper<int,smartHost> wHc(h_c,arraySize,scopeLocal);

////wrap array on device
smartArrayWrapper<int,smartDevice> wDa(d_a,arraySize,scopeLocal);
smartArrayWrapper<int,smartDevice> wDb(d_b,arraySize,scopeLocal);
smartArrayWrapper<int,smartDevice> wDc(d_c,arraySize,scopeLocal);

////transfer data from host to device 
wDa = wHa; ////quick data transfer

////copy method has extended support and overloads for other data types
wDb.copy(wHb); 
...
wDa.copy(wHa.inner_ptr(),wHa.getlen(),wHa.getType());
wDb.copy(wHb.inner_ptr(),wHb.getlen(),wHb.getType());
....
////do some work on the GPU


////transfer data back to CPU
wHc = wDc; 

```

## Local and Global scopes for automatic memory deallocation
**scopeLocal**: for Local scopes. 
**scopeGlobal**: to persist data beyond the current scope. 
**__clean_globals = ON/OFF**: to clean global wrapper data declared in current scope. 
```C++
...
const int arraySize = 1000;
int* h_1 = smartArray<int,smartHost>(arraySize);
int* h_2 = smartArray<int,smartHost>(arraySize);
{
    ////wrap host and device data
    ////use scopeLocal for managing data that is for local scopes
    ////memory is automatically freed when the wrapper goes out of scope    
    smartArrayWrapper<int,smartHost> wLocal(h_1,arraySize,scopeLocal); 

    ...
    ////use scopeGlobal for managing data that is for global scopes
    smartArrayWrapper<int,smartHost> wGlobal(h_2,arraySize,scopeGlobal);
    ....
    /////do some work
    ....
}//// data for h_1 automatically deleted because it is local scope. 
//// data for h_2 is preserved because it is global to the scope created. 

//// h_1 must be reinitialized before used again
h_1 = smartArray<int,smartHost>(arraySize);

smartArrayWrapper<int,smartHost> w2(h_1,arraySize,scopeGlobal); 
////do some work
...
////manual deletion of allocated memory
w2.destroy();

....
////use __clean_globals = ON/OFF to clean global scope wrappers when they get out of their current scope
...
Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;

///// can be replaced by 
...
Error:
    __clean_globals = ON; //// delete global wrappers and their underlying data
   //__clean_globals = OFF; //// preserve the data that the global wrappers are pointing to

    return cudaStatus;
```


## Aliases for Thread Indexing  

```C++
//// alias for __global__ 
#define __KERNEL__ __global__ 

////indexing along one dimension x,y,z = 0,1,2. default index is along the x dimension (0)
 // alias for threadIdx.x;
__device__ inline int __local_index(int dim = 0);

 // alias for blockIdx.x * blockDim.x + threadIdx.x;
__device__ inline int __global_index(int dim = 0);

 // alias for blockIdx.x;
__device__ inline int __block_index(int dim = 0);
__device__ inline int __group_index(int dim = 0);

 // alias for gridDim.x * blockDim.x;
__device__ inline int __stride(int dim = 0);

////alias functions for size of threads and blocks along the x, y, z and total size in all dimensions (i.e. 0,1,2,-1). The default is the total size along all dimensions (-1).

 // alias for gridDim.x;
__device__ inline int __num_blocks(int dim = -1 );
__device__ inline int __num_groups(int dim = -1);

 // alias for gridDim.x;
__device__ inline int __block_size(int dim = -1);
__device__ inline int __group_size(int dim = -1);
__device__ inline int __local_size(int dim = -1);

 // alias for gridDim.x * blockDim.x;
__device__ inline int __launch_size(int dim = -1);
__device__ inline int __global_size(int dim = -1);

``` 


## Indexing and Data Access (up to 4D) 
```C++
...
const int lenX = 100;
const int lenY = 100;

smartArrayWrapper<int,smartDevice> wA(d_a,lenX, lenY,scopeLocal);
smartArrayWrapper<int,smartDevice> wB(d_b,lenX, lenY,scopeLocal);
smartArrayWrapper<int,smartDevice> wC(d_c,lenX, lenY,scopeLocal);

int i = threadIdx.x + blockIdx.x * blockDim.x;
int j = threadIdx.y + blockIdx.y * blockDim.y;

////the following gives the same results
//// traditional CUDA style
////d_c[i + j * lenX] = d_a[i + j * lenX] * d_b[i + j * lenX]; 
//// traditional CUDA style possible with smart array wrapper
////wC[i + j * lenX] = wA[i + j * lenX] * wB[i + j * lenX]; 
////wC(i + j * lenX) = wA(i + j * lenX) * wB(i + j * lenX); 
////convenient indexing
wC(i,j) = wA(i,j) * wB(i,j); 

////supports upto 4 dimensions 
smartArrayWrapper<int,smartDevice> w4(d_a,lenX, lenY, lenZ, lenW, scopeLocal);
int results = w4(5,5,5,5);

////access the nth data element 
int results = w4[117];

```


## Customized Views of Pre-Allocated Data (up to 4D) 
```C++
...
const int arraySize = 1000;
int* h_1 = smartArray<int,smartHost>(arraySize);
...
////wrap array on host
smartArrayWrapper<int,smartHost> wView2D(h_1,arraySize,scopeLocal);  
wView2D.setViewDim(100,10); //// view 1D array as a 2D 100 * 10 array
int temp = wView2D.ViewAt(50,5); //// get data at 50,10 of the view (up to 4D view)
int temp2 = wView2D.ViewAt_2D(45,2); //// alternative view method (only 2D view)

smartArrayWrapper<int,smartHost> wView3D(h_1,arraySize,scopeLocal);
wView3D.setViewDim(10,10,10); //// view 1D array as a 3D 10 * 10 * 10 array
int temp3 = wView3D.ViewAt(5,5,5); //// get data at 5,5,5 of the view (up to 4D view)
int temp4 = wView3D.ViewAt_3D(4,5,2); //// alternative view method (only 3D view)

////access the original array 
//// the setViewDim and viewAt methods do not modify the layout of the array
int temp5 = wWiew3D[150]; ////access the original h_1 array at index 150 using []
int temp6 = wWiew2D(300); ////access the original h_1 array at index 300 using ()
...
```

## Navigation (up to 4D indexing) 
**Navigation using PEEK, SEEK, and ADV(advance)**
```C++
...
const int arraySize = 1000;
int* d_1 = smartArray<int,smartDevice>(arraySize);
...
////wrap array on device
smartArrayWrapper<int,smartDevice> wNav(d_1,arraySize,scopeLocal);  
...
////navigate through the data 
////peek examples
int peek1 = wNav.peek(); ////get the next data without moving the data access position
int peek5 = wNav.peek(5); ////get the next 5th position data without moving the data access position
int peek_3 = wNav.peek(-3); ////get the previous 3rd position data without moving the data access position
...
////adv examples
int adv1= wNav.seek(); ////move to the next data and return a reference to the data
int adv5 = wNav.seek(5); ////move to the next 5th position data and return a reference
int adv_3 = wNav.seek(-3); ////move to the previous 3rd position data and return a reference

...
////seek examples
int seek1 = wNav.seek(); ////move to the next data and return a reference to the data
int seek5 = wNav.seek(5); ////move to the 5th position data and return a reference
int seek_3 = wNav.seek(3); ////move to the 3rd position data and return a reference ////negative index not allow
...

```

**Navigation using PEEK4, SEEK4, and ADV4(advance4)**
```C++
....
/////other peek, seek and adv methods
////peek4(),seek4(),adv4() uses convenient indexing to navigate the data as above
int seek4 = wNav.seek4(2,5,1,3); ////move to the 2,5,1,3 position data and return a reference
int adv4 = wNav.adv4(-3,1,2); ////move to -3,1,2 position from the current index data and return a reference
int peek4 = wNav.peek4(5,7); ////get the data at index 5,7 from the current position without moving the data access position
...

```

**Navigation on Customized Views VPEEK(), PEEK4(), VSEEK(), VSEEK4(), VADV(), and VADV4()**
```C++
....
/////navigation on customized views: vpeek(), vseek() and vadv() methods
////vpeek4(),vseek4(),vadv4() uses convenient indexing to navigate customized views

...
cvNav.setViewDim(10,10,...,...); //// set a customized array view 

...
/////navigating customized views
int cseek = cvNav.vseek(); ////move to the next data and return a reference to the data
int cadv = cvNav.vadv(8); ////move to the 8th position data and return a reference
int cpeek = cvNav.vpeek(3); ////return data value at position 3 from current location

...
////multidimensional navigation of customized views
int cvseek4 = cvNav.vseek4(2,1,4,6); ////move to the 2,1,4,6 of the customized array view
int cvadv4 = cvNav.vadv4(4,1,2); ////move to 4,1,2 position from the current index data and return a reference
int cvpeek4 = cvNav.vpeek4(3,3); ////get the data at index 3,3 from the current position from the customized view without moving the data access position
...

```

**Navigation using ++ and --**
```C++
....
/////other navigation methods
int nav_ref = wNav++; ////returns a reference to the next data element and increases the data access index
int nav = ++wNav; ////returns the value of the next data element and increases the data access index

int nav_ref1 = wNav--; ////returns a reference to the previous data element and decreases the data access index
int nav1 = --wNav; ////returns the value of the previous data element and decreases the data access index


```
***
# Other Features
## Smart Device Management 
```C++
smartDeviceManager dev;
dev.mapDevice(0);
dev.setDevice();
....
dev.resetDevice();
```

## Smart Event Timings
```C++
smartEvent stopwatch;
...
		
stopwatch.recStart();
for (int i = 0; i < arraySize; i++)
{
	//// printf("index %d: %d + %d = %d\n", i, wHa[i],wHb[i], wHc[i]);
	if (wHc(i) < 0)
	{
		wHc(i) = wHa(i) + wHb(i);
	}
}
stopwatch.recStop();
stopwatch.elapsedTime();
stopwatch.printElapsedTime();


```

## Array Initializations
```C++

const unsigned int arraySize = 10000000;
int* h_a = smartArray<int,smartHost>(arraySize);
int* hp_a = smartArray<int,smartPinnedHost>(arraySize);
int* h_b = smartArray<int,smartHost>(arraySize);
int* h_c = smartArray<int,smartHost>(arraySize);

....
idx_initializeSmartArray<int>(h_a,arraySize,0,1);
idx_initializeSmartArray<int>(h_b,arraySize,-0,1);
initializeSmartArray<int>(h_c,arraySize,0);
idx_initializeSmartArray<int>(hp_a,arraySize,0);
...
initializeSmartArrayAsync<int>(d_c,arraySize,-100);
```

## Smart Kernel Configuration (Beta)
```C++
const unsigned int arraySize = 10000000;
....
smartConfig config(arraySize);
addKernel<<<config.getBlockSize(), config.getThreadSize()>>>(wDc.inner_ptr(), wDa.inner_ptr(), wDb.inner_ptr(), arraySize);

```

## Smart Random Numbers and Transformations
* New Kernel function ``` appy_func_core```, perform parallel element wise operations on allocated device arrays
```C++ 
template <typename T, class Op> __global__  
void appy_func_core(T* dev_Array, const int size, Op fn);

template <typename T, class Op> __global__
 void appy_func_core(T* dest, T* src, const int size, Op fn);

template <typename T, class Op> __global__
 void appy_func_core(T* dest, T* src1, T* src2, const int size, Op fn);

``` 

* New Kernel function ``` transform_core```, perform parallel element wise transformations on allocated device arrays. Supports up to 10 allocated device arrays
```C++
template <typename T, class Op> __global__
 void transfrom_core(T* arr, const int size, Op fn );

template <typename T, class Op> __global__
 void transfrom_core(T* arr, T* arr1, T* arr2, const int size, Op fn );

...

template <typename T, class Op> __global__
 void transfrom_core(T* arr, T* arr1, T* arr2, T* arr3, T* arr4,  T* arr5,  T* arr6, T* arr7, T* arr8, T* arr9, const int size, Op fn );

...

``` 

* New Kernel function ``` transform_core_t```, perform parallel element wise transformations on allocated device arrays of different types. Supports up to 10 allocated device arrays and types
```C++ 
template <typename T, class Op> __global__
 void transfrom_core_t(T* arr, const int size, Op fn );


template <typename T, typename T1, typename T2, class Op> __global__
 void transfrom_core_t(T* arr, T1* arr1, T2* arr2, const int size, Op fn );

...

template <typename T, typename T1, typename T2,  typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, class Op> __global__
 void transfrom_core_t(T* arr, T1* arr1, T2* arr2, T3* arr3, T4* arr4,  T5* arr5,  T6* arr6, T7* arr7, T8* arr8, T9* arr9, const int size, Op fn );

...

``` 

* Simple Reduction function and function operators for reductions on gpu. Customer implementation of smartOperator can be used with for the reduction kernel.

```C++ 
////pre-defined operators
template <typename T> class smartOperator;
template <typename T> class smartPlus: public smartOperator<T>;
template <typename T> class smartMultiply: public smartOperator<T>;
template <typename T> class smartMax: public smartOperator<T>;
template <typename T> class smartMin: public smartOperator<T>;
template <typename T> class smartOR: public smartOperator<T>;
template <typename T> class smartAND: public smartOperator<T>;

////kernel launcher
template<typename T, class Op>
void smartReduce_kl(T *answer, T *partial, const T *in, size_t N, int numBlocks, int numThreads, Op fn );


``` 

* Smart Random library for random number generation in on device kernels. Use of a default random number that can be called from any part of the code. Smart Random library provides a lightweight wrapper on cuRand library.

Functions:
```C++ 
 __device__ curandState *defaultStates;
...
__host__ cudaError_t initDefaultRand(int size = 64 * 64, int seed = time(NULL));
__host__ cudaError_t releaseDefaultRand();

``` 

Usage:
```C++ 
initDefaultRand(256*256);
....
////use defaultStates in cuRand calls
....

releaseDefaultRand(); //// called when done using defaultStates to release memory allocated;

``` 


* Other Smart Random library functions and kernels
```C++ 
__global__ void setup_rand_kernel(curandState *state, unsigned int size, unsigned int seed);

template <typename T>
__global__ void generate_uniform_kernel(T* result, int size, curandState *state );

template <typename T>
__global__ void generate_uniform_range_kernel(T* result, T lower, T upper, int size, curandState *state );

template <typename T> 
__host__ cudaError_t smartRandu(T *dev_Array, const int size, curandState *state = defaultStates);

template <typename T> 
__host__ cudaError_t smartRandu(T *dev_Array, const int sizeX, const int sizeY, curandState *state = defaultStates);

template <typename T> 
__host__ cudaError_t smartRandu(T *dev_Array, const int sizeX, const int sizeY, const int sizeZ, curandState *state = defaultStates);

template <typename T> 
__host__ cudaError_t smartRandu(T *dev_Array, const int sizeX, const int sizeY, const int sizeZ, const int sizeW, curandState *state = defaultStates);

template <typename T> 
__host__ cudaError_t smartRandr(T *dev_Array, T min, T max, const int size, curandState *state = defaultStates);

template <typename T> 
__host__ cudaError_t smartRandr(T *dev_Array, T min, T max, const int sizeX, const int sizeY, curandState *state = defaultStates);

template <typename T> 
__host__ cudaError_t smartRandr(T *dev_Array, T min, T max, const int sizeX, const int sizeY, const int sizeZ, curandState *state = defaultStates);

template <typename T> 
__host__ cudaError_t smartRandr(T *dev_Array, T min, T max, const int sizeX, const int sizeY, const int sizeZ, const int sizeW, curandState *state = defaultStates);


``` 

***

***

# Latest News
* [Smart CUDA v0.2.0 (Initial Release) - 18th January, 2014](https://github.com/markamo/Smart-Cuda/releases/tag/v0.2.0)
* [Smart CUDA v0.1.2 (Initial Release) - 19th December, 2013](https://github.com/markamo/Smart-Cuda/releases/tag/v0.1.2)
* [Smart CUDA v0.1.1 (Initial Release) -17th December, 2013](https://github.com/markamo/Smart-Cuda/releases/tag/v0.1.1)
* [SmartCUDA v 0.0.1(draft-release) - 16th December, 2013](https://github.com/markamo/Smart-Cuda/releases/tag/v0.0.1d)


### Features under consideration for future releases
- [-] Smart Kernel
- [-] Smart Device
- [-] SmartArrayWrapper.apply_func()
- [ ] SmartArrayWrapper.apply_funcAsync()
- [ ] SmartArrayWrapper.sort()
- [ ] SmartArrayWrapper.sortAsync()
- [ ] SmartArrayWrapper.reduce()
- [ ] SmartArrayWrapper.scan()
- [ ] Smart Array Wrapper basic mathematical operators
- [-] Full integration with STL::array and STL::vector
- [-] Full integration with Thrust::host_vector and Thrust::device_vector
- [ ] Basic integration with OpenCL, OpenMP, TBB, and C++ AMP 
- [ ] Integration with other CUDA libraries 
- [-] Multi-Host and Multi-Device data allocation and management
- [ ] Etc.

### Authors and Contributors
The original creator of Smart CUDA is Mark Amo-Boateng (@markamo). 

### Support or Contact
Having trouble with Smart CUDA? Check out this [Wiki] (https://github.com/markamo/Smart-Cuda/wiki). Visit http://markamo.github.io/Smart-Cuda/ or https://github.com/markamo/Smart-Cuda for latest news and source codes. Feel free to contact smartcuda@outlook.com for additional support.
