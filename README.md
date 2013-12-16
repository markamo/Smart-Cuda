![Smart CUDA](http://markamo.github.io/Smart-Cuda/logo.png)
# Welcome to Smart CUDA Library Project Page

***

Smart CUDA library is a lightweight C/C++ wrapper of the CUDA runtime API for productive natural-like CUDA programming. Smart CUDA follows the natural CUDA programming style; however, it provides low level abstraction to allow for a more convenient programming. In this way, Smart CUDA enhances developer productivity while attaining very high performance. Smart CUDA integrates seamlessly with C/C++ arrays and pointers, STL vector, and Thrust arrays. Smart CUDA only wraps the data on the device and host, therefore data on the host and device can have several wrappers and different views. This makes it very flexible, and allows easy integration with other platform and libraries.

# Why Smart CUDA library? 
Even though I am relatively new to C/C++ and CUDA programming, I realized that a lightweight wrapper could boost gpu programming productivity. Thus, I developed several wrappers that preserved the natural programming style as I went through the CUDA Programming Guide. Smart CUDA is the compilation of the basic wrappers I have currently developed. Smart CUDA library is meant to complement the efforts of other libraries such as Thrust and Magma and help boost gpu programming productivity.

***

# FEATURES

## Header-only library
```C++
#include "smartCuda\smartCuda_001d.h" ////include smart cuda version 0.0.1 draft

```
## Minimal Learning
```C++
//// memory and data allocation 
//// smartArray has overloads for allocating up to 4D data
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
int* h_a = smartArray<int,smartHost>(arraySize, cudaStatus); ////pageable host memory
int* hp_a = smartArray<int,smartPinnedHost>(arraySize, cudaStatus); ////pinned host memory
int* d_a = smartArray<int,smartDevice>(arraySize, cudaStatus); ////device memory 
```

## Multidimensional array allocation (up to 4D)
```C++
const int lenX = 10;
const int lenY = 20;
const int lenZ = 5;
const int lenW = 3;

////allocation on CPU
int* h_1D = smartArray<int,smartHost>(lenX, cudaStatus);
int* h_2D = smartArray<int,smartHost>(lenX, lenY, cudaStatus);
int* h_3D = smartArray<int,smartHost>(lenX, lenY, lenZ, cudaStatus);
int* h_4D = smartArray<int,smartHost>(lenX, lenY, lenZ, lenW, cudaStatus);

////allocation on GPU
int* d_1D = smartArray<int,smartDevice>(lenX, cudaStatus);
int* d_2D = smartArray<int,smartDevice>(lenX, lenY, cudaStatus);
int* d_3D = smartArray<int,smartDevice>(lenX, lenY, lenZ, cudaStatus);
int* d_4D = smartArray<int,smartDevice>(lenX, lenY, lenZ, lenW, cudaStatus);
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


## Convenient data transfers between CPU and GPU
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

////do some work on the GPU


////transfer data back to CPU
wHc = wDc; 

```

## Local and Global scopes for automatic memory deallocation
```C++
...
const int arraySize = 1000;
int* h_1 = smartArray<int,smartHost>(arraySize,cudaStatus);
int* h_2 = smartArray<int,smartHost>(arraySize,cudaStatus);
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
h_1 = smartArray<int,smartHost>(arraySize,cudaStatus);

smartArrayWrapper<int,smartHost> w2(h_1,arraySize,scopeGlobal); 
////do some work
...
////manual deletion of allocated memory
w2.destroy();

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


## Customized Views of Pre-Allocated Data(up to 4D) 
```C++
...
const int arraySize = 1000;
int* h_1 = smartArray<int,smartHost>(arraySize,cudaStatus);
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
** Navigation using PEEK, SEEK, and ADV(advance)**
```C++
...
const int arraySize = 1000;
int* d_1 = smartArray<int,smartDevice>(arraySize,cudaStatus);
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

....
/////other peek, seek and adv methods
////peek4(),seek4(),adv4() uses convenient indexing to navigate the data as above
int seek4 = wNav.seek4(2,5,1,3); ////move to the 2,5,1,3 position data and return a reference
int adv4 = wNav.seek(-3,1,2); ////move to -3,1,2 position from the current index data and return a reference
int peek4 = wNav.peek4(5,7); ////get the data at index 5,7 from the current position without moving the data access position
...
/////other navigation methods
int nav_ref = wNav++; ////returns a reference to the next data element and increases the data access index
int nav = ++wNav; ////returns the value of the next data element and increases the data access index

int nav_ref1 = wNav--; ////returns a reference to the previous data element and decreases the data access index
int nav1 = --wNav; ////returns the value of the previous data element and decreases the data access index


```

***

# Latest News
*SmartCUDA v 0.0.1(draft-release) - 16th December, 2013

### Features under consideration for future releases
- [ ] Smart Kernel
- [ ] Smart Device
- [ ] SmartArrayWrapper.apply_func()
- [ ] SmartArrayWrapper.apply_funcAsync()
- [ ] SmartArrayWrapper.sort()
- [ ] SmartArrayWrapper.sortAsync()
- [ ] SmartArrayWrapper.reduce()
- [ ] SmartArrayWrapper.scan()
- [ ] Smart Array Wrapper basic mathematical operators
- [ ] Basic integration with STL::array and STL::vector
- [ ] Basic integration with Thrust::host_vector and Thrust::device_vector
- [ ] Basic integration with OpenCL, OpenMP, TBB, and C++ AMP 
- [ ] Integration with other CUDA libraries 
- [ ] Multi-Host and Multi-Device data allocation and management
- [ ] Etc.

### Authors and Contributors
The original creator of Smart CUDA is Mark Amo-Boateng (@markamo). 

### Support or Contact
Having trouble with Smart CUDA? Check out the documentation at http://markamo.github.io/Smart-Cuda/ or https://github.com/markamo/Smart-Cuda or contact smartcuda@outlook.com and we’ll help you sort it out.
