
#include <cstddef>
#include <cstdio>

// class to manage device memory, to take care of allocation and deallocation.
template<typename T>
class CudaDeviceMemory
{
public:
  CudaDeviceMemory()
  {

  };

  CudaDeviceMemory(size_t size)
  {
    allocate(size);
  }
  
  ~CudaDeviceMemory()
  {
    if (device_ptr != nullptr)
      {
	cudaError_t ret = cudaFree(device_ptr);
	if (ret != cudaSuccess)
	  printf("CudaDeviceMemory: device free error\n");
	   
      }
  };

  bool allocate(size_t size)
  {
    cudaError_t ret = cudaMalloc((void**)&device_ptr, size*sizeof(T));
    return ret == cudaSuccess;
  };

  T* ptr()
  {
    return device_ptr;
  };

  T* operator*()
  {
    return device_ptr;
  };

  // zero out device memory
  bool zero()
  {
   cudaError_t ret =  cudaMemset(device_ptr, 0, sizeof(T) * size);
   return ret == cudaSuccess;
  };

  // copy data from device to local array
  bool fetch(std::unique_ptr<T[]> host_ptr)
  {
    cudaError_t ret = cudaMemcpy( *host_ptr, device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
    return ret == cudaSuccess;
  };

  // copy data from local array to device
  bool put(std::unique_ptr<T[]> * host_ptr)
  {
    cudaError_t ret = cudaMemcpy(device_ptr, *host_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
    return ret == cudaSuccess;
  };

private:
  T * device_ptr = nullptr;
  size_t size;
};
