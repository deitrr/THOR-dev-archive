
#include <cstddef>

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
	cudaFree(device_ptr);
      }
  };

  bool allocate(size_t size)
  {
    cudaMalloc((void**)&device_ptr, size*sizeof(T));
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
  void zero()
  {

  };

  // copy data from device to local array
  void fetch(std::unique_ptr<T[]> host_ptr)
  {
    
  };

  // copy data from local array to device
  void put(std::unique_ptr<T[]> * host_ptr)
  {
    
  };

private:
  T * device_ptr = nullptr;
  size_t size;
};
