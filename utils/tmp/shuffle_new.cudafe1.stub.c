#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "shuffle_new.fatbin.c"
extern void __device_stub__Z14decode_shufflePhP6__halfS1_(uint8_t *, half *, half *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z14decode_shufflePhP6__halfS1_(uint8_t *__par0, half *__par1, half *__par2){__cudaLaunchPrologue(3);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaLaunch(((char *)((void ( *)(uint8_t *, half *, half *))decode_shuffle)));}
# 36 "../shuffle_new.cu"
void decode_shuffle( uint8_t *__cuda_0,half *__cuda_1,half *__cuda_2)
# 36 "../shuffle_new.cu"
{__device_stub__Z14decode_shufflePhP6__halfS1_( __cuda_0,__cuda_1,__cuda_2);
# 94 "../shuffle_new.cu"
}
# 1 "shuffle_new.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T5) {  __nv_dummy_param_ref(__T5); __nv_save_fatbinhandle_for_managed_rt(__T5); __cudaRegisterEntry(__T5, ((void ( *)(uint8_t *, half *, half *))decode_shuffle), _Z14decode_shufflePhP6__halfS1_, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
