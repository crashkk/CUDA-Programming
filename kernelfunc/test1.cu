//one dimension of block
#include <stdio.h>

__global__ void my_first_kernel()//GPU
{
    int tid = threadIdx.x;//定义线程索引
    int bid = blockIdx.x;
    printf("Hello World from GPU(thread index:%d,block index:%d))!\n",tid,bid);
}

//GPU并行计算
//thread --> block --> grid(-->表示包含关系)
//SM(stream multi-processor)
//total threads:block_size*grid_size
int main(){
    printf("Hello World from CPU\n");
    int block_size = 3;
    int grid_size = 2;
    
    my_first_kernel<<<grid_size,block_size>>>();//调用自定义核函数
    cudaDeviceSynchronize();//必须告诉CPU核函数在GPU上执行完毕（同步操作）

    return 0;
}