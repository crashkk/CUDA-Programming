//two dimension(matrix)
#include <stdio.h>

__global__ void my_first_kernel()//GPU
{
    int tidx = threadIdx.x;//定义线程索引
    int tidy = threadIdx.y;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    printf("Hello World from GPU(thread index:(%d,%d),block index:(%d,%d)))!\n",tidy,tidx,bidy,bidx);
}

//GPU并行计算
//thread --> block --> grid(-->表示包含关系)
//SM(stream multi-processor)
//total threads:block_size*grid_size
int main(){
    printf("Hello World from CPU\n");
    dim3 block_size(3,3);
    //t00,t01,t02
    //t10,t11,t12
    //t20,t21,t22
    dim3 grid_size(2,2);
    //b00,b01
    //b10,b11
    
    my_first_kernel<<<grid_size,block_size>>>();//调用自定义核函数
    cudaDeviceSynchronize();//必须告诉CPU核函数在GPU上执行完毕（同步操作）

    return 0;
}