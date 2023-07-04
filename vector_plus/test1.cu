//index data
//z[i]=x[i]+y[i] 
//一般向量相加的实现方法：for loop
//CUDA的实现方法：使用并行计算，每个thread处理向量的一个位置的元素，步骤如下：
//memory allocation
//memory copy   gpu的mem和cpu的mem不相同
//kernel func
//memory copy

#include<stdio.h>
#include<math.h>
#include<windows.h>

//x[] +y[] =z[]
__global__ void vecAdd(const double *x,const double *y,double *z, int count)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    //t00 t01 t02 | t10 t11 t12 | t20 t21 t22
    if(index < count){
        z[index]=x[index]+y[index];
    }
}

void vecAdd_cpu(const double *x,const double *y,double *z,int count)//和cpu上的计算进行比对
{
    for(int i=0;i<count;++i){
        z[i]=x[i]+y[i];
    }
}

int main(){
    LARGE_INTEGER frequency,start,end;
    double period_gpu,period_cpu;

    const int N = 100000000;
    const int M = sizeof(double) * N;

//memory allocation
    //cpu mem alloc
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);
    double *result_cpu = (double*) malloc(M);

    for( int i = 0; i<N; ++i){
        h_x[i] = 1;
        h_y[i] = 2;
    }

    //gpu mem alloc
    double *d_x,*d_y,*d_z;
    cudaMalloc((void**) &d_x, M );//gpu的内存分配语法
    cudaMalloc((void**) &d_y, M );
    cudaMalloc((void**) &d_z, M );
//

//memory copy
    cudaMemcpy( d_x, h_x, M, cudaMemcpyHostToDevice);//从cpu传输到gpu上
    cudaMemcpy( d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = (N + block_size -1)/block_size; 
//

//kernel func
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    vecAdd<<<grid_size,block_size>>>(d_x,d_y,d_z,N);//在gpu上进行计算
    QueryPerformanceCounter(&end);
    period_gpu = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
//

//memory copy
    cudaMemcpy(h_z ,d_z ,M , cudaMemcpyDeviceToHost);//将结果z从gpu传回cpu
//
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    vecAdd_cpu(h_x,h_y,result_cpu,N);
    QueryPerformanceCounter(&end);
    period_cpu = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    
    bool error =false;

    for(int i=0;i<N; ++i){//精度比较
        if(fabs(result_cpu[i]-h_z[i])>(1.0e-10)){
            error=true;
        }
    }
    printf("Result:%s\n",error?"Errors":"Pass");
    printf("running time:GPU is %fms,while CPU is %fms",period_gpu,period_cpu);

    free(h_x);
    free(h_y);
    free(h_z);
    free(result_cpu);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

}