#include <stdio.h>
#include <math.h>
#include<windows.h>

#define BLOCK_SIZE 16 
//矩阵相乘a[][]*b[][]=c[][]
//`
void cpu_matrix_mult(int *a, int *b,int *c,const int size){//cpu实现函数
    for(int y=0;y<size;++y){
        for(int x=0;x<size;++x){
            int tmp = 0;
            for(int step =0;step<size;++step){
                tmp+=a[step+y*size]*b[x+step*size];
            }
            c[y*size+x]=tmp;
        }
    }
}
__global__ void gpu_matrix_mult(int *a, int *b,int *c,const int size){//gpu实现函数
    int y=blockDim.y*blockIdx.y+threadIdx.y;
    int x=blockDim.x*blockIdx.x+threadIdx.x;
    int tmp=0;
    if(x<size&&y<size){
        for(int step =0;step<size;++step){
            tmp+=a[step+y*size]*b[x+step*size];
        }
        c[y*size+x]=tmp;
    }
}
int main(){
    LARGE_INTEGER frequency,start,end;
    double period_gpu,period_cpu;

    int matrix_size = 1000;
    int memsize = sizeof(int)*matrix_size*matrix_size;

    int *h_a,*h_b,*h_c,*h_cc;
    cudaMallocHost((void**)&h_a,memsize);
    cudaMallocHost((void**)&h_b,memsize);
    cudaMallocHost((void**)&h_c,memsize);
    cudaMallocHost((void**)&h_cc,memsize);

    for(int y=0;y<matrix_size;++y){
        for(int x=0;x<matrix_size;++x){
            h_a[y*matrix_size+x] = rand() % 1024;
        }
    }

    int *d_a,*d_b,*d_c;
    cudaMalloc((void**) &d_a,memsize);
    cudaMalloc((void**) &d_b,memsize);
    cudaMalloc((void**) &d_c,memsize);

    cudaMemcpy(d_a,h_a,memsize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,memsize,cudaMemcpyHostToDevice);
    
    unsigned int grid_rows = (matrix_size +BLOCK_SIZE-1)/BLOCK_SIZE;
    unsigned int grid_cols = (matrix_size +BLOCK_SIZE-1)/BLOCK_SIZE;

    dim3 dimGrid(grid_cols,grid_rows);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);//warp(Gpu线程，BLOCK_SIZE^2最好设置为32的整数倍，不能大于1024)

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    gpu_matrix_mult<<<dimGrid,dimBlock>>>(d_a,d_b,d_c,matrix_size);
    QueryPerformanceCounter(&end);
    period_gpu = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;

    cudaMemcpy(h_c,d_c,memsize,cudaMemcpyDeviceToHost);

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&start);
    cpu_matrix_mult(h_a,h_b,h_cc,matrix_size);//对照组，cpu实现矩阵相乘
    QueryPerformanceCounter(&end);
    period_cpu = (end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;

    bool errors = false;
    for(int y=0;y<matrix_size;++y){
        for(int x=0;x<matrix_size;++x){
            if(fabs(h_cc[y*matrix_size+x]-h_c[y*matrix_size+x])>(1.0e-10)){
                errors = true;
            }
        }
    }
    printf("Result:%s\n",errors?"Errors":"Passed");
    printf("running time:GPU is %fms,while CPU is %fms",period_gpu,period_cpu);

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}