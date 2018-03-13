
__kernel void dotadd(
    const __global float* op1,
    const __global float* op2,
    __global float* result
)
{
    int index = get_global_id(0);
    result[index] = op1[index] + op2[index];
}

__kernel void dotaddlc(
    const float2 factors,
    const __global float* op1,
    const __global float* op2,
    __global float* result
)
{
    int index = get_global_id(0);
    float2 values= (float2)(op1[index], op2[index]);
    result[index] =  dot(factors,values);
}

__kernel void dotsubtract(
    const __global float* op1,
    const __global float* op2,
    __global float* result
)
{
    int index = get_global_id(0);
    result[index] = op1[index] - op2[index];
}

__kernel void dotmultiply(
    const __global float* op1,
    const __global float* op2,
    __global float* result
)
{
    int index = get_global_id(0);    
    result[index] = op1[index] * op2[index];
}

__kernel void randomize(
    __global float* m
)
{
    int index = get_global_id(0);
    m[index] = 1; 
}
