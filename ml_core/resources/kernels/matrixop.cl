__kernel void dotsubtract(
    const int2 dim,
    const __global float* op1,
    const __global float* op2,
    __global float* result
)
{
    int gCol = get_global_id(0);
    int gRow = get_global_id(1);
    int slice = get_global_id(2);

    int index = gRow + gCol*dim.y + slice * dim.x * dim.y;
    result[index] = op1[index] - op2[index];
}

__kernel void dotadd(
    const int2 dim,
    const __global float* op1,
    const __global float* op2,
    __global float* result
)
{
    int gCol = get_global_id(0);
    int gRow = get_global_id(1);
    int slice = get_global_id(2);

    int index = gRow + gCol*dim.y + slice * dim.x * dim.y;
    result[index] = op1[index] + op2[index];
}

__kernel void dotaddlc(
    const int2 dim,
    const float2 factors,
    const __global float* op1,
    const __global float* op2,
    __global float* result
)
{
    int gCol = get_global_id(0);
    int gRow = get_global_id(1);
    int slice = get_global_id(2);

    int index = gRow + gCol*dim.y + slice * dim.x * dim.y;
    float2 values= (float2)(op1[index], op2[index]);
    result[index] =  dot(factors,values);
}

__kernel void dotmultiply(
    const int2 dim,
    const __global float* op1,
    const __global float* op2,
    __global float* result
)
{
    int gCol = get_global_id(0);
    int gRow = get_global_id(1);
    int slice = get_global_id(2);

    int index = gRow + gCol*dim.y + slice * dim.x * dim.y;
    result[index] = op1[index] * op2[index];
}

__kernel void randomize(
    const int2 dim,
    __global float* m
)
{
    int gCol = get_global_id(0);
    int gRow = get_global_id(1);
    int slice = get_global_id(2);

    int index = gRow + gCol*dim.y + slice * dim.x * dim.y;
    m[index] = 1; 
}
