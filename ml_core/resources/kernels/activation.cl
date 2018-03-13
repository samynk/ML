__kernel void sigmoid(
    __global float* O
)
{
    int index = get_global_id(0);
    float o = O[index]; 
    O[index] = 1.0f/(1.0f+exp(-o));
}

__kernel void dsigmoid(
    __global float* O
)
{
    int index = get_global_id(0);
    float o = O[index]; 
    O[index] = o*(1-o);
}
