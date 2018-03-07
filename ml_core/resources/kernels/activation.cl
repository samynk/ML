__kernel void sigmoid(
    __global float* O,
    const int2 oDim
)
{
    int gCol = get_global_id(0);
    int gRow = get_global_id(1);
    int slice = get_global_id(2);
    
    int index = gRow+gCol*oDim.y+slice*oDim.x*oDim.y;
    float o = O[index]; 
    O[index] = 1.0f/(1.0f+exp(-o));
}

__kernel void dsigmoid(
    __global float* O,
    const int2 oDim
)
{
    int gCol = get_global_id(0);
    int gRow = get_global_id(1);
    
    int index = gRow+gCol*oDim.y;
    float o = O[index]; 
    O[index] = o*(1-o);
}
