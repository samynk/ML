__kernel void softmax(
    __global float4* O
)
{
    int index = get_global_id(0);
    float4 o = O[index]; 
    O[index] = exp(o);
}


__kernel void sigmoid(
    __global float4* O
)
{
    int index = get_global_id(0);
    float4 o = O[index]; 
    O[index] = 1.0f/(1.0f+exp(-o));
}

__kernel void dsigmoid(
    __global float4* O
)
{
    int index = get_global_id(0);
    float4 o = O[index]; 
    O[index] = o*(1-o);
}

__kernel void relu(
    __global float4* O
)
{
    int index = get_global_id(0);
    float4 v = O[index]; 
    O[index] = ( v > 0.0f )? v: (float4)(0);
}

__kernel void drelu(
    __global float4* O
)
{
    int index = get_global_id(0);
    float4 v = O[index]; 
    O[index] = ( v > 0.0f )? (float4)(1): (float4)(0);

}

__kernel void leakyrelu(
    __global float4* O,
    float alpha
)
{
    int index = get_global_id(0);
    float4 v = O[index]; 
    O[index] = ( v > 0.0f )? v: alpha*v;
}

__kernel void dleakyrelu(
    __global float4* O,
    float alpha
)
{
    int index = get_global_id(0);
    float4 v = O[index]; 
    O[index] = ( v  >  0.0f) ? (float4)(1): (float4)(alpha)  ;
}

__kernel void a_tanh(
    __global float4* O
)
{
    int index = get_global_id(0);
    float4 v = O[index]; 
    O[index] = tanh(v);
}

__kernel void dtanh(
    __global float4* O
)
{
    int index = get_global_id(0);
    float4 v = O[index]; 
    O[index] = (float4)(1) - v*v;
}