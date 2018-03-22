
__kernel void dotadd(
    const __global float* op1,
    const __global float* op2,
    __global float* result
)
{
    int index = get_global_id(0);
    result[index] = op1[index] + op2[index];
}


__kernel void sumPerRow(
    int2 dim,
    __global float* input,
    __global float* output
)
{
    int nrOfInputSlices = dim.x;
    int iHSliceSize = dim.y;
    int row = get_global_id(0);
    float sum =0;
    for(int h =0; h <nrOfInputSlices; ++h)
    {
        int index = row + h * iHSliceSize;
        sum += input[index];
    }
    output[row] = sum;
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

__kernel void adamVelocity(
    const float beta2,
    const __global float* velocity,
    const __global float* gradient,
    __global float* result
)
{
    int index = get_global_id(0);
    result[index] = beta2 * velocity[index] + (1-beta2) * gradient[index] * gradient[index];
}

__kernel void adamAdaptWeights(
    const float4 parameters,
    const __global float* moment,
    const __global float* velocity,
    __global float* weights
)
{
    int index = get_global_id(0);
    float eta = parameters.x;
    float invOneMinusBeta1 = parameters.y;
    float invOneMinusBeta2 = parameters.z;
    float epsilon = parameters.w;

    float m = moment[index];
    float v = velocity[index];
    float w = weights[index];

    weights[index] = w - ((eta * m * invOneMinusBeta1) / ( sqrt( v *invOneMinusBeta2 ) + epsilon)) ; 
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

__kernel void dotmultiplyfactor(
    const __global float* op1,
    const float factor,
    __global float* result
)
{
    int index = get_global_id(0);    
    result[index] = op1[index] * factor;
}

__kernel void randomize(
    __global float* m
)
{
    int index = get_global_id(0);
    m[index] = 1; 
}

__kernel void squared(    
    __global float4* m
)
{
    int index = get_global_id(0);
    float4 v = m[index];
    m[index] = v*v;
}

__kernel void root(    
    __global float4* m
)
{
    int index = get_global_id(0);
    float4 v = m[index];
    m[index] = sqrt(v);
}

uint wang_hash(uint seed)
{
        seed = (seed ^ 61) ^ (seed >> 16);
        seed *= 9;
        seed = seed ^ (seed >> 4);
        seed *= 0x27d4eb2d;
        seed = seed ^ (seed >> 15);
        return seed;
 }

 void wang_rnd_0(__global unsigned int * rnd_buffer,int id)                
 {
     uint maxint=0;
     maxint--;
     uint rndint=wang_hash(id);
     rnd_buffer[id]=rndint;
 }

 float wang_rnd(__global unsigned int * rnd_buffer,int id)                
 {
     uint maxint=0;
     maxint--; // not ok but works
     uint rndint=wang_hash(rnd_buffer[id]);
     rnd_buffer[id]=rndint;
     return ((float)rndint)/(float)maxint;
 }

 __kernel void rnd_init(__global unsigned int * rnd_buffer)
 {
       int id=get_global_id(0);
       wang_rnd_0(rnd_buffer,id);  // each (id) thread has its own random seed now           
 }

 __kernel void rnd_1(
    __global float * m, 
    const __global unsigned int * rnd_buffer )
 {
      int id=get_global_id(0);

      // can use this to populate a buffer with random numbers 
      // concurrently on all cores of a gpu
      m[id]=wang_rnd(rnd_buffer,id);
 }