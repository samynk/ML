__kernel void fuzzyFunction(
    // x : nrOfInputs
    // y : nrOfClasses -1
    // z : hyperSliceSize of the output matrix
    const int3 weightDim,
    const __global float* input,
    const __global float* a,
    const __global float* b,
    __global float* output
)
{
    int ohIndex = get_global_id(0);
    
    int nrOfInputs = weightDim.x;
    int classes = weightDim.y;
    int hSliceSize = weightDim.z;

    int oIndex = ohIndex % hSliceSize;
    int h = ohIndex / hSliceSize;
    
    int iIndex = oIndex / classes; 
    int oSubIndex = oIndex % classes;
    
   
    float av = a[oIndex];
    float bv = b[oIndex];
    float iv = input[iIndex + h*nrOfInputs];
    
    // float v =  (1 + exp(-av * (iv + bv)));
    float v = av*(iv+bv);
    output[ohIndex] = v;
}

__kernel void fuzzyShiftMinus(
    // x : nrOfClasses
    // y : hyperSliceSize of the input matrix.
    // z : hyperSliceSize of the output matrix.
    const int4 weightDim,
    const __global float* input,   
    __global float* output
)
{
    int ohIndex = get_global_id(0);
    
    int classes = weightDim.x;
    int ihSliceSize = weightDim.y;
    int ohSliceSize = weightDim.z;

    int oIndex = ohIndex % ohSliceSize;
    int h = ohIndex / ohSliceSize;
    int oClass = oIndex % classes;
    int oVar = oIndex / classes;

    
    int iBaseIndex = h * ihSliceSize + oVar * (classes-1) ; 
    float op1 = oClass == 0 ? 1 : input[iBaseIndex + oClass -1];
    float op2 = oClass == (classes-1) ? 0: input[iBaseIndex + oClass];

    output[ohIndex]  = op1 - op2;
}

__kernel void fuzzyShiftDeltas(
    // x : nrOfClasses
    // y : hyper slice size of input.
    // z : hyper slice size of output.
    const int4 weightDim,
    const __global float* input,   
    __global float* output
)
{
    int gIndex = get_global_id(0);
    
    int classes = weightDim.x;
    int iHSliceSize = weightDim.y;
    int oHSliceSize = weightDim.z;
 
    int hIndex = gIndex / oHSliceSize;
    int oIndex = gIndex % oHSliceSize;
    int varIndex = oIndex / (classes-1);
    int classIndex = oIndex % (classes-1);
    
    int iBase = hIndex * iHSliceSize + varIndex * classes + classIndex;

    
    output[gIndex] = input[iBase+1] - input[iBase];
}

__kernel void fuzzyBackProp(
    // x : nrOfClasses
    // y : hyper slice size of input.
    // z : hyper slice size of output.
    const int4 weightDim,
    const __global float* input,
    const __global float* weights,
    __global float* output
)
{
    int gIndex = get_global_id(0);
    
    int classes = weightDim.x;
    int iHSliceSize = weightDim.y;
    int oHSliceSize = weightDim.z;
 
    int hIndex = gIndex / oHSliceSize;
    int oIndex = gIndex % oHSliceSize; // varIndex
    
    int iSliceOffset = hIndex * iHSliceSize;
    int wIndex = oIndex *(classes-1);
    int iIndex = iSliceOffset + wIndex;
    
    float sum =0.0f;
    for(int cl =0 ; cl < (classes-1); cl++)
    {
        sum += input[iIndex+cl] * weights[iIndex];
    }
    output[gIndex] = sum;
}

__kernel void fuzzyInputAdd(
    // x : nrOfClasses
    // y : hyper slice size of input.
    // z : hyper slice size of output.
    const int4 weightDim,
    const __global float* input,
    const __global float* weights,
    __global float* output
)
{
    int gIndex = get_global_id(0);
    
    int classes = weightDim.x;
    int iHSliceSize = weightDim.y;
    int oHSliceSize = weightDim.z;
 
    int hIndex = gIndex / oHSliceSize;
    int owIndex = gIndex % oHSliceSize; 
    int varIndex = owIndex / (classes-1);
    
    int iSliceOffset = hIndex * iHSliceSize;
    
    int iIndex = iSliceOffset + varIndex;
    output[gIndex] = input[iIndex] + weights[owIndex];
}