__kernel void maxpool(
    __global float* I,
    __global float* O,
    __global int* M,
    // the dimensions of the the input matrix.
    const int2 iDim,
    // the scale of the max pool filter.
    const int2 fDim,
    // the dimensions of the output matrix.
    const int2 oDim
)
{
    int gCol = get_global_id(0);
    int gRow = get_global_id(1);
    int slice = get_global_id(2);
    
    int oIndex = gRow + gCol*oDim.y + oDim.x * oDim.y * slice;
    int iBase = iDim.x * iDim.y * slice;
    

    float m = -100000.0f;
    int iCB = gCol * fDim.x;
    int iRB = gRow * fDim.y;

    int maskIndex = 0;
    int cell = 0;
    for(int x = 0; x < fDim.x;++x)
    {
        for (int y =0; y < fDim.y;++y){        
            int iIndex = iBase + (iRB+y) + (iCB+ x) * iDim.y;
            float value = I[iIndex];
            if(value > m )
            {
                m = value;
                cell = y + x*fDim.y;
            }
            
        }
    }
    
    O[oIndex] = m ;
    M[oIndex]= cell;
}

__kernel void backpropMaxpool(
    __global float* I,
    __global float* O,
    __global int* M,
    // the dimensions of the the input matrix.
    const int2 iDim,
    // the scale of the max pool filter.
    const int2 fDim,
    // the dimensions of the output matrix.
    const int2 oDim
)
{
    int gCol = get_global_id(0);
    int gRow = get_global_id(1);
    int slice = get_global_id(2);
    
    int iCB = gCol / fDim.x;
    int iRB = gRow / fDim.y;

    int iIndex = iRB + iCB * iDim.y + slice * iDim.x * iDim.y;

    int cell = M[iIndex];

    int2 oCoord = (int2)(cell/fDim.y, cell%fDim.y);
    int2 iCoord = ((int2)(gCol%fDim.x,gRow%fDim.y) );

    int oIndex = gRow + gCol * oDim.y + slice * oDim.x * oDim.y;

    if(oCoord.x == iCoord.x && oCoord.y == iCoord.y){
        O[oIndex] = I[iIndex];
    }else{
        O[oIndex] = 0;
    }
}
