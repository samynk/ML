int2 indexToRC(int index, int nrOfRows){
    return (int2)(index%nrOfRows, index/nrOfRows);
}

int3 indexToRCS(int index, int2 dim){
    int slice = index / dim.y;
    int column = (index % dim.y) / dim.x;
    return (int3)(index% dim.x, column, slice);
}

int4 indexToRCSH(int index, int3 dim){
    int hs = index / dim.z;
    int3 rcs = indexToRCS(index%dim.z, dim.xy);
    return (int4)(rcs, hs);
}

int rcToIndex(int row, int column, int nrOfRows){
    return row + column*nrOfRows;
}

int rcsToIndex(int3 rcs, int2 dim){
    return rcs.x + rcs.y*dim.x + rcs.z*dim.y;
}

int rcshToIndex(int4 rcsh, int3 dim){
    return rcsh.x + rcsh.y*dim.x + rcsh.z*dim.y +rcsh.w*dim.z;
}

__kernel void maxpool(
    const __global float* I,
    __global float* O,
    __global int* M,
    // the dimensions of the the input matrix.
    const int3 iDim,
    // the scale of the max pool filter.
    const int2 fDim,
    // the dimensions of the output matrix.
    const int3 oDim
)
{
    int index = get_global_id(0);
    int4 rcsh = indexToRCSH(index, oDim);
    // rcsh.z contains the current filter nr.
    int iRow = rcsh.x * fDim.x;
    int iCol = rcsh.y * fDim.y;
    int slice = rcsh.z;
    int hyperSlice = rcsh.w;
    
    float m = -100000.0f;
    
    int maskIndex = 0;
    int cell = 0;
    for(int r = 0; r < fDim.x; ++r )
    {
        for (int c =0; c < fDim.y; ++c )
        {        
            int iIndex = rcshToIndex( (int4)(iRow+r,iCol+c,slice,hyperSlice) , iDim);
            float value = I[iIndex];
            if(value > m )
            {
                m = value;
                cell = r + c*fDim.x;
            }
            
        }
    }
    
    O[index] = m ;
    M[index] = cell;
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
