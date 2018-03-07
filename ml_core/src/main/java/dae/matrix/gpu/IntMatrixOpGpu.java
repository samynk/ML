/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.imatrix;
import dae.matrix.integer.intmatrix;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import static org.jocl.CL.clCreateBuffer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class IntMatrixOpGpu {
    public static cl_mem createReadMem(intmatrix matrix, int padcol, int padrow) {
        cl_mem mem = clCreateBuffer(GPU.CL_CONTEXT, CL_MEM_READ_ONLY,
                (matrix.getNrOfRows() + padrow) * (matrix.getNrOfColumns() + padcol) * matrix.getNrOfSlices()
                * Sizeof.cl_int, null, null);
        return mem;
    }

    public static cl_mem createReadWriteMem(intmatrix matrix, int padcol, int padrow) {
        cl_mem mem = clCreateBuffer(GPU.CL_CONTEXT, CL_MEM_READ_WRITE,
                (matrix.getNrOfRows() + padrow) * (matrix.getNrOfColumns() + padcol) * matrix.getNrOfSlices()
                * Sizeof.cl_int, null, null);
        return mem;
    } 
}
