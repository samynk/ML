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

    public static cl_mem createMem(intmatrix cpuBuffer, int padding, long mode) {
        int zp = cpuBuffer.getZeroPadding();
        int totalSize = (cpuBuffer.getNrOfRows() + 2 * zp)
                * (cpuBuffer.getNrOfColumns() + 2 * zp)
                * cpuBuffer.getNrOfSlices()
                * cpuBuffer.getNrOfHyperSlices() + padding;
        cl_mem mem = clCreateBuffer(GPU.CL_CONTEXT, mode,
                (totalSize + padding) * Sizeof.cl_int, null, null);
        return mem;
    }
}
