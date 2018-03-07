/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.imatrix;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clSetKernelArg;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class MatrixOpKernel extends OpenCLKernel {

    cl_kernel dotadd;
    cl_kernel dotaddlc;
    cl_kernel dotsubtract;
    cl_kernel dotmultiply;

    /**
     * Creates a new MatrixOpKernel object.
     */
    public MatrixOpKernel() {
        super("/kernels/matrixop.cl");
    }

    @Override
    public void init(cl_context context, cl_command_queue commandQueue) {
        super.init(context, commandQueue);
        dotadd = this.createKernel("dotadd");
        dotaddlc = this.createKernel("dotaddlc");
        dotsubtract = this.createKernel("dotsubtract");
        dotmultiply = this.createKernel("dotmultiply");
        super.releaseProgram();
    }

    public imatrix dotadd(imatrix O, imatrix op1, imatrix op2) {
        DeviceBuffer oDB = O.getDeviceBuffer();
        int[] oDim = oDB.getDeviceDimension();
        cl_mem memOutput = oDB.getCLReadWriteMem();

        cl_mem mem_op1 = GPU.uploadRMatrix(op1);
        cl_mem mem_op2 = GPU.uploadRMatrix(op2);

        clSetKernelArg(dotadd, 0, Sizeof.cl_int2, Pointer.to(oDim));

        clSetKernelArg(dotadd, 1, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(dotadd, 2, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(dotadd, 3, Sizeof.cl_mem, Pointer.to(memOutput));

        long globalSize[] = new long[]{oDim[0], oDim[1], O.getNrOfSlices()};
        long localSize[] = new long[]{32, 32, 1};
        clEnqueueNDRangeKernel(
                commandQueue,
                dotadd,
                3,
                null,
                globalSize,
                localSize,
                0,
                null,
                null);

        GPU.downloadRWMatrix(O);
        return O;
    }

    public imatrix dotadd(imatrix O, float factor1, imatrix op1, float factor2, imatrix op2) {
        DeviceBuffer oDB = O.getDeviceBuffer();
        int[] oDim = oDB.getDeviceDimension();
        float[] factors = new float[]{factor1, factor2};
        cl_mem memOutput = oDB.getCLReadWriteMem();

        cl_mem mem_op1 = GPU.uploadRMatrix(op1);
        cl_mem mem_op2 = GPU.uploadRMatrix(op2);

        clSetKernelArg(dotaddlc, 0, Sizeof.cl_int2, Pointer.to(oDim));
        clSetKernelArg(dotaddlc, 1, Sizeof.cl_float2, Pointer.to(factors));
        clSetKernelArg(dotaddlc, 2, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(dotaddlc, 3, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(dotaddlc, 4, Sizeof.cl_mem, Pointer.to(memOutput));

        long globalSize[] = new long[]{oDim[0], oDim[1], O.getNrOfSlices()};
        long localSize[] = new long[]{32, 32, 1};
        clEnqueueNDRangeKernel(
                commandQueue,
                dotaddlc,
                3,
                null,
                globalSize,
                localSize,
                0,
                null,
                null);

        GPU.downloadRWMatrix(O);
        return O;
    }

    public imatrix dotsubtract(imatrix O, imatrix op1, imatrix op2) {
        DeviceBuffer oDB = O.getDeviceBuffer();
        int[] oDim = oDB.getDeviceDimension();
        cl_mem memOutput = oDB.getCLReadWriteMem();

        cl_mem mem_op1 = GPU.uploadRMatrix(op1);
        cl_mem mem_op2 = GPU.uploadRMatrix(op2);

        clSetKernelArg(dotsubtract, 0, Sizeof.cl_int2, Pointer.to(oDim));

        clSetKernelArg(dotsubtract, 1, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(dotsubtract, 2, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(dotsubtract, 3, Sizeof.cl_mem, Pointer.to(memOutput));

        long globalSize[] = new long[]{oDim[0], oDim[1], O.getNrOfSlices()};
        long localSize[] = new long[]{32, 32, 1};
        clEnqueueNDRangeKernel(
                commandQueue,
                dotsubtract,
                3,
                null,
                globalSize,
                localSize,
                0,
                null,
                null);

        GPU.downloadRWMatrix(O);
        return O;
    }

    public imatrix dotmultiply(imatrix O, imatrix op1, imatrix op2) {
        DeviceBuffer oDB = O.getDeviceBuffer();
        int[] oDim = oDB.getDeviceDimension();
        cl_mem memOutput = oDB.getCLReadWriteMem();

        cl_mem mem_op1 = GPU.uploadRMatrix(op1);
        cl_mem mem_op2 = GPU.uploadRMatrix(op2);

        clSetKernelArg(dotmultiply, 0, Sizeof.cl_int2, Pointer.to(oDim));
        clSetKernelArg(dotmultiply, 1, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(dotmultiply, 2, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(dotmultiply, 3, Sizeof.cl_mem, Pointer.to(memOutput));

        long globalSize[] = new long[]{oDim[0], oDim[1], O.getNrOfSlices()};
        long localSize[] = new long[]{32, 32, 1};
        clEnqueueNDRangeKernel(
                commandQueue,
                dotmultiply,
                3,
                null,
                globalSize,
                localSize,
                0,
                null,
                null);

        GPU.downloadRWMatrix(O);
        return O;
    }

}
