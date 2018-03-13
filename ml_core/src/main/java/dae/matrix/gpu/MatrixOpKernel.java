/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.BufferSyncState;
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

    private long[] localWorkSize = new long[]{32};

    /**
     * Creates a new MatrixOpKernel object.
     */
    public MatrixOpKernel() {
        super("/kernels/neuralnet/matrixop.cl");
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
        FloatDeviceBuffer oDB = O.getDeviceBuffer();
        cl_mem memOutput = oDB.getRWMem();
        cl_mem mem_op1 = op1.getDeviceBuffer().uploadRMatrix();
        cl_mem mem_op2 = op2.getDeviceBuffer().uploadRMatrix();

        clSetKernelArg(dotadd, 0, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(dotadd, 1, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(dotadd, 2, Sizeof.cl_mem, Pointer.to(memOutput));

        clEnqueueNDRangeKernel(
                commandQueue,
                dotadd,
                1,
                null,
                oDB.getGlobalWorkSize(),
                localWorkSize,
                0,
                null,
                null);

        oDB.markRWMatrixAsMaster();
        return O;
    }

    public imatrix dotadd(imatrix O, float factor1, imatrix op1, float factor2, imatrix op2) {
        FloatDeviceBuffer oDB = O.getDeviceBuffer();
        float[] factors = new float[]{factor1, factor2};
        cl_mem memOutput = oDB.getRWMem();

        cl_mem mem_op1 = op1.getDeviceBuffer().uploadRMatrix();
        cl_mem mem_op2 = op2.getDeviceBuffer().uploadRMatrix();

        clSetKernelArg(dotaddlc, 0, Sizeof.cl_float2, Pointer.to(factors));
        clSetKernelArg(dotaddlc, 1, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(dotaddlc, 2, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(dotaddlc, 3, Sizeof.cl_mem, Pointer.to(memOutput));

        clEnqueueNDRangeKernel(
                commandQueue,
                dotaddlc,
                1,
                null,
                oDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        oDB.markRWMatrixAsMaster();
        return O;
    }

    public imatrix dotsubtract(imatrix O, imatrix op1, imatrix op2) {
        FloatDeviceBuffer oDB = O.getDeviceBuffer();
        int[] oDim = oDB.getDeviceDimension();
        cl_mem memOutput = oDB.getRWMem();

        cl_mem mem_op1 = op1.getDeviceBuffer().uploadRMatrix();
        cl_mem mem_op2 = op2.getDeviceBuffer().uploadRMatrix();

        clSetKernelArg(dotsubtract, 0, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(dotsubtract, 1, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(dotsubtract, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clEnqueueNDRangeKernel(
                commandQueue,
                dotsubtract,
                1,
                null,
                oDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        oDB.markRWMatrixAsMaster();
        return O;
    }

    public imatrix dotmultiply(imatrix O, imatrix op1, imatrix op2) {
        FloatDeviceBuffer oDB = O.getDeviceBuffer();
        int[] oDim = oDB.getDeviceDimension();
        cl_mem memOutput = oDB.getRWMem();

        cl_mem mem_op1 = op1.getDeviceBuffer().uploadRMatrix();
        cl_mem mem_op2 = op2.getDeviceBuffer().uploadRMatrix();

        clSetKernelArg(dotmultiply, 0, Sizeof.cl_mem, Pointer.to(mem_op1));
        clSetKernelArg(dotmultiply, 1, Sizeof.cl_mem, Pointer.to(mem_op2));
        clSetKernelArg(dotmultiply, 2, Sizeof.cl_mem, Pointer.to(memOutput));

        clEnqueueNDRangeKernel(
                commandQueue,
                dotmultiply,
                1,
                null,
                oDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        oDB.markRWMatrixAsMaster();
        return O;
    }
}
