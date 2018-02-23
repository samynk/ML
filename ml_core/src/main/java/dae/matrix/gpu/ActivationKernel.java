/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.imatrix;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBufferRect;
import static org.jocl.CL.clSetKernelArg;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;

/**
 * Kernels to calculate the activation functions.
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class ActivationKernel extends OpenCLKernel {

    private cl_kernel sigmoid;
    private cl_kernel dsigmoid;

    /**
     * Creates a new convolution kernel.
     *
     */
    public ActivationKernel() {
        super("/kernels/activation.cl");
    }

    @Override
    public void init(cl_context context, cl_command_queue commandQueue) {
        super.init(context, commandQueue);
        sigmoid = this.createKernel("sigmoid");
        dsigmoid = this.createKernel("dsigmoid");
        super.releaseProgram();
    }

    public void sigmoid(imatrix O) {
        int[] oDim = new int[]{O.getDeviceColumns(), O.getDeviceRows()};
        cl_mem memOutput = super.uploadRWMatrix(O);

        clSetKernelArg(sigmoid, 0, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(sigmoid, 1, Sizeof.cl_int2, Pointer.to(oDim));

        long globalSize[] = new long[]{oDim[0], oDim[1]};
        long localSize[] = new long[]{32, 32};
        clEnqueueNDRangeKernel(
                commandQueue,
                sigmoid,
                2,
                null,
                globalSize,
                localSize,
                0,
                null,
                null);

        downloadRWMatrix(O);
    }

    public void dsigmoid(imatrix O) {
        int[] oDim = new int[]{O.getDeviceColumns(), O.getDeviceRows()};
        cl_mem memOutput = super.uploadRWMatrix(O);

        clSetKernelArg(dsigmoid, 0, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(dsigmoid, 1, Sizeof.cl_int2, Pointer.to(oDim));

        long globalSize[] = new long[]{oDim[0], oDim[1]};
        long localSize[] = new long[]{32, 32};

        clEnqueueNDRangeKernel(
                commandQueue,
                dsigmoid,
                2,
                null,
                globalSize,
                localSize,
                0,
                null,
                null);

        downloadRWMatrix(O);
    }
}
