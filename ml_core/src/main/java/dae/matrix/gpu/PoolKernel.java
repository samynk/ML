/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.imatrix;
import dae.matrix.integer.intmatrix;
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
public class PoolKernel extends OpenCLKernel {

    private cl_kernel maxPool;
    private cl_kernel bpMaxPool;

    public PoolKernel() {
        super("/kernels/pool.cl");
    }

    @Override
    public void init(cl_context context, cl_command_queue commandQueue) {
        super.init(context, commandQueue);
        maxPool = this.createKernel("maxpool");
        bpMaxPool = this.createKernel("backpropMaxpool");
        super.releaseProgram();
    }

    public void maxPool(imatrix input, imatrix output, intmatrix maskLayer) {
        DeviceBuffer inputDB = input.getDeviceBuffer();
        DeviceBuffer outputDB = output.getDeviceBuffer();
        
        int[] iDim = inputDB.getDeviceDimension();
        int[] oDim = outputDB.getDeviceDimension();
        
        int scaleX = input.getNrOfColumns() / output.getNrOfColumns();
        int scaleY = input.getNrOfRows() / output.getNrOfRows();
        int[] fDim = new int[]{scaleX, scaleY};
        cl_mem memInput = GPU.uploadRMatrix(input);
        cl_mem memMask = maskLayer.getCLReadWriteMem();
        cl_mem memOutput = outputDB.getCLReadWriteMem();

        clSetKernelArg(maxPool, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(maxPool, 1, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(maxPool, 2, Sizeof.cl_mem, Pointer.to(memMask));
        clSetKernelArg(maxPool, 3, Sizeof.cl_int2, Pointer.to(iDim));
        clSetKernelArg(maxPool, 4, Sizeof.cl_int2, Pointer.to(fDim));
        clSetKernelArg(maxPool, 5, Sizeof.cl_int2, Pointer.to(oDim));

        int slices = Math.min(input.getNrOfSlices(), output.getNrOfSlices());
        long globalSize[] = new long[]{oDim[0], oDim[1], slices};
        long localSize[] = new long[]{32, 32, 1};
        clEnqueueNDRangeKernel(
                commandQueue,
                maxPool,
                3,
                null,
                globalSize,
                localSize,
                0,
                null,
                null);

        GPU.downloadRWMatrix(output);
        GPU.downloadRWMatrix(maskLayer);
    }
    
    public void backpropMaxPool(imatrix input, intmatrix maskLayer, imatrix output)
    {
        DeviceBuffer inputDB = input.getDeviceBuffer();
        DeviceBuffer outputDB = output.getDeviceBuffer();
        
        int[] iDim = inputDB.getDeviceDimension();
        int[] oDim = outputDB.getDeviceDimension();
        int scaleX = output.getNrOfColumns() / input.getNrOfColumns();
        int scaleY = output.getNrOfRows() / input.getNrOfRows();
        int[] fDim = new int[]{scaleX, scaleY};
        cl_mem memInput = GPU.uploadRMatrix(input);
        cl_mem memMask = maskLayer.getCLReadWriteMem();
        cl_mem memOutput = outputDB.getCLReadWriteMem();

        clSetKernelArg(bpMaxPool, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(bpMaxPool, 1, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(bpMaxPool, 2, Sizeof.cl_mem, Pointer.to(memMask));
        clSetKernelArg(bpMaxPool, 3, Sizeof.cl_int2, Pointer.to(iDim));
        clSetKernelArg(bpMaxPool, 4, Sizeof.cl_int2, Pointer.to(fDim));
        clSetKernelArg(bpMaxPool, 5, Sizeof.cl_int2, Pointer.to(oDim));

        int slices = Math.min(input.getNrOfSlices(), output.getNrOfSlices());
        long globalSize[] = new long[]{oDim[0], oDim[1], slices};
        long localSize[] = new long[]{32, 32, 1};
        clEnqueueNDRangeKernel(
                commandQueue,
                bpMaxPool,
                3,
                null,
                globalSize,
                localSize,
                0,
                null,
                null);

        GPU.downloadRWMatrix(output);
    }
}
