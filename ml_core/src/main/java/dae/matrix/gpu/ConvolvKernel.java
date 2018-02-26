/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.imatrix;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clEnqueueFillBuffer;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBufferRect;
import static org.jocl.CL.clEnqueueWriteBuffer;
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
public class ConvolvKernel extends OpenCLKernel {

    private cl_kernel convolution;
    private cl_kernel batchConvolution;
    private cl_kernel batchCorrelation;

    /**
     * Creates a new convolution kernel.
     *
     * @param kernelFile the location of the kernel file in the class path.
     */
    public ConvolvKernel(String kernelFile) {
        super(kernelFile);
    }

    /**
     * Initializes the kernel.
     *
     * @param context the opencl context.
     * @param commandQueue the command queue to use.
     */
    @Override
    public void init(cl_context context, cl_command_queue commandQueue) {
        super.init(context, commandQueue);
        convolution = this.createKernel("convolution");
        batchConvolution = this.createKernel("batchConvolution");
        batchCorrelation = this.createKernel("batchCorrelate");
        super.releaseProgram();
    }

    public void convolv(imatrix input, imatrix filter, imatrix output) {
        int[] iDim = new int[]{input.getNrOfColumns(), input.getNrOfRows()};
        int[] fDim = new int[]{filter.getNrOfColumns(), filter.getNrOfRows()};

        cl_mem memInput = input.getCLReadMem();
        cl_mem memFilter = filter.getCLReadMem();

        clEnqueueWriteBuffer(commandQueue, memInput, CL_TRUE, 0, input.getSliceSize()
                * Sizeof.cl_float, input.getCLPointer(), 0, null, null);
        clEnqueueWriteBuffer(commandQueue, memFilter, CL_TRUE, 0, filter.getSliceSize()
                * Sizeof.cl_float, filter.getCLPointer(), 0, null, null);

        int hostCols = output.getNrOfColumns() + output.getColPadding();
        int hostRows = output.getNrOfRows() + output.getRowPadding();

        int[] oDim = new int[]{hostCols, hostRows};
        cl_mem memOutput = output.getCLReadWriteMem();

        clSetKernelArg(convolution, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(convolution, 1, Sizeof.cl_mem, Pointer.to(memFilter));
        clSetKernelArg(convolution, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(convolution, 3, Sizeof.cl_int2, Pointer.to(iDim));
        clSetKernelArg(convolution, 4, Sizeof.cl_int2, Pointer.to(fDim));
        clSetKernelArg(convolution, 5, Sizeof.cl_int2, Pointer.to(oDim));

        long globalSize[] = new long[2];
        globalSize[0] = hostCols;
        globalSize[1] = hostRows;

        long localSize[] = new long[]{32, 32};
        clEnqueueNDRangeKernel(
                commandQueue,
                convolution,
                2,
                null,
                globalSize,
                localSize,
                0,
                null,
                null);
        long end = System.nanoTime();

        long region[] = new long[]{output.getNrOfColumns() * Float.BYTES, output.getNrOfRows(), 1};
        clEnqueueReadBufferRect(commandQueue, memOutput, CL_TRUE,
                new long[]{0, 0, 0},
                new long[]{0, 0, 0},
                region,
                // device
                hostCols * Float.BYTES, 0,
                // host
                output.getNrOfColumns() * Float.BYTES, 0,
                output.getCLPointer(),
                0,
                null,
                null);
    }

    public void batchConvolv(imatrix input, imatrix filter, imatrix output) {
        int[] iDim = new int[]{input.getNrOfColumns(), input.getNrOfRows()};
        int[] fDim = new int[]{filter.getNrOfColumns(), filter.getNrOfRows()};

        cl_mem memInput = input.getCLReadMem();
        cl_mem memFilter = filter.getCLReadMem();

        clEnqueueWriteBuffer(commandQueue, memInput, CL_TRUE, 0, input.getSliceSize()
                * Sizeof.cl_float, input.getCLPointer(), 0, null, null);
        clEnqueueWriteBuffer(commandQueue, memFilter, CL_TRUE, 0, filter.getSize()
                * Sizeof.cl_float, filter.getCLPointer(), 0, null, null);

        int deviceCols = output.getDeviceColumns();
        int deviceRows = output.getDeviceRows();

        int[] oDim = new int[]{deviceCols, deviceRows};
        cl_mem memOutput = output.getCLReadWriteMem();

        float zero[] = new float[1];
        clEnqueueFillBuffer(commandQueue, memOutput, Pointer.to(zero), Float.BYTES, 0, deviceCols * deviceRows * Float.BYTES, 0, null, null);

        clSetKernelArg(batchConvolution, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(batchConvolution, 1, Sizeof.cl_mem, Pointer.to(memFilter));
        clSetKernelArg(batchConvolution, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(batchConvolution, 3, Sizeof.cl_int2, Pointer.to(iDim));
        clSetKernelArg(batchConvolution, 4, Sizeof.cl_int2, Pointer.to(fDim));
        clSetKernelArg(batchConvolution, 5, Sizeof.cl_int2, Pointer.to(oDim));

        long globalSize[] = new long[3];
        globalSize[0] = deviceCols;
        globalSize[1] = deviceRows;
        globalSize[2] = filter.getNrOfSlices();

        long localSize[] = new long[]{32, 32, 1};
        clEnqueueNDRangeKernel(
                commandQueue,
                batchConvolution,
                3,
                null,
                globalSize,
                localSize,
                0,
                null,
                null);

        downloadRWMatrix(output);
    }

    public void batchConvolv(imatrix input, imatrix filter, int stride, imatrix output) {
        int[] iDim = new int[]{input.getNrOfColumns(), input.getNrOfRows()};
        int[] fDim = new int[]{filter.getNrOfColumns(), filter.getNrOfRows()};
        int[] ps = new int[]{stride};

        cl_mem memInput = input.getCLReadMem();
        cl_mem memFilter = filter.getCLReadMem();

        clEnqueueWriteBuffer(commandQueue, memInput, CL_TRUE, 0, input.getSliceSize()
                * Sizeof.cl_float, input.getCLPointer(), 0, null, null);
        clEnqueueWriteBuffer(commandQueue, memFilter, CL_TRUE, 0, filter.getSize()
                * Sizeof.cl_float, filter.getCLPointer(), 0, null, null);

        int deviceCols = output.getDeviceColumns();
        int deviceRows = output.getDeviceRows();

        int[] oDim = new int[]{deviceCols, deviceRows};
        cl_mem memOutput = output.getCLReadWriteMem();

        float zero[] = new float[1];
        clEnqueueFillBuffer(commandQueue, memOutput, Pointer.to(zero), Float.BYTES, 0, deviceCols * deviceRows * Float.BYTES, 0, null, null);

        int result = clSetKernelArg(batchConvolution, 0, Sizeof.cl_mem, Pointer.to(memInput));
        result = clSetKernelArg(batchConvolution, 1, Sizeof.cl_mem, Pointer.to(memFilter));
        result = clSetKernelArg(batchConvolution, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        result = clSetKernelArg(batchConvolution, 3, Sizeof.cl_int2, Pointer.to(iDim));
        result = clSetKernelArg(batchConvolution, 4, Sizeof.cl_int2, Pointer.to(fDim));
        result = clSetKernelArg(batchConvolution, 5, Sizeof.cl_int2, Pointer.to(oDim));
        result = clSetKernelArg(batchConvolution, 6, Sizeof.cl_int, Pointer.to(ps));

        long globalSize[] = new long[3];
        globalSize[0] = deviceCols;
        globalSize[1] = deviceRows;
        globalSize[2] = filter.getNrOfSlices();

        long localSize[] = new long[]{32, 32, 1};
        clEnqueueNDRangeKernel(
                commandQueue,
                batchConvolution,
                3,
                null,
                globalSize,
                localSize,
                0,
                null,
                null);

        downloadRWMatrix(output);
    }
    
    public void batchCorrelate(imatrix input, imatrix filter, int stride, imatrix output) {
        int[] iDim = new int[]{input.getNrOfColumns(), input.getNrOfRows()};
        int[] fDim = new int[]{filter.getNrOfColumns(), filter.getNrOfRows()};
        int[] ps = new int[]{stride};

        cl_mem memInput = input.getCLReadMem();
        cl_mem memFilter = filter.getCLReadMem();

        clEnqueueWriteBuffer(commandQueue, memInput, CL_TRUE, 0, input.getSliceSize()
                * Sizeof.cl_float, input.getCLPointer(), 0, null, null);
        clEnqueueWriteBuffer(commandQueue, memFilter, CL_TRUE, 0, filter.getSize()
                * Sizeof.cl_float, filter.getCLPointer(), 0, null, null);

        int deviceCols = output.getDeviceColumns();
        int deviceRows = output.getDeviceRows();

        int[] oDim = new int[]{deviceCols, deviceRows};
        cl_mem memOutput = output.getCLReadWriteMem();

        float zero[] = new float[1];
        clEnqueueFillBuffer(commandQueue, memOutput, Pointer.to(zero), Float.BYTES, 0, deviceCols * deviceRows * Float.BYTES, 0, null, null);

        int result = clSetKernelArg(batchCorrelation, 0, Sizeof.cl_mem, Pointer.to(memInput));
        result = clSetKernelArg(batchCorrelation, 1, Sizeof.cl_mem, Pointer.to(memFilter));
        result = clSetKernelArg(batchCorrelation, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        result = clSetKernelArg(batchCorrelation, 3, Sizeof.cl_int2, Pointer.to(iDim));
        result = clSetKernelArg(batchCorrelation, 4, Sizeof.cl_int2, Pointer.to(fDim));
        result = clSetKernelArg(batchCorrelation, 5, Sizeof.cl_int2, Pointer.to(oDim));
        result = clSetKernelArg(batchCorrelation, 6, Sizeof.cl_int, Pointer.to(ps));

        long globalSize[] = new long[3];
        globalSize[0] = deviceCols;
        globalSize[1] = deviceRows;
        globalSize[2] = filter.getNrOfSlices();

        long localSize[] = new long[]{32, 32, 1};
        clEnqueueNDRangeKernel(
                commandQueue,
                batchCorrelation,
                3,
                null,
                globalSize,
                localSize,
                0,
                null,
                null);

        downloadRWMatrix(output);
    }

}
