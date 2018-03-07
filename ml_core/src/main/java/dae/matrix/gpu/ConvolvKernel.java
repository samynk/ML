/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.imatrix;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clEnqueueFillBuffer;
import static org.jocl.CL.clEnqueueNDRangeKernel;
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
    private cl_kernel batchBackpropCorrelation;

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
        batchBackpropCorrelation = this.createKernel("batchBackpropCorrelate");
        super.releaseProgram();
    }

    public void convolv(imatrix input, imatrix filter, imatrix output) {
        int[] iDim = new int[]{input.getNrOfColumns(), input.getNrOfRows()};
        int[] fDim = new int[]{filter.getNrOfColumns(), filter.getNrOfRows()};

        DeviceBuffer inputDB = input.getDeviceBuffer();
        cl_mem memInput = inputDB.getCLReadMem();

        DeviceBuffer filterDB = filter.getDeviceBuffer();
        cl_mem memFilter = filterDB.getCLReadMem();

        clEnqueueWriteBuffer(commandQueue, memInput, CL_TRUE, 0, input.getSliceSize()
                * Sizeof.cl_float, inputDB.getCLPointer(), 0, null, null);
        clEnqueueWriteBuffer(commandQueue, memFilter, CL_TRUE, 0, filter.getSliceSize()
                * Sizeof.cl_float, filterDB.getCLPointer(), 0, null, null);

        DeviceBuffer outputDB = output.getDeviceBuffer();
        int[] oDim = outputDB.getDeviceDimension();
        cl_mem memOutput = outputDB.getCLReadWriteMem();

        clSetKernelArg(convolution, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(convolution, 1, Sizeof.cl_mem, Pointer.to(memFilter));
        clSetKernelArg(convolution, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(convolution, 3, Sizeof.cl_int2, Pointer.to(iDim));
        clSetKernelArg(convolution, 4, Sizeof.cl_int2, Pointer.to(fDim));
        clSetKernelArg(convolution, 5, Sizeof.cl_int2, Pointer.to(oDim));

        long globalSize[] = new long[2];
        globalSize[0] = oDim[0];
        globalSize[1] = oDim[1];

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

        GPU.downloadRWMatrix(output);
    }

    public void batchConvolv(imatrix input, imatrix filter, int stride, imatrix output) {
        DeviceBuffer inputDB = input.getDeviceBuffer();
        int[] iDim = inputDB.getDeviceDimension();
        int[] fDim = new int[]{filter.getNrOfColumns(), filter.getNrOfRows()};
        int[] ps = new int[]{stride};
        int[] fps = new int[]{filter.getNrOfSlices() / input.getNrOfSlices()};

        GPU.zeroFillR(input);
        GPU.uploadRMatrix(input);

        DeviceBuffer filterDB = filter.getDeviceBuffer();

        cl_mem memFilter = filterDB.getCLReadMem();
        clEnqueueWriteBuffer(commandQueue, memFilter, CL_TRUE, 0, filter.getSize()
                * Sizeof.cl_float, filterDB.getCLPointer(), 0, null, null);

        DeviceBuffer outputDB = output.getDeviceBuffer();
        int deviceCols = outputDB.getDeviceColumns();
        int deviceRows = outputDB.getDeviceRows();
        int[] oDim = outputDB.getDeviceDimension();
        cl_mem memOutput = outputDB.getCLReadWriteMem();
        GPU.zeroFillRW(output);

        clSetKernelArg(batchConvolution, 0, Sizeof.cl_mem, Pointer.to(inputDB.getCLReadMem()));
        clSetKernelArg(batchConvolution, 1, Sizeof.cl_mem, Pointer.to(memFilter));
        clSetKernelArg(batchConvolution, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(batchConvolution, 3, Sizeof.cl_int2, Pointer.to(iDim));
        clSetKernelArg(batchConvolution, 4, Sizeof.cl_int2, Pointer.to(fDim));
        clSetKernelArg(batchConvolution, 5, Sizeof.cl_int, Pointer.to(fps));
        clSetKernelArg(batchConvolution, 6, Sizeof.cl_int2, Pointer.to(oDim));
        clSetKernelArg(batchConvolution, 7, Sizeof.cl_int, Pointer.to(ps));

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

        GPU.downloadRWMatrix(output);
    }

    public void batchCorrelate(imatrix input, imatrix filter, int stride, imatrix output) {
        DeviceBuffer inputDB = input.getDeviceBuffer();
        
        int[] iDim = inputDB.getDeviceDimension();
        int[] fDim = new int[]{filter.getNrOfColumns(), filter.getNrOfRows()};
        int[] fps = new int[]{filter.getNrOfSlices() / input.getNrOfSlices()};
        int[] ps = new int[]{stride};

        cl_mem memInput = inputDB.getCLReadMem();
        
        DeviceBuffer filterDB = filter.getDeviceBuffer();
        cl_mem memFilter = filterDB.getCLReadMem();

        GPU.zeroFillR(input);
        GPU.uploadRMatrix(input);

        clEnqueueWriteBuffer(commandQueue, memFilter, CL_TRUE, 0, filter.getSize()
                * Sizeof.cl_float, filterDB.getCLPointer(), 0, null, null);

        
        DeviceBuffer outputDB = output.getDeviceBuffer();
        int deviceCols = outputDB.getDeviceColumns();
        int deviceRows = outputDB.getDeviceRows();

        int[] oDim = new int[]{deviceCols, deviceRows};
        cl_mem memOutput = outputDB.getCLReadWriteMem();

        float zero[] = new float[1];
        clEnqueueFillBuffer(commandQueue, memOutput, Pointer.to(zero), Float.BYTES, 0, deviceCols * deviceRows * Float.BYTES, 0, null, null);

        clSetKernelArg(batchCorrelation, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(batchCorrelation, 1, Sizeof.cl_mem, Pointer.to(memFilter));
        clSetKernelArg(batchCorrelation, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(batchCorrelation, 3, Sizeof.cl_int2, Pointer.to(iDim));
        clSetKernelArg(batchCorrelation, 4, Sizeof.cl_int2, Pointer.to(fDim));
        clSetKernelArg(batchCorrelation, 5, Sizeof.cl_int, Pointer.to(fps));
        clSetKernelArg(batchCorrelation, 6, Sizeof.cl_int2, Pointer.to(oDim));
        clSetKernelArg(batchCorrelation, 7, Sizeof.cl_int, Pointer.to(ps));

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

        GPU.downloadRWMatrix(output);
    }

    public void batchBackpropCorrelate(imatrix input, imatrix filter, int stride, imatrix output) {
        DeviceBuffer inputDB = input.getDeviceBuffer();
        int[] iDim = inputDB.getDeviceDimension();
        int[] fDim = new int[]{filter.getNrOfColumns(), filter.getNrOfRows()};
        int[] fps = new int[]{filter.getNrOfSlices() / output.getNrOfSlices()};
        int[] ps = new int[]{stride};

        cl_mem memInput = inputDB.getCLReadMem();
        
        DeviceBuffer filterDB = filter.getDeviceBuffer();
        cl_mem memFilter = filterDB.getCLReadMem();

        GPU.zeroFillR(input);
        GPU.uploadRMatrix(input);

        clEnqueueWriteBuffer(commandQueue, memFilter, CL_TRUE, 0, filter.getSize()
                * Sizeof.cl_float, filterDB.getCLPointer(), 0, null, null);

        DeviceBuffer outputDB = output.getDeviceBuffer();
        int deviceCols = outputDB.getDeviceColumns();
        int deviceRows = outputDB.getDeviceRows();

        int[] oDim = new int[]{deviceCols, deviceRows};
        cl_mem memOutput = outputDB.getCLReadWriteMem();

        GPU.zeroFillRW(output);

        clSetKernelArg(batchBackpropCorrelation, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(batchBackpropCorrelation, 1, Sizeof.cl_mem, Pointer.to(memFilter));
        clSetKernelArg(batchBackpropCorrelation, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(batchBackpropCorrelation, 3, Sizeof.cl_int2, Pointer.to(iDim));
        clSetKernelArg(batchBackpropCorrelation, 4, Sizeof.cl_int2, Pointer.to(fDim));
        clSetKernelArg(batchBackpropCorrelation, 5, Sizeof.cl_int, Pointer.to(fps));
        clSetKernelArg(batchBackpropCorrelation, 6, Sizeof.cl_int2, Pointer.to(oDim));
        clSetKernelArg(batchBackpropCorrelation, 7, Sizeof.cl_int, Pointer.to(ps));

        long globalSize[] = new long[3];
        globalSize[0] = deviceCols;
        globalSize[1] = deviceRows;
        globalSize[2] = output.getNrOfSlices();

        long localSize[] = new long[]{32, 32, 1};
        clEnqueueNDRangeKernel(
                commandQueue,
                batchBackpropCorrelation,
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
