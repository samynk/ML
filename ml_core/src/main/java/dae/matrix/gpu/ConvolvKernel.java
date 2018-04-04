/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.fmatrix;
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
public class ConvolvKernel extends OpenCLKernel {

    private cl_kernel convolution;
    private cl_kernel batchConvolution;
    private cl_kernel batchCorrelation;
    private cl_kernel batchBackpropCorrelation;
    private cl_kernel rotateKernel;
    private cl_kernel maxRotation;
    private cl_kernel inverseMaxRotation;

    private final long[] localWorkSize = new long[]{DEFAULTWORKSIZE};

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
        rotateKernel = this.createKernel("rotateKernels");
        maxRotation = this.createKernel("maxRotation");
        inverseMaxRotation = this.createKernel("inverseMaxRotation");
        super.releaseProgram();
    }

    public void convolv(imatrix input, imatrix filter, imatrix output) {
        FloatDeviceBuffer inputDB = input.getDeviceBuffer();
        cl_mem memInput = inputDB.uploadRMatrix();

        FloatDeviceBuffer filterDB = filter.getDeviceBuffer();
        cl_mem memFilter = filterDB.uploadRMatrix();

        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem memOutput = outputDB.getRWMem();

        clSetKernelArg(convolution, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(convolution, 1, Sizeof.cl_mem, Pointer.to(memFilter));
        clSetKernelArg(convolution, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(convolution, 3, Sizeof.cl_int2, Pointer.to(inputDB.getDeviceDimension()));
        clSetKernelArg(convolution, 4, Sizeof.cl_int2, Pointer.to(filterDB.getDeviceDimension()));
        clSetKernelArg(convolution, 5, Sizeof.cl_int2, Pointer.to(outputDB.getDeviceDimension()));

        clEnqueueNDRangeKernel(
                commandQueue,
                convolution,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);
        outputDB.markRWMatrixAsMaster();
    }

    public void batchConvolv(imatrix input, imatrix filter, int stride, imatrix output) {
        FloatDeviceBuffer inputDB = input.getDeviceBuffer();
        int[] fDim = new int[]{filter.getNrOfRows(), filter.getNrOfColumns()};
        int[] ps = new int[]{stride};
        int[] fps = new int[]{filter.getNrOfSlices() / input.getNrOfSlices()};

        cl_mem memInput = inputDB.uploadRMatrix();

        FloatDeviceBuffer filterDB = filter.getDeviceBuffer();
        cl_mem memFilter = filterDB.uploadRMatrix();

        FloatDeviceBuffer outputDB = output.getDeviceBuffer();

        cl_mem memOutput = outputDB.getRWMem();
        GPU.zeroFillRW(output);

        clSetKernelArg(batchConvolution, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(batchConvolution, 1, Sizeof.cl_mem, Pointer.to(memFilter));
        clSetKernelArg(batchConvolution, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(batchConvolution, 3, Sizeof.cl_int4, Pointer.to(inputDB.getDimensionSizes()));
        clSetKernelArg(batchConvolution, 4, Sizeof.cl_int2, Pointer.to(fDim));
        clSetKernelArg(batchConvolution, 5, Sizeof.cl_int4, Pointer.to(outputDB.getDimensionSizes()));
        clSetKernelArg(batchConvolution, 6, Sizeof.cl_int, Pointer.to(fps));
        clSetKernelArg(batchConvolution, 7, Sizeof.cl_int, Pointer.to(ps));

        clEnqueueNDRangeKernel(
                commandQueue,
                batchConvolution,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                localWorkSize,
                0,
                null,
                null);

        outputDB.markRWMatrixAsMaster();
    }

    public void batchCorrelate(imatrix input, imatrix filter, int stride, imatrix output) {

        int[] fDim = new int[]{filter.getNrOfRows(), filter.getNrOfColumns()};
        int[] ps = new int[]{stride};
        int[] fps = new int[]{filter.getNrOfSlices() / input.getNrOfSlices()};

        FloatDeviceBuffer inputDB = input.getDeviceBuffer();
        if (input.getZeroPadding() > 0) {
            GPU.zeroFillR(input);
        }
        cl_mem memInput = inputDB.uploadRMatrix();

        FloatDeviceBuffer filterDB = filter.getDeviceBuffer();
        cl_mem memFilter = filterDB.uploadRMatrix();

        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem memOutput = outputDB.getRWMem();

        clSetKernelArg(batchCorrelation, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(batchCorrelation, 1, Sizeof.cl_mem, Pointer.to(memFilter));
        clSetKernelArg(batchCorrelation, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(batchCorrelation, 3, Sizeof.cl_int4, Pointer.to(inputDB.getDimensionSizes()));
        clSetKernelArg(batchCorrelation, 4, Sizeof.cl_int2, Pointer.to(fDim));
        clSetKernelArg(batchCorrelation, 5, Sizeof.cl_int4, Pointer.to(outputDB.getDimensionSizes()));
        clSetKernelArg(batchCorrelation, 6, Sizeof.cl_int, Pointer.to(fps));
        clSetKernelArg(batchCorrelation, 7, Sizeof.cl_int, Pointer.to(ps));

        clEnqueueNDRangeKernel(
                commandQueue,
                batchCorrelation,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                localWorkSize,
                0,
                null,
                null);

        outputDB.markRWMatrixAsMaster();
    }

    public void batchBackpropCorrelate(imatrix input, imatrix filter, int stride, imatrix output) {
        int[] fDim = new int[]{filter.getNrOfRows(), filter.getNrOfColumns()};
        int[] ps = new int[]{stride};
        int[] fps = new int[]{filter.getNrOfSlices() / output.getNrOfSlices()};

        FloatDeviceBuffer inputDB = input.getDeviceBuffer();
        if (input.getZeroPadding() > 0) {
            GPU.zeroFillR(input);
        }
        cl_mem memInput = inputDB.uploadRMatrix();

        FloatDeviceBuffer filterDB = filter.getDeviceBuffer();
        cl_mem memFilter = filterDB.uploadRMatrix();

        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem memOutput = outputDB.getRWMem();

        clSetKernelArg(batchBackpropCorrelation, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(batchBackpropCorrelation, 1, Sizeof.cl_mem, Pointer.to(memFilter));
        clSetKernelArg(batchBackpropCorrelation, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(batchBackpropCorrelation, 3, Sizeof.cl_int4, Pointer.to(inputDB.getDimensionSizes()));
        clSetKernelArg(batchBackpropCorrelation, 4, Sizeof.cl_int2, Pointer.to(fDim));
        clSetKernelArg(batchBackpropCorrelation, 5, Sizeof.cl_int4, Pointer.to(outputDB.getDimensionSizes()));
        clSetKernelArg(batchBackpropCorrelation, 6, Sizeof.cl_int, Pointer.to(fps));
        clSetKernelArg(batchBackpropCorrelation, 7, Sizeof.cl_int, Pointer.to(ps));

        clEnqueueNDRangeKernel(
                commandQueue,
                batchBackpropCorrelation,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        outputDB.markRWMatrixAsMaster();
    }

    public void rotateKernels(imatrix filter, int nrOfFeatures, int nrOfRotations, float minAngle, float maxAngle) {
        // slices per feature
        int sPRF = filter.getNrOfSlices() / nrOfFeatures;
        // slices per rotation.
        int sPRR = sPRF / nrOfRotations;

        int[] fDim = new int[]{filter.getNrOfRows(), filter.getNrOfColumns(), sPRF, sPRR};
        int slices = filter.getNrOfSlices();
        fmatrix sincos = new fmatrix(2, slices);
        float angleStep = (maxAngle - minAngle) / (slices - 1);
        float angle = minAngle;
        for (int i = 0; i < slices; ++i) {
            float s = (float) Math.sin(angle);
            float c = (float) Math.cos(angle);
            sincos.set(0, i, s);
            sincos.set(1, i, c);
            angle += angleStep;
        }

        FloatDeviceBuffer filterDB = filter.getDeviceBuffer();
        cl_mem memFilter = filterDB.uploadRWMatrix();

        FloatDeviceBuffer sincosDB = sincos.getDeviceBuffer();
        cl_mem memSincos = sincosDB.uploadRMatrix();

        clSetKernelArg(rotateKernel, 0, Sizeof.cl_mem, Pointer.to(memFilter));
        clSetKernelArg(rotateKernel, 1, Sizeof.cl_int4, Pointer.to(fDim));
        clSetKernelArg(rotateKernel, 2, Sizeof.cl_mem, Pointer.to(memSincos));

        // first dimension is (x,y) of kernel to rotate
        // second dimension is slice of filter.
        clEnqueueNDRangeKernel(
                commandQueue,
                rotateKernel,
                1,
                null,
                filterDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        filterDB.markRWMatrixAsMaster();
    }

    public void maxRotation(imatrix input, int nrOfFeatures, int nrOfRotations, float minAngle, float maxAngle, imatrix valOutput, imatrix rotOutput) {
        // slices per feature
        int sPRF = valOutput.getNrOfSlices() / nrOfFeatures;
        int[] desc = new int[]{sPRF, nrOfRotations};

        float angleStep = (maxAngle - minAngle) / (nrOfRotations - 1);

        FloatDeviceBuffer inputDB = input.getDeviceBuffer();
        cl_mem memInput = inputDB.uploadRMatrix();

        FloatDeviceBuffer outputDB = valOutput.getDeviceBuffer();
        cl_mem memOutput = outputDB.getRWMem();

        FloatDeviceBuffer rotOutputDB = rotOutput.getDeviceBuffer();
        cl_mem memRotOutput = rotOutputDB.getRWMem();

        clSetKernelArg(maxRotation, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(maxRotation, 1, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(maxRotation, 2, Sizeof.cl_mem, Pointer.to(memRotOutput));
        clSetKernelArg(maxRotation, 3, Sizeof.cl_int4, Pointer.to(inputDB.getDimensionSizes()));
        clSetKernelArg(maxRotation, 4, Sizeof.cl_int4, Pointer.to(outputDB.getDimensionSizes()));
        clSetKernelArg(maxRotation, 5, Sizeof.cl_int2, Pointer.to(desc));
        clSetKernelArg(maxRotation, 6, Sizeof.cl_float, Pointer.to(new float[]{angleStep}));

       
        // first dimension is (x,y) of kernel to rotate
        // second dimension is slice of filter.
        clEnqueueNDRangeKernel(
                commandQueue,
                maxRotation,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        outputDB.markRWMatrixAsMaster();
        rotOutputDB.markRWMatrixAsMaster();
    }
    
    public void maxInverseRotation(imatrix valInput,imatrix rotInput,  int nrOfFeatures, int nrOfRotations, float minAngle, float maxAngle, imatrix output){
        // slices per feature
        int sPRF = valInput.getNrOfSlices() / nrOfFeatures;
        int[] desc = new int[]{sPRF, nrOfRotations};

        float angleStep = (maxAngle - minAngle) / (nrOfRotations - 1);

        FloatDeviceBuffer valInputDB = valInput.getDeviceBuffer();
        cl_mem memValInput = valInputDB.uploadRMatrix();

        FloatDeviceBuffer rotInputDB = rotInput.getDeviceBuffer();
        cl_mem memRotInput = rotInputDB.uploadRMatrix();

        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem memOutput = outputDB.getRWMem();

        clSetKernelArg(inverseMaxRotation, 0, Sizeof.cl_mem, Pointer.to(memValInput));
        clSetKernelArg(inverseMaxRotation, 1, Sizeof.cl_mem, Pointer.to(memRotInput));
        clSetKernelArg(inverseMaxRotation, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(inverseMaxRotation, 3, Sizeof.cl_int4, Pointer.to(valInputDB.getDimensionSizes()));
        clSetKernelArg(inverseMaxRotation, 4, Sizeof.cl_int4, Pointer.to(outputDB.getDimensionSizes()));
        clSetKernelArg(inverseMaxRotation, 5, Sizeof.cl_int2, Pointer.to(desc));
        clSetKernelArg(inverseMaxRotation, 6, Sizeof.cl_float, Pointer.to(new float[]{angleStep}));

       
        // first dimension is (x,y) of kernel to rotate
        // second dimension is slice of filter.
        clEnqueueNDRangeKernel(
                commandQueue,
                inverseMaxRotation,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        outputDB.markRWMatrixAsMaster();
    }
}
