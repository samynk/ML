/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import static org.jocl.CL.CL_KERNEL_NUM_ARGS;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clGetKernelInfo;
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
    private cl_kernel accumulateRotateKernel;
    private cl_kernel maxRotation;
    private cl_kernel inverseMaxRotation;
    
    private cl_kernel forwardPancake;
    private cl_kernel deltasPancake;

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
        accumulateRotateKernel = this.createKernel("accumulateRotateKernels");
        maxRotation = this.createKernel("maxRotation");
        inverseMaxRotation = this.createKernel("inverseMaxRotation");
        forwardPancake = this.createKernel("forwardPancake");
        deltasPancake = this.createKernel("deltasPancake");
        super.releaseProgram();
    }

    public void convolv(imatrix input, imatrix filter, imatrix output) {
        FloatDeviceBuffer inputDB = input.getDeviceBuffer();
        cl_mem memInput = inputDB.upload();

        FloatDeviceBuffer filterDB = filter.getDeviceBuffer();
        cl_mem memFilter = filterDB.upload();

        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem memOutput = outputDB.getMem();

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
        outputDB.markGpuAsMaster();
    }

    public void batchConvolv(imatrix input, imatrix filter, int stride, imatrix output) {
        FloatDeviceBuffer inputDB = input.getDeviceBuffer();
        int[] fDim = new int[]{filter.getNrOfRows(), filter.getNrOfColumns()};
        int[] ps = new int[]{stride};
        int[] fps = new int[]{filter.getNrOfSlices() / input.getNrOfSlices()};

        cl_mem memInput = inputDB.upload();

        FloatDeviceBuffer filterDB = filter.getDeviceBuffer();
        cl_mem memFilter = filterDB.upload();

        FloatDeviceBuffer outputDB = output.getDeviceBuffer();

        cl_mem memOutput = outputDB.getMem();
        GPU.zeroFill(output);

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

        outputDB.markGpuAsMaster();
    }

    public void batchCorrelate(imatrix input, imatrix filter, int stride, imatrix output) {

        int[] fDim = new int[]{filter.getNrOfRows(), filter.getNrOfColumns()};
        int[] ps = new int[]{stride};
        int[] fps = new int[]{filter.getNrOfSlices() / input.getNrOfSlices()};

        FloatDeviceBuffer inputDB = input.getDeviceBuffer();
        if (input.getZeroPadding() > 0) {
            GPU.zeroFill(input);
        }
        cl_mem memInput = inputDB.upload();

        FloatDeviceBuffer filterDB = filter.getDeviceBuffer();
        cl_mem memFilter = filterDB.upload();

        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem memOutput = outputDB.getMem();

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

        outputDB.markGpuAsMaster();
    }

    public void batchBackpropCorrelate(imatrix input, imatrix filter, int stride, imatrix output) {
        int[] fDim = new int[]{filter.getNrOfRows(), filter.getNrOfColumns()};
        int[] ps = new int[]{stride};
        int[] fps = new int[]{filter.getNrOfSlices() / output.getNrOfSlices()};

        FloatDeviceBuffer inputDB = input.getDeviceBuffer();
        if (input.getZeroPadding() > 0) {
            GPU.zeroFill(input);
        }
        inputDB.markCpuAsMaster();
        cl_mem memInput = inputDB.upload();

        FloatDeviceBuffer filterDB = filter.getDeviceBuffer();
        cl_mem memFilter = filterDB.upload();

        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem memOutput = outputDB.getMem();

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

        outputDB.markGpuAsMaster();
    }

    fmatrix sincos;

    public void rotateKernels(imatrix filter, int nrOfRotations, float minAngle, float maxAngle, imatrix output) {

        int[] fDim = new int[]{filter.getNrOfRows(), filter.getNrOfColumns(), nrOfRotations};
        if (sincos == null || sincos.getNrOfColumns() < nrOfRotations) {
            sincos = new fmatrix(2, nrOfRotations);
        }

        float angleStep = (maxAngle - minAngle) / (nrOfRotations - 1);
        float angle = minAngle;
        for (int i = 0; i < nrOfRotations; ++i) {
            float s = (float) Math.sin(angle);
            float c = (float) Math.cos(angle);
            sincos.set(0, i, s);
            sincos.set(1, i, c);
            angle += angleStep;
        }
        sincos.makeMaster();

        FloatDeviceBuffer filterDB = filter.getDeviceBuffer();
        cl_mem memFilter = filterDB.upload();

        FloatDeviceBuffer sincosDB = sincos.getDeviceBuffer();
        cl_mem memSincos = sincosDB.upload();

        FloatDeviceBuffer outputDB = output.getDeviceBuffer();

        clSetKernelArg(rotateKernel, 0, Sizeof.cl_mem, Pointer.to(memFilter));
        clSetKernelArg(rotateKernel, 1, Sizeof.cl_int4, Pointer.to(fDim));
        clSetKernelArg(rotateKernel, 2, Sizeof.cl_mem, Pointer.to(memSincos));
        clSetKernelArg(rotateKernel, 3, Sizeof.cl_mem, Pointer.to(outputDB.getMem()));

        // first dimension is (x,y) of kernel to rotate
        // second dimension is slice of filter.
        clEnqueueNDRangeKernel(
                commandQueue,
                rotateKernel,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        outputDB.markGpuAsMaster();
    }

    public void accumulateRotateKernels(imatrix rotatedOutput, int nrOfRotations, float minAngle, float maxAngle, imatrix kernelOutput) {
        int[] fDim = new int[]{kernelOutput.getNrOfRows(), kernelOutput.getNrOfColumns(), nrOfRotations};
        if (sincos == null || sincos.getNrOfColumns() < nrOfRotations) {
            sincos = new fmatrix(2, nrOfRotations);
        }
        float angleStep = (maxAngle - minAngle) / (nrOfRotations - 1);
        float angle = minAngle;
        for (int i = 0; i < nrOfRotations; ++i) {
            float s = (float) Math.sin(-angle);
            float c = (float) Math.cos(-angle);
            sincos.set(0, i, s);
            sincos.set(1, i, c);
            angle += angleStep;
        }
        sincos.makeMaster();

        FloatDeviceBuffer rotatedDB = rotatedOutput.getDeviceBuffer();
        cl_mem memRotated = rotatedDB.upload();

        FloatDeviceBuffer sincosDB = sincos.getDeviceBuffer();
        cl_mem memSincos = sincosDB.upload();

        GPU.zeroFill(kernelOutput);
        FloatDeviceBuffer outputDB = kernelOutput.getDeviceBuffer();
        cl_mem memOutput = outputDB.getMem();

        long[] result = new long[1];
        clGetKernelInfo(accumulateRotateKernel, CL_KERNEL_NUM_ARGS, Sizeof.cl_int, Pointer.to(result), null);

        clSetKernelArg(accumulateRotateKernel, 0, Sizeof.cl_mem, Pointer.to(memRotated));
        clSetKernelArg(accumulateRotateKernel, 1, Sizeof.cl_int4, Pointer.to(fDim));
        clSetKernelArg(accumulateRotateKernel, 2, Sizeof.cl_mem, Pointer.to(memSincos));
        clSetKernelArg(accumulateRotateKernel, 3, Sizeof.cl_mem, Pointer.to(memOutput));

        // first dimension is (x,y) of kernel to rotate
        // second dimension is slice of filter.
        clEnqueueNDRangeKernel(
                commandQueue,
                accumulateRotateKernel,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        outputDB.markGpuAsMaster();
    }

    public void maxRotation(imatrix input, int nrOfFeatures, int nrOfRotations, float minAngle, float maxAngle, imatrix valOutput, imatrix rotOutput) {
        // slices per feature
        int sPRF = valOutput.getNrOfSlices() / nrOfFeatures;
        int[] desc = new int[]{sPRF, nrOfRotations, input.getSize()};

        float angleStep = (maxAngle - minAngle) / (nrOfRotations - 1);

        FloatDeviceBuffer inputDB = input.getDeviceBuffer();
        cl_mem memInput = inputDB.upload();

        FloatDeviceBuffer outputDB = valOutput.getDeviceBuffer();
        cl_mem memOutput = outputDB.getMem();

        FloatDeviceBuffer rotOutputDB = rotOutput.getDeviceBuffer();
        cl_mem memRotOutput = rotOutputDB.getMem();

        clSetKernelArg(maxRotation, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(maxRotation, 1, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(maxRotation, 2, Sizeof.cl_mem, Pointer.to(memRotOutput));
        clSetKernelArg(maxRotation, 3, Sizeof.cl_int4, Pointer.to(inputDB.getDimensionSizes()));
        clSetKernelArg(maxRotation, 4, Sizeof.cl_int4, Pointer.to(outputDB.getDimensionSizes()));
        clSetKernelArg(maxRotation, 5, Sizeof.cl_int4, Pointer.to(desc));
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

        outputDB.markGpuAsMaster();
        rotOutputDB.markGpuAsMaster();
    }

    public void maxInverseRotation(imatrix valInput, imatrix rotInput, int nrOfFeatures, int nrOfRotations, float minAngle, float maxAngle, imatrix output) {
        // slices per feature
        int sPRF = valInput.getNrOfSlices() / nrOfFeatures;
        int[] desc = new int[]{sPRF, nrOfRotations, output.getSize()};

        float angleStep = (maxAngle - minAngle) / (nrOfRotations - 1);

        FloatDeviceBuffer valInputDB = valInput.getDeviceBuffer();
        cl_mem memValInput = valInputDB.upload();

        FloatDeviceBuffer rotInputDB = rotInput.getDeviceBuffer();
        cl_mem memRotInput = rotInputDB.upload();

        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem memOutput = outputDB.getMem();

        clSetKernelArg(inverseMaxRotation, 0, Sizeof.cl_mem, Pointer.to(memValInput));
        clSetKernelArg(inverseMaxRotation, 1, Sizeof.cl_mem, Pointer.to(memRotInput));
        clSetKernelArg(inverseMaxRotation, 2, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(inverseMaxRotation, 3, Sizeof.cl_int4, Pointer.to(valInputDB.getDimensionSizes()));
        clSetKernelArg(inverseMaxRotation, 4, Sizeof.cl_int4, Pointer.to(outputDB.getDimensionSizes()));
        clSetKernelArg(inverseMaxRotation, 5, Sizeof.cl_int4, Pointer.to(desc));
        clSetKernelArg(inverseMaxRotation, 6, Sizeof.cl_float, Pointer.to(new float[]{angleStep}));

        // first dimension is (x,y) of kernel to rotate
        // second dimension is slice of filter.
        clEnqueueNDRangeKernel(
                commandQueue,
                inverseMaxRotation,
                1,
                null,
                valInputDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        outputDB.markGpuAsMaster();
    }
    
    public void forwardPancake(imatrix input,int slicesPerGroup, int biases, imatrix weights, imatrix output)
    {
        FloatDeviceBuffer inputDB = input.getDeviceBuffer();
        cl_mem memInput = inputDB.upload();

        FloatDeviceBuffer weightDB = weights.getDeviceBuffer();
        cl_mem memWeights = weightDB.upload();

        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem memOutput = outputDB.getMem();

        clSetKernelArg(forwardPancake, 0, Sizeof.cl_mem, Pointer.to(memInput));
        clSetKernelArg(forwardPancake, 1, Sizeof.cl_int4, Pointer.to(inputDB.getDimensionSizes()));
        clSetKernelArg(forwardPancake, 2, Sizeof.cl_mem, Pointer.to(memWeights));
        clSetKernelArg(forwardPancake, 3, Sizeof.cl_int4, Pointer.to(weightDB.getDimensionSizes()));
        clSetKernelArg(forwardPancake, 4, Sizeof.cl_mem, Pointer.to(memOutput));
        clSetKernelArg(forwardPancake, 5, Sizeof.cl_int4,Pointer.to(outputDB.getDimensionSizes()));
        clSetKernelArg(forwardPancake, 6, Sizeof.cl_int2, Pointer.to(new int[]{slicesPerGroup, biases}));

        // first dimension is (x,y) of kernel to rotate
        // second dimension is slice of filter.
        clEnqueueNDRangeKernel(
                commandQueue,
                forwardPancake,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                this.localWorkSize,
                0,
                null,
                null);

        outputDB.markGpuAsMaster();
    }
    
    public void deltasPancake( imatrix input, imatrix deltas, int slicesPerGroup, int biases, imatrix deltaWeights)
    {
        
    }

}
