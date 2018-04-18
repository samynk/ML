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
public class FuzzyKernel extends OpenCLKernel {

    private cl_kernel fuzzyFunction;
    private cl_kernel fuzzyShiftMinus;
    private cl_kernel fuzzyShiftDeltas;
    private cl_kernel fuzzyBackProp;
    private cl_kernel fuzzyInputAdd;

    private static long[] LOCALWORKSIZE = new long[]{OpenCLKernel.DEFAULTWORKSIZE};

    public FuzzyKernel() {
        super("/kernels/neuralnet/fuzzy.cl");
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
        fuzzyFunction = this.createKernel("fuzzyFunction");
        fuzzyShiftMinus = this.createKernel("fuzzyShiftMinus");
        fuzzyShiftDeltas = this.createKernel("fuzzyShiftDeltas");
        fuzzyBackProp = this.createKernel("fuzzyBackProp");
        fuzzyInputAdd = this.createKernel("fuzzyInputAdd");
        super.releaseProgram();
    }

    public void fuzzyFunction(imatrix input, int classes, imatrix a, imatrix b, imatrix output) {
        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem inputMem = input.getDeviceBuffer().upload();
        cl_mem aMem = a.getDeviceBuffer().upload();
        cl_mem bMem = b.getDeviceBuffer().upload();
        cl_mem outputMem = outputDB.getMem();

        int[] dim = new int[]{input.getNrOfRows(), classes-1, output.getHyperSliceSize()};

        clSetKernelArg(fuzzyFunction, 0, Sizeof.cl_int4, Pointer.to(dim));
        clSetKernelArg(fuzzyFunction, 1, Sizeof.cl_mem, Pointer.to(inputMem));
        clSetKernelArg(fuzzyFunction, 2, Sizeof.cl_mem, Pointer.to(aMem));
        clSetKernelArg(fuzzyFunction, 3, Sizeof.cl_mem, Pointer.to(bMem));
        clSetKernelArg(fuzzyFunction, 4, Sizeof.cl_mem, Pointer.to(outputMem));

        clEnqueueNDRangeKernel(
                commandQueue,
                fuzzyFunction,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                LOCALWORKSIZE,
                0,
                null,
                null);

        outputDB.markGpuAsMaster();
    }

    public void fuzzyShiftMinus(imatrix input, int classes, imatrix output) {
        int[] dim = new int[]{classes, input.getHyperSliceSize(), output.getHyperSliceSize()};
        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem inputMem = input.getDeviceBuffer().upload();
        cl_mem outputMem = outputDB.getMem();

        clSetKernelArg(fuzzyShiftMinus, 0, Sizeof.cl_int4, Pointer.to(dim));
        clSetKernelArg(fuzzyShiftMinus, 1, Sizeof.cl_mem, Pointer.to(inputMem));
        clSetKernelArg(fuzzyShiftMinus, 2, Sizeof.cl_mem, Pointer.to(outputMem));

        clEnqueueNDRangeKernel(
                commandQueue,
                fuzzyShiftMinus,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                LOCALWORKSIZE,
                0,
                null,
                null);

        outputDB.markGpuAsMaster();
    }

    public void fuzzyShiftDeltas(imatrix input, int classes, imatrix output) {
        int[] dim = new int[]{classes, input.getHyperSliceSize(), output.getHyperSliceSize(), input.getNrOfHyperSlices()};
        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem inputMem = input.getDeviceBuffer().upload();
        // reset output matrix.
        GPU.zeroFill(output);
        cl_mem outputMem = outputDB.getMem();

        clSetKernelArg(fuzzyShiftDeltas, 0, Sizeof.cl_int4, Pointer.to(dim));
        clSetKernelArg(fuzzyShiftDeltas, 1, Sizeof.cl_mem, Pointer.to(inputMem));
        clSetKernelArg(fuzzyShiftDeltas, 2, Sizeof.cl_mem, Pointer.to(outputMem));
        
        clEnqueueNDRangeKernel(
                commandQueue,
                fuzzyShiftDeltas,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                LOCALWORKSIZE,
                0,
                null,
                null);

        outputDB.markGpuAsMaster();
    }
    
    public void fuzzyBackProp(imatrix input, imatrix weights, int classes, imatrix output) {
        int[] dim = new int[]{classes, input.getHyperSliceSize(), output.getHyperSliceSize()};
        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem inputMem = input.getDeviceBuffer().upload();
        cl_mem weightMem = weights.getDeviceBuffer().upload();
        // reset output matrix.
        GPU.zeroFill(output);
        cl_mem outputMem = outputDB.getMem();

        clSetKernelArg(fuzzyBackProp, 0, Sizeof.cl_int4, Pointer.to(dim));
        clSetKernelArg(fuzzyBackProp, 1, Sizeof.cl_mem, Pointer.to(inputMem));
        clSetKernelArg(fuzzyBackProp, 2, Sizeof.cl_mem, Pointer.to(weightMem));
        clSetKernelArg(fuzzyBackProp, 3, Sizeof.cl_mem, Pointer.to(outputMem));
        
        clEnqueueNDRangeKernel(
                commandQueue,
                fuzzyBackProp,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                LOCALWORKSIZE,
                0,
                null,
                null);

        outputDB.markGpuAsMaster();
    }
    
    public void fuzzyInputAdd(imatrix input, imatrix weights, int classes, imatrix output) {
        int[] dim = new int[]{classes, input.getHyperSliceSize(), output.getHyperSliceSize()};
        FloatDeviceBuffer outputDB = output.getDeviceBuffer();
        cl_mem inputMem = input.getDeviceBuffer().upload();
        cl_mem weightMem = weights.getDeviceBuffer().upload();
        // reset output matrix.
        GPU.zeroFill(output);
        cl_mem outputMem = outputDB.getMem();

        clSetKernelArg(fuzzyInputAdd, 0, Sizeof.cl_int4, Pointer.to(dim));
        clSetKernelArg(fuzzyInputAdd, 1, Sizeof.cl_mem, Pointer.to(inputMem));
        clSetKernelArg(fuzzyInputAdd, 2, Sizeof.cl_mem, Pointer.to(weightMem));
        clSetKernelArg(fuzzyInputAdd, 3, Sizeof.cl_mem, Pointer.to(outputMem));
        
        clEnqueueNDRangeKernel(
                commandQueue,
                fuzzyInputAdd,
                1,
                null,
                outputDB.getGlobalWorkSize(),
                LOCALWORKSIZE,
                0,
                null,
                null);

        outputDB.markGpuAsMaster();
    }
}
