/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import static org.jocl.CL.*;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_kernel;
import org.jocl.cl_program;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class OpenCLKernel {

    private final String kernelFile;
    private final String kernelSource;

    private cl_command_queue commandQueue;
    private cl_context context;
    private cl_program program;
    private cl_kernel kernel;

    /**
     *
     * @param kernelFile
     */
    public OpenCLKernel(String kernelFile) {
        this.kernelFile = kernelFile;

        InputStream is = getClass().getResourceAsStream(kernelFile);
        BufferedReader br = new BufferedReader(new InputStreamReader(is));

        StringBuilder sb = new StringBuilder();
        br.lines().forEach(s -> {
            sb.append(s);
            sb.append('\n');
        });
        kernelSource = sb.toString();

        System.out.println("Source code :");
        System.out.println(sb.toString());
    }

    /**
     * Initializes the kernel
     *
     * @param context the opencl context.
     * @param commandQueue the command queue to use.
     */
    public void init(cl_context context, cl_command_queue commandQueue) {
        this.context = context;
        this.commandQueue = commandQueue;
        program = clCreateProgramWithSource(context, 1, new String[]{kernelSource}, null, null);
        String compileOptions = "-cl-mad-enable";
        clBuildProgram(program, 0, null, compileOptions, null, null);
        kernel = clCreateKernel(program, "convolution", null);
        clReleaseProgram(program);
    }

    public void destroy() {
        clReleaseKernel(kernel);
    }

    public void convolv(imatrix input, imatrix filter, imatrix output) {
        int[] iDim = new int[]{input.getNrOfRows(), input.getNrOfColumns()};
        int[] fDim = new int[]{filter.getNrOfRows(), filter.getNrOfColumns()};
        int[] oDim = new int[]{output.getNrOfRows(), output.getNrOfColumns()};

        clEnqueueWriteBuffer(commandQueue, input.getCLReadMem(), CL_TRUE, 0, input.getSize()
                * Sizeof.cl_float, input.getCLPointer(), 0, null, null);
        clEnqueueWriteBuffer(commandQueue, filter.getCLReadMem(), CL_TRUE, 0, filter.getSize()
                * Sizeof.cl_float, filter.getCLPointer(), 0, null, null);
        clEnqueueWriteBuffer(commandQueue, output.getCLReadWriteMem(), CL_TRUE, 0, output.getSize()
                * Sizeof.cl_float, output.getCLPointer(), 0, null, null);

        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(input.getCLReadMem()));
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(filter.getCLReadMem()));
        clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(output.getCLReadWriteMem()));
        clSetKernelArg(kernel, 3, Sizeof.cl_int2, Pointer.to(iDim));
        clSetKernelArg(kernel, 4, Sizeof.cl_int2, Pointer.to(fDim));
        clSetKernelArg(kernel, 5, Sizeof.cl_int2, Pointer.to(oDim));

        long globalSize[] = new long[2];
        // globalSize[0] = round(fDim[0],output.getNrOfColumns());
        // globalSize[1] = round(fDim[1],output.getNrOfRows());
        globalSize[0] = output.getNrOfColumns();
        globalSize[1] = output.getNrOfRows();

        long localSize[] = new long[]{4,4};
        clEnqueueNDRangeKernel(
                commandQueue,
                kernel,
                2,
                null,
                globalSize,
                localSize,
                0,
                null,
                null);

        clEnqueueReadBuffer(commandQueue, output.getCLReadWriteMem(), CL_TRUE, 0, output.getSize()
                * Sizeof.cl_float, Pointer.to(output.getRawData()), 0, null, null);
    }

    private static long round(long groupSize, long globalSize) {
        long r = globalSize % groupSize;
        return r == 0 ? globalSize : globalSize + groupSize - r;
    }
}
