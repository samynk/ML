/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.imatrix;
import dae.matrix.integer.intmatrix;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import static org.jocl.CL.*;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_program;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class OpenCLKernel {
    public static final int DEFAULTWORKSIZE = 1024;
    
    
    private final String kernelFile;
    private final String kernelSource;

    protected cl_command_queue commandQueue;
    protected cl_context context;
    private cl_program program;
    
    private final long[] eGWS = new long[1];
    private final long[] eLWS = new long[1];
    private final long[] localWorkSize = new long[]{DEFAULTWORKSIZE};

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
    }
    
    /**
     * Returns the kernel file location.
     * @return a String object with the location of the kernel file.
     */
    public String getKernelFile(){
        return kernelFile;
    }

    /**
     * Initializes the kernel
     *
     * @param context the opencl context.
     * @param commandQueue the command queue to use.
     *
     */
    public void init(cl_context context, cl_command_queue commandQueue) {
        this.context = context;
        this.commandQueue = commandQueue;
        program = clCreateProgramWithSource(context, 1, new String[]{kernelSource}, null, null);
        String compileOptions = "-cl-mad-enable -cl-std=CL2.0";
        clBuildProgram(program, 0, null, compileOptions, null, null);
    }



    /**
     * Creates a kernel for a specific method in the source file.
     *
     * @param method the method to compile.
     * @return the created kernel.
     */
    public final cl_kernel createKernel(String method) {
        return clCreateKernel(program, method, null);
    }

    /**
     * Release the program.
     */
    public void releaseProgram() {
        clReleaseProgram(program);
    }

    public void destroy() {

    }

    protected void applyKernel(cl_kernel kernel, imatrix O) {
        FloatDeviceBuffer db = O.getDeviceBuffer();
        cl_mem memOutput = db.upload();
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memOutput));
        long workSize = db.getGlobalWorkSize()[0];
        long localSize = localWorkSize[0];
        // divide worksize by 4 because activation kernels use float4
        // to speed up operations.
        eGWS[0] = workSize / 4;
        if (eGWS[0] < localSize) {
            localSize = eGWS[0];
        } else if (eGWS[0] % localSize != 0) {
            do {
                localSize >>= 1;
            } while (eGWS[0] % localSize != 0);
        }
        eLWS[0] = localSize;
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, eGWS, eLWS, 0, null, null);
        db.markGpuAsMaster();
    }
    
    protected void applyKernel(cl_kernel kernel, intmatrix O) {
        IntDeviceBuffer db = O.getDeviceBuffer();
        cl_mem memOutput = db.uploadRWMatrix();
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memOutput));
        long workSize = db.getGlobalWorkSize()[0];
        long localSize = localWorkSize[0];
        // divide worksize by 4 because activation kernels use float4
        // to speed up operations.
        eGWS[0] = workSize / 4;
        if (eGWS[0] < localSize) {
            localSize = eGWS[0];
        } else if (eGWS[0] % localSize != 0) {
            do {
                localSize >>= 1;
            } while (eGWS[0] % localSize != 0);
        }
        eLWS[0] = localSize;
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, eGWS, eLWS, 0, null, null);
        db.markRWMatrixAsMaster();
    }

    protected void applyKernel(cl_kernel kernel, imatrix O, float kernelArg1) {
        clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[]{kernelArg1}));
        applyKernel(kernel, O);
    }
}
