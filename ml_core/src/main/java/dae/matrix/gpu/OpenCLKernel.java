/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.imatrix;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import static org.jocl.CL.*;
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

    private final String kernelFile;
    private final String kernelSource;

    protected cl_command_queue commandQueue;
    protected cl_context context;
    private cl_program program;

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
        String compileOptions = "-cl-mad-enable";
        clBuildProgram(program, 0, null, compileOptions, null, null);
    }

    /**
     * Writes to a buffer object on the device that is defined as a read only
     * buffer on the device.
     *
     * @param m the matrix to upload into the device.
     * @return the buffer object that is associated with the device buffer.
     */
    protected final cl_mem uploadRMatrix(imatrix m) {
        cl_mem buffer = m.getCLReadMem();
        enqueueWriteMatrix(m, buffer);
        return buffer;
    }

    /**
     * Writes to a buffer object on the device that is defined as a read/write
     * buffer on the device.
     *
     * @param m the matrix to upload into the device.
     * @return the buffer object that is associated with the device buffer.
     */
    protected final cl_mem uploadRWMatrix(imatrix m) {
        cl_mem buffer = m.getCLReadWriteMem();
        enqueueWriteMatrix(m, buffer);
        return buffer;
    }

    /**
     * Writes the matrix m into the provided device buffer on the device.
     *
     * @param m the matrix m to upload into the device.
     * @param deviceBuffer the buffer object that is associated with the
     * buffer on the device.
     * @return
     */
    private void enqueueWriteMatrix(imatrix m, cl_mem deviceBuffer) {
        int deviceCols = m.getDeviceColumns();
        int deviceRows = m.getDeviceRows();

        long region[] = new long[]{m.getNrOfColumns() * Float.BYTES, m.getNrOfRows(), m.getNrOfSlices()};
        clEnqueueWriteBufferRect(commandQueue, deviceBuffer, CL_TRUE,
                new long[]{0, 0, 0},
                new long[]{0, 0, 0},
                region,
                // device
                deviceCols * Float.BYTES, deviceCols * deviceRows * Float.BYTES,
                // host
                m.getNrOfColumns() * Float.BYTES, m.getSliceSize() * Float.BYTES,
                m.getCLPointer(),
                0,
                null,
                null);
    }

    /**
     * Downloads a padded matrix from the device that was used as a read only
     * buffer into the matrix m.
     *
     * @param m the padded matrix to download from the device
     */
    protected final void downloadRMatrix(imatrix m) {
        cl_mem memOutput = m.getCLReadMem();
        enqueueReadMatrix(m, memOutput);
    }

    /**
     * Downloads a padded matrix that was used as a read write buffer into the
     * matrix m.
     *
     * @param m the padded matrix to download from the device
     */
    protected final void downloadRWMatrix(imatrix m) {
        cl_mem memOutput = m.getCLReadWriteMem();
        enqueueReadMatrix(m, memOutput);
    }

    /**
     * Reads a matrix from the referenced device buffer object.
     *
     * @param m the matrix to read the buffer into.
     * @param deviceBuffer the device buffer to read the data from.
     */
    private void enqueueReadMatrix(imatrix m, cl_mem deviceBuffer) {
        int deviceCols = m.getDeviceColumns();
        int deviceRows = m.getDeviceRows();

        long region[] = new long[]{m.getNrOfColumns() * Float.BYTES, m.getNrOfRows(), m.getNrOfSlices()};
        clEnqueueReadBufferRect(commandQueue, deviceBuffer, CL_TRUE,
                new long[]{0, 0, 0},
                new long[]{0, 0, 0},
                region,
                // device
                deviceCols * Float.BYTES, deviceCols * deviceRows * Float.BYTES,
                // host
                m.getNrOfColumns() * Float.BYTES, m.getSliceSize() * Float.BYTES,
                m.getCLPointer(),
                0,
                null,
                null);
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
}
