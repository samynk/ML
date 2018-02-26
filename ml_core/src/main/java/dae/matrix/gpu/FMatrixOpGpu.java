package dae.matrix.gpu;

import dae.matrix.imatrix;
import dae.matrix.op.FMatrixOp;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.jocl.CL;
import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_MAX_WORK_ITEM_SIZES;
import static org.jocl.CL.CL_DEVICE_NAME;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clEnqueueWriteBuffer;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetDeviceInfo;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import static org.jocl.blast.CLBlast.CLBlastSgemm;
import static org.jocl.blast.CLBlastLayout.CLBlastLayoutColMajor;
import static org.jocl.blast.CLBlastTranspose.*;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_event;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class FMatrixOpGpu implements FMatrixOp {

    private static final cl_context context;
    private static final cl_command_queue commandQueue;

    private static final ConvolvKernel convolvKernel;
    private static final ActivationKernel activationKernel;
    private static long maxWorkItemSizes[];

    static {
        // The platform, device type and device number
        // that will be used
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_ALL;
        final int deviceIndex = 0;

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        // Create a context for the selected device
        context = clCreateContext(contextProperties, 1,
                new cl_device_id[]{device}, null, null, null);

        String deviceName = getString(device, CL_DEVICE_NAME);

        System.out.printf("CL_DEVICE_NAME: %s\n", deviceName);

        // Create a command-queue
        commandQueue = clCreateCommandQueue(context, device, 0, null);

        convolvKernel = new ConvolvKernel("/kernels/convolve.kernel");
        convolvKernel.init(context, commandQueue);

        activationKernel = new ActivationKernel();
        activationKernel.init(context, commandQueue);

        // CL_DEVICE_MAX_WORK_ITEM_SIZES
        maxWorkItemSizes = getSizes(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);

    }

    private static int getInt(cl_device_id device, int paramName) {
        long size[] = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        byte buffer[] = new byte[(int) size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer),
                null);

        return buffer[0];
    }

    private static String getString(cl_device_id device, int paramName) {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int) size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer),
                null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length - 1);
    }

    /**
     * Calculates the following product : alpha A * B + beta * C, where A*B is a
     * matrix multiplication. The result is stored in C.
     *
     * @param alpha a float value that defines the alpha value.
     * @param A the matrix A.
     * @param B the matrix B.
     * @param beta a float value that defines the beta value.
     * @param C the matrix C, where the result will be stored.
     */
    @Override
    public void sgemm(float alpha, imatrix A, imatrix B, float beta, imatrix C) {
        // Create the device input buffers
        int M = A.getNrOfRows();
        int K = B.getNrOfRows();
        int N = C.getNrOfColumns();

        cl_mem memA = A.getCLReadMem();
        cl_mem memB = B.getCLReadMem();
        cl_mem memC = C.getCLReadWriteMem();

        // Copy the host data to the device
        clEnqueueWriteBuffer(commandQueue, memA, CL_TRUE, 0, M * K
                * Sizeof.cl_float, A.getCLPointer(), 0, null, null);
        clEnqueueWriteBuffer(commandQueue, memB, CL_TRUE, 0, K * N
                * Sizeof.cl_float, B.getCLPointer(), 0, null, null);
        clEnqueueWriteBuffer(commandQueue, memC, CL_TRUE, 0, M * N
                * Sizeof.cl_float, C.getCLPointer(), 0, null, null);

        int lda = A.getNrOfRows();
        if (A.isTransposed()) {
            lda = A.getNrOfColumns();
        }

        int ldb = B.getNrOfRows();
        if (B.isTransposed()) {
            ldb = B.getNrOfColumns();
        }
        // Execute GEMM:
        // C = alpha * A * B + beta * C
        cl_event event = new cl_event();
        CLBlastSgemm(CLBlastLayoutColMajor,
                A.isTransposed() ? CLBlastTransposeYes : CLBlastTransposeNo,
                B.isTransposed() ? CLBlastTransposeYes : CLBlastTransposeNo,
                M, N, K,
                alpha,
                memA, 0, lda,
                memB, 0, ldb,
                beta,
                memC, 0, M,
                commandQueue, event);

        clEnqueueReadBuffer(commandQueue, memC, CL_TRUE, 0, M * N
                * Sizeof.cl_float, C.getCLPointer(), 0, null, null);
    }

    /**
     * Applies a convolution filter on the input matrix.
     *
     * @param input the matrix to convolve.
     * @param filter the filter to apply.
     * @param stride the stride with which to advance the filter.
     * @param output the matrix where the output is stored.
     */
    @Override
    public void convolve(imatrix input, imatrix filter, int stride, imatrix output) {
        convolvKernel.convolv(input, filter, output);
    }

    /**
     * Applies a convolution filter on the input matrix, with the slices taken
     * into account.
     *
     * @param input the matrix to convolve.
     * @param filter the filter to apply.
     * @param stride the stride with which to advance the filter.
     * @param output the matrix where the output is stored.
     */
    @Override
    public void batchConvolve(imatrix input, imatrix filter, int stride, imatrix output) {
        convolvKernel.batchConvolv(input, filter, stride, output);
    }

    /**
     * Applies a correlation filter on the input matrix, with the slices taken
     * into account.
     *
     * @param input the matrix to convolve.
     * @param filter the filter to apply.
     * @param stride the stride with which to advance the filter.
     * @param output the matrix where the output is stored.
     */
    @Override
    public void batchCorrelate(imatrix input, imatrix filter, int stride, imatrix output) {
        convolvKernel.batchCorrelate(input, filter, stride, output);
    }

    /**
     * Calculates the sigmoid activation function. The result is stored back
     * into the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    @Override
    public void sigmoid(imatrix O) {
        activationKernel.sigmoid(O);
    }

    /**
     * Calculates the derivative of the sigmoid activation function. The result
     * is stored back into the given matrix.
     *
     * @param O the matrix to apply the sigmoid activation function to.
     */
    @Override
    public void dsigmoid(imatrix O) {
        activationKernel.dsigmoid(O);
    }

    public static void cleanup() {
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }

    public static cl_mem createReadMem(imatrix matrix, int padcol, int padrow) {
        cl_mem mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                (matrix.getNrOfRows() + padrow) * (matrix.getNrOfColumns() + padcol) * matrix.getNrOfSlices()
                * Sizeof.cl_float, null, null);
        return mem;
    }

    public static cl_mem createReadWriteMem(imatrix matrix, int padcol, int padrow) {
        cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                (matrix.getNrOfRows() + padrow) * (matrix.getNrOfColumns() + padcol) * matrix.getNrOfSlices()
                * Sizeof.cl_float, null, null);
        return mem;
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    private static long[] getSizes(cl_device_id device, int paramName, int numValues) {
        // The size of the returned data has to depend on 
        // the size of a size_t, which is handled here
        ByteBuffer buffer = ByteBuffer.allocate(
                numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
        clGetDeviceInfo(device, paramName, Sizeof.size_t * numValues,
                Pointer.to(buffer), null);
        long values[] = new long[numValues];
        if (Sizeof.size_t == 4) {
            for (int i = 0; i < numValues; i++) {
                values[i] = buffer.getInt(i * Sizeof.size_t);
            }
        } else {
            for (int i = 0; i < numValues; i++) {
                values[i] = buffer.getLong(i * Sizeof.size_t);
            }
        }
        return values;
    }
}
