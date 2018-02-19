package dae.matrix.blas;

import dae.matrix.fmatrix;
import dae.matrix.imatrix;
import java.nio.FloatBuffer;
import org.jocl.CL;
import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_MAX_COMPUTE_UNITS;
import static org.jocl.CL.CL_DEVICE_NAME;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
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
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clWaitForEvents;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import static org.jocl.blast.CLBlast.CLBlastSgemm;
import static org.jocl.blast.CLBlastLayout.CLBlastLayoutColMajor;
import static org.jocl.blast.CLBlastLayout.CLBlastLayoutRowMajor;
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
public class OCLBlas {

    private static cl_context context;
    private static cl_command_queue commandQueue;

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

    public static void sgemm(float alpha, imatrix A, imatrix B, float beta, imatrix C) {
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

        // Wait for the computation to be finished
        // XXX CLBlast does not set the event properly.
        // This would cause a CL_INVALID_EVENT error
        // clWaitForEvents( 1, new cl_event[] { event });
        // Copy the result data back to the host
        clEnqueueReadBuffer(commandQueue, memC, CL_TRUE, 0, M * N
                * Sizeof.cl_float, C.getCLPointer(), 0, null, null);

        // Clean up
        // clReleaseMemObject(memA);
        // clReleaseMemObject(memB);
        //clReleaseMemObject(memC);
        // clReleaseCommandQueue(commandQueue);
        // clReleaseContext(context);
    }

    public static void cleanup() {
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }

    public static cl_mem createReadMem(imatrix matrix) {
        cl_mem mem = clCreateBuffer(context, CL_MEM_READ_ONLY, matrix.getNrOfRows() * matrix.getNrOfColumns()
                * Sizeof.cl_float, null, null);
        return mem;
    }

    public static cl_mem createReadWriteMem(imatrix matrix) {
        cl_mem mem = clCreateBuffer(context, CL_MEM_READ_WRITE, matrix.getNrOfRows() * matrix.getNrOfColumns()
                * Sizeof.cl_float, null, null);
        return mem;
    }
}
