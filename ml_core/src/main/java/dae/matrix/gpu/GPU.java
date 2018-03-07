/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.imatrix;
import dae.matrix.integer.intmatrix;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.jocl.CL;
import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_MAX_WORK_ITEM_SIZES;
import static org.jocl.CL.CL_DEVICE_NAME;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clEnqueueFillBuffer;
import static org.jocl.CL.clEnqueueReadBufferRect;
import static org.jocl.CL.clEnqueueWriteBufferRect;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetDeviceInfo;
import static org.jocl.CL.clGetPlatformIDs;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class GPU {

    protected static final cl_context CL_CONTEXT;
    protected static final cl_command_queue CL_COMMAND_QUEUE;

    public static final ConvolvKernel KERNEL_CONVOLV;
    public static final ActivationKernel KERNEL_ACTIVATION;
    public static final PoolKernel KERNEL_POOL;
    public static final MatrixOpKernel KERNEL_MATRIX_OP;

    private static final long maxWorkItemSizes[];

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
        CL_CONTEXT = clCreateContext(contextProperties, 1,
                new cl_device_id[]{device}, null, null, null);

        String deviceName = getString(device, CL_DEVICE_NAME);

        System.out.printf("CL_DEVICE_NAME: %s\n", deviceName);

        // Create a command-queue
        CL_COMMAND_QUEUE = clCreateCommandQueue(CL_CONTEXT, device, 0, null);

        KERNEL_CONVOLV = new ConvolvKernel("/kernels/convolve.kernel");
        KERNEL_CONVOLV.init(CL_CONTEXT, CL_COMMAND_QUEUE);

        KERNEL_ACTIVATION = new ActivationKernel();
        KERNEL_ACTIVATION.init(CL_CONTEXT, CL_COMMAND_QUEUE);

        KERNEL_POOL = new PoolKernel();
        KERNEL_POOL.init(CL_CONTEXT, CL_COMMAND_QUEUE);
        
        KERNEL_MATRIX_OP = new MatrixOpKernel();
        KERNEL_MATRIX_OP.init(CL_CONTEXT,CL_COMMAND_QUEUE);

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

    public static final void zeroFillR(imatrix m) {
        DeviceBuffer mDB = m.getDeviceBuffer();
        int deviceRows = mDB.getDeviceRows();
        int deviceCols = mDB.getDeviceColumns();
        zeroFill(mDB.getCLReadMem(), deviceRows * deviceCols * m.getNrOfSlices());
    }

    public static final void zeroFillRW(imatrix m) {
        DeviceBuffer mDB = m.getDeviceBuffer();
        int deviceRows = mDB.getDeviceRows();
        int deviceCols = mDB.getDeviceColumns();
        zeroFill(mDB.getCLReadWriteMem(), deviceRows * deviceCols * m.getNrOfSlices());
    }

    private static void zeroFill(cl_mem buffer, int size) {
        float zero[] = new float[1];
        clEnqueueFillBuffer(CL_COMMAND_QUEUE, buffer, Pointer.to(zero), Float.BYTES, 0, size * Float.BYTES, 0, null, null);
    }

    /**
     * Writes to a buffer object on the device that is defined as a read only
     * buffer on the device.
     *
     * @param m the matrix to upload into the device.
     * @return the buffer object that is associated with the device buffer.
     */
    public static final cl_mem uploadRMatrix(imatrix m) {
        cl_mem buffer = m.getDeviceBuffer().getCLReadMem();
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
    public static final cl_mem uploadRWMatrix(imatrix m) {
        cl_mem buffer = m.getDeviceBuffer().getCLReadWriteMem();
        enqueueWriteMatrix(m, buffer);
        return buffer;
    }

    /**
     * Writes the matrix m into the provided device buffer on the device.
     *
     * @param m the matrix m to upload into the device.
     * @param deviceBuffer the buffer object that is associated with the buffer
     * on the device.
     * @return
     */
    private static void enqueueWriteMatrix(imatrix m, cl_mem deviceBuffer) {
        DeviceBuffer mdb = m.getDeviceBuffer();
        int deviceCols = mdb.getDeviceColumns();
        int deviceRows = mdb.getDeviceRows();

        long region[] = new long[]{m.getNrOfRows() * Float.BYTES, m.getNrOfColumns(), m.getNrOfSlices()};
        long hostOffset[] = new long[]{m.getZeroPadding() * Float.BYTES, m.getZeroPadding(), 0};
        clEnqueueWriteBufferRect(CL_COMMAND_QUEUE, deviceBuffer, CL_TRUE,
                hostOffset,
                new long[]{0, 0, 0},
                region,
                // device
                deviceRows * Float.BYTES, deviceCols * deviceRows * Float.BYTES,
                // host
                m.getNrOfRows() * Float.BYTES, m.getSliceSize() * Float.BYTES,
                mdb.getCLPointer(),
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
    public static final void downloadRMatrix(imatrix m) {
        cl_mem memOutput = m.getDeviceBuffer().getCLReadMem();
        enqueueReadMatrix(m, memOutput);
    }

    /**
     * Downloads a padded matrix that was used as a read write buffer into the
     * matrix m.
     *
     * @param m the padded matrix to download from the device
     */
    public static final void downloadRWMatrix(imatrix m) {
        cl_mem memOutput = m.getDeviceBuffer().getCLReadWriteMem();
        enqueueReadMatrix(m, memOutput);
    }

    /**
     * Downloads a padded matrix from the device that was used as a read only
     * buffer into the matrix m.
     *
     * @param m the padded matrix to download from the device
     */
    public static final void downloadRMatrix(intmatrix m) {
        cl_mem memOutput = m.getCLReadMem();
        enqueueReadMatrix(m, memOutput);
    }

    /**
     * Downloads a padded matrix that was used as a read write buffer into the
     * matrix m.
     *
     * @param m the padded matrix to download from the device
     */
    public static final void downloadRWMatrix(intmatrix m) {
        cl_mem memOutput = m.getCLReadWriteMem();
        enqueueReadMatrix(m, memOutput);
    }

    /**
     * Reads a matrix from the referenced device buffer object.
     *
     * @param m the matrix to read the buffer into.
     * @param deviceBuffer the device buffer to read the data from.
     */
    public static void enqueueReadMatrix(imatrix m, cl_mem deviceBuffer) {
        DeviceBuffer db = m.getDeviceBuffer();
        int deviceCols = db.getDeviceColumns();
        int deviceRows = db.getDeviceRows();

        long region[] = new long[]{m.getNrOfRows() * Float.BYTES, m.getNrOfColumns(), m.getNrOfSlices()};
        long bufferOffset[] = new long[]{m.getZeroPadding()*Float.BYTES, m.getZeroPadding(), 0};
        clEnqueueReadBufferRect(CL_COMMAND_QUEUE, deviceBuffer, CL_TRUE,
                bufferOffset,
                new long[]{0, 0, 0},
                region,
                // device, bufferRowPitch
                deviceRows * Float.BYTES, deviceCols * deviceRows * Float.BYTES,
                // host
                m.getNrOfRows() * Float.BYTES, m.getSliceSize() * Float.BYTES,
                db.getCLPointer(),
                0,
                null,
                null);
    }

    /**
     * Reads a matrix from the referenced device buffer object.
     *
     * @param m the matrix to read the buffer into.
     * @param deviceBuffer the device buffer to read the data from.
     */
    private static void enqueueReadMatrix(intmatrix m, cl_mem deviceBuffer) {
        int deviceCols = m.getDeviceColumns();
        int deviceRows = m.getDeviceRows();

        long region[] = new long[]{m.getNrOfRows() * Integer.BYTES, m.getNrOfColumns(), m.getNrOfSlices()};
        clEnqueueReadBufferRect(CL_COMMAND_QUEUE, deviceBuffer, CL_TRUE,
                new long[]{m.getZeroPadding(), m.getZeroPadding(), 0},
                new long[]{0, 0, 0},
                region,
                // device
                deviceCols * Integer.BYTES, deviceCols * deviceRows * Integer.BYTES,
                // host
                m.getNrOfColumns() * Integer.BYTES, m.getSliceSize() * Integer.BYTES,
                m.getCLPointer(),
                0,
                null,
                null);
    }
}
