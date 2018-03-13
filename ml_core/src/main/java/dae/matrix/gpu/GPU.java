/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.imatrix;
import dae.matrix.integer.intmatrix;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.jocl.CL;
import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_MAX_WORK_ITEM_SIZES;
import static org.jocl.CL.CL_DEVICE_NAME;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clEnqueueCopyBuffer;
import static org.jocl.CL.clEnqueueCopyBufferRect;
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
        KERNEL_MATRIX_OP.init(CL_CONTEXT, CL_COMMAND_QUEUE);

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
        FloatDeviceBuffer mDB = m.getDeviceBuffer();
        zeroFill(mDB.getRMem(), mDB.getDeviceSize());
    }

    public static final void zeroFillRW(imatrix m) {
        FloatDeviceBuffer mDB = m.getDeviceBuffer();
        zeroFill(mDB.getRWMem(), mDB.getDeviceSize());
    }

    private static void zeroFill(cl_mem buffer, int size) {
        float zero[] = new float[1];
        clEnqueueFillBuffer(CL_COMMAND_QUEUE, buffer, Pointer.to(zero), Float.BYTES, 0, size, 0, null, null);
    }

    /**
     * Writes to a buffer object on the device that is defined as a read only
     * buffer on the device.
     *
     * @param m the matrix to upload into the device.
     * @return the buffer object that is associated with the device buffer.
     */
    public static final cl_mem uploadRMatrix(imatrix m) {
        Logger.getLogger(GPU.class.getName()).log(Level.INFO, "Uploading read only matrix {0} to gpu.", m.getName());
        cl_mem buffer = m.getDeviceBuffer().getRMem();
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
        Logger.getLogger(GPU.class.getName()).log(Level.INFO, "Uploading read write matrix {0} to gpu.", m.getName());
        cl_mem buffer = m.getDeviceBuffer().getRWMem();
        enqueueWriteMatrix(m, buffer);
        return buffer;
    }
    
        /**
     * Writes to a buffer object on the device that is defined as a read only
     * buffer on the device.
     *
     * @param m the matrix to upload into the device.
     * @return the buffer object that is associated with the device buffer.
     */
    public static final cl_mem uploadRMatrix(intmatrix m) {
        Logger.getLogger(GPU.class.getName()).log(Level.INFO, "Uploading read only matrix {0} to gpu.", m.getName());
        cl_mem buffer = m.getDeviceBuffer().getRMem();
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
    public static final cl_mem uploadRWMatrix(intmatrix m) {
        Logger.getLogger(GPU.class.getName()).log(Level.INFO, "Uploading read write matrix {0} to gpu.", m.getName());
        cl_mem buffer = m.getDeviceBuffer().getRWMem();
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
        FloatDeviceBuffer mdb = m.getDeviceBuffer();
        clEnqueueWriteBufferRect(CL_COMMAND_QUEUE, deviceBuffer, CL_TRUE,
                mdb.getDeviceOffset(),
                mdb.getHostOffset(),
                mdb.getHostRegion(),
                mdb.getDeviceRowPitch(), mdb.getDeviceSlicePitch(),
                mdb.getHostRowPitch(), mdb.getHostSlicePitch(),
                mdb.getCLPointer(),
                0,
                null,
                null);
    }
    
     /**
     * Writes the matrix m into the provided device buffer on the device.
     *
     * @param m the matrix m to upload into the device.
     * @param deviceBuffer the buffer object that is associated with the buffer
     * on the device.
     * @return
     */
    private static void enqueueWriteMatrix(intmatrix m, cl_mem deviceBuffer) {
        IntDeviceBuffer mdb = m.getDeviceBuffer();
        clEnqueueWriteBufferRect(CL_COMMAND_QUEUE, deviceBuffer, CL_TRUE,
                mdb.getDeviceOffset(),
                mdb.getHostOffset(),
                mdb.getHostRegion(),
                mdb.getDeviceRowPitch(), mdb.getDeviceSlicePitch(),
                mdb.getHostRowPitch(), mdb.getHostSlicePitch(),
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
        Logger.getLogger(GPU.class.getName()).log(Level.INFO, "Downloading read only matrix {0} from gpu.", m.getName());
        cl_mem memOutput = m.getDeviceBuffer().getRMem();
        enqueueReadMatrix(m, memOutput);
    }

    /**
     * Downloads a padded matrix that was used as a read write buffer into the
     * matrix m.
     *
     * @param m the padded matrix to download from the device
     */
    public static final void downloadRWMatrix(imatrix m) {
        Logger.getLogger(GPU.class.getName()).log(Level.INFO, "Downloading read write matrix {0} from gpu.", m.getName());
        cl_mem memOutput = m.getDeviceBuffer().getRWMem();
        enqueueReadMatrix(m, memOutput);
    }

    /**
     * Downloads a padded matrix from the device that was used as a read only
     * buffer into the matrix m.
     *
     * @param m the padded matrix to download from the device
     */
    public static final void downloadRMatrix(intmatrix m) {
        cl_mem memOutput = m.getDeviceBuffer().getRMem();
        enqueueReadMatrix(m, memOutput);
    }

    /**
     * Downloads a padded matrix that was used as a read write buffer into the
     * matrix m.
     *
     * @param m the padded matrix to download from the device
     */
    public static final void downloadRWMatrix(intmatrix m) {
        cl_mem memOutput = m.getDeviceBuffer().getRWMem();
        enqueueReadMatrix(m, memOutput);
    }

    /**
     * Reads a matrix from the referenced device buffer object.
     *
     * @param m the matrix to read the buffer into.
     * @param deviceBuffer the device buffer to read the data from.
     */
    public static void enqueueReadMatrix(imatrix m, cl_mem deviceBuffer) {
        FloatDeviceBuffer db = m.getDeviceBuffer();
        clEnqueueReadBufferRect(CL_COMMAND_QUEUE, deviceBuffer, CL_TRUE,
                db.getDeviceOffset(),
                db.getHostOffset(),
                db.getHostRegion(),
                db.getDeviceRowPitch(), db.getDeviceSlicePitch(),
                db.getHostRowPitch(), db.getHostSlicePitch(),
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
        IntDeviceBuffer db = m.getDeviceBuffer();
        clEnqueueReadBufferRect(CL_COMMAND_QUEUE, deviceBuffer, CL_TRUE,
                db.getDeviceOffset(),
                db.getHostOffset(),
                db.getHostRegion(),
                db.getDeviceRowPitch(), db.getDeviceSlicePitch(),
                db.getHostRowPitch(), db.getHostSlicePitch(),
                db.getCLPointer(),
                0,
                null,
                null);
    }

    static void copyRWBuffer(imatrix cpuBuffer) {
        FloatDeviceBuffer db = cpuBuffer.getDeviceBuffer();
        clEnqueueCopyBuffer(CL_COMMAND_QUEUE,
                db.getRWMem(),
                db.getRMem(),
                0, 0,
                db.getDeviceSize(), 0, null, null);
    }
    
    static void copyRWBuffer(intmatrix cpuBuffer){
        IntDeviceBuffer db = cpuBuffer.getDeviceBuffer();
        clEnqueueCopyBuffer(CL_COMMAND_QUEUE,
                db.getRWMem(),
                db.getRMem(),
                0, 0,
                db.getDeviceSize(), 0, null, null);
    }

    public static void enqueueCopyMatrix(imatrix src, imatrix dst) {
        FloatDeviceBuffer dbSrc = src.getDeviceBuffer();
        int srcCols = dbSrc.getDeviceColumns();
        int srcRows = dbSrc.getDeviceRows();

        FloatDeviceBuffer dbDst = dst.getDeviceBuffer();
        int dstCols = dbDst.getDeviceColumns();
        int dstRows = dbDst.getDeviceRows();

        int numRows = Math.min(src.getNrOfRows(), dst.getNrOfRows());
        int numCols = Math.min(src.getNrOfColumns(), dst.getNrOfColumns());

        int srcSlices = src.getNrOfSlices() * src.getNrOfHyperSlices();
        int dstSlices = dst.getNrOfSlices() * dst.getNrOfHyperSlices();
        int numSlices = Math.min(srcSlices, dstSlices);

        dbSrc.uploadRMatrix();
        dbDst.uploadRMatrix();

        long region[] = new long[]{numRows * Float.BYTES, numCols, numSlices};

        if (srcCols != dstCols
                || srcRows != dstRows
                || srcSlices != dstSlices
                || src.getZeroPadding() != dst.getZeroPadding()) {
            clEnqueueCopyBufferRect(CL_COMMAND_QUEUE,
                    dbSrc.getRMem(),
                    dbDst.getRMem(),
                    dbSrc.getDeviceOffset(),
                    dbDst.getDeviceOffset(),
                    region,
                    dbSrc.getDeviceRowPitch(), dbSrc.getDeviceSlicePitch(),
                    dbDst.getDeviceRowPitch(), dbDst.getDeviceSlicePitch(),
                    0, null, null);
        } else {
            clEnqueueCopyBuffer(CL_COMMAND_QUEUE,
                    dbSrc.getRMem(),
                    dbDst.getRMem(),
                    0, 0,
                    dbSrc.getDeviceSize(), 0, null, null
            );
        }
        dbDst.markRMatrixAsMaster();
    }
}
