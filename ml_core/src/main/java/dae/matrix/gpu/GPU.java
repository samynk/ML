/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.imatrix;
import dae.matrix.integer.intmatrix;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
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
    public static final FuzzyKernel KERNEL_FUZZY;

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
        cl_platform_id platform = platforms[platforms.length - 1];

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

        KERNEL_FUZZY = new FuzzyKernel();
        KERNEL_FUZZY.init(CL_CONTEXT, CL_COMMAND_QUEUE);

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

    public static final void zeroFill(imatrix m) {
        FloatDeviceBuffer mDB = m.getDeviceBuffer();
        zeroFill(mDB.getMem(), mDB.getDeviceSize());
    }

    public static void fillR(imatrix O, float f) {
        float[] pattern = new float[]{f};
        FloatDeviceBuffer db = O.getDeviceBuffer();

        clEnqueueFillBuffer(CL_COMMAND_QUEUE, O.getDeviceBuffer().getMem(),
                Pointer.to(pattern), Sizeof.cl_float, 0, db.getDeviceSize(), 0, null, null);
    }

    private static void zeroFill(cl_mem buffer, int size) {
        float zero[] = new float[1];
        clEnqueueFillBuffer(CL_COMMAND_QUEUE, buffer, Pointer.to(zero), Sizeof.cl_float, 0, size, 0, null, null);
    }

    /**
     * Writes to a buffer object on the device that is defined as a read only
     * buffer on the device.
     *
     * @param m the matrix to upload into the device.
     * @return the buffer object that is associated with the device buffer.
     */
    public static final cl_mem upload(imatrix m) {
        Logger.getLogger(GPU.class.getName()).log(Level.INFO, "Uploading matrix {0} to gpu.", m.getName());
        cl_mem buffer = m.getDeviceBuffer().getMem();
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
    public static final void download(imatrix m) {
        Logger.getLogger(GPU.class.getName()).log(Level.INFO, "Downloading read only matrix {0} from gpu.", m.getName());
        cl_mem memOutput = m.getDeviceBuffer().getMem();
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

    static void copyBuffer(imatrix cpuBuffer) {
        FloatDeviceBuffer db = cpuBuffer.getDeviceBuffer();
        clEnqueueCopyBuffer(CL_COMMAND_QUEUE,
                db.getMem(),
                db.getMem(),
                0, 0,
                db.getDeviceSize(), 0, null, null);
    }

    static void copyRBuffer(intmatrix cpuBuffer) {
        IntDeviceBuffer db = cpuBuffer.getDeviceBuffer();
        clEnqueueCopyBuffer(CL_COMMAND_QUEUE,
                db.getRMem(),
                db.getRWMem(),
                0, 0,
                db.getDeviceSize(), 0, null, null);
    }

    static void copyRWBuffer(intmatrix cpuBuffer) {
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

        dbSrc.upload();
        dbDst.upload();

        long region[] = new long[]{numRows * Float.BYTES, numCols, numSlices};

        if (srcCols != dstCols
                || srcRows != dstRows
                || srcSlices != dstSlices
                || src.getZeroPadding() != dst.getZeroPadding()) {
            clEnqueueCopyBufferRect(CL_COMMAND_QUEUE,
                    dbSrc.getMem(),
                    dbDst.getMem(),
                    dbSrc.getDeviceOffset(),
                    dbDst.getDeviceOffset(),
                    region,
                    dbSrc.getDeviceRowPitch(), dbSrc.getDeviceSlicePitch(),
                    dbDst.getDeviceRowPitch(), dbDst.getDeviceSlicePitch(),
                    0, null, null);
        } else {
            clEnqueueCopyBuffer(CL_COMMAND_QUEUE,
                    dbSrc.getMem(),
                    dbDst.getMem(),
                    0, 0,
                    dbSrc.getDeviceSize(), 0, null, null
            );
        }
        dbDst.markGpuAsMaster();
    }

    public static void enqueueCopySliceMatrix(imatrix src, imatrix dst) {
        FloatDeviceBuffer dbSrc = src.getDeviceBuffer();
        FloatDeviceBuffer dbDst = dst.getDeviceBuffer();

        if (src.getHyperSliceSize() == dst.getHyperSliceSize()) {
            dbSrc.upload();
            int deviceSize = Math.min(dbSrc.getDeviceSize(), dbDst.getDeviceSize());

            if (dst.getZeroPadding() == 0) {
                clEnqueueCopyBuffer(CL_COMMAND_QUEUE,
                        dbSrc.getMem(),
                        dbDst.getMem(),
                        0, 0,
                        deviceSize, 0, null, null
                );
            } else {
                int numRows = dst.getNrOfRows();
                int numCols = dst.getNrOfColumns();
                int numSlices = dst.getNrOfSlices() * dst.getNrOfHyperSlices();
                long region[] = new long[]{numRows * Sizeof.cl_float, numCols, numSlices};

                clEnqueueCopyBufferRect(CL_COMMAND_QUEUE,
                        dbSrc.getMem(),
                        dbDst.getMem(),
                        dbSrc.getDeviceOffset(),
                        dbDst.getDeviceOffset(),
                        region,
                        numRows * Sizeof.cl_float, numRows * numCols * Sizeof.cl_float,
                        dbDst.getDeviceRowPitch(), dbDst.getDeviceSlicePitch(),
                        0, null, null);
            }
            dbDst.markGpuAsMaster();
        } else if (src.getHyperSliceSize() < dst.getHyperSliceSize()) {
            dbSrc.upload();
            int numSlicesDst = dst.getNrOfHyperSlices();
            int numSlicesSrc = src.getNrOfHyperSlices();
            int numSlices = Math.min(numSlicesDst, numSlicesSrc);
            int hSlicePitch = src.getHyperSliceSize() * Sizeof.cl_float;
            long region[] = new long[]{hSlicePitch, numSlices, 1};
            clEnqueueCopyBufferRect(CL_COMMAND_QUEUE,
                    dbSrc.getMem(),
                    dbDst.getMem(),
                    dbSrc.getDeviceOffset(),
                    dbDst.getDeviceOffset(),
                    region,
                    hSlicePitch, 0,
                    dbDst.getDeviceSlicePitch() * dst.getNrOfSlices(), 0,
                    0, null, null);
            dbDst.markGpuAsMaster();
        } else {
            enqueueCopyMatrix(src, dst);
        }
    }

    public static void zip(imatrix matrix1, imatrix matrix2, imatrix dest) {
        FloatDeviceBuffer dbSrc1 = matrix1.getDeviceBuffer();
        FloatDeviceBuffer dbSrc2 = matrix2.getDeviceBuffer();
        FloatDeviceBuffer dbDst = dest.getDeviceBuffer();

        // copy dbSrc1 with interleave of size of dbSrc2 and offset 0.
        dbSrc1.upload();
        dbSrc2.upload();
        // determin number of hyper slices to copy.
        int numSlicesDst = dest.getNrOfHyperSlices();
        int numSlicesSrc1 = matrix1.getNrOfHyperSlices();
        int numSlicesSrc2 = matrix2.getNrOfHyperSlices();
        int numHSlices = Math.min(Math.min(numSlicesDst, numSlicesSrc1), numSlicesSrc2);

        int slicePitch1 = matrix1.getSliceSize() * Sizeof.cl_float;
        int slicePitch2 = matrix2.getSliceSize() * Sizeof.cl_float;

        // copy first matrix
        long region[] = new long[]{slicePitch1, numHSlices * matrix1.getNrOfSlices(), 1};
        clEnqueueCopyBufferRect(CL_COMMAND_QUEUE,
                dbSrc1.getMem(),
                dbDst.getMem(),
                dbSrc1.getDeviceOffset(),
                dbDst.getDeviceOffset(),
                region,
                slicePitch1, 0,
                slicePitch1 + slicePitch2, 0,
                0, null, null);
        // copy second matrix;
        region[0] = slicePitch2;
        region[1] = numHSlices * matrix2.getNrOfSlices();

        long[] dstOffset = dbDst.getDeviceOffset();
        long[] eOffset = new long[3];

        eOffset[0] = dstOffset[0] + slicePitch1;
        eOffset[1] = dstOffset[1];
        eOffset[2] = dstOffset[2];

        clEnqueueCopyBufferRect(CL_COMMAND_QUEUE,
                dbSrc2.getMem(),
                dbDst.getMem(),
                dbSrc1.getDeviceOffset(),
                eOffset,
                region,
                slicePitch2, 0,
                slicePitch1 + slicePitch2, 0,
                0, null, null);
        dbDst.markGpuAsMaster();
    }

    public static void zip(List<imatrix> srcMatrices, imatrix dest) {
        int hSlices = dest.getNrOfHyperSlices();
        int rows = dest.getNrOfRows();
        int columns = dest.getNrOfColumns();
        int slices = Integer.MAX_VALUE;
        for (imatrix im : srcMatrices) {
            if (im.getNrOfHyperSlices() < hSlices) {
                hSlices = im.getNrOfHyperSlices();
            }
            if (im.getNrOfRows() < rows) {
                rows = im.getNrOfRows();
            }
            if (im.getNrOfColumns() < columns) {
                columns = im.getNrOfColumns();
            }
            if (im.getNrOfSlices() < slices) {
                slices = im.getNrOfSlices();
            }
        }

        FloatDeviceBuffer dbDst = dest.getDeviceBuffer();
        long region[] = new long[3];
        long[] dstOffset = dbDst.getDeviceOffset();
        long[] eOffset = new long[3];

        eOffset[0] = dstOffset[0];
        eOffset[1] = dstOffset[1];
        eOffset[2] = dstOffset[2];

        int destSlicePitch = dest.getSliceSize() * Sizeof.cl_float;
        for (int mi = 0; mi < srcMatrices.size(); ++mi) {
            imatrix current = srcMatrices.get(mi);

            FloatDeviceBuffer dbSrc = current.getDeviceBuffer();
            dbSrc.upload();
            int slicePitch = current.getSliceSize() * Sizeof.cl_float;
            region[0] = slicePitch;
            region[1] = hSlices * slices;
            region[2] = 1;

            clEnqueueCopyBufferRect(CL_COMMAND_QUEUE,
                    dbSrc.getMem(),
                    dbDst.getMem(),
                    dbSrc.getDeviceOffset(),
                    eOffset,
                    region,
                    slicePitch, 0,
                    destSlicePitch, 0,
                    0, null, null);

            eOffset[0] += slicePitch;
        }
        dbDst.markGpuAsMaster();
    }

    /**
     * Unzips the src matrix into two destination matrices per slice. The even
     * slices will be copied into the first destination matrix, the uneven
     * slices will be copied into the second destination matrix.
     *
     * @param src the source matrix.
     * @param dest1 the first destination matrix.
     * @param dest2 the second destination matrix.
     */
    public static void unzip(imatrix src, imatrix dest1, imatrix dest2) {
        FloatDeviceBuffer dbDest1 = dest1.getDeviceBuffer();
        FloatDeviceBuffer dbDest2 = dest2.getDeviceBuffer();
        FloatDeviceBuffer dbSrc = src.getDeviceBuffer();

        // copy dbSrc1 with interleave of size of dbSrc2 and offset 0.
        dbDest1.upload();
        dbDest2.upload();
        // determin number of hyper slices to copy.
        int numSlicesSrc = src.getNrOfHyperSlices();
        int numSlicesDst1 = dest1.getNrOfHyperSlices();
        int numSlicesDst2 = dest2.getNrOfHyperSlices();
        int numHSlices = Math.min(Math.min(numSlicesSrc, numSlicesDst1), numSlicesDst2);

        int slicePitch1 = dest1.getSliceSize() * Sizeof.cl_float;
        int slicePitch2 = dest2.getSliceSize() * Sizeof.cl_float;

        // copy first matrix
        long region[] = new long[]{slicePitch1, numHSlices * dest1.getNrOfSlices(), 1};
        clEnqueueCopyBufferRect(CL_COMMAND_QUEUE,
                dbSrc.getMem(),
                dbDest1.getMem(),
                dbSrc.getDeviceOffset(),
                dbDest1.getDeviceOffset(),
                region,
                slicePitch1 + slicePitch2, 0,
                slicePitch1, 0,
                0, null, null);
        // copy second matrix;
        region[0] = slicePitch2;
        region[1] = numHSlices * dest2.getNrOfSlices();

        long[] dstOffset = dbSrc.getDeviceOffset();
        long[] eOffset = new long[3];

        eOffset[0] = dstOffset[0] + slicePitch1;
        eOffset[1] = dstOffset[1];
        eOffset[2] = dstOffset[2];

        clEnqueueCopyBufferRect(CL_COMMAND_QUEUE,
                dbSrc.getMem(),
                dbDest2.getMem(),
                eOffset,
                dbDest1.getDeviceOffset(),
                region,
                slicePitch1 + slicePitch2, 0,
                slicePitch2, 0,
                0, null, null);
        dbDest1.markGpuAsMaster();
        dbDest2.markGpuAsMaster();
    }

    /**
     * Unzips the src matrix into two destination matrices per slice. The slices
     * will be distributed over the destination matrices.
     *
     * @param src the source matrix.
     * @param dst the list of matrix to unzip the errors into.
     */
    static void unzip(imatrix src, List<imatrix> dstMatrices) {
        int hSlices = src.getNrOfHyperSlices();
        int rows = src.getNrOfRows();
        int columns = src.getNrOfColumns();
        int slices = Integer.MAX_VALUE;
        for (imatrix im : dstMatrices) {
            if (im.getNrOfHyperSlices() < hSlices) {
                hSlices = im.getNrOfHyperSlices();
            }
            if (im.getNrOfRows() < rows) {
                rows = im.getNrOfRows();
            }
            if (im.getNrOfColumns() < columns) {
                columns = im.getNrOfColumns();
            }
            if (im.getNrOfSlices() < slices) {
                slices = im.getNrOfSlices();
            }
        }

        FloatDeviceBuffer dbSrc = src.getDeviceBuffer();
        dbSrc.upload();
        long region[] = new long[3];

        long[] dstOffset = dbSrc.getDeviceOffset();
        long[] eOffset = new long[3];

        eOffset[0] = dstOffset[0];
        eOffset[1] = dstOffset[1];
        eOffset[2] = dstOffset[2];

        int srcSlicePitch = src.getSliceSize() * Sizeof.cl_float;

        for (int mi = 0; mi < dstMatrices.size(); ++mi) {
            imatrix current = dstMatrices.get(mi);

            FloatDeviceBuffer dbDst = current.getDeviceBuffer();
            int slicePitch = current.getSliceSize() * Sizeof.cl_float;

            region[0] = slicePitch;
            region[1] = slices * hSlices;
            region[2] = 1;

            clEnqueueCopyBufferRect(CL_COMMAND_QUEUE,
                    dbSrc.getMem(),
                    dbDst.getMem(),
                    eOffset,
                    dbDst.getDeviceOffset(),
                    region,
                    srcSlicePitch, 0,
                    slicePitch, 0,
                    0, null, null);
            eOffset[0] += slicePitch;
            dbDst.markGpuAsMaster();
        }

    }

}
