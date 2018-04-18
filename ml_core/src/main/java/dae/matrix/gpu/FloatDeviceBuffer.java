/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.BufferSyncState;
import dae.matrix.imatrix;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class FloatDeviceBuffer {

    private final imatrix cpuBuffer;

    private cl_mem memPadded;

    private int padding;
    private int deviceSize;
    private final long[] globalWorkSize = new long[1];
    private static final int LOCALSIZE = OpenCLKernel.DEFAULTWORKSIZE;

    private BufferSyncState cpuBufferState = BufferSyncState.UPTODATE;
    private BufferSyncState gpuBufferState = BufferSyncState.OUTOFDATE;

    private final int[] deviceDimension = new int[2];

    private final long[] deviceOffset = new long[3];
    private final long[] hostOffset = new long[3];
    private final long[] hostRegion = new long[3];

    private final int[] dimensions = new int[3];

    public FloatDeviceBuffer(imatrix cpuBuffer) {
        this.cpuBuffer = cpuBuffer;
        initPaddedMem();
    }

    private void initPaddedMem() {
        int zp = cpuBuffer.getZeroPadding();
        deviceDimension[0] = cpuBuffer.getNrOfRows() + 2 * zp;
        deviceDimension[1] = cpuBuffer.getNrOfColumns() + 2 * zp;

        dimensions[0] = cpuBuffer.getNrOfRows() + 2 * zp;
        dimensions[1] = dimensions[0] * (cpuBuffer.getNrOfColumns() + 2 * zp);
        dimensions[2] = dimensions[1] * cpuBuffer.getNrOfSlices();

        int totalSize = dimensions[2] * cpuBuffer.getNrOfHyperSlices();

        padding = LOCALSIZE - (totalSize % LOCALSIZE);
        globalWorkSize[0] = (totalSize + padding);
        deviceSize = (totalSize + padding) * Sizeof.cl_float;

        hostRegion[0] = cpuBuffer.getNrOfRows() * Float.BYTES;
        hostRegion[1] = cpuBuffer.getNrOfColumns();
        hostRegion[2] = cpuBuffer.getNrOfSlices() * cpuBuffer.getNrOfHyperSlices();

        deviceOffset[0] = zp * Float.BYTES;
        deviceOffset[1] = zp;
        deviceOffset[2] = 0;

        hostOffset[0] = 0;
        hostOffset[1] = 0;
        hostOffset[2] = 0;

    }

    /**
     * Gets the region on the host. This is a utility method to facilitate
     * uploads and downloads.
     *
     * @return an array with the host region.
     */
    protected long[] getHostRegion() {
        return hostRegion;
    }

    /**
     * Gets the offset of each slice on the device.
     *
     * @return the offset of each slice on the device.
     */
    protected long[] getDeviceOffset() {
        return deviceOffset;
    }

    /**
     * Gets the offset of each slice on the host.
     *
     * @return the offset of each host on the device.
     */
    protected long[] getHostOffset() {
        return hostOffset;
    }

    protected int getDeviceRowPitch() {
        return deviceDimension[0] * Float.BYTES;
    }

    protected int getDeviceSlicePitch() {
        return deviceDimension[0] * deviceDimension[1] * Float.BYTES;
    }

    protected int getHostRowPitch() {
        return cpuBuffer.getNrOfRows() * Float.BYTES;
    }

    protected int getHostSlicePitch() {
        return cpuBuffer.getSliceSize() * Float.BYTES;
    }

    /**
     * Returns the size in bytes of the buffer on the device.
     *
     * @return the device size.
     */
    public int getDeviceSize() {
        return deviceSize;
    }

    /**
     * Returns the global work size for this buffer.
     *
     * @return the global work size for the buffer.
     */
    public long[] getGlobalWorkSize() {
        return globalWorkSize;
    }

    public cl_mem getMem() {
        if (memPadded == null) {
            memPadded = FMatrixOpGpu.createMem(cpuBuffer, padding, CL_MEM_READ_WRITE);
        }
        return memPadded;
    }

    public cl_mem upload() {
        cl_mem rmem = getMem();

        if (gpuBufferState == BufferSyncState.OUTOFDATE) {
            GPU.upload(cpuBuffer);
            gpuBufferState = BufferSyncState.UPTODATE;
        }
        return rmem;
    }

    public void download() {
        if (cpuBufferState == BufferSyncState.OUTOFDATE) {
            GPU.download(cpuBuffer);
            cpuBufferState = BufferSyncState.UPTODATE;
        }
    }

    public Pointer getCLPointer() {
        return Pointer.to(cpuBuffer.getHostData().array());
    }

    /**
     * Get the number of columns on the device.
     *
     * @return the number of columns on the gpu device.
     */
    public int getDeviceColumns() {
        return deviceDimension[0];
    }

    /**
     * Get the number of rows on the device.
     *
     * @return the number of rows on the gpu device.
     */
    public int getDeviceRows() {
        return deviceDimension[1];
    }

    /**
     * Returns the dimension of the device as an int array where the first
     * element is the number of columns and the second element is the number of
     * rows.
     *
     * @return the device dimension as an int array.
     */
    public int[] getDeviceDimension() {
        return deviceDimension;
    }

    /**
     * Returns an array of 3 elements with the size of the different dimensions.
     * the first element is the number of rows. the second element is the size
     * of one slice (rows*columns) the third element is the size of one hyper
     * slice (rows*columns*slices).
     *
     * @return an array with sizes of the dimensions of this matrix.
     */
    public int[] getDimensionSizes() {
        return dimensions;
    }

    /**
     * A String representation of this device buffer.
     *
     * @return
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Source matrix\n");
        sb.append("_____________\n");
        sb.append("Dimension: [ ").append(cpuBuffer.getNrOfRows()).append(" ; ");
        sb.append(cpuBuffer.getNrOfColumns()).append(" ; ");
        sb.append(cpuBuffer.getNrOfSlices()).append(" ; ");
        sb.append(cpuBuffer.getNrOfHyperSlices()).append(" ; ");
        sb.append(" ]\n");
        sb.append("Zero padding : ").append(cpuBuffer.getZeroPadding()).append("\n\n");

        sb.append("Device :\n");
        sb.append("________\n");
        sb.append("Device origin : [").append(deviceOffset[0]).append(";").append(deviceOffset[1]).append(";").append(deviceOffset[2]).append("]\n");
        sb.append("Host   offset : [").append(hostOffset[0]).append(";").append(hostOffset[1]).append(";").append(hostOffset[2]).append("]\n");
        sb.append("Host   region : [").append(hostRegion[0]).append(";").append(hostRegion[1]).append(";").append(hostRegion[2]).append("]\n");
        sb.append("Device pitch [ row ; slice ] : [ ").append(getDeviceRowPitch()).append(";").append(getDeviceSlicePitch()).append("]\n");
        sb.append("Host   pitch [ row ; slice ] : [ ").append(getHostRowPitch()).append(";").append(getHostSlicePitch()).append("]\n");
        return sb.toString();
    }

    /**
     * Synchronizes the host with the buffer that is stored on the gpu if
     * necessary.
     */
    public void syncHost() {
        download();
    }

    public void markGpuAsMaster() {
        this.cpuBufferState = BufferSyncState.OUTOFDATE;
        this.gpuBufferState = BufferSyncState.UPTODATE;
        
    }

    public void markCpuAsMaster() {
        this.cpuBufferState = BufferSyncState.UPTODATE;
        this.gpuBufferState = BufferSyncState.OUTOFDATE;
    }

}
