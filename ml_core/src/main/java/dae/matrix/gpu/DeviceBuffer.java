/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.gpu;

import dae.matrix.BufferSyncState;
import dae.matrix.imatrix;
import org.jocl.Pointer;
import org.jocl.cl_mem;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class DeviceBuffer {

    private final imatrix cpuBuffer;

    private cl_mem rMem;
    private cl_mem rwMem;

    private BufferSyncState rBufferState = BufferSyncState.CPU;
    private BufferSyncState rwBufferState = BufferSyncState.GPU;

    private final int colPadding;
    private final int rowPadding;

    private final int[] deviceDimension = new int[2];

    public DeviceBuffer(imatrix cpuBuffer) {

        this.cpuBuffer = cpuBuffer;
        int zp = cpuBuffer.getZeroPadding();
        colPadding = 2 * zp + 32 - ((cpuBuffer.getNrOfColumns() + 2 * zp) % 32);
        rowPadding = 2 * zp + 32 - ((cpuBuffer.getNrOfRows() + 2 * zp) % 32);

        deviceDimension[0] = cpuBuffer.getNrOfColumns() + colPadding;
        deviceDimension[1] = cpuBuffer.getNrOfRows() + rowPadding;

    }

    /**
     * Returns the buffersync state which indicates where the most up to date
     * version of the buffer can be found.
     *
     * @return the buffer sync state object.
     */
    public BufferSyncState getRBufferSyncState() {
        return rBufferState;
    }

    /**
     * Sets the buffersync state which indicates where the most up to date
     * version of the buffer can be found.
     *
     * @param state the buffer sync state object.
     */
    public void setRBufferSyncState(BufferSyncState state) {
        this.rBufferState = state;
    }

    /**
     * Returns the buffersync state which indicates where the most up to date
     * version of the buffer can be found.
     *
     * @return the buffer sync state object.
     */
    public BufferSyncState getRWBufferSyncState() {
        return rwBufferState;
    }

    /**
     * Sets the buffersync state which indicates where the most up to date
     * version of the buffer can be found.
     *
     * @param state the buffer sync state object.
     */
    public void setRWBufferSyncState(BufferSyncState state) {
        this.rwBufferState = state;
    }

    public cl_mem getCLReadMem() {
        if (rMem == null) {
            System.out.println("creating new mem r buffer");
            rMem = FMatrixOpGpu.createReadMem(cpuBuffer, this.colPadding, this.rowPadding);
        }
        return rMem;
    }

    public cl_mem getCLReadWriteMem() {
        if (rwMem == null) {
            System.out.println("creating new mem rw buffer");
            rwMem = FMatrixOpGpu.createReadWriteMem(cpuBuffer, this.colPadding, this.rowPadding);
        }
        return rwMem;
    }

    public Pointer getCLPointer() {
        return Pointer.to(cpuBuffer.getHostData().array());
    }

    /**
     * Get the padding for the columns.
     *
     * @return the padding for the columns.
     */
    public int getColPadding() {
        return colPadding;
    }

    /**
     * Get the padding for the rows.
     *
     * @return the padding for the rows.
     */
    public int getRowPadding() {
        return rowPadding;
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
     * element is the number of columns and the second element is the number
     * of rows.
     * @return the device dimension as an int array.
     */
    public int[] getDeviceDimension(){
        return deviceDimension;
    }
}
