/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix;

import dae.matrix.gpu.DeviceBuffer;
import dae.matrix.gpu.FMatrixOpGpu;
import dae.neuralnet.activation.Function;
import java.nio.FloatBuffer;
import org.jocl.Pointer;
import org.jocl.cl_mem;

/**
 * A zero padding wrapper around another matrix. This is a utility class that
 * makes it easier to do a full convolution without copying a regular matrix
 * into a zero padded matrix.
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class zpmatrix implements imatrix {

    private final imatrix source;
    private final int zeroPadding;

    private cl_mem rMem;
    private cl_mem rwMem;

    private final int[] padding = new int[2];

    public zpmatrix(imatrix source, int zp) {
        this.source = source;
        this.zeroPadding = zp;
        padding[0] = 2 * zeroPadding + 32 - ((source.getNrOfColumns() + 2 * zeroPadding) % 32);
        padding[1] = 2 * zeroPadding + 32 - ((source.getNrOfRows() + 2 * zeroPadding) % 32);
    }

    @Override
    public boolean isRowVector() {
        return source.isRowVector();
    }

    @Override
    public void set(int row, int column, float value) {
        source.set(row, column, value);
    }

    @Override
    public void set(int row, int column, int slice, float value) {
        source.set(row, column, slice, value);
    }

    @Override
    public void setRow(int row, float[] values) {
        source.setRow(row, values);
    }

    @Override
    public void getRow(int row, imatrix rowStorage) {
        source.getRow(row, rowStorage);
    }

    @Override
    public void getRow(int row, int targetRow, imatrix rowStorage) {
        source.getRow(row, targetRow, rowStorage);
    }

    @Override
    public void setColumn(int column, float... values) {
        source.setColumn(column, values);
    }

    @Override
    public void getColumn(int column, imatrix columnStorage) {
        source.getColumn(column, columnStorage);
    }

    @Override
    public void getColumn(int column, int targetColumn, imatrix columnStorage) {
        source.getColumn(column, targetColumn, columnStorage);
    }

    @Override
    public float get(int row, int column) {
        return source.get(row, column);
    }

    @Override
    public float get(int row, int column, int slice) {
        return source.get(row, column, slice);
    }

    @Override
    public int getNrOfRows() {
        return source.getNrOfRows();
    }

    @Override
    public int getNrOfColumns() {
        return source.getNrOfColumns();
    }

    @Override
    public int getNrOfSlices() {
        return source.getNrOfSlices();
    }

    @Override
    public int getSliceSize() {
        return source.getSliceSize();
    }

    @Override
    public int getSize() {
        return source.getSize();
    }

    @Override
    public int getZeroPadding() {
        return this.zeroPadding;
    }

    @Override
    public Cell max() {
        return source.max();
    }

    @Override
    public Cell max(Cell result) {
        return source.max(result);
    }

    @Override
    public Cell min() {
        return source.min();
    }

    @Override
    public Cell min(Cell result) {
        return source.min(result);
    }

    @Override
    public void multiply(float factor) {
        source.multiply(factor);
    }

    @Override
    public String getSizeAsString() {
        return source.getSizeAsString();
    }

    @Override
    public void applyFunction(Function f) {
        source.applyFunction(f);
    }

    @Override
    public imatrix copy() {
        return new zpmatrix(source, zeroPadding);
    }

    @Override
    public FloatBuffer getHostData() {
        return source.getHostData();
    }

    /**
     * Returns the DeviceBuffer object.
     *
     * @return the DeviceBuffer object.
     */
    @Override
    public DeviceBuffer getDeviceBuffer() {
        return source.getDeviceBuffer();
    }

    @Override
    public boolean isTransposed() {
        return source.isTransposed();
    }

}
