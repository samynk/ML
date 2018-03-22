package dae.matrix;

import dae.matrix.gpu.FloatDeviceBuffer;
import dae.neuralnet.activation.Function;
import java.nio.FloatBuffer;

/**
 * A matrix that offers a transposed view on another matrix.
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class tmatrix implements imatrix {

    private imatrix source;
    private final static tmatrix HELPER = new tmatrix(null);

    public tmatrix(imatrix source) {
        this.source = source;
    }

    /**
     * Returns a name for the matrix object.
     *
     * @return the name of the matrix object.
     */
    @Override
    public String getName() {
        return source.getName() + "t";
    }

    /**
     * Checks if this matrix is a row vector.
     *
     * @return true if the matrix is a row vector, false otherwise.
     */
    @Override
    public boolean isRowVector() {
        return source.getNrOfColumns() == 1 && getNrOfSlices() == 1;
    }
    
    /**
     * Checks if this matrix is a batch matrix, in other words it is a multiple
     * of a row vector.
     *
     * @return true if this matrix is a row vector with a number of hyperslices
     * that is bigger than 1.
     */
    @Override
    public boolean isBatchMatrix() {
        return source.isBatchMatrix();
    }

    public void setSource(imatrix source) {
        this.source = source;
    }

    @Override
    public void set(int row, int column, float value) {
        source.set(column, row, value);
    }

    @Override
    public void set(int row, int column, int slice, float value) {
        source.set(column, row, slice, value);
    }

    @Override
    public void set(int row, int column, int slice, int hyperslice, float value) {
        source.set(column, row, hyperslice, slice, value);
    }

    @Override
    public void setRow(int row, float[] values) {
        source.setColumn(row, values);
    }

    /**
     * Resets all the values in the matrix to zero.
     */
    @Override
    public void reset() {
        source.reset();
    }

    @Override
    public void getRow(int row, imatrix rowStorage) {
        HELPER.setSource(rowStorage);
        source.getColumn(row, HELPER);
    }

    @Override
    public void getRow(int row, int targetRow, imatrix rowStorage) {
        HELPER.setSource(rowStorage);
        source.getColumn(row, targetRow, HELPER);
    }

    @Override
    public void setColumn(int column, float... values) {
        source.setRow(column, values);
    }

    @Override
    public void getColumn(int column, imatrix columnStorage) {
        HELPER.setSource(columnStorage);
        source.getRow(column, HELPER);
    }

    @Override
    public void getColumn(int column, int targetColumn, imatrix columnStorage) {
        HELPER.setSource(columnStorage);
        source.getRow(column, targetColumn, HELPER);
    }

    @Override
    public float get(int row, int column) {
        return source.get(column, row);
    }

    @Override
    public float get(int row, int column, int slice) {
        return source.get(column, row, slice);
    }

    /**
     * Gets the value of a cell.
     *
     * @param row the row of the cell.
     * @param column the column of the cell.
     * @param slice the slice of the cell.
     * @param hyperslice the hyperslice of the cell.
     * @return the value of the cell.
     */
    @Override
    public float get(int row, int column, int slice, int hyperslice) {
        return source.get(column, row, slice, hyperslice);
    }

    @Override
    public int getNrOfRows() {
        return source.getNrOfColumns();
    }

    @Override
    public int getNrOfColumns() {
        return source.getNrOfRows();
    }

    @Override
    public int getNrOfSlices() {
        return source.getNrOfSlices();
    }

    @Override
    public int getNrOfHyperSlices() {
        return source.getNrOfHyperSlices();
    }

    @Override
    public int getZeroPadding() {
        return source.getZeroPadding();
    }

    /**
     * Returns the total number of cells in this matrix.
     *
     * @return the total number of cells in the matrix.
     */
    @Override
    public int getSliceSize() {
        return source.getSliceSize();
    }
    
    @Override
    public int getHyperSliceSize() {
        return source.getHyperSliceSize();
    }

    /**
     * Returns the number of cells in one slice of this matrix.
     *
     * @return the number of cells in one slice of the matrix.
     */
    @Override
    public int getSize() {
        return source.getSize();
    }

    @Override
    public String getSizeAsString() {
        return "[" + getNrOfRows() + "," + getNrOfColumns() + "]";
    }

    @Override
    public void applyFunction(Function f) {
        source.applyFunction(f);
    }

    @Override
    public void multiply(float factor) {
        source.multiply(factor);
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
    public FloatDeviceBuffer getDeviceBuffer() {
        return source.getDeviceBuffer();
    }

    /**
     * Synchronizes the host buffer with the device buffer if necessary.
     */
    @Override
    public void sync() {
        source.sync();
    }

    /**
     * Make the cpu buffer the most current.
     */
    @Override
    public void makeMaster() {
        source.makeMaster();
    }

    /**
     * Checks if this is a transposed view on the source data.
     */
    @Override
    public boolean isTransposed() {
        return !source.isTransposed();
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
    public imatrix copy() {
        return new tmatrix(source.copy());
    }

    @Override
    public String toString() {
        return fmatrix.print(this);
    }
}
