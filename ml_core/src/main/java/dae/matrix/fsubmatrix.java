package dae.matrix;

import dae.matrix.gpu.FloatDeviceBuffer;
import dae.neuralnet.activation.Function;
import java.nio.FloatBuffer;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class fsubmatrix implements imatrix {

    private final imatrix source;
    private final int rb;
    private final int cb;
    private final int sb;
    private final int hb;

    private final int rows;
    private final int columns;
    private final int slices;
    private final int hyperslices;

    /**
     * Creates a sub matrix which uses another matrix as backing source.
     *
     * @param source the source matrix of this matrix.
     * @param rb the row base of the submatrix.
     * @param cb the column base of the submatrix.
     * @param rows the number of rows in the submatrix.
     * @param columns the number of columns in the submatrix.
     */
    public fsubmatrix(imatrix source, int rb, int cb, int rows, int columns) {
        this(source, rb, cb, 0, rows, columns, source.getNrOfSlices());
    }

    /**
     * Creates a sub matrix which uses another matrix as backing source.
     *
     * @param source the source matrix of this matrix.
     * @param rb the row base of the submatrix.
     * @param cb the column base of the submatrix.
     * @param sb the slice base of the submatrix.
     * @param rows the number of rows in the submatrix.
     * @param columns the number of columns in the submatrix.
     * @param slices the number of slices in the submatrix.
     */
    public fsubmatrix(imatrix source, int rb, int cb, int sb, int rows, int columns, int slices) {
        this(source, rb, cb, sb, 0, rows, columns, slices, 1);
    }

    /**
     * Creates a sub matrix which uses another matrix as backing source.
     *
     * @param source the source matrix of this matrix.
     * @param rb the row base of the submatrix.
     * @param cb the column base of the submatrix.
     * @param sb the slice base of the submatrix.
     * @param hb the hyper slice base of the submatrix.
     * @param rows the number of rows in the submatrix.
     * @param columns the number of columns in the submatrix.
     * @param slices the number of slices in the submatrix.
     * @param hyperslices the number of hyperslices in the submatrix.
     */
    public fsubmatrix(imatrix source, int rb, int cb, int sb, int hb, int rows, int columns, int slices, int hyperslices) {
        this.source = source;
        this.rb = Math.max(rb, 0);
        this.cb = Math.max(cb, 0);
        this.sb = Math.max(sb, 0);
        this.hb = Math.max(hb, 0);

        this.rows = Math.min(source.getNrOfRows(), Math.max(rows - rb, 0));
        this.columns = Math.min(source.getNrOfColumns(), Math.max(columns - cb, 0));
        this.slices = Math.min(source.getNrOfSlices(), Math.max(slices - sb, 0));
        this.hyperslices = Math.min(source.getNrOfHyperSlices(), Math.max(hyperslices - hb, 0));
    }

    /**
     * Returns a name for the matrix object.
     *
     * @return the name of the matrix object.
     */
    @Override
    public String getName() {
        return source.getName();
    }

    /**
     * The zero padding in the matrix. Not supported by this class, this method
     * always returns zero.
     *
     * @return the zero padding in the matrix.
     */
    @Override
    public int getZeroPadding() {
        return 0;
    }

    /**
     * Checks if this matrix is a row vector.
     *
     * @return true if the matrix is a row vector, false otherwise.
     */
    @Override
    public boolean isRowVector() {
        return rows == 1 && slices == 1;
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
        return isRowVector() && this.hyperslices > 1;
    }

    @Override
    public void set(int row, int column, float value) {
        source.set(row - rb, column - cb, value);
    }

    @Override
    public void set(int row, int column, int slice, float value) {
        source.set(row - rb, column - cb, slice - sb, value);
    }

    @Override
    public void set(int row, int column, int slice, int hyperslice, float value) {
        source.set(row - rb, column - cb, slice - sb, hyperslice - hb, value);
    }

    @Override
    public void setRow(int row, float[] values) {
        source.setRow(row - rb, values);
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
        source.getRow(row - rb, rowStorage);
    }

    @Override
    public void getRow(int row, int targetRow, imatrix rowStorage) {
        source.getRow(row - rb, targetRow, rowStorage);
    }

    @Override
    public void setColumn(int column, float... values) {
        source.setColumn(column - cb, values);
    }

    @Override
    public void getColumn(int column, imatrix columnStorage) {
        source.getColumn(column - cb, columnStorage);
    }

    @Override
    public void getColumn(int column, int targetColumn, imatrix columnStorage) {
        source.getColumn(column - cb, targetColumn, columnStorage);
    }

    @Override
    public float get(int row, int column) {
        return source.get(row - rb, column - cb);
    }

    @Override
    public float get(int row, int column, int slice) {
        return source.get(row - rb, column - cb, slice - sb);
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
        return source.get(row - rb, column - cb, slice - sb, hyperslice - hb);
    }

    @Override
    public int getNrOfRows() {
        return rows;
    }

    @Override
    public int getNrOfColumns() {
        return columns;
    }

    @Override
    public int getNrOfSlices() {
        return slices;
    }

    @Override
    public int getNrOfHyperSlices() {
        return hyperslices;
    }

    /**
     * Returns the size of one slice in this matrix.
     *
     * @return the size of one slice in this matrix.
     */
    @Override
    public int getSize() {
        return rows * columns;
    }

    /**
     * Returns the total number of cells in this matrix.
     *
     * @return the total number of cells in the matrix.
     */
    @Override
    public int getSliceSize() {
        return rows * columns * slices;
    }

    @Override
    public Cell max() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Cell max(Cell result) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Cell min() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public Cell min(Cell result) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void multiply(float factor) {
        applyFunction(x -> x * factor);
    }

    @Override
    public String getSizeAsString() {
        return "[" + getNrOfRows() + "," + getNrOfColumns() + "]";
    }

    @Override
    public void applyFunction(Function f) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < columns; ++c) {
                float value = get(r, c);
                float fvalue = f.evaluate(value);
                set(r, c, fvalue);
            }
        }
    }

    @Override
    public imatrix copy() {
        fmatrix copy = new fmatrix(rows, columns);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < columns; ++c) {
                copy.set(r, c, get(r, c));
            }
        }
        return copy;
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
        return this.source.getDeviceBuffer();
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

    @Override
    public boolean isTransposed() {
        return source.isTransposed();
    }

    @Override
    public String toString() {
        return fmatrix.print(this);
    }

}
