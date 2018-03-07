package dae.matrix;

import dae.matrix.gpu.DeviceBuffer;
import dae.neuralnet.activation.Function;
import java.nio.FloatBuffer;
import org.jocl.Pointer;
import org.jocl.cl_mem;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public interface imatrix {

    /**
     * Checks if this matrix is a row vector.
     *
     * @return true if the matrix is a row vector, false otherwise.
     */
    public boolean isRowVector();

    /**
     * Sets a cell in this matrix to the given value.
     *
     * @param row the row to set.
     * @param column the column to set.
     * @param value the new value for the cell.
     */
    public void set(int row, int column, float value);

    /**
     * Sets a cell in this matrix to the given value.
     *
     * @param row the row to set.
     * @param column the column to set.
     * @param slice the slice to set.
     * @param value the new value for the cell.
     */
    public void set(int row, int column, int slice, float value);

    /**
     * Sets a row in the matrix to the given values.
     *
     * @param row the row to set.
     * @param values the new row values in the matrix.
     */
    public void setRow(int row, float[] values);

    /**
     * Stores a row of this matrix into the provided fmatrix storage.
     *
     * @param row the row to get.
     * @param rowStorage the rowstorage to store the row in.
     */
    public void getRow(int row, imatrix rowStorage);

    /**
     * Stores a row of this matrix into the target row of the fmatrix storage.
     *
     * @param targetRow the target row for the fmatrix storage.
     * @param row the row to get.
     * @param rowStorage the rowstorage to store the row in.
     */
    public void getRow(int row, int targetRow, imatrix rowStorage);

    /**
     * Sets a column in the matrix to the given values.
     *
     * @param column the column to set.
     * @param values the new column values.
     */
    public void setColumn(int column, float... values);

    /**
     * Stores a column of this matrix into the provided fmatrix storage.
     *
     * @param column the column to get.
     * @param columnStorage the column storage to store the row in.
     */
    public void getColumn(int column, imatrix columnStorage);

    /**
     * Stores a row of this matrix into the target row of the imatrix storage.
     *
     * @param column the column to get.
     * @param targetColumn the target row for the imatrix storage.
     * @param columnStorage the imatrix to store the column in.
     */
    public void getColumn(int column, int targetColumn, imatrix columnStorage);

    /**
     * Gets the value of a cell.
     *
     * @param row the row of the cell.
     * @param column the column of the cell.
     * @return the value of the cell.
     */
    public float get(int row, int column);

    /**
     * Gets the value of a cell.
     *
     * @param row the row of the cell.
     * @param column the column of the cell.
     * @param slice the slice of the cell.
     * @return the value of the cell.
     */
    public float get(int row, int column, int slice);

    /**
     * Returns the number of rows.
     *
     * @return the number of rows.
     */
    public int getNrOfRows();

    /**
     * Returns the number of columns.
     *
     * @return the number of columns.
     */
    public int getNrOfColumns();

    /**
     * Returns the number of slices in the matrix.
     *
     * @return the number of slices.
     */
    public int getNrOfSlices();

    /**
     * Returns the total number of cells in this matrix in a single slice.
     *
     * @return the total number of cells in the matrix.
     */
    public int getSliceSize();

    /**
     * Returns the total number of cells in this matrix.
     *
     * @return the total number of cells in the matrix.
     */
    public int getSize();

    /**
     * Gets the zero padding around the matrix.
     *
     * @return the zero padding around the matrix.
     */
    public int getZeroPadding();

    /**
     * Gets the maximum value in the matrix.
     *
     * @return a Cell object with the maximum value.
     */
    public Cell max();

    /**
     * Gets the maximum value in the matrix.
     *
     * @param result a Cell object that will store the result.
     * @return the result parameter.
     */
    public Cell max(Cell result);

    /**
     * Gets the minimum value in the matrix.
     *
     * @return a Cell object with the maximum value.
     */
    public Cell min();

    /**
     * Gets the maximum value in the matrix.
     *
     * @param result a Cell object that will store the result.
     * @return the result parameter.
     */
    public Cell min(Cell result);

    /**
     * Multiplies all the cells in this matrix with the given value.
     *
     * @param factor the factor to multiply all the cells with.
     */
    public void multiply(float factor);

    /**
     * Returns the dimension of this matrix in a string representation.
     *
     * @return
     */
    public String getSizeAsString();

    /**
     * Applies a function to all the cells in the matrix.
     *
     * @param f the function to apply.
     */
    public void applyFunction(Function f);

    /**
     * Creates a copy of this matrix.
     *
     * @return a copy of this matrix.
     */
    public imatrix copy();

    /**
     * Gets the raw data of the matrix.
     *
     * @return the raw data.
     */
    public FloatBuffer getHostData();

    /**
     * Checks if this is a transposed view on the source data.
     *
     * @return true if the matrix is transposed, false othersise.
     */
    public boolean isTransposed();

    /**
     * Returns the DeviceBuffer object.
     * @return the DeviceBuffer object.
     */
    public DeviceBuffer getDeviceBuffer();

}
