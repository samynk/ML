package dae.matrix;

import dae.matrix.gpu.FloatDeviceBuffer;
import dae.matrix.gpu.FMatrixOpGpu;
import dae.matrix.integer.intmatrix;
import dae.matrix.op.FMatrixOp;
import dae.neuralnet.Layer;
import dae.neuralnet.activation.ActivationFunction;
import dae.neuralnet.activation.Function;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class fmatrix implements imatrix {

    /**
     * The name of the matrix.
     */
    private String name;
    /**
     * The dimensions of the matrix
     */
    private final int rows;
    private final int columns;
    private final int slices;
    private final int hyperSlices;

    /**
     * Zero padding that will be applied to the rows and columns.
     */
    private final int zeropadding;
    /**
     * The sizes of the various dimensions.
     */
    private final int sliceSize;
    private final int hyperSliceSize;
    private final int size;

    // private float[] data;
    private final FloatBuffer data;
    private FloatDeviceBuffer deviceBuffer;

    private static FMatrixOp matrixOp = new FMatrixOpGpu();
    private static int MATRIXCOUNT = 0;

    /**
     * Creates a new fmatrix object with the given rows and columns,1 slice and
     * 1 hyperslice.
     *
     * @param rows the number of rows in the matrix.
     * @param columns the number of columns in the matrix.
     */
    public fmatrix(int rows, int columns) {
        this(rows, columns, 1, 1);
    }

    /**
     * Creates a new fmatrix object with the given rows, columns and slices.
     *
     * @param rows the number of rows in the matrix.
     * @param columns the number of columns in the matrix.
     * @param slices the number of slices in the matrix.
     */
    public fmatrix(int rows, int columns, int slices) {
        this(rows, columns, slices, 1);
    }

    /**
     * Creates a new fmatrix object with the given rows, columns, slices and
     * hyperslices.
     *
     * @param rows the number of rows in the matrix.
     * @param columns the number of columns in the matrix.
     * @param slices the number of slices in the matrix.
     * @param hyperSlices the number of hyperslices in this matrix.
     */
    public fmatrix(int rows, int columns, int slices, int hyperSlices) {
        this(rows, columns, slices, hyperSlices, 0);
    }

    /**
     * Creates a new fmatrix object with the given rows and columns.
     *
     * The zero padding adds a virtual number of rows and columns around the
     * fmatrix. This means that the zero padded row is a negative row on the
     * upper side of the matrix and a negative column on the left side of the
     * matrix. On the right and bottom side of the matrix the zero padded column
     * and row have indexes greater than the number of rows and columns in this
     * matrix.
     *
     * @param rows the number of rows in the matrix.
     * @param columns the number of columns in the matrix.
     * @param slices the number of slices in the matrix.
     * @param hyperSlices the number of hyperslices in this matrix.
     * @param zeropadding the zero padding to add around this matrix.
     */
    public fmatrix(int rows, int columns, int slices, int hyperSlices, int zeropadding) {
        this.rows = rows;
        this.columns = columns;
        this.slices = slices;
        this.hyperSlices = hyperSlices;
        this.zeropadding = zeropadding;
        this.sliceSize = this.rows * this.columns;
        this.hyperSliceSize = sliceSize * slices;
        this.size = sliceSize * slices * hyperSlices;
        data = FloatBuffer.allocate(size);
        deviceBuffer = new FloatDeviceBuffer(this);
        name = "matrix" + MATRIXCOUNT++;
    }

    /**
     * Creates a new fmatrix from the given parameter.
     *
     * @param toCopy the fmatrix to copy.
     */
    public fmatrix(fmatrix toCopy) {
        this(toCopy.rows, toCopy.columns, toCopy.slices, toCopy.hyperSlices, toCopy.zeropadding);
        toCopy.data.rewind();
        this.data.rewind();

        while (this.data.hasRemaining()) {
            this.data.put(toCopy.data.get());
        }
    }

    /**
     * Returns a name for the matrix object.
     *
     * @return the name of the matrix object.
     */
    @Override
    public String getName() {
        return name;
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
        return getNrOfHyperSlices() > 1;
    }

    /**
     * Gets the raw data of the matrix.
     *
     * @return the raw data.
     */
    @Override
    public FloatBuffer getHostData() {
        return data;
    }

    /**
     * Sets all the elements in this matrix to zero.
     */
    public void reset() {
        matrixOp.reset(this);
    }

    /**
     * Converts a row and column coordinate to a 1D coordinate. The slice number
     * and hyperslice number is assumed to be zero.
     *
     * @param r the row of the cell.
     * @param c the column of the cell.
     * @return the index of the cell in the 1D float backing array.
     */
    private int rcToIndex(int r, int c) {
        return c * rows + r;
    }

    /**
     * Converts a row, column and slice coordinate to a 1D coordinate.
     *
     * @param r the row of the cell.
     * @param c the column of the cell.
     * @param s the slice number of the cell.
     * @return the index of the cell in the 1D float backing array.
     */
    private int rcsToIndex(int r, int c, int s) {
        return rcToIndex(r, c) + s * sliceSize;
    }

    /**
     * Converts a row, column, slice and hyperslice coordinate to a 1D
     * coordinate.
     *
     * @param r the row of the cell.
     * @param c the column of the cell.
     * @param s the slice number of the cell.
     * @return the index of the cell in the 1D float backing array.
     */
    private int rcshToIndex(int r, int c, int s, int h) {
        return rcsToIndex(r, c, s) + h * hyperSliceSize;
    }

    /**
     * Checks if this is a transposed view on the source data.
     *
     * @return true if this is transposed view of the source data, false
     * otherwise.
     */
    @Override
    public boolean isTransposed() {
        return false;
    }

    /**
     * Calls the cell method on the cell iterator.
     *
     * @param it the CellIterator object.
     */
    public void iterateCells(CellIterator it) {
        for (int hyperslice = 0; hyperslice < getNrOfSlices(); ++hyperslice) {
            for (int slice = 0; slice < getNrOfSlices(); ++slice) {
                for (int row = 0; row < getNrOfRows(); ++row) {
                    for (int column = 0; column < getNrOfColumns(); ++column) {
                        it.cell(this, row, column, slice, get(row, column, slice));
                    }
                }
            }
        }
    }

    /**
     * Sets a cell in this matrix to the given value. The slice number is
     * assumed to be zero.
     *
     * @param row the row to set.
     * @param column the column to set.
     * @param value the new value for the cell.
     */
    @Override
    public void set(int row, int column, float value) {
        data.put(rcToIndex(row, column), value);
    }

    /**
     * Sets a cell in this matrix to the given value.
     *
     * @param row the row to set.
     * @param column the column to set.
     * @param slice the slice to set.
     * @param value the new value for the cell.
     */
    @Override
    public void set(int row, int column, int slice, float value) {
        data.put(rcsToIndex(row, column, slice), value);
    }

    /**
     * Sets a cell in this matrix to the given value.
     *
     * @param row the row to set.
     * @param column the column to set.
     * @param slice the slice to set.
     * @param hyperslice the slice to set.
     * @param value the new value for the cell.
     */
    @Override
    public void set(int row, int column, int slice, int hyperslice, float value) {
        data.put(rcshToIndex(row, column, slice, hyperslice), value);
    }

    /**
     * Sets a row in the matrix to the given values. The slice number is assumed
     * to be zero.
     *
     * @param row the row to set.
     * @param values the new row values in the matrix.
     */
    @Override
    public void setRow(int row, float[] values) {
        int limit = Math.min(this.columns, values.length);
        for (int c = 0; c < limit; ++c) {
            set(row, c, values[c]);
        }
    }

    /**
     * Sets all the cells in the row in the matrix to the given value.
     *
     * @param row the row to set.
     * @param value the value.
     */
    public void setRow(int row, float value) {
        for (int c = 0; c < columns; ++c) {
            set(row, c, value);
        }
    }

    /**
     * Sets a column in the matrix to the given values.
     *
     * @param column the column to set.
     * @param values the new column values.
     */
    @Override
    public void setColumn(int column, float... values) {
        int maxIndex = Math.min(values.length, rows);
        for (int row = 0; row < maxIndex; ++row) {
            set(row, column, values[row]);
        }

    }

    /**
     * Gets the value of a cell. The slice number is one.
     *
     * @param row the row of the cell.
     * @param column the column of the cell.
     * @return the value of the cell.
     */
    @Override
    public float get(int row, int column) {
        int index = rcToIndex(row, column);
        if (index < data.limit()) {
            return data.get(index);
        } else {
            return 0.0f;
        }
    }

    /**
     * Gets the value of a cell.
     *
     * @param row the row of the cell.
     * @param column the column of the cell.
     * @param slice the slice of the cell.
     * @return the value of the cell.
     */
    @Override
    public float get(int row, int column, int slice) {
        int index = rcsToIndex(row, column, slice);
        if (index < data.limit()) {
            return data.get(index);
        } else {
            return 0.0f;
        }
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
        int index = rcshToIndex(row, column, slice, hyperslice);
        if (index < data.limit()) {
            return data.get(index);
        } else {
            return 0.0f;
        }
    }

    /**
     * Gets the maximum value in the matrix.
     *
     * @return a Cell object with the maximum value.
     */
    @Override
    public Cell max() {
        Cell result = new Cell();
        return max(result);
    }

    /**
     * Returns the maximum values and the index of the cell per row.
     *
     * @param maxCells
     */
    public void maxPerRow(ArrayList<Cell> maxCells) {
        for (int row = 0; row < getNrOfRows(); ++row) {
            float max = Float.MIN_VALUE;
            for (int column = 0; column < getNrOfColumns(); ++column) {
                float value = this.get(row, column);
                Cell c = maxCells.get(row);
                if (value > max) {
                    max = value;
                    c.value = max;
                    c.row = row;
                    c.column = column;
                }
            }
        }
    }

    /**
     * Returns the maximum values and the index of the cell per row.
     *
     * @param maxCells
     */
    public void maxPerColumn(ArrayList<Cell> maxCells) {
        for (int h = 0; h < getNrOfHyperSlices(); ++h) {
            float max = Float.MIN_VALUE;
            Cell c = maxCells.get(h);
            for (int r = 0; r < getNrOfRows(); ++r) {
                float value = this.get(r, 0, 0, h);
                if (value > max) {
                    max = value;
                    c.value = max;
                    c.row = r;
                    c.column = h;
                }
            }
        }
    }

    /**
     * Gets the maximum value in the matrix.
     *
     * @param result a Cell object that will store the result.
     * @return the result parameter.
     */
    @Override
    public Cell max(Cell result) {
        float max = Float.MIN_VALUE;
        for (int slice = 0; slice < getNrOfSlices(); ++slice) {
            for (int row = 0; row < getNrOfRows(); ++row) {
                for (int column = 0; column < getNrOfColumns(); ++column) {
                    float value = this.get(row, column);
                    if (value > max) {
                        max = value;
                        result.value = max;
                        result.row = row;
                        result.column = column;
                    }
                }
            }
        }
        return result;
    }

    /**
     * Gets the minimum value in the matrix.
     *
     * @return a Cell object with the minimum value.
     */
    @Override
    public Cell min() {
        Cell result = new Cell();
        return min(result);
    }

    /**
     * Gets the minimum value in the matrix.
     *
     * @param result a Cell object that will store the result.
     * @return the result parameter.
     */
    @Override
    public Cell min(Cell result) {
        float min = Float.MAX_VALUE;
        for (int slice = 0; slice < getNrOfSlices(); ++slice) {
            for (int row = 0; row < getNrOfRows(); ++row) {
                for (int column = 0; column < getNrOfColumns(); ++column) {
                    float value = this.get(row, column);
                    if (value < min) {
                        min = value;
                        result.value = min;
                        result.row = row;
                        result.column = column;
                    }
                }
            }
        }
        return result;
    }

    /**
     * Gets a row from the matrix, as an fmatrix object.
     *
     * @param row the row to return.
     * @return the row of the matrix as a single row matrix.
     */
    public fmatrix getRow(int row) {
        fmatrix result = new fmatrix(1, getNrOfColumns());
        for (int c = 0; c < columns; ++c) {
            result.set(0, c, get(row, c));
        }
        return result;
    }

    /**
     * Stores a row of this matrix into the provided imatrix storage.
     *
     * @param row the row to get.
     * @param rowStorage the rowstorage to store the row in.
     */
    @Override
    public void getRow(int row, imatrix rowStorage) {
        getRow(row, 0, rowStorage);
    }

    /**
     * Stores a row of this matrix into the target row of the fmatrix storage.
     *
     * @param targetRow the target row for the fmatrix storage.
     * @param row the row to get.
     * @param rowStorage the rowstorage to store the row in.
     */
    @Override
    public void getRow(int row, int targetRow, imatrix rowStorage) {
        for (int c = 0; c < columns; ++c) {
            rowStorage.set(targetRow, c, get(row, c));
        }
    }

    /**
     * Stores a row of this matrix into the target row of the fmatrix storage.
     *
     * @param targetRow the target row for the fmatrix storage.
     * @param row the row to get.
     * @param targetSlice
     * @param targetHyperSlice
     * @param rowStorage the rowstorage to store the row in.
     */
    public void getRow(int row, int targetRow, int targetSlice, int targetHyperSlice, imatrix rowStorage) {
        for (int c = 0; c < columns; ++c) {
            rowStorage.set(targetRow, c, targetSlice, targetHyperSlice, get(row, c));
        }
    }

    /**
     * Returns the column of the matrix as a matrix.
     *
     * @param column the column of the matrix.
     * @return the column of the matrix as a single column matrix.
     */
    public fmatrix getColumn(int column) {
        fmatrix result = new fmatrix(getNrOfRows(), 1);
        for (int row = 0; row < rows; ++row) {
            result.set(row, 0, this.get(row, column));
        }
        return result;
    }

    /**
     * Stores a row of this matrix into the provided fmatrix storage.
     *
     * @param column the column to get.
     * @param columnStorage a fmatrix object to store the column in.
     */
    @Override
    public void getColumn(int column, imatrix columnStorage) {
        getColumn(column, 0, columnStorage);
    }

    /**
     * Stores a row of this matrix into the target row of the imatrix storage.
     *
     * @param column the column to get.
     * @param targetColumn the target row for the imatrix storage.
     * @param columnStorage the imatrix to store the column in.
     */
    @Override
    public void getColumn(int column, int targetColumn, imatrix columnStorage) {
        for (int row = 0; row < rows; ++row) {
            columnStorage.set(row, targetColumn, this.get(row, column));
        }
    }

    /**
     * Stores a row of this matrix into the target row of the fmatrix storage.
     *
     * @param column the row to get.
     * @param targetColumn the target row for the fmatrix storage.
     * @param targetSlice
     * @param targetHyperSlice
     * @param columnStorage the rowstorage to store the row in.
     */
    public void getColumn(int column, int targetColumn, int targetSlice, int targetHyperSlice, imatrix columnStorage) {
        for (int r = 0; r < this.getNrOfRows(); ++r) {
            columnStorage.set(r, targetColumn, targetSlice, targetHyperSlice, get(r, column));
        }
    }

    /**
     * Stores a slice into the provided storage.
     *
     * @param hyperslice the hyperslice to get.
     * @param targetHyperSlice the hyperslice to store the data in.
     * @param storage the storage for the slice.
     */
    public void getHyperSlice(int hyperslice, int targetHyperSlice, imatrix storage) {
        if (isTransposed() == storage.isTransposed()
                && getSliceSize() == storage.getSliceSize()) {
            // row - column layout is the same.
            int srcStart = this.rcshToIndex(0, 0, 0, hyperslice);
            int dstStart = storage.getSliceSize() * targetHyperSlice;
            int tocopy = Math.min(getSliceSize(), storage.getSliceSize());
            float[] src = this.data.array();
            float[] dst = storage.getHostData().array();
            System.arraycopy(src, srcStart, dst, dstStart, tocopy);
        } else {
            // no assumptions possible.
            int iRows = Math.min(this.getNrOfRows(), storage.getNrOfRows());
            int iCols = Math.min(this.getNrOfColumns(), storage.getNrOfColumns());
            int iSlices = Math.min(this.getNrOfSlices(), storage.getNrOfSlices());
            for (int r = 0; r < iRows; ++r) {
                for (int c = 0; c < iCols; ++c) {
                    for (int s = 0; s < iSlices; ++s) {
                        float value = this.get(r, c, s, hyperslice);
                        storage.set(r, c, s, targetHyperSlice, value);
                    }
                }
            }
        }
    }

    /**
     * Returns the number of rows.
     *
     * @return the number of rows.
     */
    @Override
    public int getNrOfRows() {
        return rows;
    }

    /**
     * Returns the number of columns.
     *
     * @return the number of columns.
     */
    @Override
    public int getNrOfColumns() {
        return columns;
    }

    /**
     * Returns the number of slices in the matrix.
     *
     * @return the number of slices.
     */
    @Override
    public int getNrOfSlices() {
        return slices;
    }

    /**
     * Returns the number of slices in the matrix.
     *
     * @return the number of slices.
     */
    @Override
    public int getNrOfHyperSlices() {
        return hyperSlices;
    }

    /**
     * Gets the amount of zero padding in this matrix.
     *
     * @return the zero padding in this matrix.
     */
    @Override
    public int getZeroPadding() {
        return zeropadding;
    }

    /**
     * Returns the total number of cells in this matrix in a single slice.
     *
     * @return the total number of cells in the matrix in a single slice.
     */
    @Override
    public int getSliceSize() {
        return rows * columns;
    }

    /**
     * Returns the size of the matrix.
     *
     * @return the size of the matrix.
     */
    @Override
    public int getSize() {
        return this.size;
    }

    /**
     * Makes the summation of all the cells in this matrix.
     *
     * @return the total sum of the cells in this matrix.
     */
    public float sum() {
        float sum = 0.0f;

        for (int row = 0; row < getNrOfRows(); ++row) {
            for (int column = 0; column < getNrOfColumns(); ++column) {
                sum += get(row, column);
            }
        }
        return sum;
    }

    /**
     * Multiplies all the cells in this matrix with the given value.
     *
     * @param factor the factcor to multiply all the cells with.
     */
    @Override
    public void multiply(float factor) {
        this.applyFunction(x -> x * factor);
    }

    /**
     * Adds a value to all the cells in this matrix.
     *
     * @param value the value to add.
     */
    public void add(float value) {
        this.applyFunction(x -> x + value);
    }

    public void log() {
        this.applyFunction(x -> (float) Math.log(x));
    }

    /**
     * Clamps all the values in this matrix between the given minimum and
     * maximum.
     *
     * @param min The minimum to clamp the values to.
     * @param max The maximum to clamp the values to.
     */
    public void clamp(float min, float max) {
        this.applyFunction(x -> x > max ? max : (x < min ? min : x));
    }

    /**
     * Adds the given matrix to this matrix. The add operation will only add the
     * common cells of the two matrices.
     *
     * @param op2 the matrix to add.
     * @return true if the operations was succesful (dimensions must agree),
     * false otherwise.
     */
    public boolean add(fmatrix op2) {
        if (equalDimension(this, op2)) {
            for (int row = 0; row < rows; ++row) {
                for (int column = 0; column < columns; ++column) {
                    float v1 = get(row, column);
                    float v2 = op2.get(row, column);
                    set(row, column, v1 + v2);
                }
            }
            return true;
        } else {
            return false;
        }
    }

    /**
     * Calculates the e^x for every x in this matrix.
     */
    public void exp() {
        this.applyFunction(x -> (float) Math.exp(x));
    }

    /**
     * Calculates the tanh for every element in this matrix.
     */
    public void tanh() {
        this.applyFunction(x -> (float) Math.tanh(x));
    }

    /**
     * Calculates the tangent for every cell in this matrix.
     */
    public void tan() {
        this.applyFunction(x -> (float) Math.tan(x));
    }

    /**
     * Calculates the derivative of the tanh of every cell in this matrix.
     */
    public void difftanh() {
        this.applyFunction(x -> (float) Math.tanh(x));
        this.applyFunction(x -> (1 - x * x));
    }

    /**
     * Calculates the soft max function column wise.
     */
    public void softMaxPerColumn() {
        // first exp on all the elements.
        exp();
        for (int column = 0; column < getNrOfColumns(); ++column) {
            float sum = 0;
            for (int row = 0; row < getNrOfRows(); ++row) {
                float value = get(row, column);
                sum += value;
            }

            for (int row = 0; row < getNrOfRows(); ++row) {
                float current = get(row, column) / sum;
//                if ( Float.isNaN(current)){
//                    System.out.println("soft max problem 2, sum is :"+sum);
//                    current = get(row+1,column+1);
//                }
                set(row, column, current);
            }
        }
    }

    public static void softMaxPerRow(imatrix m) {
        m.applyFunction(x -> (float) Math.exp(x));
        for (int h = 0; h < m.getNrOfHyperSlices(); ++h) {
            for (int slices = 0; slices < m.getNrOfSlices(); ++slices) {
                for (int r = 0; r < m.getNrOfRows(); ++r) {
                    float sum = 0;
                    for (int c = 0; c < m.getNrOfColumns(); ++c) {
                        float value = m.get(r, c);
                        sum += value;
                    }
                    for (int c = 0; c < m.getNrOfColumns(); ++c) {
                        float value = m.get(r, c) / sum;
                        m.set(r, c, value);
                    }
                }
            }
        }
    }

    @Override
    public void applyFunction(Function f) {
        for (int i = 0; i < data.limit(); ++i) {
            float v = data.get(i);
            float vf = f.evaluate(v);
            data.put(i, vf);
        }
    }

    /**
     * Apply a function to every cell in the matrix, with the coordinates of the
     * cell as extra parameters.
     *
     * @param f the function to apply.
     */
    public void applyCellFunction(IndexedFunction f) {
        this.iterateCells((fmatrix source, int row, int column, int slice, float currentValue) -> {
            float vf = f.evaluate(row, column, slice, currentValue);
            set(row, column, slice, vf);
        });
    }

    /**
     * Copies this matrix into another matrix.
     *
     * @return a new fmatrix with the same number of rows and columns and the
     * same values.
     */
    @Override
    public imatrix copy() {
        fmatrix result = new fmatrix(getNrOfRows(), getNrOfColumns(), getNrOfSlices());
        for (int slice = 0; slice < getNrOfSlices(); ++slice) {
            for (int row = 0; row < getNrOfRows(); ++row) {
                for (int column = 0; column < getNrOfColumns(); ++column) {
                    result.set(row, column, slice, get(row, column, slice));
                }
            }
        }
        return result;
    }

    /**
     * Creates a transposed copy of this matrix.
     *
     * @return a new transposed copy of this matrix.
     */
    public imatrix tcopy() {
        imatrix result = copy();
        return new tmatrix(result);
    }

    /**
     * Creates a new fmatrix filled with zeros.
     *
     * @param rows the number of rows.
     * @param columns the number of columns.
     * @return
     */
    public static fmatrix zeros(int rows, int columns) {
        fmatrix result = new fmatrix(rows, columns);
        return result;
    }

    /**
     * Creates a new matrix with the diagonal set to one, the rest of the cells
     * set to zero.
     *
     * @param rows the number of rows.
     * @param columns the number of columns.
     * @return a new fmatrix object.
     */
    public static fmatrix eye(int rows, int columns) {
        fmatrix result = new fmatrix(rows, columns);
        for (int i = 0; i < rows && i < columns; ++i) {
            result.set(i, i, 1);
        }
        return result;
    }

    /**
     * Creates a new fmatrix filled with ones.
     *
     * @param rows the number of rows.
     * @param columns the number of columns.
     * @return
     */
    public static fmatrix ones(int rows, int columns) {
        fmatrix result = new fmatrix(rows, columns);
        result.applyFunction(x -> 1);
        return result;
    }

    /**
     * Creates a new fmatrix filled with ones.
     *
     * @param rows the number of rows.
     * @param columns the number of columns.
     * @param minValue the minimum value for the random number.
     * @param maxValue the maximum value for the random number.
     * @return
     */
    public static fmatrix random(int rows, int columns, final float minValue, float maxValue) {
        fmatrix result = new fmatrix(rows, columns);
        final Random r = new Random(System.currentTimeMillis());
        final float diff = (maxValue - minValue);

        result.iterateCells((fmatrix source, int row, int column, int slice, float currentValue) -> {
            float value = (r.nextFloat() * diff) + minValue;
            source.set(row, column, slice, value);
        });
        return result;
    }

    public static void randomize(imatrix m, Random r, float min, float max) {
        m.applyFunction(x -> min + (r.nextFloat() * (max - min)));
    }

    public static fmatrix construct(String range) {
        Range r = parseRange(range);
        return construct(r);
    }

    public static fmatrix construct(Range range) {
        if (range.singleton) {
            fmatrix result = new fmatrix(1, 1);
            result.set(0, 0, range.startOfRange);
            return result;
        } else {
            float diff = range.endOfRange - range.startOfRange;
            int nrOfElements = (int) Math.floor(diff / range.increment) + 1;
            if (nrOfElements < 0) {
                return new fmatrix(1, 1);
            } else {
                fmatrix result = new fmatrix(nrOfElements, 1);
                for (int i = 0; i < nrOfElements; ++i) {
                    float value = range.startOfRange + i * range.increment;
                    result.set(i, 0, value);
                }
                return result;
            }
        }
    }

    public static imatrix multiply(imatrix op1, imatrix op2) {
        if (op1.getNrOfColumns() != op2.getNrOfRows()) {
            String op1dim = "[" + op1.getNrOfRows() + "," + op1.getNrOfColumns() + "]";
            String op2dim = "[" + op2.getNrOfRows() + "," + op2.getNrOfColumns() + "]";
            System.out.println("Multiply Error : dimension do not agree " + op1dim + "*" + op2dim + "\n");
            return null;
        }
        fmatrix result = new fmatrix(op1.getNrOfRows(), op2.getNrOfColumns());
        return multiply(result, op1, op2);
    }

    public static imatrix multiply(imatrix result, imatrix op1, imatrix op2) {
        if (op1.getNrOfColumns() != op2.getNrOfRows()) {
            System.out.println("Multiplay Error , inner dimension must agree: " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }
        int maxRows = Math.min(op1.getNrOfRows(), result.getNrOfRows());
        int maxColumns = Math.min(op2.getNrOfColumns(), result.getNrOfColumns());
        for (int c_row = 0; c_row < maxRows; ++c_row) {
            for (int c_column = 0; c_column < maxColumns; ++c_column) {
                float sum = 0;
                for (int index = 0; index < op1.getNrOfColumns(); ++index) {
                    sum += op1.get(c_row, index) * op2.get(index, c_column);
                }
                result.set(c_row, c_column, sum);
            }
        }

        return result;
    }

    public static imatrix dotmultiply(imatrix op1, imatrix op2) {
        if (!equalDimension(op1, op2)) {
            return null;
        }
        fmatrix result = new fmatrix(op1.getNrOfRows(), op1.getNrOfColumns(), op1.getNrOfSlices());
        return dotmultiply(result, op1, op2);
    }

    public static imatrix dotmultiply(imatrix result, imatrix op1, imatrix op2) {
        if (equalDimension(op1, op2) && equalDimension(result, op1)) {
            return matrixOp.dotmultiply(result, op1, op2);
        } else {
            return null;
        }
    }

    public static imatrix dotmultiply(imatrix result, fmatrix op1, float factor) {
        return matrixOp.dotmultiply(result, op1, factor);
    }

    public static imatrix sgemm(float alpha, imatrix a, imatrix b, float beta, imatrix c) {
        matrixOp.sgemm(alpha, a, b, beta, c);
        return c;
    }

    public static void sigmoid(imatrix o) {
        matrixOp.sigmoid(o);
    }

    public static void dsigmoid(imatrix o) {
        matrixOp.dsigmoid(o);
    }

    public static void batchConvolve(imatrix input, imatrix filter, int stride, imatrix output) {
        matrixOp.batchConvolve(input, filter, stride, output);
    }

    public static void batchCorrelate(imatrix input, imatrix filter, int stride, imatrix output) {
        matrixOp.batchCorrelate(input, filter, stride, output);
    }

    public static void batchBackpropCorrelate(imatrix input, imatrix filter, int stride, imatrix output) {
        matrixOp.batchBackpropCorrelate(input, filter, stride, output);
    }

    public static void convolve(imatrix input, imatrix filter, int stride, imatrix output) {
        matrixOp.convolve(input, filter, stride, output);
    }

    /**
     * Calculates a fuzzification layer.
     *
     * @param input the inputs to fuzzify.
     * @param a the weights that determine the slopes of the transition.
     * @param b the weights that determine the crossing point between two
     * classes.
     * @param output the fuzzified input.
     */
    public static void fuzzyFunction(imatrix input, imatrix a, imatrix b, imatrix output) {
        matrixOp.fuzzyFunction(input, a, b, output);
    }

    /**
     * Expands the elements of the input into the output with the following
     * algorithm:
     *
     * o1 = 1-i1 o2 = i1-i2 o3 = i3-i2 ... on = i(n-1)
     *
     * This also means that for every classes-1 input elements an extra output
     * element will be created.
     *
     * size(outputs) = classes * (size(inputs)/(classes-1))
     *
     * @param input the input matrix which is a row vector.
     * @param output the output matrix which is also a row vector.
     */
    public static void fuzzyShiftMinus(imatrix input, imatrix output) {
        int nrOfVariables = input.getNrOfRows();
        int lc = output.getNrOfColumns() - 1;
        for (int rv = 0; rv < nrOfVariables; ++rv) {

            float previous = 1;
            for (int ic = 0; ic < input.getNrOfColumns(); ++ic) {
                float iv = input.get(rv, ic);
                output.set(rv, ic, previous - iv);
                previous = iv;
            }
            output.set(rv, lc, previous);
        }
    }

    public static void fuzzyShiftDeltas(imatrix deltas, imatrix output) {
        int nrOfVariables = deltas.getNrOfRows();
        for (int v = 0; v < nrOfVariables; ++v) {
            for (int c = 0; c < output.getNrOfColumns(); ++c) {
                float dn = deltas.get(v, c);
                float dnp1 = deltas.get(v, c + 1);
                output.set(v, c, dnp1 - dn);
            }
        }
    }

    /**
     * Applies a max pool on the input matrix and stores it into the output
     * matrix. It is assumed that the dimensions of the output matrix are
     * dividers of the dimensions of the input matrix.
     *
     * The resulting maskLayer can be used to back propagate deltas to the
     * previous layer. The maskLayer will be filled with zeros and the location
     * of the maximum per filter will be set to 1.
     *
     * @param input the input matrix.
     * @param output the output matrix.
     * @param maskLayer a matrix with the same dimension as the input layer
     * which can be used to determine which input pixels contributed to the
     * output.
     */
    public static void batchMaxPool(imatrix input, imatrix output, intmatrix maskLayer) {
        matrixOp.batchMaxPool(input, output, maskLayer);
    }

    /**
     * Scales up the input matrix to the dimensions of the output matrix. Only
     * the cells that are defined in the masking layer are applied to output.
     *
     * @param input the input matrix.
     * @param maskLayer a matrix with the same dimension as the input layer
     * which can be used to determine which input pixels contributed to the
     * output.
     * @param output the output matrix.
     *
     */
    public static void batchBackpropMaxPool(imatrix input, intmatrix maskLayer, imatrix output) {
        matrixOp.batchBackpropMaxPool(input, maskLayer, output);
    }

    /**
     * Applies the activation function on the given matrix.
     *
     * @param function the function that defines the derived activation
     * function.
     * @param m the matrix to apply the activation function to.
     */
    public static void applyActivation(ActivationFunction function, fmatrix m) {
        matrixOp.applyActivation(function, m);
    }

    /**
     * Applies the derived activation function on the given matrix.
     *
     * @param function the function that defines the derived activation
     * function.
     * @param m the matrix to apply the activation function to.
     */
    public static void applyDerivedActivation(ActivationFunction function, fmatrix m) {
        matrixOp.applyDerivedActivation(function, m);
    }

    public static fmatrix dotdivide(fmatrix op1, fmatrix op2) {
        if (op1.getNrOfRows() != op2.getNrOfRows() || op1.getNrOfColumns() != op2.getNrOfColumns()) {
            System.out.println("DotDivide Error , matrix dimension are not the same " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }
        fmatrix result = new fmatrix(op1.getNrOfRows(), op2.getNrOfRows());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row, column);
                float op2value = op2.get(row, column);
                result.set(row, column, op1value / op2value);
            }
        }
        return result;
    }

    public static fmatrix dotadd(imatrix op1, imatrix op2) {
        if (!equalDimension(op1, op2)) {
            System.out.println("DotAdd Error , matrix dimension are not the same " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }
        fmatrix result = new fmatrix(op1.getNrOfRows(), op1.getNrOfColumns());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row, column);
                float op2value = op2.get(row, column);
                result.set(row, column, op1value + op2value);
            }
        }
        return result;
    }

    public static fmatrix dotaddrow(int row1, imatrix op1, int row2, imatrix op2) {
        if (op1.getNrOfColumns() != op2.getNrOfColumns()) {
            System.out.println("DotAddRow Error , column sizes are not the same : " + op1.getNrOfColumns() + " != " + op2.getNrOfColumns());
            return null;
        }
        fmatrix result = new fmatrix(1, op1.getNrOfColumns());

        for (int column = 0; column < result.getNrOfColumns(); ++column) {
            float op1value = op1.get(row1, column);
            float op2value = op2.get(row2, column);
            result.set(0, column, op1value + op2value);
        }

        return result;
    }

    /**
     * Calculates the element by element addition of op1 and op2.
     *
     * @param result the matrix to store the result.
     * @param op1 the first operand.
     * @param op2 the second operand.
     * @return the result matrix
     */
    public static imatrix dotadd(imatrix result, imatrix op1, imatrix op2) {
        if (!equalDimension(op1, op2) || !equalDimension(result, op1)) {
            return null;
        }
        return matrixOp.dotadd(result, op1, op2);
    }

    /**
     * Calculates the element by element addition of factor1 * op1 and factor2 *
     * op2.
     *
     * @param result the matrix to store the result.
     * @param factor1 the first factor.
     * @param op1 the first operand.
     * @param factor2 the second factor.
     * @param op2 the second operand.
     * @return the result matrix
     */
    public static imatrix dotadd(imatrix result, float factor1, imatrix op1, float factor2, imatrix op2) {
        if (!equalDimension(op1, op2) || !equalDimension(result, op1)) {
            return null;
        }
        return matrixOp.dotadd(result, factor1, op1, factor2, op2);
    }

    public static imatrix dotaddrow(int rrow, imatrix result, int row1, imatrix op1, int row2, imatrix op2) {
        if (op1.getNrOfColumns() != op2.getNrOfColumns()) {
            System.out.println("DotAddRow Error , column sizes are not the same : " + op1.getNrOfColumns() + " != " + op2.getNrOfColumns());
            return null;
        }

        for (int column = 0; column < result.getNrOfColumns(); ++column) {
            float op1value = op1.get(row1, column);
            float op2value = op2.get(row2, column);
            result.set(rrow, column, op1value + op2value);
        }
        return result;
    }

    public static fmatrix dotadd(imatrix op1, float op2) {
        fmatrix result = new fmatrix(op1.getNrOfRows(), op1.getNrOfColumns());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row, column);
                result.set(row, column, op1value + op2);
            }
        }
        return result;
    }

    public static imatrix dotsubtract(imatrix op1, imatrix op2) {
        if (!equalDimension(op1, op2)) {
            return null;
        }
        fmatrix result = new fmatrix(op1.getNrOfRows(), op1.getNrOfColumns(), op1.getNrOfSlices());
        return dotsubtract(result, op1, op2);
    }

    public static imatrix dotsubtract(imatrix result, imatrix op1, imatrix op2) {
        if (!equalDimension(op1, op2) || !equalDimension(result, op1)) {
            return null;
        } else {
            return matrixOp.dotsubtract(result, op1, op2);
        }
    }

    public static imatrix dotsubtract(imatrix op1, float op2) {
        fmatrix result = new fmatrix(op1.getNrOfRows(), op1.getNrOfColumns());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row, column);
                result.set(row, column, op1value - op2);
            }
        }
        return result;
    }

    public static void adamVelocity(imatrix velocity, float beta2, imatrix previousVelocity, imatrix deltaWeights) {
        matrixOp.adamVelocity(velocity, beta2, previousVelocity, deltaWeights);
    }

    public static void adamAdaptWeights(imatrix weights, float factor, float beta1, float beta2, float epsilon, imatrix moment, imatrix velocity) {
        matrixOp.adamAdaptWeights(weights, factor, beta1, beta2, epsilon, moment, velocity);
    }

    public static imatrix mergeRows(imatrix op1, imatrix op2) {
        int cs = Math.max(op1.getNrOfColumns(), op2.getNrOfColumns());
        fmatrix result = new fmatrix(op1.getNrOfRows() + op2.getNrOfRows(), cs);

        for (int rows = 0; rows < op1.getNrOfRows(); ++rows) {
            for (int columns = 0; columns < op1.getNrOfColumns(); ++columns) {
                result.set(rows, columns, op1.get(rows, columns));
            }
        }

        for (int rows = 0; rows < op2.getNrOfRows(); ++rows) {
            for (int columns = 0; columns < op2.getNrOfColumns(); ++columns) {
                result.set(rows + op1.getNrOfRows(), columns, op2.get(rows, columns));
            }
        }
        return result;
    }

    public static imatrix mergeColumns(imatrix op1, imatrix op2) {
        int rs = Math.max(op1.getNrOfRows(), op2.getNrOfRows());
        fmatrix result = new fmatrix(rs, op1.getNrOfColumns() + op2.getNrOfColumns());

        for (int rows = 0; rows < op1.getNrOfRows(); ++rows) {
            for (int columns = 0; columns < op1.getNrOfColumns(); ++columns) {
                result.set(rows, columns, op1.get(rows, columns));
            }
        }

        for (int rows = 0; rows < op2.getNrOfRows(); ++rows) {
            for (int columns = 0; columns < op2.getNrOfColumns(); ++columns) {
                result.set(rows, columns + op1.getNrOfColumns(), op2.get(rows, columns));
            }
        }
        return result;
    }

    public static void sumPerColumn(imatrix source, imatrix sums) {
        if (source.getNrOfColumns() != sums.getNrOfColumns()) {
            System.out.println("SumPerColumn Error: number of columns is not the same: " + source.getNrOfColumns() + "!=" + sums.getNrOfColumns());
        }
        for (int c = 0; c < source.getNrOfColumns(); ++c) {
            float sum = 0;
            for (int r = 0; r < source.getNrOfRows(); ++r) {
                sum += source.get(r, c);
            }
            sums.set(0, c, sum);
        }
    }

    public static void sumPerRow(fmatrix source, fmatrix sums) {
//        if (source.getNrOfRows() != sums.getNrOfRows()) {
//            System.out.println("SumPerColumn Error: number of columns is not the same: " + source.getNrOfColumns() + "!=" + sums.getNrOfColumns());
//        }
        for (int r = 0; r < source.getNrOfRows(); ++r) {
            float sum = 0;
            for (int c = 0; c < source.getNrOfColumns(); ++c) {
                sum += source.get(r, c);
            }
            sums.set(0, r, sum);
        }
    }

    public static void copyInto(imatrix toCopy, imatrix dest) {
        matrixOp.copyInto(toCopy, dest);
    }

    /**
     * Copies the first row of every hyperslice of the source matrix into the
     * destination matrix according to a column major ordering.
     *
     * @param source the source matrix.
     * @param dest the destination matrix.
     */
    public static void rowVectorToMatrix(imatrix source, imatrix dest) {

        int eDstRows = dest.getNrOfRows();
        int eDstCols = dest.getNrOfColumns();
        int sliceSize = eDstRows * eDstCols;
        int hyperSlices = Math.min(source.getNrOfHyperSlices(), dest.getNrOfHyperSlices());

        int eSrcCols = source.getNrOfColumns();
        for (int hp = 0; hp < hyperSlices; ++hp) {
            for (int i = 0; i < eSrcCols; ++i) {
                float tc = source.get(0, i, 0, hp);
                int slice = i / sliceSize;
                int sliceIndex = i % sliceSize;
                int row = sliceIndex % eDstRows;
                int col = sliceIndex / eDstRows;
                dest.set(row, col, slice, hp, tc);
            }
        }
    }

    /**
     * Copies the source matrix into the first row of the destination matrix
     * according to column major ordering.
     *
     * @param source the matrix to copy.
     * @param dest the destination of the
     */
    public static void matrixToRowVector(imatrix source, imatrix dest) {
        if (dest.getNrOfRows() > 1) {
            Logger.getLogger(fmatrix.class
                    .getName()).log(Level.INFO, "matrixToRowVector: Destination matrix has more than one row.");
        }
        FloatBuffer src = source.getHostData();
        FloatBuffer dst = dest.getHostData();
        int copies = Math.min(source.getSize(), dest.getNrOfColumns() * dest.getNrOfHyperSlices());
        dst.rewind();
        dst.put(src.array(), 0, copies);
    }

    public static String print(imatrix m) {
        StringBuilder result = new StringBuilder();
        for (int hyperSlice = 0; hyperSlice < m.getNrOfHyperSlices(); ++hyperSlice) {
            result.append("| Hyperslice ");
            result.append((hyperSlice + 1));
            result.append(" |\n");
            for (int slice = 0; slice < m.getNrOfSlices(); ++slice) {
                String[][] cells;
                result.append("| Slice ");
                result.append((slice + 1));
                result.append(" |\n");
                cells = new String[m.getNrOfRows()][m.getNrOfColumns()];
                int[] widths = new int[m.getNrOfColumns()];
                for (int row = 0; row < m.getNrOfRows(); ++row) {
                    for (int column = 0; column < m.getNrOfColumns(); ++column) {
                        String fs = Float.toString(m.get(row, column, slice, hyperSlice));
                        cells[row][column] = fs;
                        if (fs.length() > widths[column]) {
                            widths[column] = fs.length();
                        }
                    }
                }
                for (int i = 0; i < widths.length; ++i) {
                    widths[i] = (widths[i] / 8 + 1) * 8;
                }

                for (int row = 0; row < m.getNrOfRows(); ++row) {
                    for (int column = 0; column < m.getNrOfColumns(); ++column) {
                        int maxwidth = widths[column];
                        String toAdd = cells[row][column];
                        int charsToAdd = maxwidth - toAdd.length();
                        result.append(toAdd);
                        for (int i = 0; i < charsToAdd; ++i) {
                            result.append(' ');
                        }
                    }
                    result.append('\n');
                }
            }
        }
        return result.toString();
    }

    /**
     * Checks if the two matrices have equal dimensions.
     *
     * @param op1 the first matrix.
     * @param op2 the second matrix.
     * @return true if the dimensions are equal, false otherwise.
     */
    public static boolean equalDimension(imatrix op1, imatrix op2) {
        return op1.getNrOfRows() == op2.getNrOfRows()
                && op1.getNrOfColumns() == op2.getNrOfColumns()
                && op1.getNrOfSlices() == op2.getNrOfSlices();
    }

    private static Range parseRange(String range) {
        Range r = new Range();
        int firstColon = range.indexOf(':');
        if (firstColon > -1) {
            r.singleton = false;
            r.startOfRange = Float.parseFloat(range.substring(0, firstColon));
            int secondColon = range.indexOf(':', firstColon + 1);
            if (secondColon > -1) {
                r.increment = Float.parseFloat(range.substring(firstColon + 1, secondColon));
                r.endOfRange = Float.parseFloat(range.substring(secondColon + 1));
            } else {
                r.endOfRange = Float.parseFloat(range.substring(firstColon + 1));
                r.increment = (r.endOfRange > r.startOfRange) ? 1 : -1;
            }
        } else {
            // try singleton
            try {
                float number = Float.parseFloat(range);
                r.startOfRange = number;
                r.endOfRange = number;
                r.singleton = true;
            } catch (NumberFormatException ex) {
                ex.printStackTrace();
            }
        }
        return r;
    }

    @Override
    public String getSizeAsString() {
        return "[" + getNrOfRows() + "," + getNrOfColumns() + "]";
    }

    @Override
    public String toString() {
        return fmatrix.print(this);
    }

    public void randomize(float min, float max) {
        Random r = new Random();
        this.applyFunction(x -> (r.nextFloat() * (max - min)) + min);
    }

    @Override
    public FloatDeviceBuffer getDeviceBuffer() {
        return deviceBuffer;
    }

    public static void writeAs2DImage(imatrix m, Path location) {
        float max = m.max().value;
        float min = m.min().value;

        float factor = 255f / (max - min);
        System.out.println("factor: " + factor);

        BufferedImage bi = new BufferedImage(m.getNrOfColumns(), m.getNrOfRows(), BufferedImage.TYPE_BYTE_GRAY);

        for (int r = 0; r < m.getNrOfRows(); ++r) {
            for (int c = 0; c < m.getNrOfColumns(); ++c) {
                float p = (m.get(r, c) - min) * factor;
                int pi = (int) Math.round(p);
                bi.setRGB(c, r, (pi << 16) + (pi << 8) + pi);
            }
        }

        String homeDir = System.getProperty("user.home");
        Path exportPath = Paths.get(homeDir, ".nn", location + ".png");
        try {
            Files.createDirectories(exportPath);
            ImageIO.write(bi, "png", exportPath.toFile());

        } catch (IOException ex) {
            Logger.getLogger(Layer.class
                    .getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void writeAs3DImage(imatrix m, int slicesPerRow, int padding, Path location) {
        float max = m.max().value;
        float min = m.min().value;

        float factor = 255f / (max - min);
        // (x-min)*factor

        int nrOfSlicesCols = (m.getNrOfSlices() / slicesPerRow) + 1;

        BufferedImage bi = new BufferedImage(
                (m.getNrOfColumns() + 5) * nrOfSlicesCols,
                (m.getNrOfRows() + 5) * slicesPerRow,
                BufferedImage.TYPE_BYTE_GRAY);
        for (int slice = 0; slice < m.getNrOfSlices(); ++slice) {
            int imageRowB = (slice % slicesPerRow) * (m.getNrOfRows() + padding);
            int imageColB = (slice / slicesPerRow) * (m.getNrOfColumns() + padding);

            for (int r = 0; r < m.getNrOfRows(); ++r) {
                for (int c = 0; c < m.getNrOfColumns(); ++c) {
                    float p = (m.get(r, c, slice) - min) * factor;
                    int pi = (int) Math.round(p);
                    bi.setRGB(imageColB + c, imageRowB + r, (pi << 16) + (pi << 8) + pi);
                }
            }
        }

        String homeDir = System.getProperty("user.home");
        Path exportPath = Paths.get(homeDir, ".nn", location + ".png");
        try {
            Files.createDirectories(exportPath);
            ImageIO.write(bi, "png", exportPath.toFile());

        } catch (IOException ex) {
            Logger.getLogger(Layer.class
                    .getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Synchronizes the host buffer with the device buffer if necessary.
     */
    @Override
    public void sync() {
        this.deviceBuffer.syncHost();
    }

    public void makeMaster() {
        deviceBuffer.markCpuMatrixAsMatrix();
    }
}
