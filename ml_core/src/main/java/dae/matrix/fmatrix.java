package dae.matrix;

import dae.matrix.gpu.FMatrixOpGpu;
import dae.matrix.op.FMatrixOp;
import dae.neuralnet.activation.Function;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Random;
import org.jocl.Pointer;
import org.jocl.cl_mem;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public strictfp class fmatrix implements imatrix {

    private final int rows;
    private final int columns;
    private final int slices;

    private final int sliceSize;
    private final int size;

    // private float[] data;
    private final FloatBuffer data;
    private ByteBuffer bb;
    private cl_mem rMem;
    private cl_mem rwMem;

    private final int[] padding = new int[2];

    private static FMatrixOp matrixOp = new FMatrixOpGpu();

    /**
     * Creates a new fmatrix object with the given rows and columns and 1 slice.
     *
     * @param rows the number of rows in the matrix.
     * @param columns the number of columns in the matrix.
     */
    public fmatrix(int rows, int columns) {
        this(rows, columns, 1);
    }

    /**
     * Creates a new fmatrix object with the given rows and columns.
     *
     * @param rows the number of rows in the matrix.
     * @param columns the number of columns in the matrix.
     * @param slices the number of slices in the matrix.
     */
    public fmatrix(int rows, int columns, int slices) {
        this.rows = rows;
        this.columns = columns;
        this.slices = slices;
        this.sliceSize = rows * columns;
        this.size = sliceSize * slices;
        data = FloatBuffer.allocate(size);

        padding[0] = 32 - (columns % 32);
        padding[1] = 32 - (rows % 32);
    }

    public fmatrix(fmatrix toCopy) {
        this(toCopy.rows, toCopy.columns, toCopy.slices);
        toCopy.data.rewind();

        while (this.data.hasRemaining()) {
            this.data.put(toCopy.data.get());
        }
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
        data.rewind();
        for (int i = 0; i < data.limit(); ++i) {
            data.put(0);
        }
        data.rewind();
    }

    /**
     * Converts a row and column coordinate to a 1D coordinate. The slice number
     * is assumed to be zero.
     *
     * @param r the row of the cell.
     * @param c the column of the cell.
     * @return the index of the cell in the 1D float backing array.
     */
    private int rcToIndex(int r, int c) {
        return c * rows + r;
    }

    /**
     * Converts a row and column coordinate to a 1D coordinate.
     *
     * @param r the row of the cell.
     * @param c the column of the cell.
     * @param s the slice number of the cell.
     * @return the index of the cell in the 1D float backing array.
     */
    private int rcsToIndex(int r, int c, int s) {
        return r + c * rows + s * sliceSize;
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
        for (int slice = 0; slice < getNrOfSlices(); ++slice) {
            for (int row = 0; row < getNrOfRows(); ++row) {
                for (int column = 0; column < getNrOfColumns(); ++column) {
                    it.cell(this, row, column, slice, get(row, column, slice));
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
     * Sets a cell in this matrix to the given value. The slice number is
     * assumed to be zero.
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
        return data.get(rcToIndex(row, column));
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
        return data.get(rcsToIndex(row, column, slice));
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
     * Gets the maximum value in the matrix.
     *
     * @param result a Cell object that will store the result.
     * @return the result parameter.
     */
    @Override
    public Cell max(Cell result) {
        float max = Float.MIN_VALUE;
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

    public void softMaxPerRow() {
        exp();
        for (int r = 0; r < getNrOfRows(); ++r) {
            float sum = 0;
            for (int c = 0; c < getNrOfColumns(); ++c) {
                float value = get(r, c);
                sum += value;
            }
            for (int c = 0; c < getNrOfColumns(); ++c) {
                float value = get(r, c) / sum;
                set(r, c, value);
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
        fmatrix result = new fmatrix(getNrOfRows(), getNrOfColumns());
        for (int row = 0; row < getNrOfRows(); ++row) {
            for (int column = 0; column < getNrOfColumns(); ++column) {
                result.set(row, column, get(row, column));
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

    public static fmatrix dotmultiply(fmatrix op1, fmatrix op2) {
        if (op1.getNrOfRows() != op2.getNrOfRows() || op1.getNrOfColumns() != op2.getNrOfColumns()) {
            System.out.println("DotMultiply Error , matrix dimension are not the same " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }
        fmatrix result = new fmatrix(op1.getNrOfRows(), op2.getNrOfRows());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row, column);
                float op2value = op2.get(row, column);
                result.set(row, column, op1value * op2value);
            }
        }
        return result;
    }

    public static imatrix dotmultiply(imatrix result, imatrix op1, imatrix op2) {
        if (result.getNrOfRows() != op2.getNrOfRows() || result.getNrOfColumns() != op2.getNrOfColumns()
                || op1.getNrOfRows() != op2.getNrOfRows() || op1.getNrOfColumns() != op2.getNrOfColumns()) {
            System.out.println("DotMultiply Error , matrix dimension are not the same " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }
        //fmatrix result = new fmatrix(op1.getNrOfRows(), op2.getNrOfRows());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row, column);
                float op2value = op2.get(row, column);
                result.set(row, column, op1value * op2value);
            }
        }
        return result;
    }

    public static fmatrix dotmultiply(fmatrix op1, float op2) {
        fmatrix result = new fmatrix(op1.getNrOfRows(), op1.getNrOfColumns());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row + 1, column + 1);
                result.set(row, column, op1value * op2);
            }
        }
        return result;
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

    public static imatrix dotadd(imatrix result, imatrix op1, imatrix op2) {
        if (!equalDimension(op1, op2) || !equalDimension(result, op1)) {
            System.out.println("DotAdd Error , matrix dimension are not the same " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }

        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row, column);
                float op2value = op2.get(row, column);
                result.set(row, column, op1value + op2value);
            }
        }
        return result;
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
            System.out.println("DotSubtract Error , matrix dimension are not the same " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }
        fmatrix result = new fmatrix(op1.getNrOfRows(), op1.getNrOfColumns());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row, column);
                float op2value = op2.get(row, column);
                result.set(row, column, op1value - op2value);
            }
        }
        return result;
    }

    public static imatrix dotsubtract(imatrix result, imatrix op1, imatrix op2) {
        if (!equalDimension(op1, op2) || !equalDimension(result, op1)) {
            System.out.println("DotSubtract Error , matrix dimension are not the same " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }

        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row, column);
                float op2value = op2.get(row, column);
                result.set(row, column, op1value - op2value);
            }
        }
        return result;
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

    public static void copyInto(imatrix toCopy, imatrix dest) {
        if (toCopy.isTransposed() == dest.isTransposed()
                && equalDimension(toCopy, dest)) {
            FloatBuffer srcData = toCopy.getHostData();
            FloatBuffer destData = dest.getHostData();
            srcData.rewind();
            destData.rewind();

            destData.put(srcData);

        } else {
            int maxRow = toCopy.getNrOfRows() < dest.getNrOfRows() ? toCopy.getNrOfRows() : dest.getNrOfRows();
            int maxColumn = toCopy.getNrOfColumns() < dest.getNrOfColumns() ? toCopy.getNrOfColumns() : dest.getNrOfColumns();
            for (int row = 0; row < maxRow; ++row) {
                for (int column = 0; column < maxColumn; ++column) {
                    dest.set(row, column, toCopy.get(row, column));
                }
            }
        }
    }

    public static String print(imatrix m) {
        StringBuilder result = new StringBuilder();

        for (int slice = 0; slice < m.getNrOfSlices(); ++slice) {
            String[][] cells;
            result.append("Slice ");
            result.append((slice + 1));
            result.append("\n-------\n");
            cells = new String[m.getNrOfRows()][m.getNrOfColumns()];
            int[] widths = new int[m.getNrOfColumns()];
            for (int row = 0; row < m.getNrOfRows(); ++row) {
                for (int column = 0; column < m.getNrOfColumns(); ++column) {
                    String fs = Float.toString(m.get(row, column, slice));
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
        return op1.getNrOfRows() == op2.getNrOfRows() && op1.getNrOfColumns() == op2.getNrOfColumns();
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

    public void randomize(int i, int i0) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public cl_mem getCLReadMem() {
        if (rMem == null) {
            System.out.println("creating new mem r buffer");
            rMem = FMatrixOpGpu.createReadMem(this, padding[0], padding[1]);
        }
        return rMem;
    }

    @Override
    public cl_mem getCLReadWriteMem() {
        if (rwMem == null) {
            System.out.println("creating new mem rw buffer");
            rwMem = FMatrixOpGpu.createReadWriteMem(this, padding[0], padding[1]);
        }
        return rwMem;
    }

    @Override
    public Pointer getCLPointer() {
        return Pointer.to(data.array());
    }

    /**
     * Get the padding for the columns.
     *
     * @return the padding for the columns.
     */
    @Override
    public int getColPadding() {
        return padding[0];
    }

    /**
     * Get the padding for the rows.
     *
     * @return the padding for the rows.
     */
    @Override
    public int getRowPadding() {
        return padding[1];
    }

    /**
     * Get the number of columns on the device.
     *
     * @return the number of columns on the gpu device.
     */
    @Override
    public int getDeviceColumns() {
        return columns + getColPadding();
    }

    /**
     * Get the number of rows on the device.
     *
     * @return the number of rows on the gpu device.
     */
    @Override
    public int getDeviceRows() {
        return rows + getRowPadding();
    }
}
