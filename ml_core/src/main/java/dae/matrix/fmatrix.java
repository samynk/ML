package dae.matrix;

import dae.matrix.gpu.FMaxtrixOpGpu;
import dae.neuralnet.activation.Function;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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

    // private float[] data;
    private FloatBuffer data;
    private ByteBuffer bb;
    private cl_mem rMem;
    private cl_mem rwMem;

    /**
     * Creates a new fmatrix object with the given rows and columns.
     *
     * @param rows the number of rows in the matrix.
     * @param columns the number of columns in the matrix.
     */
    public fmatrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        
        data = FloatBuffer.allocate(rows * columns);
    }

    /**
     * Gets the raw data of the matrix.
     *
     * @return the raw data.
     */
    @Override
    public FloatBuffer getRawData() {
        return data;
    }

    @Override
    public ByteBuffer getBuffer() {
        return bb;
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

    private int rcToIndex(int r, int c) {
        return c * rows + r;
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
        for (int row = 0; row < getNrOfRows(); ++row) {
            for (int column = 0; column < getNrOfColumns(); ++column) {
                it.cell(this, row, column, get(row, column));
            }
        }
    }

    /**
     * Sets a cell in this matrix to the given value.
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
     * Sets a row in the matrix to the given values.
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
     * Gets the value of a cell.
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
     * Returns the total number of cells in this matrix.
     *
     * @return the total number of cells in the matrix.
     */
    @Override
    public int getSize() {
        return rows * columns;
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

    public void applyCellFunction(IndexedFunction f) {
        Cell temp = new Cell();
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < columns; ++c) {
                float v = get(r, c);
                temp.column = c;
                temp.row = r;
                temp.value = v;
                float vf = f.evaluate(temp);
                set(r, c, vf);
            }
        }
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

        result.iterateCells((fmatrix source, int row, int column, float currentValue) -> {
            float value = (r.nextFloat() * diff) + minValue;
            source.set(row, column, value);
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
        //NativeBlas.sgemm('N', 'N', c.rows, c.columns, a.columns, alpha, a.data, 0,
        //			a.rows, b.data, 0, b.rows, beta, c.data, 0, c.rows);
        FMaxtrixOpGpu.sgemm(alpha, a, b, beta, c);
//        char opA = 'N';
//        int lda = a.getNrOfRows();
//        if (a.isTransposed()) {
//            opA = 'T';
//            lda = a.getNrOfColumns();
//        }
//        char opB = 'N';
//        int ldb = b.getNrOfRows();
//        if (b.isTransposed()) {
//            opB = 'T';
//            ldb = b.getNrOfColumns();
//        }

//        NativeBlas.sgemm(opA, opB, c.getNrOfRows(), c.getNrOfColumns(), a.getNrOfColumns(), alpha, a.getRawData().array(), 0,
//                lda, b.getRawData().array(), 0, ldb, beta, c.getRawData().array(), 0, c.getNrOfRows());
        return c;
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
            FloatBuffer srcData = toCopy.getRawData();
            FloatBuffer destData = dest.getRawData();
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

    public fmatrix submatrix(String rowRange, String columnRange) {
        return null;
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
        String[][] cells;
        cells = new String[getNrOfRows()][getNrOfColumns()];
        int[] widths = new int[getNrOfColumns()];
        for (int row = 0; row < getNrOfRows(); ++row) {
            for (int column = 0; column < getNrOfColumns(); ++column) {
                String fs = Float.toString(get(row, column));
                cells[row][column] = fs;
                if (fs.length() > widths[column]) {
                    widths[column] = fs.length();
                }
            }
        }
        for (int i = 0; i < widths.length; ++i) {
            widths[i] = (widths[i] / 8 + 1) * 8;
        }
        StringBuilder result = new StringBuilder();

        for (int row = 0; row < getNrOfRows(); ++row) {
            for (int column = 0; column < getNrOfColumns(); ++column) {
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

        return result.toString();
    }

    public void randomize(int i, int i0) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public cl_mem getCLReadMem() {
        if (rMem == null) {
            rMem = FMaxtrixOpGpu.createReadMem(this);
        }
        return rMem;
    }

    @Override
    public cl_mem getCLReadWriteMem() {
        if (rwMem == null) {
            rwMem = FMaxtrixOpGpu.createReadWriteMem(this);
        }
        return rwMem;
    }

    @Override
    public Pointer getCLPointer() {
        return Pointer.to(data.array());
    }
}
