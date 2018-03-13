/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix.integer;

import dae.matrix.gpu.IntDeviceBuffer;
import dae.matrix.gpu.IntMatrixOpGpu;
import java.nio.IntBuffer;
import org.jocl.Pointer;
import org.jocl.cl_mem;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class intmatrix {

    private String name = "intmatrix" + (count++);
    private static int count = 0;

    private final int rows;
    private final int columns;
    private final int slices;
    private final int hyperslices;

    private final int zeropadding;

    private final int sliceSize;
    private final int hyperSliceSize;
    private final int size;

    // private float[] data;
    private final IntBuffer data;
    private final IntDeviceBuffer deviceBuffer;

    /**
     * Creates a new intmatrix object with the given rows and columns and 1
     * slice.
     *
     * @param rows the number of rows in the matrix.
     * @param columns the number of columns in the matrix.
     */
    public intmatrix(int rows, int columns) {
        this(rows, columns, 1, 1, 0);
    }

    /**
     * Creates a new intmatrix object with the given rows and columns.
     *
     * @param rows the number of rows in the matrix.
     * @param columns the number of columns in the matrix.
     * @param slices the number of slices in the matrix.
     */
    public intmatrix(int rows, int columns, int slices) {
        this(rows, columns, slices, 1, 0);
    }

    /**
     * Creates a new intmatrix object with the given rows and columns.
     *
     * The zero padding adds a virtual number of rows and columns around the
     * intmatrix. This means that the zero padded row is a negative row on the
     * upper side of the matrix and a negative column on the left side of the
     * matrix. On the right and bottom side of the matrix the zero padded column
     * and row have indexes greater than the number of rows and columns in this
     * matrix.
     *
     * @param rows the number of rows in the matrix.
     * @param columns the number of columns in the matrix.
     * @param slices the number of slices in the matrix.
     * @param hyperslices the number of hyperslices in the matrix.
     * @param zeropadding the zero padding to add around this matrix.
     */
    public intmatrix(int rows, int columns, int slices, int hyperslices, int zeropadding) {
        this.rows = rows;
        this.columns = columns;
        this.slices = slices;
        this.hyperslices = hyperslices;
        this.zeropadding = zeropadding;
        this.sliceSize = this.rows * this.columns;
        this.hyperSliceSize = sliceSize * this.slices;
        this.size = hyperSliceSize * hyperslices;
        data = IntBuffer.allocate(size);
        deviceBuffer = new IntDeviceBuffer(this);
    }

    /**
     * Creates a new intmatrix from the given parameter.
     *
     * @param toCopy the intmatrix to copy.
     */
    public intmatrix(intmatrix toCopy) {
        this(toCopy.rows, toCopy.columns, toCopy.slices, toCopy.hyperslices, toCopy.zeropadding);
        toCopy.data.rewind();
        this.data.rewind();

        while (this.data.hasRemaining()) {
            this.data.put(toCopy.data.get());
        }
    }
    
    /**
     * Returns the name of this int matrix.
     * @return the name.
     */
    public String getName(){
        return name;
    }

    /**
     * Gets the raw data of the matrix.
     *
     * @return the raw data.
     */
    public IntBuffer getHostData() {
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
     * Converts a row,column and slice coordinate to a 1D coordinate.
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
     * Converts a row,column and slice coordinate to a 1D coordinate.
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
    public boolean isTransposed() {
        return false;
    }

    /**
     * Calls the cell method on the cell iterator.
     *
     * @param it the CellIterator object.
     */
    public void iterateCells(IntCellIterator it) {
        for (int hyperSlice = 0; hyperSlice < getNrOfSlices(); ++hyperSlice) {
            for (int slice = 0; slice < getNrOfSlices(); ++slice) {
                for (int row = 0; row < getNrOfRows(); ++row) {
                    for (int column = 0; column < getNrOfColumns(); ++column) {
                        it.cell(this, row, column, slice, hyperSlice, get(row, column, slice, hyperSlice));
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
    public void set(int row, int column, int value) {
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
    public void set(int row, int column, int slice, int value) {
        data.put(rcsToIndex(row, column, slice), value);
    }

    /**
     * Gets the value of a cell. The slice number is one.
     *
     * @param row the row of the cell.
     * @param column the column of the cell.
     * @return the value of the cell.
     */
    public int get(int row, int column) {
        int index = rcToIndex(row, column);
        if (index < data.limit()) {
            return data.get(index);
        } else {
            return 0;
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
    public int get(int row, int column, int slice) {
        int index = rcsToIndex(row, column, slice);
        if (index < data.limit()) {
            return data.get(index);
        } else {
            return 0;
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
    public int get(int row, int column, int slice, int hyperslice) {
        int index = rcshToIndex(row, column, slice, hyperslice);
        if (index < data.limit()) {
            return data.get(index);
        } else {
            return 0;
        }
    }

    /**
     * Returns the number of rows.
     *
     * @return the number of rows.
     */
    public int getNrOfRows() {
        return rows;
    }

    /**
     * Returns the number of columns.
     *
     * @return the number of columns.
     */
    public int getNrOfColumns() {
        return columns;
    }

    /**
     * Returns the number of slices in the matrix.
     *
     * @return the number of slices.
     */
    public int getNrOfSlices() {
        return slices;
    }

    /**
     * Returns the number of slices in the matrix.
     *
     * @return the number of slices.
     */
    public int getNrOfHyperSlices() {
        return slices;
    }

    /**
     * Gets the amount of zero padding in this matrix.
     *
     * @return the zero padding in this matrix.
     */
    public int getZeroPadding() {
        return zeropadding;
    }

    /**
     * Returns the total number of cells in this matrix in a single slice.
     *
     * @return the total number of cells in the matrix in a single slice.
     */
    public int getSliceSize() {
        return rows * columns;
    }

    /**
     * Returns the size of the matrix.
     *
     * @return the size of the matrix.
     */
    public int getSize() {
        return this.size;
    }

    public Pointer getCLPointer() {
        return Pointer.to(data.array());
    }

    public IntDeviceBuffer getDeviceBuffer() {
        return deviceBuffer;
    }
    
    public void sync(){
        deviceBuffer.syncHost();
    }

    @Override
    public String toString() {
        return print(this);
    }

    // static functions
    public static String print(intmatrix m) {
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
                        String fs = Integer.toString(m.get(row, column, slice));
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
}
