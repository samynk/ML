package dae.matrix.integer;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public interface IntCellIterator {
    /**
     * Iterator method that can be used to iterate over all the values of the matrix.
     * @param source the fmatrix source object.
     * @param row the row of the current cell.
     * @param column the column of the current cell.
     * @param slice the slice of the current cell.
     * @param hyperSlice the hyperslice of the current cell.
     * @param currentValue the current value of the cell.
     */
    public void cell(intmatrix source, int row, int column, int slice, int hyperSlice, int currentValue);
}
