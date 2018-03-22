/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix;

import dae.matrix.gpu.FloatDeviceBuffer;
import dae.neuralnet.activation.Function;
import java.nio.FloatBuffer;

/**
 * Provides a different view on the same underlying matrix in terms of rows,
 * columns and slices. The number of hyper slices is not changed.
 *
 * The matrix view must have the same hyperslice size of the underlying matrix,
 * otherwise the operation will fail.
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class fmatrixview implements imatrix {

    private final imatrix source;

    private final int rows;
    private final int columns;
    private final int slices;

    private final int sliceSize;
    private final int hyperSliceSize;

    public fmatrixview(int rows, int columns, int slices, imatrix source) {
        this.rows = rows;
        this.columns = columns;
        this.slices = slices;

        this.sliceSize = rows * columns;
        this.hyperSliceSize = sliceSize * slices;

        if (hyperSliceSize != source.getHyperSliceSize()) {
            throw new IllegalArgumentException("Hyperslice sizes do not correspond, " + hyperSliceSize + " != " + source.getHyperSliceSize());
        }
        this.source = source;
    }
    
    public imatrix getSource(){
        return source;
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

    @Override
    public String getName() {
        return source.getName() + "_view";
    }

    @Override
    public boolean isRowVector() {
        return source.isRowVector();
    }

    @Override
    public boolean isBatchMatrix() {
        return source.isBatchMatrix();
    }

    @Override
    public void set(int row, int column, float value) {

    }

    @Override
    public void set(int row, int column, int slice, float value) {

    }

    @Override
    public void set(int row, int column, int slice, int hyperslice, float value) {

    }

    @Override
    public void setRow(int row, float[] values) {
    }

    @Override
    public void reset() {
        source.reset();
    }

    @Override
    public void getRow(int row, imatrix rowStorage) {

    }

    @Override
    public void getRow(int row, int targetRow, imatrix rowStorage) {

    }

    @Override
    public void setColumn(int column, float... values) {

    }

    @Override
    public void getColumn(int column, imatrix columnStorage) {

    }

    @Override
    public void getColumn(int column, int targetColumn, imatrix columnStorage) {

    }

    @Override
    public float get(int row, int column) {
        return source.getHostData().get(rcToIndex(row, column));
    }

    @Override
    public float get(int row, int column, int slice) {
        return source.getHostData().get(rcsToIndex(row, column, slice));
    }

    @Override
    public float get(int row, int column, int slice, int hyperslice) {
        return source.getHostData().get(rcshToIndex(row, column, slice, hyperslice));
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
        return source.getNrOfHyperSlices();
    }

    @Override
    public int getSliceSize() {
        return sliceSize;
    }

    @Override
    public int getHyperSliceSize() {
        return hyperSliceSize;
    }

    @Override
    public int getSize() {
        return source.getSize();
    }

    @Override
    public int getZeroPadding() {
        return source.getZeroPadding();
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

    }

    @Override
    public String getSizeAsString() {
        return "[ " + getNrOfRows() + " , " + getNrOfColumns() + " , " + getNrOfSlices() +" , " +getNrOfHyperSlices() + " ]";
    }

    @Override
    public void applyFunction(Function f) {
        source.applyFunction(f);
    }

    @Override
    public imatrix copy() {
        return null;
    }

    @Override
    public FloatBuffer getHostData() {
        return source.getHostData();
    }

    @Override
    public boolean isTransposed() {
        return source.isTransposed();
    }

    @Override
    public FloatDeviceBuffer getDeviceBuffer() {
        return source.getDeviceBuffer();
    }

    @Override
    public void sync() {
        source.sync();
    }

    @Override
    public void makeMaster() {
        source.makeMaster();
    }
}
