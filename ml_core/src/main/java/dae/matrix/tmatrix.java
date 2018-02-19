package dae.matrix;

import dae.neuralnet.activation.Function;
import java.nio.FloatBuffer;
import org.jocl.Pointer;
import org.jocl.cl_mem;

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

    public void setSource(imatrix source) {
        this.source = source;
    }

    @Override
    public void set(int row, int column, float value) {
        source.set(column, row, value);
    }

    @Override
    public void setRow(int row, float[] values) {
        source.setColumn(row, values);
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
    public int getNrOfRows() {
        return source.getNrOfColumns();
    }

    @Override
    public int getNrOfColumns() {
        return source.getNrOfRows();
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
    public FloatBuffer getRawData() {
        return source.getRawData();
    }

    /**
     * Checks if this is a transposed view on the source data.
     */
    @Override
    public boolean isTransposed() {
        return !source.isTransposed();
    }

    @Override
    public cl_mem getCLReadMem() {
        return source.getCLReadMem();
    }
    
    @Override
    public cl_mem getCLReadWriteMem() {
        return source.getCLReadWriteMem();
    }
    
    @Override
    public Pointer getCLPointer(){
        return source.getCLPointer();
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
}
