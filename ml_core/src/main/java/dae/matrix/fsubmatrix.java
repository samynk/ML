/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dae.matrix;

import dae.neuralnet.activation.Function;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import org.jocl.Pointer;
import org.jocl.cl_mem;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public class fsubmatrix implements imatrix {

    private final imatrix source;
    private final int rb;
    private final int cb;
    private final int rows;
    private final int columns;

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
        this.source = source;
        this.rb = Math.max(rb, 0);
        this.cb = Math.max(cb, 0);

        this.rows = Math.min(source.getNrOfRows(), Math.max(rows, 0));
        this.columns = Math.min(source.getNrOfColumns(), Math.max(columns, 0));
    }

    @Override
    public void set(int row, int column, float value) {
        source.set(row - rb, column - cb, value);
    }

    @Override
    public void setRow(int row, float[] values) {
        source.setRow(row - rb, values);
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
    public int getNrOfRows() {
        return rows;
    }

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
    public int getSize(){
        return rows*columns;
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
                copy.set(r,c, get(r,c));
            }
        }
        return copy;
    }

    @Override
    public FloatBuffer getRawData() {
        return source.getRawData();
    }

    @Override
    public boolean isTransposed() {
        return source.isTransposed();
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
    public Pointer getCLPointer() {
        return source.getCLPointer();
    }

    @Override
    public ByteBuffer getBuffer() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
