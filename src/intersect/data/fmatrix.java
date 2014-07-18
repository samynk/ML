/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package intersect.data;

import java.util.ArrayList;
import java.util.Random;



/**
 *
 * @author Koen
 */
public strictfp class fmatrix {

    
    private int rows;
    private int columns;

    private float[][] data;
    private boolean transposed = false;

    public fmatrix(int rows, int columns){
        this.rows = rows;
        this.columns = columns;
        data = new float[rows][columns];
    }
    
    public void reset(){
         for (int row = 0; row < getNrOfRows(); ++row)
            for (int column = 0; column < getNrOfColumns(); ++column)
            {
                set(row+1, column+1, 0);
            }
    }
    
    /**
     * Adds a row to the bottom (or in the case of a transposed matrix to the
     * right.
     * @param values 
     */
    public void addRow(ArrayList<Float> values){
        if ( transposed ){
            int nrOfRows = values.size();
            int newRowSize = nrOfRows > rows ? nrOfRows : rows;
            float[][] newdata = new float[newRowSize][columns+1];
            
            for (int row = 0; row < rows; ++row) {
                System.arraycopy(data[row], 0, newdata[row], 0, columns);
            }
            
            for (int r = 0; r < nrOfRows;++r)
                newdata[r][columns] = values.get(r);
            
            ++columns;
            rows = newRowSize;
            data = newdata;
        }else{
            int nrOfCols = values.size();

            int newColSize = nrOfCols > columns ? nrOfCols : columns;
            // resize needed
            float[][] newdata = new float[rows + 1][newColSize];
            // copy old data
            for (int row = 0; row < rows; ++row) {
                System.arraycopy(data[row], 0, newdata[row], 0, columns);
            }
            for (int c = 0; c < nrOfCols; ++c) {
                newdata[rows][c] = values.get(c);
            }
            rows = rows + 1;
            columns = newColSize;
            data = newdata;
        }
    }

    public void iterateCells(CellIterator it)
    {
        for (int row = 0; row < getNrOfRows(); ++row)
            for (int column = 0; column < getNrOfColumns(); ++column)
            {
                it.cell(this,row+1, column+1, get(row+1,column+1));
            }
    }

    public void set(int row, int column, float value){
        if ( transposed )
            data[column-1][row-1]=value;
        else
            data[row-1][column-1]=value;
    }
    
    public void setRow(int row, float[] values){
        if ( transposed ){
            int maxIndex = values.length > rows?values.length:rows;
            for (int column = 1; column <= maxIndex; ++column)
                set(row,column,values[column]);
        }else{
            System.arraycopy(values, 0, this.data[row-1],0, this.columns);
        }
    }
    
    public void setColumn(int column, float[] values){
        if (transposed){
            System.arraycopy(values,0,this.data[column-1],0,this.rows);
        }else{
            int maxIndex = values.length > getNrOfRows()?values.length:getNrOfRows();
            for (int row = 1; row <= maxIndex; ++row)
                set(row,column,values[row-1]);
        }
    }

    public float get(int row, int column){
        if ( transposed)
            return data[column-1][row-1];
        else
            return data[row-1][column-1];
    }
    
    public  Cell max(){
        Cell result =new Cell();
        float max = Float.MIN_VALUE;
           for (int row = 0; row < getNrOfRows(); ++row)
            for (int column = 0; column < getNrOfColumns(); ++column)
            {
                float value = this.get(row+1,column+1);
                if ( value > max){
                    max = value;
                    result.value= max;
                    result.row = row+1;
                    result.column = column+1;
                }
                
            }
           return result;
    }
    
    public fmatrix getRow(int row){
        System.out.println("gettring row of matrix = " + transposed);
        if ( transposed ){
            
            fmatrix result = new fmatrix(1,getNrOfColumns());
            for ( int column = 1 ;  column <= getNrOfColumns(); ++column)
                result.set(1,column,this.get(row,column));
            return result;
        }else{
            fmatrix result = new fmatrix(1,getNrOfColumns());
            result.setRow(1,data[row-1]);
            return result;
        }
    }
    
    public fmatrix getColumn(int column){
        if (transposed){
            fmatrix result = new fmatrix(getNrOfRows(),1);
            float[] values = data[column-1];
            result.setColumn(1,values);
            return result;
        }else{
            fmatrix result = new fmatrix(getNrOfRows(),1);
            for (int row=1;row <= rows;++row)
                result.set(row, 1, this.get(row,column));
            return result;
        }
    }

    public int getNrOfRows(){
        if ( transposed )
            return columns;
        else
            return rows;
    }

    public int getNrOfColumns(){
        if ( transposed)
            return rows;
        else
            return columns;
    }

    public void transpose(){
        transposed = !transposed;
    }

    public float sum(){
        float sum = 0.0f;
        for (int row = 0; row < getNrOfRows(); ++row)
            for (int column = 0; column < getNrOfColumns(); ++column)
            {
                sum+= get(row+1, column+1);
            }
        return sum;
    }

    public void multiply(float value){
        for (int row = 0; row < getNrOfRows(); ++row)
            for (int column = 0; column < getNrOfColumns(); ++column)
            {
                set( row+1,column+1, get(row+1, column+1) * value);
            }
    }

    public void add(float value){
        for (int row = 0; row < getNrOfRows(); ++row)
            for (int column = 0; column < getNrOfColumns(); ++column)
            {
                set( row+1,column+1, get(row+1, column+1) + value);
            }
    }
    
    public void clamp(float min, float max){
        for (int row = 0; row < getNrOfRows(); ++row)
            for (int column = 0; column < getNrOfColumns(); ++column)
            {
                float value = get(row+1, column+1);
                if ( value < min)
                    value = min;
                if ( value > max)
                    value = max;
                set( row+1,column+1, value);
            }
    }
    
    public void add(fmatrix op2){
        int maxRow = getNrOfRows()  < op2.getNrOfRows()?getNrOfRows():op2.getNrOfRows();
        int maxColumn = getNrOfColumns()  < op2.getNrOfColumns()?getNrOfColumns():op2.getNrOfColumns();
         for (int row = 0; row < maxRow; ++row)
            for (int column = 0; column < maxColumn; ++column)
            {
                set( row+1,column+1, get(row+1, column+1) + op2.get(row+1, column+1));
            }
    }

    public void exp(){
         for (int row = 0; row < getNrOfRows(); ++row)
            for (int column = 0; column < getNrOfColumns(); ++column)
            {
                float value = (float)Math.exp(get(row+1,column+1));
//                if (Float.isNaN(value)){
//                    System.out.println("exp : nan value");
//                }
//                if (Float.isInfinite(value)){
//                    System.out.println("tanh : infinite value is " + get(row+1,column+1));
//                }
                set( row+1,column+1, (float)Math.exp(get(row+1, column+1)));
            }
    }
    
    public void tanh() {
        for (int row = 0; row < getNrOfRows(); ++row) {
            for (int column = 0; column < getNrOfColumns(); ++column) {
                float value = (float)Math.tanh(get(row+1,column+1));
//                if (Float.isNaN(value)){
//                    System.out.println("tanh : nan value is " + get(row+1,column+1));
//                }
//                if (Float.isInfinite(value)){
//                    System.out.println("tanh : infinite value is " + get(row+1,column+1));
//                }
                set(row + 1, column + 1, value);
            }
        }
    }
    
    public void tan() {
       for (int row = 0; row < getNrOfRows(); ++row) {
            for (int column = 0; column < getNrOfColumns(); ++column) {
                float value = (float)Math.tan(get(row+1,column+1));
                set(row + 1, column + 1, value);
            }
        }
    }

    
    public void difftanh() {
        for (int row = 0; row < getNrOfRows(); ++row) {
            for (int column = 0; column < getNrOfColumns(); ++column) {
                float tanh = (float) Math.tanh(get(row + 1, column + 1));
                set(row + 1, column + 1, 1-tanh*tanh);
            }
        }
    }
    
    public void softMaxPerColumn(){
        // first exp on all the elements.
        exp();
        for( int column = 0; column < getNrOfColumns();++column)
        {
            float sum = 0;
            for (int row = 0 ; row < getNrOfRows();++row)
            {
                float value = get(row+1,column+1);
//                if ( Float.isNaN(value)){
//                    System.out.println("soft max problem 1 , value is : "+value);
//                }
//                if ( Float.isInfinite(value)){
//                    System.out.println("soft max problem 1b, values is infinite : "+value);
//                }
                sum += value;
            }
            
            for (int row = 0 ; row < getNrOfRows();++row)
            {
                float current = get(row+1,column+1)/sum;
//                if ( Float.isNaN(current)){
//                    System.out.println("soft max problem 2, sum is :"+sum);
//                    current = get(row+1,column+1);
//                }
                set(row+1,column+1,current);
            }
        }
    }

    public fmatrix copy(){
        fmatrix result = new fmatrix(getNrOfRows(),getNrOfColumns());
        for (int row = 0; row < getNrOfRows(); ++row)
            for (int column = 0; column < getNrOfColumns(); ++column)
            {
                result.set( row+1,column+1, get(row+1, column+1));
            }
        return result;
    }

    public fmatrix tcopy(){
        transpose();
        fmatrix result = copy();
        transpose();
        return result;
    }

    public static fmatrix zeros(int rows, int columns){
        fmatrix result = new fmatrix(rows,columns);
        return result;
    }

    public static fmatrix eye(int rows, int columns){
        fmatrix result = new fmatrix(rows, columns);
        for( int i = 0; i < rows && i < columns; ++i)
            result.set(i+1,i+1,1);
        return result;
    }

    public static fmatrix ones(int rows, int columns){
        fmatrix result = new fmatrix(rows,columns);
        for (int row = 0; row < result.getNrOfRows(); ++row)
            for (int column = 0; column < result.getNrOfColumns();++column )
                result.set(row+1,column+1,1);

        return result;
    }

    public static fmatrix random(int rows, int column, final float minValue, float maxValue){
        fmatrix result = new fmatrix(rows,column);
        final Random r = new Random(System.currentTimeMillis());
        final float diff = (maxValue-minValue);

        result.iterateCells(new CellIterator(){
            public void cell(fmatrix source, int row, int column, float currentValue) {
                float value = (r.nextFloat() * diff) + minValue;
                source.set(row, column,value );
            }
        });
        return result;
    }


    public static fmatrix construct(String range){
        Range r = parseRange(range);
        return construct(r);
    }

    public static fmatrix construct(Range range){
        if (range.singleton){
            fmatrix result = new fmatrix(1,1);
            result.set(1,1,range.startOfRange);
            return result;
        }else{
            float diff = range.endOfRange - range.startOfRange;
            int nrOfElements = (int)Math.floor(diff/range.increment)+ 1 ;
            if ( nrOfElements < 0 ){
                fmatrix result = new fmatrix(1,1);
                result.set(1,1,range.startOfRange);
                return new fmatrix(1,1);
            }else{
                fmatrix result = new fmatrix(nrOfElements,1);
                for (int i = 0 ; i < nrOfElements;++i){
                   float value = range.startOfRange + i*range.increment;
                   result.set(i+1,1,value);
                }
                return result;
            }
        }
    }

    public static fmatrix multiply(fmatrix op1, fmatrix op2)
    {
        if ( op1.getNrOfColumns() != op2.getNrOfRows())
        {
            String op1dim = "["+op1.getNrOfRows()+","+op1.getNrOfColumns()+"]";
            String op2dim = "["+op2.getNrOfRows()+","+op2.getNrOfColumns()+"]";
            System.out.println("ERROR : dimension do not agree " +  op1dim +"*"+op2dim+"\n");
            return null;
        }
        fmatrix result = new fmatrix(op1.getNrOfRows(), op2.getNrOfColumns());
        return multiply(result, op1, op2);
    }

    public static fmatrix multiply(fmatrix result, fmatrix op1, fmatrix op2)
    {
        if (op1.getNrOfColumns() != op2.getNrOfRows()) {
            System.out.println("Error , inner dimension must agree: " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }
        for( int c_row = 0; c_row < op1.getNrOfRows(); ++c_row)
        {
            for (int c_column = 0; c_column < op2.getNrOfColumns();++c_column)
            {
                float sum = 0;
                for(int index = 0; index < op1.getNrOfColumns();++index)
                {
                    sum += op1.get(c_row+1, index+1)*op2.get(index+1, c_column+1);
                }
                result.set(c_row+1, c_column+1, sum);
            }
        }
        return result;
    }

    public static fmatrix dotmultiply(fmatrix op1, fmatrix op2) {
        if (op1.getNrOfRows() != op2.getNrOfRows() || op1.getNrOfColumns() != op2.getNrOfColumns()) {
            System.out.println("Error , matrix dimension are not the same " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }
        fmatrix result = new fmatrix(op1.getNrOfRows(), op2.getNrOfRows());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row + 1, column + 1);
                float op2value = op2.get(row + 1, column + 1);
                result.set(row + 1, column + 1, op1value * op2value);
            }
        }
        return result;
    }
    
     public static fmatrix dotmultiply(fmatrix result, fmatrix op1, fmatrix op2) {
        if ( result.getNrOfRows() != op2.getNrOfRows() || result.getNrOfColumns() != op2.getNrOfColumns() ||
                op1.getNrOfRows() != op2.getNrOfRows() || op1.getNrOfColumns() != op2.getNrOfColumns()) {
            System.out.println("Error , matrix dimension are not the same " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }
        //fmatrix result = new fmatrix(op1.getNrOfRows(), op2.getNrOfRows());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row + 1, column + 1);
                float op2value = op2.get(row + 1, column + 1);
                result.set(row + 1, column + 1, op1value * op2value);
            }
        }
        return result;
    }
    
    public static fmatrix dotmultiply(fmatrix op1, float op2) {
        fmatrix result = new fmatrix(op1.getNrOfRows(), op1.getNrOfColumns());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row + 1, column + 1);
                result.set(row + 1, column + 1, op1value * op2);
            }
        }
        return result;
    }

    public static fmatrix dotdivide(fmatrix op1, fmatrix op2) {
        if (op1.getNrOfRows() != op2.getNrOfRows() || op1.getNrOfColumns() != op2.getNrOfColumns()) {
            System.out.println("Error , matrix dimension are not the same " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }
        fmatrix result = new fmatrix(op1.getNrOfRows(), op2.getNrOfRows());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row + 1, column + 1);
                float op2value = op2.get(row + 1, column + 1);
                result.set(row + 1, column + 1, op1value / op2value);
            }
        }
        return result;
    }

    public static fmatrix dotadd(fmatrix op1, fmatrix op2) {
        if (op1.getNrOfRows() != op2.getNrOfRows() || op1.getNrOfColumns() != op2.getNrOfColumns()) {
            System.out.println("Error , matrix dimension are not the same " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }
        fmatrix result = new fmatrix(op1.getNrOfRows(), op1.getNrOfColumns());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row + 1, column + 1);
                float op2value = op2.get(row + 1, column + 1);
                result.set(row + 1, column + 1, op1value + op2value);
            }
        }
        return result;
    }

    public static fmatrix dotadd(fmatrix op1, float op2){
        fmatrix result = new fmatrix(op1.getNrOfRows(), op1.getNrOfColumns());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row + 1, column + 1);
                result.set(row + 1, column + 1, op1value + op2);
            }
        }
        return result;
    }

    public static fmatrix dotsubtract(fmatrix op1, fmatrix op2) {
        if (op1.getNrOfRows() != op2.getNrOfRows() || op1.getNrOfColumns() != op2.getNrOfColumns()) {
            System.out.println("Error , matrix dimension are not the same " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }
        fmatrix result = new fmatrix(op1.getNrOfRows(), op1.getNrOfColumns());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row + 1, column + 1);
                float op2value = op2.get(row + 1, column + 1);
                result.set(row + 1, column + 1, op1value - op2value);
            }
        }
        return result;
    }
    
    public static fmatrix dotsubtract(fmatrix result, fmatrix op1, fmatrix op2) {
        if (op1.getNrOfRows() != result.getNrOfRows() || op1.getNrOfColumns() != result.getNrOfColumns() ||
                op1.getNrOfRows() != op2.getNrOfRows() || op1.getNrOfColumns() != op2.getNrOfColumns()) {
            System.out.println("Error , matrix dimension are not the same " + op1.getSizeAsString() + " != " + op2.getSizeAsString());
            return null;
        }
        
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row + 1, column + 1);
                float op2value = op2.get(row + 1, column + 1);
                result.set(row + 1, column + 1, op1value - op2value);
            }
        }
        return result;
    }

    public static fmatrix dotsubtract(fmatrix op1, float op2){
        fmatrix result = new fmatrix(op1.getNrOfRows(), op1.getNrOfColumns());
        for (int row = 0; row < result.getNrOfRows(); ++row) {
            for (int column = 0; column < result.getNrOfColumns(); ++column) {
                float op1value = op1.get(row + 1, column + 1);
                result.set(row + 1, column + 1, op1value - op2);
            }
        }
        return result;
    }

    public static fmatrix mergeRows(fmatrix op1, fmatrix op2){
        int cs = Math.max(op1.getNrOfColumns(),op2.getNrOfColumns());
        fmatrix result = new fmatrix(op1.getNrOfRows()+op2.getNrOfRows(),cs);

        for (int rows = 0; rows < op1.getNrOfRows();++rows){
            for(int columns = 0; columns < op1.getNrOfColumns();++columns)
            {
                result.set(rows+1, columns+1, op1.get(rows+1, columns+1));
            }
        }

        for (int rows = 0; rows < op2.getNrOfRows();++rows){
            for(int columns = 0; columns < op2.getNrOfColumns();++columns)
            {
                result.set(rows+1+op1.getNrOfRows(), columns+1, op2.get(rows+1, columns+1));
            }
        }
        return result;
    }
    
    public static fmatrix mergeColumns(fmatrix op1, fmatrix op2){
        int rs = Math.max(op1.getNrOfRows(),op2.getNrOfRows());
        fmatrix result = new fmatrix(rs,op1.getNrOfColumns()+op2.getNrOfColumns());

        for (int rows = 0; rows < op1.getNrOfRows();++rows){
            for(int columns = 0; columns < op1.getNrOfColumns();++columns)
            {
                result.set(rows+1, columns+1, op1.get(rows+1, columns+1));
            }
        }

        for (int rows = 0; rows < op2.getNrOfRows();++rows){
            for(int columns = 0; columns < op2.getNrOfColumns();++columns)
            {
                result.set(rows+1, columns+1+op1.getNrOfColumns(), op2.get(rows+1, columns+1));
            }
        }
        return result;
    }

    public static void copyInto(fmatrix toCopy, fmatrix dest){
        int maxRow = toCopy.getNrOfRows()  < dest.getNrOfRows()?toCopy.getNrOfRows():dest.getNrOfRows();
        int maxColumn = toCopy.getNrOfColumns()  < dest.getNrOfColumns()?toCopy.getNrOfColumns():dest.getNrOfColumns();
         for (int row = 0; row < maxRow; ++row)
            for (int column = 0; column < maxColumn; ++column)
            {
                dest.set( row+1,column+1, toCopy.get(row+1, column+1));
            }
    }



    public fmatrix submatrix(String rowRange, String columnRange){
        return null;
    }

    private static Range parseRange(String range){
        Range r = new Range();
        int firstColon = range.indexOf(':');
        if ( firstColon > -1 ){
            r.singleton = false;
            r.startOfRange = Float.parseFloat(range.substring(0,firstColon));
            int secondColon = range.indexOf(':',firstColon+1);
            if ( secondColon > -1){
                r.increment = Float.parseFloat(range.substring(firstColon+1,secondColon));
                r.endOfRange = Float.parseFloat(range.substring(secondColon+1));
            }else{
                r.endOfRange = Float.parseFloat(range.substring(firstColon+1));
                r.increment = (r.endOfRange>r.startOfRange)?1:-1;
            }
        }else{
            // try singleton
            try{
                float number = Float.parseFloat(range);
                r.startOfRange = number;
                r.endOfRange = number;
                r.singleton = true;
            }catch(NumberFormatException ex){
                ex.printStackTrace();
            }
        }
        return r;
    }

    public String getSizeAsString(){
        return "["+getNrOfRows()+","+getNrOfColumns()+"]";
    }

    @Override
    public String toString() {
        String[][] cells;
        cells = new String[getNrOfRows()][getNrOfColumns()];
        int[] widths = new int[getNrOfColumns()];
        for (int row =0; row < getNrOfRows(); ++row) {
            for (int column = 0; column < getNrOfColumns(); ++column) {
                String fs = Float.toString(get(row+1, column+1));
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

    public static void main(String[] args) {
        Range r = new Range();
        r.startOfRange = 2.57f;
        r.endOfRange = 11.2f;
        r.increment = 0.2f;
        fmatrix result = fmatrix.construct(r);
        System.out.println(result);

        result.transpose();
        System.out.println(result);

        fmatrix identity = fmatrix.eye(3,3);
        System.out.println(identity);

        fmatrix op1 = fmatrix.random(3,2,-5.0f,1.0f);
        fmatrix op2 = fmatrix.random(2,1,8.0f,20.0f);

        fmatrix result2 = fmatrix.multiply(op1, op2);

        System.out.println(op1);
        System.out.println(op2);
        System.out.println(result2);

        fmatrix test = fmatrix.construct("1:10");
        System.out.println(test);
        System.out.println("Sum is : "+test.sum());
        test.multiply(2.0f);
        System.out.println("Sum is : "+test.sum());

        op1 = fmatrix.random(4,3,0,100);
        op2 = fmatrix.random(2,5,-50,-10);

        fmatrix merge = fmatrix.mergeRows(op1, op2);
        System.out.println(op1);
        System.out.println(op2);
        System.out.println(merge);

        fmatrix merge2 = fmatrix.mergeColumns(op1, op2);
        System.out.println(merge2);

        System.out.println("To copy : ");
        System.out.println(op1);
        fmatrix op1_copy = op1.copy();
        System.out.println(op1_copy);
        fmatrix op1_copy_t = op1.tcopy();
        System.out.println(op1_copy_t);
        
        System.out.println("addRow test");
        fmatrix rowTest1 = fmatrix.random(2,2,-1,2);
        
        System.out.println("before add");
        System.out.println(rowTest1);
        /*
        //ArrayList<Float> rowToAdd = new ArrayList<>();
        rowToAdd.add(12.22f);
        rowToAdd.add(13.2f);
        rowToAdd.add(17.1f);
        
        rowTest1.addRow(rowToAdd);
        System.out.println("after add");
        System.out.println(rowTest1);
        */
        
        fmatrix op3 = fmatrix.random(2,7,-50,-10);
        op3.transpose();
        fmatrix row1 = op3.getRow(1);
        
        System.out.println("op3:\n"+op3);
        System.out.println("row1 :\n" +row1);
        System.out.println("column2 : \n" + op3.getColumn(2));
    }

    
    

   
}
