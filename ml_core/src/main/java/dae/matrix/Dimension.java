/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class Dimension {

    private final int r;
    private final int c;
    private final int s;
    private final int h;

    /**
     * Creates a new dimension object with the given rows, columns, slices and
     * hyperslices.
     *
     * @param r the number of rows in the Dimension object.
     * @param c the number of columns in the Dimension object.
     * @param s the number of slices in the Dimension object.
     * @param h the number of hyperslices in the Dimension object.
     */
    public Dimension(int r, int c, int s, int h) {
        this.r = r;
        this.c = c;
        this.s = s;
        this.h = h;
    }
    
    public int getRows(){
        return r;
    }
    
    public int getColumns(){
        return c;
    }
    
    public int getSlices(){
        return s;
    }
    
    public int getHyperSlices(){
        return h;
    }

    public static Dimension Dim(int r, int c) {
        return new Dimension(r, c, 1, 1);
    }

    public static Dimension Dim(int r, int c, int s) {
        return new Dimension(r, c, s, 1);
    }

    public static Dimension Dim(int r, int c, int s, int h) {
        return new Dimension(r, c, s, h);
    }
}
