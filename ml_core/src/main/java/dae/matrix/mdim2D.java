/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix;

/**
 * Describes a 2D matrix dimension.
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class mdim2D {

    public int rows;
    public int columns;
    public boolean transposed;
    public int ld;

    public String toString() {
        return "Dimension\n[rows,columns]=[ " + rows + "," + columns + " ]"
                + "\ntranspose = " + transposed
                + "\nleading = " + ld + "\n";
    }
}
