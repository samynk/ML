/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.matrix;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public interface IndexedFunction {
    float evaluate(int row, int column, int slice, float value);
}
