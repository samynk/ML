package dae.matrix;

/**
 *
 * @author Koen Samyn (samyn.koen@gmail.com)
 */
public interface CellIterator {
    public void cell(fmatrix source, int row, int column, float currentValue);
}
