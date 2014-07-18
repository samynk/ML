/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.layer;

import java.util.ArrayList;

/**
 * Validates the incoming weights of a group of neural net cells.
 * @author Koen
 */
public interface WeightValidator {
    public void validate(ArrayList<NeuralNetCell> cells);
}
