/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package mlproject.animation;

import intersect.data.fmatrix;
import mlproject.fuzzy.FuzzySystem;
import mlproject.fuzzy.io.ParseFCL;
import mlproject.layer.FuzzyToNeural;
import mlproject.layer.MultiLayerNN;

/**
 *
 * @author samyn_000
 */
public class FuzzyController implements Controller{
    private FuzzySystem controller;
    private MultiLayerNN neuralController;
    private fmatrix neuralInputs;
    
    
    public FuzzyController(String controllerFile){
        ParseFCL fclParser = new ParseFCL(controllerFile);
        this.controller = fclParser.getResult();
        neuralController = FuzzyToNeural.FuzzyToNeural2(this.controller);
        if (neuralController != null) {
            neuralInputs = new fmatrix(neuralController.getInputLayer().getNrOfCells(), 1);
        }
    }

    @Override
    public void evaluate(float time) {
        controller.evaluate();
    }

    @Override
    public void setInput(String name, float value) {
        controller.setFuzzyInput(name, value);
        neuralController.setInput(name, value);
    }

    @Override
    public float getOutput(String name) {
        return controller.getFuzzyOutput(name);
    }
    
    @Override
    public boolean isAbsoluteAngle(){
        return false;
    }

    public FuzzySystem getFuzzySystem() {
       return this.controller;
    }
}
