/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class LearningRateDecay implements LearningRate{
    private final float baseLearningRate;
    private float learningRate;
    private final float decay;
    
    public LearningRateDecay(float baseLearningRate, float decay){
        this.baseLearningRate = baseLearningRate;
        this.learningRate = baseLearningRate;
        this.decay = decay;
    }
    
    @Override
    public float getLearningRate(int iteration) {
        learningRate = baseLearningRate / (1.0f + decay *iteration);
        return learningRate;
    }
    
}
