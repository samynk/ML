/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class LearningRateConst implements LearningRate {

    private final float learningRate;

    public LearningRateConst(float learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public float getLearningRate(int iteration) {
        return learningRate;
    }

}
