/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class LearningRateDecay implements LearningRate {

    private final float baseLearningRate;
    private float learningRate;
    private final float decay;

    public LearningRateDecay(float baseLearningRate, float decay) {
        this.baseLearningRate = baseLearningRate;
        this.learningRate = baseLearningRate;
        this.decay = decay;
    }

    @Override
    public float getLearningRate(int iteration) {
        learningRate = baseLearningRate / (1.0f + decay * iteration);
        return learningRate;
    }

    public float getBaseLearningRate() {
        return baseLearningRate;
    }

    public float getDecay() {
        return decay;
    }

    @Override
    public boolean equals(Object other) {
        if (other instanceof LearningRateDecay) {
            LearningRateDecay lrd = (LearningRateDecay) other;
            return Math.abs(lrd.baseLearningRate - baseLearningRate) < 0.00001f
                    && Math.abs(lrd.decay - decay) < 0.00001f;
        } else {
            return false;
        }
    }
}
