/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public interface LearningRate {
    /**
     * Gets the learning for the given iteration.
     * @param iteration the iteration number.
     * @return the learning rate.
     */
    public float getLearningRate(int iteration);
}
