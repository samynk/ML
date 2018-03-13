/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet.analysis;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class WeightAnalysis {

    private float mean;
    private float dev;
    private float min;
    private float max;

    public WeightAnalysis(float mean, float dev, float min, float max) {
        this.mean = mean;
        this.dev = dev;
        this.min = min;
        this.max = max;
    }

    /**
     * @return the mean
     */
    public float getMean() {
        return mean;
    }

    /**
     * @return the dev
     */
    public float getDev() {
        return dev;
    }

    /**
     * @return the min
     */
    public float getMin() {
        return min;
    }

    /**
     * @return the max
     */
    public float getMax() {
        return max;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("All values in range [").append(min).append(" ; ").append(max).append("]\n");
        result.append("Mean : ").append(mean).append("\n");
        result.append("Variance : ").append(this.dev).append("\n");
        return result.toString();
    }
}
