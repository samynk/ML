/*
 * Digital Arts and Entertainment 2018.
 * www.digitalartsandentertainment.be
 */
package dae.neuralnet.analysis;

import dae.matrix.Cell;
import dae.matrix.imatrix;
import java.nio.FloatBuffer;

/**
 *
 * @author Koen Samyn <samyn.koen@gmail.com>
 */
public class WeightAnalyzer {

    public static WeightAnalysis analyzeMatrix(imatrix matrix) {
        Cell min = matrix.min();
        Cell max = matrix.max();

        FloatBuffer data = matrix.getHostData();

        data.rewind();
        float N = 1;
        float mean = 0;
        while (data.hasRemaining()) {

            float d = data.get();
            mean = (mean * ((N - 1.0f) / N) + d / N);
            N++;
        }

        // calculate variance
        data.rewind();
        float var = 0;
        float dataSize = data.limit();
        while (data.hasRemaining()) {
            float d = data.get();
            var += (((d - mean) * (d - mean)) / dataSize);
        }
        var = (float) Math.sqrt(var);

        return new WeightAnalysis(mean, var, min.value, max.value);
    }
}
