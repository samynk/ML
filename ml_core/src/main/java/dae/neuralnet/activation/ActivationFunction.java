package dae.neuralnet.activation;

/**
 *
 * @author Koen
 */
public enum ActivationFunction {
    TANH(
            x -> (float) Math.tanh(x),
            x -> 1 - x * x
    ),
    SOFTMAX(
            x -> (float) Math.exp(x),
            x -> 1
    ),
    IDENTITY(
            x -> x,
            x -> 1
    ),
    SIGMOID(
            x -> (float) (1 / (1 + Math.exp(-x))),
            x -> x * (1 - x)
    ),
    CESIGMOID(
            x -> (float) (1 / (1 + Math.exp(-x))),
            x -> 1
    ),
    LEAKYRELU(
            x -> (x < 0) ? 0.001f * x :  x,
            x -> (x < 0) ? 0.001f : 1
    ),
    RELU(
            x -> (x < 0) ? 0 : x,
            x -> (x < 0) ? 0 : 1
    );

    private final Function a;
    private final Function da;

    ActivationFunction(Function a, Function da) {
        this.a = a;
        this.da = da;
    }

    public Function getActivation() {
        return a;
    }

    public Function getDerivedActivation() {
        return da;
    }
}
