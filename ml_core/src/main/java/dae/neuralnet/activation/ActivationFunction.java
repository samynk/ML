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
    LEAKYRELU(
            x -> (x < 0) ? 0.01f * x : 0.1f * x,
            x -> (x < 0) ? 0.01f : .1f
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
