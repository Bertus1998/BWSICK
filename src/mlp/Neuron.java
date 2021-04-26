package mlp;

import java.io.Serializable;
import java.util.Random;

public class Neuron implements Serializable {

    public double value;
    public double[] weights;
    public double bias;
    public double delta;

    public Neuron(int prevLayerSize) {

        Random r = new Random();

        weights = new double[prevLayerSize];
        bias = 1;
        delta = 0;
        value = 0;

        for (int i = 0; i < weights.length; i++){
            weights[i] = r.nextDouble();
        }
    }
}
