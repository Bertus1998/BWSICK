package mlp;

import java.io.Serializable;

public class Layer implements Serializable {

    public Neuron[] neurons;
    public int length;

    public Layer(int length, int prevLayerLength) {

        this.length = length;
        neurons = new Neuron[length];

        for (int i = 0; i < length; i++) {
            neurons[i] = new Neuron(prevLayerLength);
        }
    }
}
