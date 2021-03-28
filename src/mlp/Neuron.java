package mlp;

import java.util.ArrayList;

public class Neuron {

    static float bias = 1;

    ArrayList<Synapse> prevNeurons;
    ArrayList<Synapse> nextNeurons;
    float value;

    public Neuron() {
        prevNeurons = new ArrayList<>();
        nextNeurons = new ArrayList<>();
        value = 0;
    }

    // Applies bias to summed value
    public void ApplyBias() {
        value += bias;
    }
    // Activates value
    public void ActivateValue() {
        value = Sigmoid();
    }
    // Passes value
    public void PassValue() {
        for (Synapse s : nextNeurons)
        {
            s.PassValue();
        }
    }


    // ACTIVATION FUNCTIONS

    // Activation function
    float Sigmoid() {
        return (float)(1.0 / (1.0 + Math.exp(-value)));
    }
}
