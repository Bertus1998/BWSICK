package mlp;

import java.util.Random;

public class Synapse {

    public Neuron inputNeuron;      // Reference to neuron that provides data
    public Neuron outputNeuron;     // Reference to neuron data has to be passed to
    public float weight;            // Weight of the synapse

    // Constructor that initiates weight with random value and assigns values to input and output
    public Synapse(Neuron input, Neuron output) {
        Random rand = new Random();
        weight = rand.nextFloat();      // Weight value from 0 to 1

        inputNeuron = input;            // Assign values to input and output
        outputNeuron = output;
        input.nextNeurons.add(this);    // Save reference to this synapse in neurons' objects
        output.prevNeurons.add(this);
    }

    // Passes the value from input to output while multiplying value by weight
    public void PassValue()
    {
        outputNeuron.value += inputNeuron.value * weight;
    }
}
