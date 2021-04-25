package mlp;

public class Network {

    private final Neuron[][] network;
    int outputLayerNumber;
    float learningRate = 0.001f;

    // Constructor, creates neurons and synapses
    public Network(int... amountOfNeurons) {

        // Set number of output layer
        outputLayerNumber = amountOfNeurons.length - 1;

        // Fill array with neurons' objects
        network = new Neuron[amountOfNeurons.length][];
        for (int i = 0; i < network.length; i++)
        {
            network[i] = new Neuron[amountOfNeurons[i]];
            for (int j = 0; j < network[i].length; j++)
            {
                network[i][j] = new Neuron();
            }
        }

        // Iterate through array to create synapses
        for (int i = 0; i < network.length - 1; i++)
        {
            for (int j = 0; j < network[i].length; j++)
            {
                for (int k = 0; k< network[i + 1].length; k++)
                {
                    new Synapse(network[i][j], network[i + 1][k]);
                }
            }
        }
    }

    // Assigns values to neurons in input layer
    public void SetInputData(float[] inputData) {
        if (inputData.length != network[0].length) {
            throw new IllegalArgumentException("Amount of input data in not equal to amount of input neurons.");
        }

        for (int  i = 0; i < inputData.length; i++) {
            network[0][i].sum = inputData[i];
        }
    }
    public void RunForward() {

        for (Neuron n : network[0]) {
            n.PassValue();
        }

        for (int layer = 1; layer < outputLayerNumber; layer++) {
            for (Neuron n : network[layer]) {
                n.ApplyBias();
                n.ActivateValue();
                n.PassValue();
            }
        }

        for (Neuron n : network[outputLayerNumber]) {
            n.ActivateValue();
        }

    }
    public void RunBackward(float[] expectedValues) {

        // Calculate error comparing results with expected values
        int i = 0;
        for (Neuron n : network[outputLayerNumber]) {
            n.CalculateError_Output(expectedValues[i++]);
            n.PassError(learningRate);
        }

        // Pass errors to the hidden layers
        for (int layer = outputLayerNumber - 1; layer >= 1; layer--) {
            for (Neuron n : network[layer]) {
                n.PassError(learningRate);
            }
        }
    }
    public void ResetNetwork() {
        for (Neuron[] l : network) {
            for (Neuron n : l) {
                n.ResetNeuron();
            }
        }
    }
}
