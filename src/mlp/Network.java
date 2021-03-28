package mlp;

public class Network {

    private Neuron[][] network;

    // Constructor, creates neurons and synapses
    public Network(int... amountOfNeurons) {
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
            network[0][i].value = inputData[i];
        }
    }
    public void Run() {

        int outputLayerNumber = network.length - 1;

        for (int neuron = 0; neuron < network[0].length; neuron++) {
            network[0][neuron].PassValue();
        }

        for (int layer = 1; layer < outputLayerNumber; layer++) {
            for (int neuron = 0; neuron < network[layer].length; neuron++) {
                network[layer][neuron].ApplyBias();
                network[layer][neuron].ActivateValue();
                network[layer][neuron].PassValue();
            }
        }

        for (int neuron = 0; neuron < network[outputLayerNumber].length; neuron++) {
            network[outputLayerNumber][neuron].ActivateValue();
        }
    }
}
