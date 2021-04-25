package mlp;

import java.util.ArrayList;

public class Neuron {

    float bias = 1;
    ArrayList<Synapse> prevNeurons;
    ArrayList<Synapse> nextNeurons;

    // FORWARD PROPAGATION
    float sum;
    float biasedSum;
    float activatedSum;


    // BACKWARD PROPAGATION
    float error;
    float activatedSumDerived;
    float sumOfInputWeights;

    public Neuron() {
        prevNeurons = new ArrayList<>();
        nextNeurons = new ArrayList<>();
        sum = 0;
    }

    // FORWARD PROPAGATION

    // Applies bias to summed value
    public void ApplyBias() {
        biasedSum = sum + bias;
    }
    // Activates value
    public void ActivateValue() {
        activatedSum = Sigmoid();
    }
    // Passes value
    public void PassValue() {
        for (Synapse s : nextNeurons) {
            s.PassValue();
        }
    }


    // BACKWARD PROPAGATION

    // Calculates error
    public void CalculateError_Output(float expectedValue) {
        activatedSumDerived = activatedSum * (1 - activatedSum);
        error = (expectedValue - activatedSum) * activatedSumDerived;
        System.out.println("ExpectedValue: " + expectedValue);
        System.out.println("ActivatedSum: " + activatedSum);
        System.out.println("ActivatedSumDerived: " + activatedSumDerived);
        System.out.println("Error: " + error);
    }
    public void PassError(float learningRate) {
        for (Synapse s : prevNeurons) {
            s.PassError(error, learningRate);
        }
    }
    public void ResetNeuron() {
        sum = 0;
        sumOfInputWeights = 0;
    }


    // ACTIVATION FUNCTIONS

    // Activation function
    float Sigmoid() {
        return (float)(1.0 / (1.0 + Math.exp(-biasedSum)));
    }
}
