package mlp;

import sun.nio.ch.Net;
import sun.security.jgss.spnego.NegTokenInit;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class Network implements Serializable {

    double learningRate = 0.6;
    Layer[] layers;


    public Network(int... numberOfNeurons) {

        layers = new Layer[numberOfNeurons.length];

        for (int i = 0; i < layers.length; i++) {

            if (i == 0) {
                layers[i] = new Layer(numberOfNeurons[i], 0);
            } else {
                layers[i] = new Layer(numberOfNeurons[i], numberOfNeurons[i - 1]);
            }
        }
    }

    public double[] PropagateForward(double[] input) {

        double[] output = new double[layers[layers.length - 1].length];

        // Set inputs
        for (int n = 0; n < layers[0].length; n++) {
            layers[0].neurons[n].value = input[n];
        }

        for (int l = 1; l < layers.length; l++) {

            for (int n = 0; n < layers[l].length; n++) {

                double newVal = layers[l].neurons[n].bias;
                ;
                for (int pn = 0; pn < layers[l - 1].length; pn++) {
                    newVal += layers[l].neurons[n].weights[pn] * layers[l - 1].neurons[pn].value;
                }

                layers[l].neurons[n].value = Evaluate(newVal);

            }
        }

        for (int n = 0; n < layers[layers.length - 1].length; n++) {
            output[n] = layers[layers.length - 1].neurons[n].value;
        }

        return output;
    }

    public double PropagateBackward(double[] input, double[] expected, double[] recieved) {

        for (int n = 0; n < layers[layers.length - 1].length; n++) {

            double error = expected[n] - recieved[n];
            layers[layers.length - 1].neurons[n].delta = error * EvaluateDerivative(recieved[n]);
        }

        for (int l = layers.length - 2; l >= 0; l--) {

            for (int n = 0; n < layers[l].length; n++) {

                double error = 0;
                for (int nn = 0; nn < layers[l + 1].length; nn++) {
                    error += layers[l + 1].neurons[nn].delta * layers[l + 1].neurons[nn].weights[n];
                }

                layers[l].neurons[n].delta = error * EvaluateDerivative(layers[l].neurons[n].value);
            }

            for (int nn = 0; nn < layers[l + 1].length; nn++) {

                for (int n = 0; n < layers[l].length; n++) {
                    layers[l + 1].neurons[nn].weights[n] += learningRate * layers[l + 1].neurons[nn].delta * layers[l].neurons[n].value;
                }

                layers[l + 1].neurons[nn].bias += learningRate * layers[l + 1].neurons[nn].delta;
            }
        }

        double error = 0;
        for (int n = 0; n < expected.length; n++) {
            error += Math.abs(expected[n] - recieved[n]);
        }
        error /= expected.length;
        return error;
    }

    public int Predict(double[] inputs) {
        double[] output = PropagateForward(inputs);

        int maxIndex = 0;
        double maxValue = 0;
        for (int i = 0; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxIndex = i;
                maxValue = output[i];
            }
        }

        return maxIndex;
    }


    double Evaluate(double value) {
        return 1 / (1 + Math.exp(-value));
    }

    double EvaluateDerivative(double value) {
        return value * (1 - value);
    }

        /*
        public void SaveToFile(String path) {

            try {
                File f = new File(path);
                FileWriter fw = new FileWriter(f);

                if (f.exists()) {
                    f.delete();
                }

                f.createNewFile();

                fw.write("MLP File\n");
                fw.write("Structure:\n");

                StringBuilder sb = new StringBuilder();
                for (Layer layer : layers) {
                    sb.append(layer.length).append(" ");
                }

                fw.write(sb.toString());

                fw.write("\n");
                fw.write("\n");
                fw.write("Weights:\n");

                sb = new StringBuilder();

                for (Layer l : layers) {
                    for (int n = 0; n < l.length; n++) {
                        for (int w = 0; w < l.neurons[n].weights.length; w++) {
                            sb.append(l.neurons[n].weights[w]).append(" ");
                        }
                        sb.append(l.neurons[n].bias).append("\n");
                    }
                    fw.write(sb.toString());
                }


                fw.close();
            } catch (Exception e) {
                System.out.println("ERROR while");
                System.out.println(e.getMessage());
            }
        }
        public static Network CreateFromFile(String path) {

            Network network;

            try {

                List<String> lines = Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);

                String[] structure = lines.get(2).split(" ");
                int[] layerStructure = new int[structure.length];
                for (int i = 0; i < layerStructure.length; i++) {
                    layerStructure[i] = Integer.parseInt(structure[i]);
                }

                network = new Network(layerStructure);

                int currLine = 5;
                for (int l = 0; l < network.layers.length; l++) {
                    for (int n = 0; n < network.layers[l].length; n++) {

                        String[] neuronData = lines.get(currLine++).split(" ");
                        for (int w = 0; w < network.layers[l].neurons[n].weights.length; w++) {
                            network.layers[l].neurons[n].weights[w] = Double.parseDouble(neuronData[w]);
                        }
                        network.layers[l].neurons[n].bias = Double.parseDouble(neuronData[neuronData.length - 1]);
                    }
                }
            } catch (Exception e) {
                System.out.println("ERROR");
                e.printStackTrace();
                return null;
            }

            return network;
        }
         */

    public void SaveToFile(String path) {

        try {
            FileOutputStream fos = new FileOutputStream(path);
            ObjectOutputStream oos = new ObjectOutputStream(fos);

            oos.writeObject(this);
            oos.close();
            fos.close();
            System.out.println("Network saved to file.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Network CreateFromFile(String path) {

        Network network;

        try {

            FileInputStream fis = new FileInputStream(path);
            ObjectInputStream ois = new ObjectInputStream(fis);
            network = (Network) ois.readObject();
            ois.close();
            fis.close();
            System.out.println("Network loaded from file.");

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

        return network;
    }
}

