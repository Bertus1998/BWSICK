package mlp;

import sun.nio.ch.Net;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class BWICK {

    static DataBase dataBaseOfPhotos;
    static DataBase dataBaseOfSound;
    static ArrayList<Iris> irises;
    static boolean networkFromFile = true;

    public static void main(String[] args) {
        //LoadData();

        String pathToFile = "trainedNetwork_Irises.mlp";
        Network irisNetwork;

        while (true) {

            LoadData();

            //irisNetwork = TrainNetwork(new Network(4, 5, 3), irises);
            irisNetwork = Network.CreateFromFile(pathToFile);


            int correct = 0;
            int iStart = 0;
            int iEnd = 150;

            for (int i = iStart; i < iEnd; i++) {

                Iris iris = irises.get(i);
                int guess = irisNetwork.Predict(iris.GetInput());

                if (iris.GetOutput()[guess] == 1) {
                    correct++;
                }
            }
            System.out.println("RESULT: " + correct + " from " + (iEnd - iStart) + " | " + (correct * 100 / (iEnd - iStart)) + "%");

            if (correct == 50 && !networkFromFile) {
                irisNetwork.SaveToFile(pathToFile);
                break;
            }

            if (networkFromFile) {
                break;
            }
        }
    }


    private static Network TrainNetwork(Network n, List<Iris> irises) {
        for (int e = 0; e < 1000; e++) {
            for (int i = 0; i < 100; i++) {

                Iris iris = irises.get(i);
                double[] result = n.PropagateForward(iris.GetInput());
                n.PropagateBackward(iris.GetInput(), iris.GetOutput(), result);
            }
        }
        networkFromFile = false;
        return n;
    }

    private static void LoadData() {

        // Load files
        //File folderOfImages = new File("C:\\Users\\Dominik\\Desktop\\BWSICK\\DataBase\\BD_zdjecia\\train");
        //File folderOfSound = new File("C:\\Users\\Dominik\\Desktop\\BWSICK\\DataBase\\BD_dzwiek\\train");
        File folderOfImages = new File("C:\\Users\\Adam\\Desktop\\Studia\\1 Semestr\\Biometryczne wspomaganie interakcji człowiek-komputer\\BWICK\\DataBase\\BD_zdjecia\\train");
        File folderOfSound = new File("C:\\Users\\Adam\\Desktop\\Studia\\1 Semestr\\Biometryczne wspomaganie interakcji człowiek-komputer\\BWICK\\DataBase\\BD_dzwiek\\train");
        DataBase dataBaseOfPhotos = DataBase.loadDataBase(folderOfImages);
        DataBase dataBaseOfSound = DataBase.loadDataBase(folderOfSound);

        try {
            //irises = Iris.GetIrises("C:\\Users\\Dominik\\Desktop\\BWSICK\\DataBase\\Irises\\iris.data");
            irises = Iris.GetIrises("C:\\Users\\Adam\\Desktop\\Studia\\1 Semestr\\Biometryczne wspomaganie interakcji człowiek-komputer\\BWICK\\DataBase\\Irises\\iris.data");
        } catch (IOException e)
        {
            System.out.println("Nie znaleziono pliku");
            return;
        }
    }

    private static float[] ImageToGrayScaleArray(BufferedImage image) {

        final byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        final int width = image.getWidth();
        final int height = image.getHeight();
        final boolean hasAlphaChannel = image.getAlphaRaster() != null;
        float greyscale;
        int count =0;
        float[] result = new float[width*height];
        if (hasAlphaChannel) {
            final int pixelLength = 4;
            for (int pixel = 0, row = 0, col = 0; pixel + 3 < pixels.length; pixel += pixelLength) {
             //   0.21f*r + 0.71f*g + 0.07f*b;
                result[count] =  ((int) pixels[pixel] & 0xff)*0.07f + (((int) pixels[pixel + 1] & 0xff) << 8)*0.71f + (((int) pixels[pixel + 2] & 0xff) << 16)*0.21f;         col++;
                count++;
                if (col == width) {
                    col = 0;
                    row++;
                }
            }
        } else {
            final int pixelLength = 3;
            for (int pixel = 0, row = 0, col = 0; pixel + 2 < pixels.length; pixel += pixelLength) {
                int argb = 0;

                result[count] = ((int) pixels[pixel] & 0xff)*0.07f + (((int) pixels[pixel + 1] & 0xff) << 8)*0.71f + (((int) pixels[pixel + 2] & 0xff) << 16)*0.21f;
                count++;

                col++;
                if (col == width) {
                    col = 0;
                    row++;
                }
            }
        }

        return result;
    }
}
