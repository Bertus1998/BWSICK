package mlp;

import javax.imageio.ImageIO;
import javax.xml.crypto.Data;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class BWICK {

    public static void main(String[] args)
    {


        File folderOfImages = new File("C:\\Users\\Dominik\\Desktop\\BWSICK\\DataBase\\BD_zdjecia\\train");
        File folderOfSound= new File("C:\\Users\\Dominik\\Desktop\\BWSICK\\DataBase\\BD_dzwiek\\train");
        DataBase dataBaseOfPhotos = DataBase.loadDataBase(folderOfImages);
        DataBase dataBaseOfSound = DataBase.loadDataBase(folderOfSound);
        //On input amount of pixels, on output amount of possible people (625)
        Network networkForPhotos = new Network(640*480, 1,1,1, 625);
        float[] arrray = new float[625];
        for(int i =0;i<2; i++)
        {
            Arrays.fill(arrray,0f);
            arrray[i] = 1;
            BufferedImage img = null;
            try {
                img = ImageIO.read(dataBaseOfPhotos.listOfFiles.get(i));
               // System.out.println(ImageToGrayScaleArray(img).toString());
                networkForPhotos.SetInputData(ImageToGrayScaleArray(img));
                networkForPhotos.RunForward();
                networkForPhotos.RunBackward(arrray);

            } catch (IOException e) {
                System.out.println(img + "was not loaded");
            }


        }

        //On input .. on output amount of possible gender, (2) <? XD
     //   Network networkForSound = new Network(640*480, 4, 2, 2);
     //   network.SetInputData(new float[] {1, 2, 3});
  //      network.RunForward();
     //   network.RunBackward(new float[] {1, 1, 1});

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
