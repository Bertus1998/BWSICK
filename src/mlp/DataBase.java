package mlp;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class DataBase {
    List<File> listOfFiles = new ArrayList<>();
    public static DataBase loadDataBase(File folder)
    {
        DataBase dataBase = new DataBase();
        for (final File fileEntry : folder.listFiles()) {
            if (!fileEntry.isDirectory()) {
                dataBase.listOfFiles.add(fileEntry);

            }
        }
        return dataBase;
    }
}
