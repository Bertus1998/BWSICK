package mlp;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

public class Iris {
    float v1, v2, v3, v4;
    int type;

    public float[] GetInput() {
        return new float[] {v1, v2, v3, v4};
    }
    public float[] GetOutput() {
        float[] res = new float[] {0, 0, 0};
        res[type] = 1;
        return res;
    }

    public static ArrayList<Iris> GetIrises(String path) throws IOException {

        ArrayList<Iris> irises = new ArrayList<>();
        List<String> lines = Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);
        lines.remove(lines.size() - 1);
        Collections.shuffle(lines);

        for (String s : lines) {
            String[] values = s.split(",");
            Iris i = new Iris();
            i.v1 = Float.parseFloat(values[0]);
            i.v2 = Float.parseFloat(values[1]);
            i.v3 = Float.parseFloat(values[2]);
            i.v4 = Float.parseFloat(values[3]);

            switch (values[4]) {
                case "Iris-setosa":
                    i.type = 0;
                    break;
                case "Iris-versicolor":
                    i.type = 1;
                    break;
                case "Iris-virginica":
                    i.type = 2;
                    break;
                default:
                    throw new IllegalArgumentException("Unknown iris type: " + values[4]);
            }
            irises.add(i);
        }
        return irises;
    }
}
