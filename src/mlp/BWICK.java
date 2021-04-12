package mlp;

public class BWICK {

    public static void main(String[] args)
    {
        Network network = new Network(3, 4, 2, 1);
        network.SetInputData(new float[] {1, 2, 3});
        network.RunForward();
        network.RunBackward(new float[] {1, 1, 1});

        System.out.println("KONIEC");
    }
}
