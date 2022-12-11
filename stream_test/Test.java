import java.util.Scanner;

public class Test {

    final static int RUNS = 1;

    public static void main(String[] args) {
        DdnnifeWrapper auto1 = new DdnnifeWrapper("../example_input/auto1_d4.nnf", 2513);
        DdnnifeWrapper vp9 = new DdnnifeWrapper("../example_input/VP9_d4.nnf", 42);

        Scanner scanner = new Scanner(System.in);
        String response = "";

        while (!response.equals("ENDE \\ü/")) {
            System.out.print(">> ");
            String line = scanner.nextLine();
            for (int i = 0; i < RUNS; ++i) {
                response = vp9.compute(line);
            }
            for (int i = 0; i < RUNS; ++i) {
                System.out.println(i + ": " + response);
            }
        }

        scanner.close();
        auto1.endProcess();
    }
}