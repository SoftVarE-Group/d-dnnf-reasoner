import java.util.Scanner;

public class Test {
    public static void main(String[] args) {
        DdnnifeWrapper auto1 = new DdnnifeWrapper("../example_input/auto1_d4.nnf", 2513);

        Scanner scanner = new Scanner(System.in);
        String response = "";

        while (!response.equals("ENDE \\ü/")) {
            System.out.print(">> ");
            response = auto1.compute(scanner.nextLine());
            System.out.println(response);
        }

        scanner.close();
        auto1.endProcess();
    }
}
