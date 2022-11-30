import java.io.*;

public class Test {
    public static void main(String[] args) throws IOException {
        ProcessBuilder builder = new ProcessBuilder("../target/release/ddnnife", "../example_input/auto1_d4.nnf", "-o", "2513", "--stream");
        // ProcessBuilder builder = new ProcessBuilder("java", "Test2");

        final Process process = builder.start();

        StreamGobbler error = new StreamGobbler(process.getErrorStream());
        error.start();

        // Watch the process
        watch(process);
    }

    final static int QUERIES = 10;

    private static void watch(final Process process) {
        new Thread() {
            public void run() {
                final BufferedReader sysIn = new BufferedReader(new InputStreamReader(System.in));
                final BufferedReader prcIn = new BufferedReader(new InputStreamReader(process.getInputStream()));
                final BufferedWriter prcOut = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));

                String line;
                try {
                    while (!(line = prcIn.readLine()).equals("")) {
                        System.out.println(line);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }

                while(true) {
                    try {
                        System.out.print(">> ");
                        String userInput = sysIn.readLine();
                        for (int i = 0; i <= QUERIES; ++i) {
                            prcOut.write(userInput + "\n");
                            prcOut.flush();
                        }
                        for (int i = 0; i <= QUERIES; ++i) {
                            while (!(line = prcIn.readLine()).equals("")) {
                                System.out.println("i: " + i + " " + line);
                            }
                        }
                    } catch (Exception e) {
                        System.out.println("Shutting down without an exception :D");
                        System.out.println("Exception: " + e);
                        break;
                    }
                }
            }
        }.start();
    }
}