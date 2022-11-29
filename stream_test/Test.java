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

    private static void watch(final Process process) {
        new Thread() {
            public void run() {
                final BufferedReader sysIn = new BufferedReader(new InputStreamReader(System.in));
                final BufferedReader prcIn = new BufferedReader(new InputStreamReader(process.getInputStream()));
                final BufferedWriter prcOut = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));

                while(true) {
                    try {
                        String line;
                        while (!(line = prcIn.readLine()).equals("")) {
                            System.out.println(line);
                        }

                        System.out.print(">> ");
                        prcOut.write(sysIn.readLine() + "\n");
                        prcOut.flush();
                    } catch (NullPointerException e) {
                        System.out.println("Shutting down without an exception :D");
                        break;
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }.start();
    }
}