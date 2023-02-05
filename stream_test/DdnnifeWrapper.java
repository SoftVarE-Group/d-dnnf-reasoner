import java.io.*;

public class DdnnifeWrapper {

    private String ddnnf_path;
    private int features;

    private ProcessBuilder builder;
    private Process process;

    private BufferedReader prcIn;
    private BufferedWriter prcOut;

    public DdnnifeWrapper(String ddnnf_path, int features) {
        this.ddnnf_path = ddnnf_path;
        this.features = features;

        this.builder = new ProcessBuilder(
            "../target/release/ddnnife", ddnnf_path, "-o", Integer.toString(features), "--stream");
        this.startProcess();
    }

    public DdnnifeWrapper(String ddnnf_path) {
        this.ddnnf_path = ddnnf_path;
        this.features = -1;

        this.builder = new ProcessBuilder(
                "../target/release/ddnnife", ddnnf_path, "--stream");
        this.startProcess();
    }

    private void startProcess() {
        try {
            this.process = this.builder.start();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        this.prcIn = new BufferedReader(new InputStreamReader(process.getInputStream()));
        this.prcOut = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
    }

    public void endProcess() {
        try {
            this.prcIn.close();
            this.prcOut.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        this.process.destroy();
    }

    public String compute(String query) {
        try {
            this.prcOut.write(query + "\n");
            this.prcOut.flush();

            return this.prcIn.readLine();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}