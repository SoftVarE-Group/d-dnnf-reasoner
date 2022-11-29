import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

class StreamGobbler extends Thread
{
    private InputStream is;
    private String myMessage;

    public StreamGobbler(InputStream istream) {
        this.is = istream;
    }

    public String getMessage() {
        return this.myMessage;
    }

    @Override
    public void run() {
        StringBuilder buffer = new StringBuilder();
        BufferedReader br = new BufferedReader(new InputStreamReader(is));

        int size = 1024 * 1024;
        char[] ch = new char[size];
        int read = 0;
        try {
            while ((read = br.read(ch, 0, size)) >= 0) {
                buffer.append(ch, 0, read);
            }
        }
        catch (Exception ioe) {
            ioe.printStackTrace();
        }
        finally {
            try {
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        this.myMessage = buffer.toString();
    }
}