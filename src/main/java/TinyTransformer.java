import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

public class TinyTransformer {

    Embedding embedding;
    Embedding positionalEmbedding;
    TransformerBlock block;
    double[][] output;
    int[] lastTokens;
    double[][] lastX, lastPos, lastBlockOut;

    TinyTransformer(int vocab, int dim) {
        embedding = new Embedding(vocab, dim);
        positionalEmbedding = new Embedding(1024, dim); // Support up to 1024 context length
        block = new TransformerBlock(dim);

        output = new double[dim][vocab];
        double scale = Math.sqrt(2.0 / (dim + vocab));
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < vocab; j++)
                output[i][j] = (Math.random() - 0.5) * 2 * scale;
    }

    double[][] forward(int[] tokens) {
        this.lastTokens = tokens;
        this.lastX = embedding.forward(tokens);
        
        // Add positional embeddings
        int[] positions = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) positions[i] = i % 1024;
        this.lastPos = positionalEmbedding.forward(positions);
        double[][] x = Tensor.add(lastX, lastPos);

        this.lastBlockOut = block.forward(x);
        return Tensor.softmax(Tensor.matmul(lastBlockOut, output));
    }

    public void saveWeights(String path) {
        try (FileOutputStream fos = new FileOutputStream(path);
             FileChannel channel = fos.getChannel()) {
            saveMatrix(channel, embedding.table);
            saveMatrix(channel, positionalEmbedding.table);
            saveMatrix(channel, block.attention.Wq);
            saveMatrix(channel, block.attention.Wk);
            saveMatrix(channel, block.attention.Wv);
            saveMatrix(channel, block.ff.W1);
            saveMatrix(channel, block.ff.W2);
            saveMatrix(channel, output);
        } catch (IOException e) {
            System.err.println("Failed to save weights: " + e.getMessage());
        }
    }

    public void loadWeights(String path) {
        File file = new File(path);
        if (!file.exists()) return;
        try (FileInputStream fis = new FileInputStream(path);
             FileChannel channel = fis.getChannel()) {
            loadMatrix(channel, embedding.table);
            loadMatrix(channel, positionalEmbedding.table);
            loadMatrix(channel, block.attention.Wq);
            loadMatrix(channel, block.attention.Wk);
            loadMatrix(channel, block.attention.Wv);
            loadMatrix(channel, block.ff.W1);
            loadMatrix(channel, block.ff.W2);
            loadMatrix(channel, output);
            System.out.println("Weights loaded from " + path);
        } catch (IOException e) {
            System.err.println("Failed to load weights: " + e.getMessage());
        }
    }

    private void saveMatrix(FileChannel channel, double[][] m) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(m.length * m[0].length * 8);
        for (double[] row : m) {
            for (double v : row) buf.putDouble(v);
        }
        buf.flip();
        channel.write(buf);
    }

    private void loadMatrix(FileChannel channel, double[][] m) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(m.length * m[0].length * 8);
        channel.read(buf);
        buf.flip();
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[0].length; j++) m[i][j] = buf.getDouble();
        }
    }
}
