import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

public class TinyTransformer {

    Embedding embedding;
    Embedding positionalEmbedding;
    TransformerBlock[] blocks;
    double[][] outputWeights;
    int[] lastTokens;
    double[][] lastX, lastPos;
    double[][][] lastBlockOutputs;

    TinyTransformer(int vocab, int dim, int nHeads, int nBlocks) {
        embedding = new Embedding(vocab, dim);
        positionalEmbedding = new Embedding(1024, dim);
        blocks = new TransformerBlock[nBlocks];
        for (int i = 0; i < nBlocks; i++) {
            blocks[i] = new TransformerBlock(dim, nHeads);
        }

        outputWeights = new double[dim][vocab];
        double scale = Math.sqrt(2.0 / (dim + vocab));
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < vocab; j++)
                outputWeights[i][j] = (Math.random() - 0.5) * 2 * scale;
    }

    double[][] forward(int[] tokens) {
        this.lastTokens = tokens;
        this.lastX = embedding.forward(tokens);
        
        int[] positions = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) positions[i] = i % 1024;
        this.lastPos = positionalEmbedding.forward(positions);
        double[][] x = Tensor.add(lastX, lastPos);

        lastBlockOutputs = new double[blocks.length + 1][tokens.length][x[0].length];
        lastBlockOutputs[0] = x;
        for (int i = 0; i < blocks.length; i++) {
            x = blocks[i].forward(x);
            lastBlockOutputs[i+1] = x;
        }
        
        return Tensor.softmax(Tensor.matmul(x, outputWeights));
    }

    public void saveWeights(String path) {
        try (FileOutputStream fos = new FileOutputStream(path);
             FileChannel channel = fos.getChannel()) {
            saveMatrix(channel, embedding.table);
            saveMatrix(channel, positionalEmbedding.table);
            for (TransformerBlock block : blocks) {
                for (int h = 0; h < block.attention.nHeads; h++) {
                    saveMatrix(channel, block.attention.Wq[h]);
                    saveMatrix(channel, block.attention.Wk[h]);
                    saveMatrix(channel, block.attention.Wv[h]);
                }
                saveMatrix(channel, block.attention.Wo);
                saveArray(channel, block.ln1.gamma);
                saveArray(channel, block.ln1.beta);
                saveMatrix(channel, block.ff.W1);
                saveMatrix(channel, block.ff.W2);
                saveArray(channel, block.ln2.gamma);
                saveArray(channel, block.ln2.beta);
            }
            saveMatrix(channel, outputWeights);
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
            for (TransformerBlock block : blocks) {
                for (int h = 0; h < block.attention.nHeads; h++) {
                    loadMatrix(channel, block.attention.Wq[h]);
                    loadMatrix(channel, block.attention.Wk[h]);
                    loadMatrix(channel, block.attention.Wv[h]);
                }
                loadMatrix(channel, block.attention.Wo);
                loadArray(channel, block.ln1.gamma);
                loadArray(channel, block.ln1.beta);
                loadMatrix(channel, block.ff.W1);
                loadMatrix(channel, block.ff.W2);
                loadArray(channel, block.ln2.gamma);
                loadArray(channel, block.ln2.beta);
            }
            loadMatrix(channel, outputWeights);
            System.out.println("Weights loaded from " + path);
        } catch (IOException e) {
            System.err.println("Failed to load weights: " + e.getMessage());
        }
    }

    private void saveArray(FileChannel channel, double[] a) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(a.length * 8);
        for (double v : a) buf.putDouble(v);
        buf.flip();
        channel.write(buf);
    }

    private void loadArray(FileChannel channel, double[] a) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(a.length * 8);
        channel.read(buf);
        buf.flip();
        for (int i = 0; i < a.length; i++) a[i] = buf.getDouble();
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
