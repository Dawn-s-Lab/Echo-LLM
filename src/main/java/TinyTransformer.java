public class TinyTransformer {

    Embedding embedding;
    Embedding positionalEmbedding;
    TransformerBlock block;
    double[][] output;

    TinyTransformer(int vocab, int dim) {
        embedding = new Embedding(vocab, dim);
        positionalEmbedding = new Embedding(1024, dim); // Support up to 1024 context length
        block = new TransformerBlock(dim);

        output = new double[dim][vocab];
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < vocab; j++)
                output[i][j] = Math.random() - 0.5;
    }

    double[][] forward(int[] tokens) {
        double[][] x = embedding.forward(tokens);
        
        // Add positional embeddings
        int[] positions = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) positions[i] = i % 1024;
        double[][] pos = positionalEmbedding.forward(positions);
        x = Tensor.add(x, pos);

        x = block.forward(x);
        return Tensor.softmax(Tensor.matmul(x, output));
    }
}
