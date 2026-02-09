public class Embedding {

    double[][] table;

    Embedding(int vocabSize, int dim) {
        table = new double[vocabSize][dim];
        for (int i = 0; i < vocabSize; i++)
            for (int j = 0; j < dim; j++)
                table[i][j] = Math.random() - 0.5;
    }

    double[][] forward(int[] tokens) {
        double[][] out = new double[tokens.length][table[0].length];
        for (int i = 0; i < tokens.length; i++) {
            int token = tokens[i];
            // Basic bound check for vocab
            if (token < 0 || token >= table.length) token = 0; 
            out[i] = table[token].clone();
        }
        return out;
    }
}
