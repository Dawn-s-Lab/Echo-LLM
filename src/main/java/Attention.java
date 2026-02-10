public class Attention {

    double[][] Wq, Wk, Wv;
    double[][] lastX, lastWeights, lastV, lastScores;

    Attention(int dim) {
        Wq = random(dim, dim);
        Wk = random(dim, dim);
        Wv = random(dim, dim);
    }

    double[][] random(int rows, int cols) {
        double[][] m = new double[rows][cols];
        double scale = Math.sqrt(2.0 / (rows + cols));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i][j] = (Math.random() - 0.5) * 2 * scale;
        return m;
    }

    double[][] forward(double[][] X) {
        this.lastX = X;
        double[][] Q = Tensor.matmul(X, Wq);
        double[][] K = Tensor.matmul(X, Wk);
        this.lastV = Tensor.matmul(X, Wv);

        double[][] scores = Tensor.matmul(Q, Tensor.transpose(K));

        // scale
        double scale = 1.0 / Math.sqrt(Wq[0].length);
        for (int i = 0; i < scores.length; i++)
            for (int j = 0; j < scores[0].length; j++)
                scores[i][j] *= scale;

        // causal mask
        for (int i = 0; i < scores.length; i++) {
            for (int j = i + 1; j < scores[0].length; j++) {
                scores[i][j] = -1e9;
            }
        }
        this.lastScores = scores;
        this.lastWeights = Tensor.softmax(scores);

        return Tensor.matmul(lastWeights, lastV);
    }
}
