public class Attention {

    double[][] Wq, Wk, Wv;

    Attention(int dim) {
        Wq = random(dim);
        Wk = random(dim);
        Wv = random(dim);
    }

    double[][] random(int dim) {
        double[][] m = new double[dim][dim];
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
                m[i][j] = Math.random() - 0.5;
        return m;
    }

    double[][] forward(double[][] X) {

        double[][] Q = Tensor.matmul(X, Wq);
        double[][] K = Tensor.matmul(X, Wk);
        double[][] V = Tensor.matmul(X, Wv);

        double[][] scores = Tensor.matmul(Q, transpose(K));

        // scale
        double scale = 1.0 / Math.sqrt(X[0].length);
        for (int i = 0; i < scores.length; i++)
            for (int j = 0; j < scores[0].length; j++)
                scores[i][j] *= scale;

        double[][] weights = Tensor.softmax(scores);

        // causal mask
        for (int i = 0; i < weights.length; i++) {
            for (int j = i + 1; j < weights[0].length; j++) {
                weights[i][j] = 0;
            }
            // re-normalize
            double sum = 0;
            for (int j = 0; j <= i; j++) sum += weights[i][j];
            for (int j = 0; j <= i; j++) weights[i][j] /= sum;
        }

        return Tensor.matmul(weights, V);
    }

    double[][] transpose(double[][] M) {
        double[][] t = new double[M[0].length][M.length];
        for (int i = 0; i < M.length; i++)
            for (int j = 0; j < M[0].length; j++)
                t[j][i] = M[i][j];
        return t;
    }
}
