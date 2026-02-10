public class FeedForward {

    double[][] W1, W2;
    double[][] lastX, lastH;

    FeedForward(int dim) {
        W1 = random(dim, dim * 4);
        W2 = random(dim * 4, dim);
    }

    double[][] random(int in, int out) {
        double[][] m = new double[in][out];
        double scale = Math.sqrt(2.0 / (in + out));
        for (int i = 0; i < in; i++)
            for (int j = 0; j < out; j++)
                m[i][j] = (Math.random() - 0.5) * 2 * scale;
        return m;
    }

    double[][] forward(double[][] X) {
        this.lastX = X;
        this.lastH = Tensor.relu(Tensor.matmul(X, W1));
        return Tensor.matmul(lastH, W2);
    }
}
