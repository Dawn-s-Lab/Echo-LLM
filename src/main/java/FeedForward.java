public class FeedForward {

    double[][] W1, W2;

    FeedForward(int dim) {
        W1 = random(dim, dim * 2);
        W2 = random(dim * 2, dim);
    }

    double[][] random(int in, int out) {
        double[][] m = new double[in][out];
        for (int i = 0; i < in; i++)
            for (int j = 0; j < out; j++)
                m[i][j] = Math.random() - 0.5;
        return m;
    }

    double[][] forward(double[][] X) {
        return Tensor.matmul(
                Tensor.relu(Tensor.matmul(X, W1)),
                W2
        );

    }
}
