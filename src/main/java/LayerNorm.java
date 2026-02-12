public class LayerNorm {
    double[] gamma;
    double[] beta;
    double[][] lastX;
    double[] lastMean;
    double[] lastVar;
    double eps = 1e-5;

    LayerNorm(int dim) {
        gamma = new double[dim];
        beta = new double[dim];
        for (int i = 0; i < dim; i++) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    double[][] forward(double[][] X) {
        this.lastX = X;
        int n = X.length;
        int d = X[0].length;
        double[][] out = new double[n][d];
        lastMean = new double[n];
        lastVar = new double[n];

        for (int i = 0; i < n; i++) {
            double mean = 0;
            for (int j = 0; j < d; j++) mean += X[i][j];
            mean /= d;
            lastMean[i] = mean;

            double var = 0;
            for (int j = 0; j < d; j++) var += (X[i][j] - mean) * (X[i][j] - mean);
            var /= d;
            lastVar[i] = var;

            for (int j = 0; j < d; j++) {
                out[i][j] = (X[i][j] - mean) / Math.sqrt(var + eps) * gamma[j] + beta[j];
            }
        }
        return out;
    }
}