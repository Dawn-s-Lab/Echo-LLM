public class Tensor {

    static double[][] matmul(double[][] A, double[][] B) {
        int m = A.length, n = B[0].length, p = B.length;
        double[][] out = new double[m][n];

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < p; k++)
                    out[i][j] += A[i][k] * B[k][j];

        return out;
    }

    static double[][] add(double[][] A, double[][] B) {
        double[][] out = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                out[i][j] = A[i][j] + B[i][j];
        return out;
    }

    static double[][] softmax(double[][] X) {
        double[][] out = new double[X.length][X[0].length];

        for (int i = 0; i < X.length; i++) {
            double sum = 0;
            for (int j = 0; j < X[0].length; j++) {
                out[i][j] = Math.exp(X[i][j]);
                sum += out[i][j];
            }
            for (int j = 0; j < X[0].length; j++)
                out[i][j] /= sum;
        }

        return out;
    }

    static double[][] relu(double[][] X) {
        double[][] out = new double[X.length][X[0].length];
        for (int i = 0; i < X.length; i++)
            for (int j = 0; j < X[0].length; j++)
                out[i][j] = Math.max(0, X[i][j]);
        return out;
    }

    static double[][] subtract(double[][] A, double[][] B) {
        double[][] out = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                out[i][j] = A[i][j] - B[i][j];
        return out;
    }

    static double[][] mul(double[][] A, double scalar) {
        double[][] out = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                out[i][j] = A[i][j] * scalar;
        return out;
    }
}
