public class Tensor {

    static double[][] matmul(double[][] A, double[][] B) {
        int m = A.length, n = B[0].length, p = B.length;
        double[][] out = new double[m][n];

        for (int i = 0; i < m; i++) {
            double[] out_i = out[i];
            double[] A_i = A[i];
            for (int k = 0; k < p; k++) {
                double a_ik = A_i[k];
                double[] B_k = B[k];
                for (int j = 0; j < n; j++) {
                    out_i[j] += a_ik * B_k[j];
                }
            }
        }

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
            double max = X[i][0];
            for (int j = 1; j < X[0].length; j++) if (X[i][j] > max) max = X[i][j];

            double sum = 0;
            for (int j = 0; j < X[0].length; j++) {
                out[i][j] = Math.exp(X[i][j] - max);
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

    static double[][] mul(double[][] A, double[][] B) {
        double[][] out = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                out[i][j] = A[i][j] * B[i][j];
        return out;
    }

    static double[][] div(double[][] A, double scalar) {
        double[][] out = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                out[i][j] = A[i][j] / scalar;
        return out;
    }

    static double[][] transpose(double[][] M) {
        double[][] t = new double[M[0].length][M.length];
        for (int i = 0; i < M.length; i++)
            for (int j = 0; j < M[0].length; j++)
                t[j][i] = M[i][j];
        return t;
    }

    static double sum(double[][] M) {
        double s = 0;
        for (double[] row : M) for (double v : row) s += v;
        return s;
    }
}
