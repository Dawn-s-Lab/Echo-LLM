public class Attention {
    int dim;
    int nHeads;
    int headDim;
    double[][][] Wq, Wk, Wv; // [head][dim][headDim]
    double[][] Wo; // [dim][dim]
    
    double[][] lastX;
    double[][] lastOut;
    double[][][] lastQ, lastK, lastV;
    double[][][] lastWeights;

    Attention(int dim, int nHeads) {
        this.dim = dim;
        this.nHeads = nHeads;
        this.headDim = dim / nHeads;
        
        Wq = new double[nHeads][dim][headDim];
        Wk = new double[nHeads][dim][headDim];
        Wv = new double[nHeads][dim][headDim];
        for (int h = 0; h < nHeads; h++) {
            Wq[h] = random(dim, headDim);
            Wk[h] = random(dim, headDim);
            Wv[h] = random(dim, headDim);
        }
        Wo = random(dim, dim);
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
        int T = X.length;
        
        lastQ = new double[nHeads][T][headDim];
        lastK = new double[nHeads][T][headDim];
        lastV = new double[nHeads][T][headDim];
        lastWeights = new double[nHeads][T][T];
        
        double[][] concat = new double[T][dim];
        
        double scale = 1.0 / Math.sqrt(headDim);

        for (int h = 0; h < nHeads; h++) {
            lastQ[h] = Tensor.matmul(X, Wq[h]);
            lastK[h] = Tensor.matmul(X, Wk[h]);
            lastV[h] = Tensor.matmul(X, Wv[h]);

            double[][] scores = Tensor.matmul(lastQ[h], Tensor.transpose(lastK[h]));
            
            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    scores[i][j] *= scale;
                    if (j > i) scores[i][j] = -1e9; // Causal mask
                }
            }
            
            lastWeights[h] = Tensor.softmax(scores);
            double[][] headOut = Tensor.matmul(lastWeights[h], lastV[h]);
            
            for (int i = 0; i < T; i++) {
                System.arraycopy(headOut[i], 0, concat[i], h * headDim, headDim);
            }
        }
        
        this.lastOut = Tensor.matmul(concat, Wo);
        return lastOut;
    }
}
