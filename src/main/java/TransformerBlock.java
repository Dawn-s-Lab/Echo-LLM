public class TransformerBlock {

    Attention attention;
    FeedForward ff;
    double[][] lastX, lastAttn;

    TransformerBlock(int dim) {
        attention = new Attention(dim);
        ff = new FeedForward(dim);
    }

    double[][] forward(double[][] X) {
        this.lastX = X;
        this.lastAttn = attention.forward(X);
        X = Tensor.add(X, lastAttn);
        X = Tensor.add(X, ff.forward(X));
        return X;
    }
}
