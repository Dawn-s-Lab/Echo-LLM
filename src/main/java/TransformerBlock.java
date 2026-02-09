public class TransformerBlock {

    Attention attention;
    FeedForward ff;

    TransformerBlock(int dim) {
        attention = new Attention(dim);
        ff = new FeedForward(dim);
    }

    double[][] forward(double[][] X) {
        X = Tensor.add(X, attention.forward(X));
        X = Tensor.add(X, ff.forward(X));
        return X;
    }
}
