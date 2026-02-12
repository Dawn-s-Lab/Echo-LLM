public class TransformerBlock {

    Attention attention;
    FeedForward ff;
    LayerNorm ln1, ln2;
    double[][] lastX, lastAttn, lastLN1, lastLN2;

    TransformerBlock(int dim, int nHeads) {
        attention = new Attention(dim, nHeads);
        ff = new FeedForward(dim);
        ln1 = new LayerNorm(dim);
        ln2 = new LayerNorm(dim);
    }

    double[][] forward(double[][] X) {
        this.lastX = X;
        this.lastLN1 = ln1.forward(X);
        this.lastAttn = attention.forward(lastLN1);
        double[][] x_res1 = Tensor.add(X, lastAttn);
        this.lastLN2 = ln2.forward(x_res1);
        X = Tensor.add(x_res1, ff.forward(lastLN2));
        return X;
    }
}
