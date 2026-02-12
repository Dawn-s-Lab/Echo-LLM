import java.util.ArrayList;
import java.util.List;

public class Trainer {

    private static List<String> trainingUrls = new ArrayList<>();

    public static void addTrainingUrl(String url) {
        trainingUrls.add(url);
    }

    public static void clearTrainingUrls() {
        trainingUrls.clear();
    }

    public static void trainWithUrls(TinyTransformer model, int epochs, double lr) {
        StringBuilder allData = new StringBuilder();
        for (String url : trainingUrls) {
            String data;
            if (url.contains("wikipedia.org")) {
                // Extract title if it's a wikipedia link, or just use as is if it's a title
                String title = url.substring(url.lastIndexOf("/") + 1);
                data = WikiData.fetchFromWikipedia(title);
            } else {
                data = WikiData.fetchFromUrl(url);
            }
            if (data != null && !data.startsWith("Error")) {
                allData.append(data).append(" ");
            }
        }
        train(model, allData.toString(), epochs, lr);
    }

    static void train(TinyTransformer model, String data, int epochs, double lr) {
        if (data == null || data.length() <= 16) {
            System.out.println("Insufficient training data.");
            return;
        }
        int[] tokens = data.chars().toArray();
        int contextWindow = 16;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            int count = 0;
            for (int i = 0; i < tokens.length - contextWindow - 1; i += contextWindow) {
                int[] input = new int[contextWindow];
                System.arraycopy(tokens, i, input, 0, contextWindow);
                
                int[] targets = new int[contextWindow];
                for (int t = 0; t < contextWindow; t++) {
                    int nextToken = tokens[i + t + 1];
                    if (nextToken < 0 || nextToken >= 256) nextToken = ' ';
                    targets[t] = nextToken;
                }

                // Forward
                double[][] probs = model.forward(input);
                
                // Loss (Cross Entropy) over all timesteps
                for (int t = 0; t < contextWindow; t++) {
                    double p = probs[t][targets[t]];
                    if (p < 1e-15) p = 1e-15;
                    if (p > 1.0) p = 1.0;
                    totalLoss += -Math.log(p);
                    count++;
                }

                // Backward pass
                double[][] dLogits = new double[probs.length][256];
                for(int t=0; t<probs.length; t++) {
                    for(int v=0; v<256; v++) {
                        dLogits[t][v] = probs[t][v];
                    }
                    dLogits[t][targets[t]] -= 1.0;
                }

                // dLoss/dOutputWeights
                double[][] lastBlockOut = model.lastBlockOutputs[model.blocks.length];
                double[][] dOutput = Tensor.matmul(Tensor.transpose(lastBlockOut), dLogits);
                // dLoss/dX_last
                double[][] dX = Tensor.matmul(dLogits, Tensor.transpose(model.outputWeights));

                // Backprop through blocks
                for (int b = model.blocks.length - 1; b >= 0; b--) {
                    TransformerBlock block = model.blocks[b];
                    
                    // X = x_res1 + FF(LN2(x_res1))
                    // dLoss/dFF_out = dX
                    // dLoss/dx_res1 = dX + dLoss/dFF_in
                    
                    FeedForward ff = block.ff;
                    LayerNorm ln2 = block.ln2;
                    
                    // FF update
                    double[][] dW2 = Tensor.matmul(Tensor.transpose(ff.lastH), dX);
                    double[][] dH = Tensor.matmul(dX, Tensor.transpose(ff.W2));
                    double[][] dZ = new double[ff.lastH.length][ff.lastH[0].length];
                    for(int r=0; r<dH.length; r++) {
                        for(int c=0; c<dH[0].length; c++) {
                            if(ff.lastH[r][c] > 0) dZ[r][c] = dH[r][c];
                        }
                    }
                    double[][] dW1 = Tensor.matmul(Tensor.transpose(ff.lastX), dZ);
                    double[][] dX_ff_in = Tensor.matmul(dZ, Tensor.transpose(ff.W1));
                    
                    update(ff.W2, dW2, lr);
                    update(ff.W1, dW1, lr);
                    
                    // LN2 backprop
                    double[][] dX_ln2_in = new double[dX.length][dX[0].length];
                    for (int row = 0; row < dX.length; row++) {
                        double mean = ln2.lastMean[row];
                        double var = ln2.lastVar[row];
                        double stdInv = 1.0 / Math.sqrt(var + ln2.eps);
                        int D = ln2.gamma.length;
                        
                        double dot = 0;
                        for (int j = 0; j < D; j++) {
                            double x_hat = (ln2.lastX[row][j] - mean) * stdInv;
                            dot += dX[row][j] * ln2.gamma[j] * x_hat;
                        }
                        
                        double sum_grad = 0;
                        for (int j = 0; j < D; j++) {
                            sum_grad += dX[row][j] * ln2.gamma[j];
                        }

                        for (int j = 0; j < D; j++) {
                            double x_hat = (ln2.lastX[row][j] - mean) * stdInv;
                            double grad = dX[row][j];
                            if (Double.isNaN(grad) || Double.isInfinite(grad)) grad = 0;
                            
                            // Parameter updates
                            ln2.gamma[j] -= lr * Math.max(-5.0, Math.min(5.0, grad * x_hat));
                            ln2.beta[j] -= lr * Math.max(-5.0, Math.min(5.0, grad));
                            
                            // Input gradient
                            dX_ln2_in[row][j] = (ln2.gamma[j] * stdInv / D) * (D * grad - sum_grad - x_hat * dot);
                        }
                    }
                    
                    // dX for the residual connection before FF
                    double[][] dX_res1 = Tensor.add(dX, dX_ln2_in);

                    // x_res1 = X_prev + Attn(LN1(X_prev))
                    // dLoss/dAttn_out = dX_res1
                    // dLoss/dX_prev = dX_res1 + dLoss/dAttn_in
                    
                    Attention attn = block.attention;
                    LayerNorm ln1 = block.ln1;
                    
                    // Attention update
                    int T_len = attn.lastX.length;
                    double[][] dWo_real = Tensor.matmul(Tensor.transpose(attn.lastConcat), dX_res1);
                    double[][] dConcat = Tensor.matmul(dX_res1, Tensor.transpose(attn.Wo));
                    
                    double[][] dX_attn_in = new double[T_len][attn.dim];
                    
                    for (int h = 0; h < attn.nHeads; h++) {
                        double[][] dHeadOut = new double[T_len][attn.headDim];
                        for (int row = 0; row < T_len; row++) {
                            System.arraycopy(dConcat[row], h * attn.headDim, dHeadOut[row], 0, attn.headDim);
                        }
                        
                        // headOut = weights * V
                        double[][] dWv = Tensor.matmul(Tensor.transpose(attn.lastWeights[h]), dHeadOut);
                        double[][] dWeights = Tensor.matmul(dHeadOut, Tensor.transpose(attn.lastV[h]));
                        double[][] dV = Tensor.matmul(attn.lastWeights[h], dHeadOut);
                        
                        // weights = softmax(Q*K^T / scale)
                        double[][] dScores = new double[T_len][T_len];
                        for (int row = 0; row < T_len; row++) {
                            double dot_weights = 0;
                            for (int col = 0; col < T_len; col++) {
                                dot_weights += dWeights[row][col] * attn.lastWeights[h][row][col];
                            }
                            for (int col = 0; col < T_len; col++) {
                                dScores[row][col] = attn.lastWeights[h][row][col] * (dWeights[row][col] - dot_weights);
                            }
                        }
                        
                        double scale = 1.0 / Math.sqrt(attn.headDim);
                        double[][] dQ = Tensor.mul(Tensor.matmul(dScores, attn.lastK[h]), scale);
                        double[][] dK = Tensor.mul(Tensor.matmul(Tensor.transpose(dScores), attn.lastQ[h]), scale);
                        
                        update(attn.Wq[h], Tensor.matmul(Tensor.transpose(attn.lastX), dQ), lr);
                        update(attn.Wk[h], Tensor.matmul(Tensor.transpose(attn.lastX), dK), lr);
                        update(attn.Wv[h], Tensor.matmul(Tensor.transpose(attn.lastX), dV), lr);
                        
                        // Accumulate dX_attn_in
                        double[][] dX_q = Tensor.matmul(dQ, Tensor.transpose(attn.Wq[h]));
                        double[][] dX_k = Tensor.matmul(dK, Tensor.transpose(attn.Wk[h]));
                        double[][] dX_v = Tensor.matmul(dV, Tensor.transpose(attn.Wv[h]));
                        dX_attn_in = Tensor.add(dX_attn_in, Tensor.add(dX_q, Tensor.add(dX_k, dX_v)));
                    }
                    
                    update(attn.Wo, dWo_real, lr);
                    
                    // LN1 backprop
                    double[][] dX_ln1_in = new double[dX_res1.length][dX_res1[0].length];
                    for (int row = 0; row < dX_res1.length; row++) {
                        double mean = ln1.lastMean[row];
                        double var = ln1.lastVar[row];
                        double stdInv = 1.0 / Math.sqrt(var + ln1.eps);
                        int D = ln1.gamma.length;
                        
                        double dot = 0;
                        for (int j = 0; j < D; j++) {
                            double x_hat = (ln1.lastX[row][j] - mean) * stdInv;
                            dot += dX_attn_in[row][j] * ln1.gamma[j] * x_hat;
                        }
                        
                        double sum_grad = 0;
                        for (int j = 0; j < D; j++) {
                            sum_grad += dX_attn_in[row][j] * ln1.gamma[j];
                        }

                        for (int j = 0; j < D; j++) {
                            double x_hat = (ln1.lastX[row][j] - mean) * stdInv;
                            double grad = dX_attn_in[row][j];
                            if (Double.isNaN(grad) || Double.isInfinite(grad)) grad = 0;
                            
                            ln1.gamma[j] -= lr * Math.max(-5.0, Math.min(5.0, grad * x_hat));
                            ln1.beta[j] -= lr * Math.max(-5.0, Math.min(5.0, grad));
                            
                            dX_ln1_in[row][j] = (ln1.gamma[j] * stdInv / D) * (D * grad - sum_grad - x_hat * dot);
                        }
                    }
                    
                    // Final dX for the next block (previous block in forward)
                    dX = Tensor.add(dX_res1, dX_ln1_in);
                }

                update(model.outputWeights, dOutput, lr);
                
                // Update embeddings
                for(int t=0; t<input.length; t++) {
                    int tok = input[t];
                    if(tok >= 0 && tok < 256) {
                        for(int d=0; d<model.embedding.table[tok].length; d++) {
                            double g = dX[t][d];
                            if (g > 5.0) g = 5.0;
                            if (g < -5.0) g = -5.0;
                            model.embedding.table[tok][d] -= lr * g;
                        }
                    }
                    
                    // Positional embeddings update
                    int pos = t % 1024;
                    for(int d=0; d<model.positionalEmbedding.table[pos].length; d++) {
                        double g = dX[t][d];
                        if (g > 5.0) g = 5.0;
                        if (g < -5.0) g = -5.0;
                        model.positionalEmbedding.table[pos][d] -= lr * g;
                    }
                }
            }
            if (epoch % 10 == 0) {
                System.out.println("Epoch " + epoch + " Loss: " + (totalLoss / count));
            }
        }
    }

    private static void update(double[][] weight, double[][] grad, double lr) {
        double clipValue = 5.0;
        for(int i=0; i<weight.length; i++) {
            for(int j=0; j<weight[0].length; j++) {
                double g = grad[i][j];
                if (g > clipValue) g = clipValue;
                if (g < -clipValue) g = -clipValue;
                weight[i][j] -= lr * g;
            }
        }
    }
}
