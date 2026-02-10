public class Trainer {

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
            for (int i = 0; i < tokens.length - contextWindow; i++) {
                int[] input = new int[contextWindow];
                System.arraycopy(tokens, i, input, 0, contextWindow);
                
                int nextToken = tokens[i + contextWindow];
                if (nextToken < 0 || nextToken >= 256) nextToken = ' ';

                // Forward
                double[][] probs = model.forward(input);
                double[] lastProbs = probs[probs.length - 1];
                
                // Loss (Cross Entropy)
                totalLoss += -Math.log(lastProbs[nextToken] + 1e-10);
                count++;

                // Backward pass (simplified backprop)
                // dLoss/dlogits = probs - target
                double[][] dLogits = new double[probs.length][256];
                for(int t=0; t<probs.length; t++) {
                    for(int v=0; v<256; v++) {
                        dLogits[t][v] = probs[t][v];
                    }
                }
                dLogits[probs.length-1][nextToken] -= 1.0;

                // dLoss/dOutputWeights = lastBlockOut^T * dLogits
                double[][] dOutput = Tensor.matmul(Tensor.transpose(model.lastBlockOut), dLogits);
                // dLoss/dLastBlockOut = dLogits * OutputWeights^T
                double[][] dBlockOut = Tensor.matmul(dLogits, Tensor.transpose(model.output));

                // Backward through TransformerBlock
                // Residual connections: dX = dBlockOut + dFF_out + dAttn_out
                // FF backward
                FeedForward ff = model.block.ff;
                double[][] dFFOut = dBlockOut;
                double[][] dW2 = Tensor.matmul(Tensor.transpose(ff.lastH), dFFOut);
                double[][] dH = Tensor.matmul(dFFOut, Tensor.transpose(ff.W2));
                // ReLU backward
                double[][] dZ = new double[ff.lastH.length][ff.lastH[0].length];
                for(int row=0; row<dH.length; row++) {
                    for(int col=0; col<dH[0].length; col++) {
                        if(ff.lastH[row][col] > 0) dZ[row][col] = dH[row][col];
                    }
                }
                double[][] dW1 = Tensor.matmul(Tensor.transpose(ff.lastX), dZ);
                double[][] dFF_in = Tensor.matmul(dZ, Tensor.transpose(ff.W1));

                // Attention backward (very simplified)
                Attention attn = model.block.attention;
                double[][] dAttnOut = dBlockOut;
                double[][] dWv = Tensor.matmul(Tensor.transpose(attn.lastX), Tensor.matmul(Tensor.transpose(attn.lastWeights), dAttnOut));
                double[][] dAttnIn = Tensor.matmul(Tensor.matmul(dAttnOut, Tensor.transpose(attn.lastV)), attn.lastWeights); // heuristic

                // Update weights (SGD)
                update(model.output, dOutput, lr);
                update(ff.W2, dW2, lr);
                update(ff.W1, dW1, lr);
                update(attn.Wv, dWv, lr);
                
                // Update embeddings
                for(int t=0; t<input.length; t++) {
                    int tok = input[t];
                    if(tok >= 0 && tok < 256) {
                        for(int d=0; d<model.embedding.table[tok].length; d++) {
                            model.embedding.table[tok][d] -= lr * dBlockOut[t][d];
                        }
                    }
                }
            }
            if (epoch % 10 == 0) {
                System.out.println("Epoch " + epoch + " Loss: " + (totalLoss / count));
            }
        }
    }

    private static void update(double[][] weight, double[][] grad, double lr) {
        for(int i=0; i<weight.length; i++) {
            for(int j=0; j<weight[0].length; j++) {
                weight[i][j] -= lr * grad[i][j];
            }
        }
    }
}
