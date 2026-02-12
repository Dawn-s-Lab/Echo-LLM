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

                // Backward pass (simplified)
                double[][] dLogits = new double[probs.length][256];
                for(int t=0; t<probs.length; t++) {
                    for(int v=0; v<256; v++) {
                        dLogits[t][v] = probs[t][v];
                    }
                }
                dLogits[probs.length-1][nextToken] -= 1.0;

                // dLoss/dOutputWeights
                double[][] lastBlockOut = model.lastBlockOutputs[model.blocks.length];
                double[][] dOutput = Tensor.matmul(Tensor.transpose(lastBlockOut), dLogits);
                // dLoss/dX_last
                double[][] dX = Tensor.matmul(dLogits, Tensor.transpose(model.outputWeights));

                // Backprop through blocks
                for (int b = model.blocks.length - 1; b >= 0; b--) {
                    TransformerBlock block = model.blocks[b];
                    // Very simplified backprop for demonstration
                    // In a real LLM, we'd go through LayerNorm and FF precisely
                    
                    // FF update
                    FeedForward ff = block.ff;
                    double[][] dW2 = Tensor.matmul(Tensor.transpose(ff.lastH), dX);
                    double[][] dH = Tensor.matmul(dX, Tensor.transpose(ff.W2));
                    double[][] dZ = new double[ff.lastH.length][ff.lastH[0].length];
                    for(int row=0; row<dH.length; row++) {
                        for(int col=0; col<dH[0].length; col++) {
                            if(ff.lastH[row][col] > 0) dZ[row][col] = dH[row][col];
                        }
                    }
                    double[][] dW1 = Tensor.matmul(Tensor.transpose(ff.lastX), dZ);
                    
                    // Attention update (heuristic)
                    Attention attn = block.attention;
                    for (int h = 0; h < attn.nHeads; h++) {
                        double[][] dWv = Tensor.matmul(Tensor.transpose(attn.lastX), Tensor.matmul(Tensor.transpose(attn.lastWeights[h]), dX));
                        update(attn.Wv[h], dWv, lr);
                    }
                    
                    update(ff.W2, dW2, lr);
                    update(ff.W1, dW1, lr);
                }

                update(model.outputWeights, dOutput, lr);
                
                // Update embeddings
                for(int t=0; t<input.length; t++) {
                    int tok = input[t];
                    if(tok >= 0 && tok < 256) {
                        for(int d=0; d<model.embedding.table[tok].length; d++) {
                            model.embedding.table[tok][d] -= lr * dX[t][d];
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
