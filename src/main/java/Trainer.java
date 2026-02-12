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
                    totalLoss += -Math.log(probs[t][targets[t]] + 1e-10);
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
