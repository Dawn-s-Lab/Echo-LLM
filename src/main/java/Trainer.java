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
                if (nextToken >= 256) nextToken = 0;

                // Forward
                double[][] probs = model.forward(input);
                double[] lastProbs = probs[probs.length - 1];
                
                // Loss (Cross Entropy)
                totalLoss += -Math.log(lastProbs[nextToken] + 1e-10);
                count++;

                // Slightly better heuristic: nudge towards target, away from others
                for (int d = 0; d < model.output.length; d++) {
                    model.output[d][nextToken] += lr * 0.1;
                    // Push away from other likely tokens slightly to sharpen distribution
                    for (int v = 0; v < 256; v++) {
                        if (v != nextToken && lastProbs[v] > 0.1) {
                            model.output[d][v] -= lr * 0.02;
                        }
                    }
                }
                
                // Update embeddings for the context
                for (int t : input) {
                    if (t < 256) {
                        for (int d = 0; d < model.embedding.table[t].length; d++) {
                            model.embedding.table[t][d] += lr * 0.01;
                        }
                    }
                }
            }
            if (epoch % 20 == 0) {
                System.out.println("Epoch " + epoch + " Loss: " + (totalLoss / count));
            }
        }
    }
}
