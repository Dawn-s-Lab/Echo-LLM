import java.util.Random;
public class Sampler {
    static Random rng = new Random();
    static int sample(double[] probs) {
        // Temperature sampling
        double temp = 0.8; // Slightly higher for more variety
        double[] expProbs = new double[probs.length];
        double sum = 0;
        for (int i = 0; i < probs.length; i++) {
            expProbs[i] = Math.pow(probs[i], 1.0 / temp);
            sum += expProbs[i];
        }
        for (int i = 0; i < probs.length; i++) expProbs[i] /= sum;

        double r = rng.nextDouble();
        double cumulative = 0;
        for (int i = 0; i < expProbs.length; i++) {
            cumulative += expProbs[i];
            if (r < cumulative) {
                return i;
            }
        }
        return expProbs.length - 1; // fallback
    }
}
