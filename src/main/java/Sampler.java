import java.util.Random;
public class Sampler {
    static Random rng = new Random();
    static int sample(double[] probs) {
        double r = rng.nextDouble();
        double cumulative = 0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r < cumulative) {
                return i;
            }
        }
        return probs.length - 1; // fallback
    }
}
