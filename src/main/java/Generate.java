public class Generate {
    static String generate(
            TinyTransformer model,
            String seed,
            int length
    ) {

        StringBuilder text = new StringBuilder(seed);

        for (int i = 0; i < length; i++) {

            int[] tokens = text
                    .chars()
                    .toArray();

            double[][] probs = model.forward(tokens);

            double[] last = probs[probs.length - 1];

            int next = Sampler.sample(last);

            text.append((char) next);
        }

        return text.toString();
    }

}
