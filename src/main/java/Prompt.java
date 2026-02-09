public class Prompt {
    static String prompt(
            TinyTransformer model,
            String seed,
            int steps
    ) {

        StringBuilder text = new StringBuilder(seed);

        for (int i = 0; i < steps; i++) {

            // convert current text to tokens
            int[] tokens = text.chars().toArray();

            // run model
            double[][] probs = model.forward(tokens);

            // sample next token
            double[] last = probs[probs.length - 1];
            int next = Sampler.sample(last);

            // append
            text.append((char) next);
        }

        return text.toString();
    }

}
