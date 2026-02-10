public class Prompt {
    static String prompt(
            TinyTransformer model,
            String seed,
            int steps
    ) {

        StringBuilder text = new StringBuilder(seed);

        for (int i = 0; i < steps; i++) {

            // convert current text to tokens
            int[] rawChars = text.chars().toArray();
            int[] tokens = new int[rawChars.length];
            for (int j = 0; j < rawChars.length; j++) {
                tokens[j] = (rawChars[j] >= 0 && rawChars[j] < 256) ? rawChars[j] : ' ';
            }

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
