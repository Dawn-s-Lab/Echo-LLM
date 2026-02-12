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
            int contextStart = Math.max(0, tokens.length - 1024);
            int[] context = new int[tokens.length - contextStart];
            System.arraycopy(tokens, contextStart, context, 0, context.length);
            
            double[][] probs = model.forward(context);

            // sample next token
            double[] last = probs[probs.length - 1];
            int next = Sampler.sample(last);

            // append
            text.append((char) next);
        }

        return text.toString();
    }

}
