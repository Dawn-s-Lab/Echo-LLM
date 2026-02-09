import java.util.Arrays;

public class Main {
    public static void main(String[] args) {

        TinyTransformer model = new TinyTransformer(256, 64);

        String trainingData = WikiData.getSampleData();
        if (trainingData == null) trainingData = "";
        System.out.println("Fetched " + trainingData.length() + " characters from Wikipedia.");
        System.out.println("Training on Wikipedia data...");
        Trainer.train(model, trainingData, 50, 0.05);

        System.out.println("\nGenerating:");
        String output = Prompt.prompt(model, "Transformers are ", 50);

        System.out.println(output);
    }
}
