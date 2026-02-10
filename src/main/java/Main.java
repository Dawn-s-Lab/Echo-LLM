import java.util.Scanner;

public class Main {
    public static void main(String[] args) {

        TinyTransformer model = new TinyTransformer(256, 128);

        Scanner scanner = new Scanner(System.in);
        System.out.print("Do you want to re-train the model? (y/n): ");
        String answer = scanner.nextLine().trim().toLowerCase();

        if (answer.equals("y")) {
            String trainingData = WikiData.getSampleData();
            if (trainingData == null) trainingData = "";
            System.out.println("Fetched " + trainingData.length() + " characters from Wikipedia.");
            System.out.println("Training on Wikipedia data...");
            Trainer.train(model, trainingData, 200, 0.05);
            model.saveWeights("weights.bin");
        } else {
            model.loadWeights("weights.bin");
            System.out.println("Skipping training.");
        }

        System.out.println("\nGenerating:");
        String output = Prompt.prompt(model, "Transformers are ", 50);

        System.out.println(output);
    }
}
