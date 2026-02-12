import java.util.Scanner;

public class Main {
    public static void main(String[] args) {

        TinyTransformer model = new TinyTransformer(256, 128, 4, 4);

        Scanner scanner = new Scanner(System.in);
        System.out.print("Do you want to re-train the model? (y/n): ");
        String answer = scanner.nextLine().trim().toLowerCase();

        if (answer.equals("y")) {
            System.out.println("Do you want to use custom URLs for training? (y/n): ");
            String custom = scanner.nextLine().trim().toLowerCase();
            if (custom.equals("y")) {
                System.out.println("Enter URLs (one per line, empty line to finish):");
                while (true) {
                    String url = scanner.nextLine().trim();
                    if (url.isEmpty()) break;
                    Trainer.addTrainingUrl(url);
                }
                System.out.println("Training on custom URLs...");
                Trainer.trainWithUrls(model, 20, 0.0005);
            } else {
                String trainingData = WikiData.getSampleData();
                if (trainingData == null) trainingData = "";
                System.out.println("Fetched " + trainingData.length() + " characters from Wikipedia.");
                System.out.println("Training on Wikipedia data...");
                Trainer.train(model, trainingData, 20, 0.0005);
            }
            model.saveWeights("weights.bin");
        } else {
            model.loadWeights("weights.bin");
            System.out.println("Skipping training.");
        }

        System.out.println("\nGenerating:");
        String output = Prompt.prompt(model, "Transformers are", 100);

        System.out.println(output);
    }
}
