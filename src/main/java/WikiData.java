import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class WikiData {
    private static final String STORAGE_PATH = "training_data.txt";

    public static String getSampleData() {
        String data = loadFromStorage();
        if (data != null) {
            System.out.println("Loaded training data from storage.");
            return data;
        }

        data = fetchFromWikipedia("Transformer_(deep_learning_architecture)");
        if (data != null && !data.startsWith("Error") && !data.startsWith("Wikipedia")) {
            saveToStorage(data);
        }
        return data;
    }

    private static void saveToStorage(String data) {
        try {
            Files.writeString(Paths.get(STORAGE_PATH), data);
            System.out.println("Saved training data to storage.");
        } catch (IOException e) {
            System.err.println("Failed to save data to storage: " + e.getMessage());
        }
    }

    private static String loadFromStorage() {
        Path path = Paths.get(STORAGE_PATH);
        if (Files.exists(path)) {
            try {
                return Files.readString(path);
            } catch (IOException e) {
                System.err.println("Failed to load data from storage: " + e.getMessage());
            }
        }
        return null;
    }

    public static String fetchFromUrl(String urlString) {
        try {
            HttpClient client = HttpClient.newBuilder()
                    .followRedirects(HttpClient.Redirect.NORMAL)
                    .build();

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(urlString))
                    .header("User-Agent", "Echo/1.0")
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            return response.body();
        } catch (Exception e) {
            e.printStackTrace();
            return "Error fetching from URL: " + e.getMessage();
        }
    }

    public static String fetchFromWikipedia(String title) {
        String url = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&format=json&redirects=1&titles=" + title;
        String body = fetchFromUrl(url);
        
        if (body.startsWith("Error")) return body;
        
        System.out.println("DEBUG: Wikipedia Response Body: " + (body.length() > 100 ? body.substring(0, 100) + "..." : body));

        // Simple regex to extract the "extract" field from Wikipedia API JSON
            // Using DOTALL to match across lines and a non-greedy match.
            // Note: Wikipedia API usually escapes double quotes as \", so (.*?) might over-consume if not careful,
            // but for a single "extract" field in this specific API call it should be fine.
            Pattern pattern = Pattern.compile("\"extract\":\"(.*?)\"(?:,|})", Pattern.DOTALL);
            Matcher matcher = pattern.matcher(body);
            if (matcher.find()) {
                String extract = matcher.group(1);
                // Basic unescape for JSON characters and remove non-ASCII
                String cleaned = extract.replace("\\n", " ")
                        .replace("\\\"", "\"")
                        .replace("\\\\", "\\")
                        .replaceAll("[^\\x20-\\x7E]", " ");
                
                if (cleaned.isEmpty()) {
                    return "Wikipedia article is empty or contains no ASCII text.";
                }
                return cleaned;
            }
            
            if (body.contains("\"missing\":\"\"") || body.contains("\"missing\":true")) {
                return "Wikipedia page not found.";
            }

            return "Failed to parse Wikipedia response. Body: " + (body.length() > 100 ? body.substring(0, 100) : body);
    }
}
