import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class WikiData {
    public static String getSampleData() {
        return fetchFromWikipedia("Transformer_(deep_learning_architecture)");
    }

    public static String fetchFromWikipedia(String title) {
        try {
            HttpClient client = HttpClient.newBuilder()
                    .followRedirects(HttpClient.Redirect.NORMAL)
                    .build();
            String url = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&format=json&redirects=1&titles=" + title;
            
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .header("User-Agent", "TinyTransformerBot/1.0")
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            String body = response.body();
            // System.out.println("DEBUG: Wikipedia Response Body: " + body);

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
                        .replaceAll("[^\\x00-\\x7F]", "");
                
                if (cleaned.isEmpty()) {
                    return "Wikipedia article is empty or contains no ASCII text.";
                }
                return cleaned;
            }
            
            if (body.contains("\"missing\":\"\"") || body.contains("\"missing\":true")) {
                return "Wikipedia page not found.";
            }

            return "Failed to parse Wikipedia response. Body: " + (body.length() > 100 ? body.substring(0, 100) : body);
        } catch (Exception e) {
            e.printStackTrace();
            return "Error fetching from Wikipedia: " + e.getMessage();
        }
    }
}
