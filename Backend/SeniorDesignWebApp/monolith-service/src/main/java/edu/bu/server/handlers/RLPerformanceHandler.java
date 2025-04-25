package edu.bu.server.handlers;
import com.google.api.core.ApiFuture;
import com.google.auth.oauth2.ServiceAccountCredentials;
import com.google.cloud.Timestamp;
import com.google.cloud.firestore.*;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.tinylog.Logger;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.DayOfWeek;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class RLPerformanceHandler implements HttpHandler{
    private final Firestore database;
    private final String ALPHA_VANTAGE_API_KEY = "NL9PDOM5JWRPAT9O";

    public RLPerformanceHandler() {
        Firestore tempDb = null;
        try {
            FileInputStream serviceAccount =
                    new FileInputStream("C:/Users/tenni/documents/SeniorDesign/serviceAccount.json");

            FirestoreOptions firestoreOptions =
                    FirestoreOptions.newBuilder()
                            .setProjectId("seniordesign-35b8b")
                            .setCredentials(ServiceAccountCredentials.fromStream(serviceAccount))
                            .build();

            tempDb = firestoreOptions.getService();
        } catch (IOException e) {
            Logger.error(e, "Error initializing Firestore with service account credentials.");
        }
        database = tempDb;
    }

    @Override
    public void handle(HttpExchange exchange) throws IOException {
        if (!exchange.getRequestMethod().equalsIgnoreCase("GET")) {
            exchange.sendResponseHeaders(405, -1); // 405 Method Not Allowed
            return;
        }

        JSONArray rlPerformanceArray = new JSONArray();

        try {
            CollectionReference signalsRef = database.collection("StockSignals");
            ApiFuture<QuerySnapshot> tickersFuture = signalsRef.get();
            List<QueryDocumentSnapshot> tickerDocs = tickersFuture.get().getDocuments();

            for (DocumentSnapshot tickerDoc : tickerDocs) {
                String tickerSymbol = tickerDoc.getId();

                CollectionReference predictionsRef = signalsRef.document(tickerSymbol).collection("predictions");
                ApiFuture<QuerySnapshot> predictionsFuture = predictionsRef.get();

                // Sort predictions by next_trading_date
                List<DocumentSnapshot> predictionDocs = new ArrayList<>(predictionsFuture.get().getDocuments());

                predictionDocs.sort((d1, d2) -> {
                    String date1 = (String) d1.get("next_trading_date");
                    String date2 = (String) d2.get("next_trading_date");
                    return date1.compareTo(date2);
                });

                double portfolioValue = 10000.0;
                double sharesHeld = 0.0;
                boolean initialized = false;

                for (DocumentSnapshot predictionDoc : predictionDocs) {
                    Map<String, Object> data = predictionDoc.getData();
                    if (data == null || !data.containsKey("rl_recommendation") || !data.containsKey("next_trading_date"))
                        continue;
                    try {
                        String recommendation = data.get("rl_recommendation").toString().toLowerCase().trim();
                        String nextTradingDate = data.get("next_trading_date").toString();

                        String tradeDate = getPreviousTradingDate(nextTradingDate);  // Buy/Sell happens one day *before*
                        Logger.info("Processing {} on trade date {} (next_trading_date was {})", tickerSymbol, tradeDate, nextTradingDate);

                        double closePrice = fetchClosePrice(tickerSymbol, tradeDate);
                        if (closePrice == -1) continue;

                        if (!initialized) {
                            portfolioValue = 10000.0;
                            sharesHeld = 0.0;
                            initialized = true;
                        }

                        if (recommendation.equals("buy") && sharesHeld == 0) {
                            sharesHeld = portfolioValue / closePrice;
                            portfolioValue = 0.0;
                        } else if (recommendation.equals("sell") && sharesHeld > 0) {
                            portfolioValue = sharesHeld * closePrice;
                            sharesHeld = 0.0;
                        }

                        // Always include unrealized value
                        double totalValue = portfolioValue + (sharesHeld * closePrice);
                        double cumulativePL = ((totalValue - 10000.0) / 10000.0) * 100;

                        JSONObject dailyJson = new JSONObject();
                        dailyJson.put("ticker", tickerSymbol);
                        dailyJson.put("date", nextTradingDate);  // Output remains under the original next trading date
                        dailyJson.put("cumulative_pl", cumulativePL);
                        rlPerformanceArray.add(dailyJson);

                    } catch (Exception innerEx) {
                        Logger.error(innerEx, "Failed while processing prediction for {}", tickerSymbol);
                    }
                }
            }

            JSONObject responseJson = new JSONObject();
            responseJson.put("rl_performance", rlPerformanceArray);
            String response = responseJson.toJSONString();

            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(200, response.getBytes().length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(response.getBytes());
            }

        } catch (Exception e) {
            Logger.error(e, "Error fetching RL performance data.");
            JSONObject errorJson = new JSONObject();
            errorJson.put("error", "Error fetching RL data: " + e.getMessage());
            String response = errorJson.toJSONString();
            exchange.getResponseHeaders().add("Content-Type", "application/json");
            exchange.sendResponseHeaders(500, response.getBytes().length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(response.getBytes());
            }
        }
    }


    double fetchClosePrice(String ticker, String targetDate) {
        try {
            String apiUrl = String.format(
                    "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&apikey=%s",
                    ticker, ALPHA_VANTAGE_API_KEY);

            URL url = new URL(apiUrl);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.connect();

            Scanner scanner = new Scanner(url.openStream(), StandardCharsets.UTF_8);
            String inline = scanner.useDelimiter("\\A").next();
            scanner.close();

            org.json.simple.parser.JSONParser parser = new org.json.simple.parser.JSONParser();
            JSONObject json = (JSONObject) parser.parse(inline);

            if (!json.containsKey("Time Series (Daily)")) {
                Logger.warn("Alpha Vantage: no daily time series for {}", ticker);
                return -1;
            }

            JSONObject dailyPrices = (JSONObject) json.get("Time Series (Daily)");

            // ✅ Only allow exact match
            if (dailyPrices.containsKey(targetDate)) {
                JSONObject dayData = (JSONObject) dailyPrices.get(targetDate);
                return Double.parseDouble((String) dayData.get("4. close"));
            }

            Logger.warn("No close price found for {} on {} — skipping", ticker, targetDate);
            return -1;

        } catch (Exception e) {
            Logger.error(e, "Error fetching close price for {} on {}", ticker, targetDate);
            return -1;
        }
    }
    private String getPreviousTradingDate(String date) {
        LocalDate currentDate = LocalDate.parse(date);
        do {
            currentDate = currentDate.minusDays(1);
        } while (currentDate.getDayOfWeek() == DayOfWeek.SATURDAY || currentDate.getDayOfWeek() == DayOfWeek.SUNDAY);
        return currentDate.toString();
    }
}
