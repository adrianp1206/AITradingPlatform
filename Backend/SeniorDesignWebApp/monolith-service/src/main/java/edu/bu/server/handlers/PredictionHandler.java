package edu.bu.server.handlers;

import com.google.api.core.ApiFuture;
import com.google.auth.oauth2.ServiceAccountCredentials;
import com.google.cloud.Timestamp;
import com.google.cloud.firestore.*;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.List;
import java.util.Map;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.tinylog.Logger;

public class PredictionHandler implements HttpHandler {
  private final Firestore database;

  public PredictionHandler() {
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

    try {

      ZonedDateTime now = ZonedDateTime.now(ZoneId.systemDefault());
      ZonedDateTime yesterdayFivePM = now.minusDays(1).withHour(17).withMinute(0).withSecond(0).withNano(0);
      ZonedDateTime yesterdayEnd = now.minusDays(1).withHour(23).withMinute(59).withSecond(59).withNano(999999999);

      Timestamp startTimestamp = Timestamp.ofTimeSecondsAndNanos(
              yesterdayFivePM.toInstant().getEpochSecond(), yesterdayFivePM.toInstant().getNano());
      Timestamp endTimestamp = Timestamp.ofTimeSecondsAndNanos(
              yesterdayEnd.toInstant().getEpochSecond(), yesterdayEnd.toInstant().getNano());

      JSONArray predictionsArray = new JSONArray();


      CollectionReference signalsRef = database.collection("StockSignals");
      ApiFuture<QuerySnapshot> tickersFuture = signalsRef.get();
      List<QueryDocumentSnapshot> tickerDocs = tickersFuture.get().getDocuments();

      for (DocumentSnapshot tickerDoc : tickerDocs) {
        String tickerSymbol = tickerDoc.getId();

        CollectionReference predictionsRef = signalsRef.document(tickerSymbol).collection("predictions");

        Query predictionsQuery = predictionsRef
                .whereGreaterThanOrEqualTo("updated_at", startTimestamp)
                .whereLessThanOrEqualTo("updated_at", endTimestamp);

        ApiFuture<QuerySnapshot> predictionsFuture = predictionsQuery.get();
        List<QueryDocumentSnapshot> predictionDocs = predictionsFuture.get().getDocuments();

        for (DocumentSnapshot predictionDoc : predictionDocs) {
          Map<String, Object> data = predictionDoc.getData();
          JSONObject predictionJson = new JSONObject();

          predictionJson.put("ticker", tickerSymbol);
          predictionJson.put("prediction_id", predictionDoc.getId());

          if (data != null) {
            for (Map.Entry<String, Object> entry : data.entrySet()) {
              if (entry.getValue() instanceof Timestamp) {
                predictionJson.put(entry.getKey(), ((Timestamp) entry.getValue()).toDate().toInstant().toString());
              } else {
                predictionJson.put(entry.getKey(), entry.getValue());
              }
            }
          }
          predictionsArray.add(predictionJson);
        }
      }

      JSONObject responseJson = new JSONObject();
      responseJson.put("predictions", predictionsArray);
      String response = responseJson.toJSONString();

      Logger.info("Fetched {} prediction documents from Firestore.", predictionsArray.size());

      exchange.getResponseHeaders().add("Content-Type", "application/json");
      exchange.sendResponseHeaders(200, response.getBytes().length);

      try (OutputStream os = exchange.getResponseBody()) {
        os.write(response.getBytes());
      }
    } catch (Exception e) {
      Logger.error(e, "Error fetching predictions from Firestore.");
      JSONObject errorJson = new JSONObject();
      errorJson.put("error", "Error fetching predictions: " + e.getMessage());
      String response = errorJson.toJSONString();
      exchange.getResponseHeaders().add("Content-Type", "application/json");
      exchange.sendResponseHeaders(500, response.getBytes().length);

      try (OutputStream os = exchange.getResponseBody()) {
        os.write(response.getBytes());
      }
    }
  }
}
