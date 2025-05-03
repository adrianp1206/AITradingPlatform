package edu.bu.server.handlers;

import com.google.api.core.ApiFuture;
import com.google.auth.oauth2.ServiceAccountCredentials;
import com.google.cloud.Timestamp;
import com.google.cloud.firestore.*;
import com.google.cloud.firestore.FirestoreOptions;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.tinylog.Logger;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.util.List;
import java.util.Map;

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
    if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
      exchange.sendResponseHeaders(405, -1);
      return;
    }

    try {

      ZonedDateTime now = ZonedDateTime.now(ZoneId.systemDefault());
      LocalDate baseDate;
      if (now.toLocalTime().isBefore(LocalTime.of(16, 0))) {

        baseDate = now.minusDays(1).toLocalDate();
      } else {

        baseDate = now.toLocalDate();
      }

      String nextTradeDate = getNextTradingDate(baseDate);

      JSONArray predictionsArray = new JSONArray();
      CollectionReference signalsRef = database.collection("StockSignals");
      ApiFuture<QuerySnapshot> tickersFuture = signalsRef.get();
      List<QueryDocumentSnapshot> tickerDocs = tickersFuture.get().getDocuments();

      for (DocumentSnapshot tickerDoc : tickerDocs) {
        String tickerSymbol = tickerDoc.getId();
        CollectionReference predsRef =
                signalsRef.document(tickerSymbol).collection("predictions");

        ApiFuture<QuerySnapshot> predsFuture =
                predsRef
                        .whereEqualTo("next_trading_date", nextTradeDate)
                        .get();
        List<QueryDocumentSnapshot> docs = predsFuture.get().getDocuments();

        for (DocumentSnapshot doc : docs) {
          Map<String,Object> data = doc.getData();
          if (data == null) continue;

          JSONObject json = new JSONObject();
          json.put("ticker",        tickerSymbol);
          json.put("prediction_id", doc.getId());

          for (Map.Entry<String,Object> e : data.entrySet()) {
            Object v = e.getValue();
            if (v instanceof Timestamp) {
              json.put(e.getKey(),
                      ((Timestamp)v).toDate()
                              .toInstant()
                              .toString());
            } else {
              json.put(e.getKey(), v);
            }
          }
          predictionsArray.add(json);
        }
      }

      JSONObject resp = new JSONObject();
      resp.put("predictions", predictionsArray);
      String body = resp.toJSONString();
      Logger.info("Fetched {} predictions (for next_trading_date={})",
              predictionsArray.size(), nextTradeDate);

      exchange.getResponseHeaders().add("Content-Type","application/json");
      exchange.sendResponseHeaders(200, body.getBytes().length);
      try (OutputStream os = exchange.getResponseBody()) {
        os.write(body.getBytes());
      }

    } catch (Exception e) {
      Logger.error(e, "Error fetching predictions from Firestore.");
      JSONObject err = new JSONObject();
      err.put("error", "Error fetching predictions: " + e.getMessage());
      String body = err.toJSONString();

      exchange.getResponseHeaders().add("Content-Type","application/json");
      exchange.sendResponseHeaders(500, body.getBytes().length);
      try (OutputStream os = exchange.getResponseBody()) {
        os.write(body.getBytes());
      }
    }
  }

  private String getNextTradingDate(LocalDate date) {
    LocalDate next = date.plusDays(1);
    while (next.getDayOfWeek() == DayOfWeek.SATURDAY
            || next.getDayOfWeek() == DayOfWeek.SUNDAY) {
      next = next.plusDays(1);
    }
    return next.toString();  // "YYYY-MM-DD"
  }
}
