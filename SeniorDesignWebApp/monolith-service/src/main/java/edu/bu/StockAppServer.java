package edu.bu;

import edu.bu.server.BasicWebServer;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;
import org.tinylog.Logger;

public class StockAppServer {
  static final String MOCK_FINHUB_ARGUMENT = "mockFinhub";
  static final String QUEUE_SERVICE_URL = "http://localhost:8010/dequeue";
  static final String WEBHOOK_URI = "wss://ws.finnhub.io";
  static final String API_TOKEN = "cq1vjm1r01ql95nces30cq1vjm1r01ql95nces3g";

  // StockAppServer
  public static void main(String[] args) throws IOException, URISyntaxException {
    Logger.info("Starting StockAppServer with arguments: {}", List.of(args));

    // start web server
    BasicWebServer webServer = new BasicWebServer();
    webServer.start();

  }
}
