package edu.bu.server.handlers;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import java.io.IOException;

public class CORSHandler implements HttpHandler {
  final HttpHandler handler;

  public CORSHandler(HttpHandler handler) {
    this.handler = handler;
  }

  @Override
  public void handle(HttpExchange exchange) throws IOException {
    exchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*");

    if (exchange.getRequestMethod().equalsIgnoreCase("OPTIONS")) {
      exchange
          .getResponseHeaders()
          .add("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
      exchange
          .getResponseHeaders()
          .add("Access-Control-Allow-Headers", "Content-Type,Authorization");
      exchange.sendResponseHeaders(204, -1);
      return;
    }

    handler.handle(exchange);
  }
}
