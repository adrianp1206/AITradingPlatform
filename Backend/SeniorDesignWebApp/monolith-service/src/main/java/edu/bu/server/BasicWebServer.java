package edu.bu.server;

import com.sun.net.httpserver.HttpServer;
import edu.bu.server.handlers.*;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.HashMap;
import java.util.Map;
import org.tinylog.Logger;

public class BasicWebServer {
    public BasicWebServer(){

    }
    public void start() throws IOException {
        // Create an HttpServer instance
        HttpServer server = HttpServer.create(new InetSocketAddress(8000), 0);

        server.createContext("/predictions", new CORSHandler(new PredictionHandler()));

        server.createContext("/rl-performance", new CORSHandler(new RLPerformanceHandler()));
        // Start the server
        server.setExecutor(null); // Use the default executor
        server.start();

        Logger.info("Server is running on port 8000");
    }
}
