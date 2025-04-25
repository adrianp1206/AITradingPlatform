package edu.bu.server.handlers;

public class Test {
    public static void main(String[] args) {
        RLPerformanceHandler handler = new RLPerformanceHandler();
        double price = handler.fetchClosePrice("TSLA", "2025-04-17");
        System.out.println("Close price: " + price);
    }
}
