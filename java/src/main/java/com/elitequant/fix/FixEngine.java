/**
 * Elite Quant System - Java FIX Engine
 * ======================================
 * Enterprise-grade FIX protocol implementation.
 *
 * Java is used for:
 * - FIX protocol messaging (industry standard)
 * - Enterprise infrastructure
 * - Scalable multi-threaded systems
 * - Cross-platform portability
 *
 * FIX (Financial Information eXchange) is the standard protocol
 * for electronic trading communication.
 */

package com.elitequant.fix;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.*;
import java.util.function.Consumer;
import java.io.*;
import java.net.*;

/**
 * Order side enumeration.
 */
enum Side {
    BUY('1'),
    SELL('2');
    
    private final char fixValue;
    
    Side(char fixValue) {
        this.fixValue = fixValue;
    }
    
    public char getFixValue() {
        return fixValue;
    }
    
    public static Side fromFixValue(char value) {
        return value == '1' ? BUY : SELL;
    }
}

/**
 * Order type enumeration.
 */
enum OrderType {
    MARKET('1'),
    LIMIT('2'),
    STOP('3'),
    STOP_LIMIT('4');
    
    private final char fixValue;
    
    OrderType(char fixValue) {
        this.fixValue = fixValue;
    }
    
    public char getFixValue() {
        return fixValue;
    }
}

/**
 * Execution report status.
 */
enum ExecType {
    NEW('0'),
    PARTIAL_FILL('1'),
    FILL('2'),
    CANCELED('4'),
    REJECTED('8');
    
    private final char fixValue;
    
    ExecType(char fixValue) {
        this.fixValue = fixValue;
    }
    
    public char getFixValue() {
        return fixValue;
    }
}

/**
 * Order representation.
 */
class Order {
    private final String orderId;
    private final String symbol;
    private final Side side;
    private final OrderType orderType;
    private final double quantity;
    private final Double price;  // null for market orders
    private final Instant timestamp;
    
    private double filledQuantity = 0.0;
    private double avgFillPrice = 0.0;
    private ExecType status = ExecType.NEW;
    
    public Order(String orderId, String symbol, Side side, OrderType orderType,
                 double quantity, Double price) {
        this.orderId = orderId;
        this.symbol = symbol;
        this.side = side;
        this.orderType = orderType;
        this.quantity = quantity;
        this.price = price;
        this.timestamp = Instant.now();
    }
    
    // Getters
    public String getOrderId() { return orderId; }
    public String getSymbol() { return symbol; }
    public Side getSide() { return side; }
    public OrderType getOrderType() { return orderType; }
    public double getQuantity() { return quantity; }
    public Double getPrice() { return price; }
    public Instant getTimestamp() { return timestamp; }
    public double getFilledQuantity() { return filledQuantity; }
    public double getAvgFillPrice() { return avgFillPrice; }
    public ExecType getStatus() { return status; }
    
    public void fill(double qty, double price) {
        double totalValue = this.avgFillPrice * this.filledQuantity + price * qty;
        this.filledQuantity += qty;
        this.avgFillPrice = totalValue / this.filledQuantity;
        
        if (this.filledQuantity >= this.quantity) {
            this.status = ExecType.FILL;
        } else {
            this.status = ExecType.PARTIAL_FILL;
        }
    }
    
    public void cancel() {
        this.status = ExecType.CANCELED;
    }
    
    public void reject() {
        this.status = ExecType.REJECTED;
    }
}

/**
 * FIX message builder.
 */
class FixMessageBuilder {
    private static final char SOH = '\u0001';  // FIX field separator
    private final StringBuilder message = new StringBuilder();
    private int checksum = 0;
    
    public FixMessageBuilder addField(int tag, String value) {
        String field = tag + "=" + value + SOH;
        message.append(field);
        for (char c : field.toCharArray()) {
            checksum += c;
        }
        return this;
    }
    
    public FixMessageBuilder addField(int tag, char value) {
        return addField(tag, String.valueOf(value));
    }
    
    public FixMessageBuilder addField(int tag, int value) {
        return addField(tag, String.valueOf(value));
    }
    
    public FixMessageBuilder addField(int tag, double value) {
        return addField(tag, String.format("%.2f", value));
    }
    
    public String build(String msgType, String senderCompId, String targetCompId, int seqNum) {
        StringBuilder header = new StringBuilder();
        
        // Build header
        String bodyAndTrailer = message.toString();
        
        // BeginString (8), BodyLength (9), MsgType (35)
        header.append("8=FIX.4.4").append(SOH);
        
        // Calculate body length
        StringBuilder body = new StringBuilder();
        body.append("35=").append(msgType).append(SOH);
        body.append("49=").append(senderCompId).append(SOH);
        body.append("56=").append(targetCompId).append(SOH);
        body.append("34=").append(seqNum).append(SOH);
        body.append("52=").append(getTimestamp()).append(SOH);
        body.append(bodyAndTrailer);
        
        header.append("9=").append(body.length()).append(SOH);
        
        String fullMsg = header.toString() + body.toString();
        
        // Calculate checksum
        int sum = 0;
        for (char c : fullMsg.toCharArray()) {
            sum += c;
        }
        String checksumStr = String.format("%03d", sum % 256);
        
        return fullMsg + "10=" + checksumStr + SOH;
    }
    
    private String getTimestamp() {
        return LocalDateTime.now().format(
            DateTimeFormatter.ofPattern("yyyyMMdd-HH:mm:ss.SSS")
        );
    }
}

/**
 * FIX message parser.
 */
class FixMessageParser {
    private static final char SOH = '\u0001';
    private final Map<Integer, String> fields = new HashMap<>();
    
    public FixMessageParser(String message) {
        parse(message);
    }
    
    private void parse(String message) {
        String[] parts = message.split(String.valueOf(SOH));
        for (String part : parts) {
            if (part.isEmpty()) continue;
            int equalPos = part.indexOf('=');
            if (equalPos > 0) {
                int tag = Integer.parseInt(part.substring(0, equalPos));
                String value = part.substring(equalPos + 1);
                fields.put(tag, value);
            }
        }
    }
    
    public String getField(int tag) {
        return fields.get(tag);
    }
    
    public int getIntField(int tag) {
        String value = fields.get(tag);
        return value != null ? Integer.parseInt(value) : 0;
    }
    
    public double getDoubleField(int tag) {
        String value = fields.get(tag);
        return value != null ? Double.parseDouble(value) : 0.0;
    }
    
    public char getCharField(int tag) {
        String value = fields.get(tag);
        return value != null && !value.isEmpty() ? value.charAt(0) : '\0';
    }
    
    public String getMsgType() {
        return getField(35);
    }
}

/**
 * FIX session handler.
 */
class FixSession {
    private final String senderCompId;
    private final String targetCompId;
    private final AtomicInteger outgoingSeqNum = new AtomicInteger(1);
    private final AtomicInteger incomingSeqNum = new AtomicInteger(1);
    private final Map<String, Order> orders = new ConcurrentHashMap<>();
    private Consumer<String> messageHandler;
    
    public FixSession(String senderCompId, String targetCompId) {
        this.senderCompId = senderCompId;
        this.targetCompId = targetCompId;
    }
    
    public void setMessageHandler(Consumer<String> handler) {
        this.messageHandler = handler;
    }
    
    /**
     * Send new order single (MsgType = D).
     */
    public String sendNewOrderSingle(Order order) {
        FixMessageBuilder builder = new FixMessageBuilder()
            .addField(11, order.getOrderId())     // ClOrdID
            .addField(55, order.getSymbol())      // Symbol
            .addField(54, order.getSide().getFixValue())  // Side
            .addField(40, order.getOrderType().getFixValue())  // OrdType
            .addField(38, order.getQuantity());   // OrderQty
        
        if (order.getPrice() != null) {
            builder.addField(44, order.getPrice());  // Price
        }
        
        builder.addField(60, Instant.now().toString());  // TransactTime
        
        String message = builder.build("D", senderCompId, targetCompId, 
                                       outgoingSeqNum.getAndIncrement());
        
        orders.put(order.getOrderId(), order);
        
        if (messageHandler != null) {
            messageHandler.accept(message);
        }
        
        return message;
    }
    
    /**
     * Send order cancel request (MsgType = F).
     */
    public String sendCancelRequest(String origOrderId, String symbol, Side side) {
        String cancelId = "CANCEL-" + origOrderId;
        
        FixMessageBuilder builder = new FixMessageBuilder()
            .addField(11, cancelId)               // ClOrdID
            .addField(41, origOrderId)            // OrigClOrdID
            .addField(55, symbol)                 // Symbol
            .addField(54, side.getFixValue())     // Side
            .addField(60, Instant.now().toString());  // TransactTime
        
        String message = builder.build("F", senderCompId, targetCompId,
                                       outgoingSeqNum.getAndIncrement());
        
        if (messageHandler != null) {
            messageHandler.accept(message);
        }
        
        return message;
    }
    
    /**
     * Process incoming execution report (MsgType = 8).
     */
    public void processExecutionReport(FixMessageParser parser) {
        String orderId = parser.getField(11);
        char execType = parser.getCharField(150);
        
        Order order = orders.get(orderId);
        if (order == null) return;
        
        switch (execType) {
            case '1':  // Partial Fill
            case '2':  // Fill
                double lastQty = parser.getDoubleField(32);
                double lastPrice = parser.getDoubleField(31);
                order.fill(lastQty, lastPrice);
                break;
            case '4':  // Canceled
                order.cancel();
                break;
            case '8':  // Rejected
                order.reject();
                break;
        }
    }
    
    /**
     * Send heartbeat (MsgType = 0).
     */
    public String sendHeartbeat() {
        FixMessageBuilder builder = new FixMessageBuilder();
        return builder.build("0", senderCompId, targetCompId,
                            outgoingSeqNum.getAndIncrement());
    }
    
    /**
     * Send logon (MsgType = A).
     */
    public String sendLogon(int heartbeatInterval) {
        FixMessageBuilder builder = new FixMessageBuilder()
            .addField(98, 0)                      // EncryptMethod (none)
            .addField(108, heartbeatInterval);    // HeartBtInt
        
        return builder.build("A", senderCompId, targetCompId,
                            outgoingSeqNum.getAndIncrement());
    }
    
    public Order getOrder(String orderId) {
        return orders.get(orderId);
    }
}

/**
 * Order manager with threading support.
 */
class OrderManager {
    private final FixSession session;
    private final ExecutorService executor;
    private final BlockingQueue<Order> orderQueue;
    private final AtomicLong orderIdCounter = new AtomicLong(1);
    private volatile boolean running = true;
    
    public OrderManager(FixSession session, int threadPoolSize) {
        this.session = session;
        this.executor = Executors.newFixedThreadPool(threadPoolSize);
        this.orderQueue = new LinkedBlockingQueue<>();
        
        // Start order processing thread
        executor.submit(this::processOrders);
    }
    
    public String generateOrderId() {
        return "ORD-" + orderIdCounter.getAndIncrement();
    }
    
    public CompletableFuture<Order> submitOrder(String symbol, Side side, 
                                                 OrderType type, double quantity,
                                                 Double price) {
        return CompletableFuture.supplyAsync(() -> {
            Order order = new Order(generateOrderId(), symbol, side, type, 
                                    quantity, price);
            orderQueue.offer(order);
            return order;
        }, executor);
    }
    
    private void processOrders() {
        while (running) {
            try {
                Order order = orderQueue.poll(100, TimeUnit.MILLISECONDS);
                if (order != null) {
                    session.sendNewOrderSingle(order);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }
    
    public void shutdown() {
        running = false;
        executor.shutdown();
        try {
            executor.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }
}

/**
 * Main FIX Engine class.
 */
public class FixEngine {
    private final FixSession session;
    private final OrderManager orderManager;
    
    public FixEngine(String senderCompId, String targetCompId) {
        this.session = new FixSession(senderCompId, targetCompId);
        this.orderManager = new OrderManager(session, 4);
        
        // Set up message logging
        session.setMessageHandler(msg -> {
            System.out.println("[FIX OUT] " + msg.replace('\u0001', '|'));
        });
    }
    
    public CompletableFuture<Order> submitMarketOrder(String symbol, Side side, 
                                                       double quantity) {
        return orderManager.submitOrder(symbol, side, OrderType.MARKET, 
                                        quantity, null);
    }
    
    public CompletableFuture<Order> submitLimitOrder(String symbol, Side side,
                                                      double quantity, double price) {
        return orderManager.submitOrder(symbol, side, OrderType.LIMIT, 
                                        quantity, price);
    }
    
    public void cancelOrder(String orderId) {
        Order order = session.getOrder(orderId);
        if (order != null) {
            session.sendCancelRequest(orderId, order.getSymbol(), order.getSide());
        }
    }
    
    public void shutdown() {
        orderManager.shutdown();
    }
    
    /**
     * Main entry point for testing.
     */
    public static void main(String[] args) {
        System.out.println("Elite Quant System - Java FIX Engine");
        System.out.println("=====================================");
        
        // Create FIX engine
        FixEngine engine = new FixEngine("ELITE_QUANT", "EXCHANGE");
        
        // Submit some test orders
        System.out.println("\nSubmitting test orders...\n");
        
        engine.submitMarketOrder("AAPL", Side.BUY, 100.0)
              .thenAccept(order -> {
                  System.out.println("Market order submitted: " + order.getOrderId());
              });
        
        engine.submitLimitOrder("MSFT", Side.SELL, 50.0, 450.00)
              .thenAccept(order -> {
                  System.out.println("Limit order submitted: " + order.getOrderId());
              });
        
        // Wait for orders to process
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        // Shutdown
        engine.shutdown();
        System.out.println("\n✓ FIX Engine test complete");
    }
}

