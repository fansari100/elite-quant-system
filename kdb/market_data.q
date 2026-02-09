/ Elite Quant System - KDB+/Q Market Data Layer
/ ================================================
/ High-performance time series database for tick data
/
/ KDB+ is the industry standard for quant finance:
/ - Used by: Citadel, Two Sigma, Jane Street, DE Shaw, Goldman, JPM
/ - Handles billions of rows with sub-millisecond queries
/ - Columnar storage optimized for time series
/
/ To run: q market_data.q
/ To load: \l market_data.q

/ ============================================================================
/ SCHEMA DEFINITIONS
/ ============================================================================

/ Trade tick schema
trade:([]
    time:`timestamp$();       / Nanosecond timestamp
    sym:`symbol$();           / Symbol
    price:`float$();          / Trade price
    size:`long$();            / Trade size
    exchange:`symbol$();      / Exchange code
    side:`symbol$();          / Buy/Sell indicator
    conditions:`symbol$()     / Trade conditions
);

/ Quote tick schema (Level 1)
quote:([]
    time:`timestamp$();
    sym:`symbol$();
    bid:`float$();            / Best bid price
    ask:`float$();            / Best ask price
    bidsize:`long$();         / Bid size
    asksize:`long$();         / Ask size
    exchange:`symbol$()
);

/ Level 2 order book
l2book:([]
    time:`timestamp$();
    sym:`symbol$();
    level:`int$();            / Price level (0 = best)
    side:`symbol$();          / bid/ask
    price:`float$();
    size:`long$();
    numorders:`int$()         / Number of orders at this level
);

/ Daily OHLCV bars
daily:([]
    date:`date$();
    sym:`symbol$();
    open:`float$();
    high:`float$();
    low:`float$();
    close:`float$();
    volume:`long$();
    vwap:`float$();           / Volume-weighted average price
    trades:`long$()           / Number of trades
);

/ Intraday bars (1-minute)
bar1m:([]
    time:`timestamp$();
    sym:`symbol$();
    open:`float$();
    high:`float$();
    low:`float$();
    close:`float$();
    volume:`long$();
    vwap:`float$();
    trades:`long$()
);

/ Portfolio positions
position:([]
    time:`timestamp$();
    account:`symbol$();
    sym:`symbol$();
    quantity:`float$();
    avgPrice:`float$();
    marketValue:`float$();
    unrealizedPnl:`float$();
    realizedPnl:`float$()
);

/ Orders
orders:([]
    orderId:`long$();
    time:`timestamp$();
    sym:`symbol$();
    side:`symbol$();
    orderType:`symbol$();
    price:`float$();
    quantity:`float$();
    filledQty:`float$();
    status:`symbol$();
    account:`symbol$()
);

/ ============================================================================
/ UTILITY FUNCTIONS
/ ============================================================================

/ Get current timestamp in nanoseconds
now:{.z.p}

/ Convert Python timestamps to q timestamps
fromPython:{`timestamp$x*1e9}

/ Convert q timestamps to Python (nanoseconds since epoch)
toPython:{`long$x}

/ Safe division (avoid divide by zero)
sdiv:{$[y=0;0f;x%y]}

/ ============================================================================
/ DATA INGESTION
/ ============================================================================

/ Insert trade tick
insertTrade:{[sym_;price_;size_;exchange_;side_]
    `trade insert (now[];sym_;price_;size_;exchange_;side_;`)
};

/ Insert quote tick
insertQuote:{[sym_;bid_;ask_;bidsize_;asksize_;exchange_]
    `quote insert (now[];sym_;bid_;ask_;bidsize_;asksize_;exchange_)
};

/ Bulk insert trades (from Python)
bulkInsertTrades:{[data]
    `trade insert data
};

/ ============================================================================
/ BAR AGGREGATION
/ ============================================================================

/ Aggregate trades to 1-minute bars
aggTo1mBars:{[sym_]
    select 
        open:first price,
        high:max price,
        low:min price,
        close:last price,
        volume:sum size,
        vwap:size wavg price,
        trades:count i
    by time:1 xbar `minute$time, sym 
    from trade 
    where sym=sym_
};

/ Aggregate to daily bars
aggToDailyBars:{[sym_]
    select 
        open:first price,
        high:max price,
        low:min price,
        close:last price,
        volume:sum size,
        vwap:size wavg price,
        trades:count i
    by date:`date$time, sym 
    from trade 
    where sym=sym_
};

/ Real-time bar aggregation (update on new tick)
updateBar:{[t;s;p;sz]
    / t=table, s=sym, p=price, sz=size
    barTime:1 xbar `minute$.z.p;
    existing:select from bar1m where time=barTime, sym=s;
    $[count existing;
        [/ Update existing bar
         update high:max(high;p), low:min(low;p), close:p, 
                volume:volume+sz, trades:trades+1 
         from `bar1m where time=barTime, sym=s];
        [/ Insert new bar
         `bar1m insert (barTime;s;p;p;p;p;sz;p;1)]]
};

/ ============================================================================
/ TECHNICAL INDICATORS
/ ============================================================================

/ Simple Moving Average
sma:{[n;x] mavg[n;x]}

/ Exponential Moving Average
ema:{[n;x] ema[2%1+n;x]}

/ Relative Strength Index
rsi:{[n;x]
    d:deltas x;
    gains:0|d;
    losses:0|-d;
    rs:sdiv[mavg[n;gains];mavg[n;losses]];
    100-100%1+rs
};

/ Bollinger Bands
bollinger:{[n;k;x]
    m:mavg[n;x];
    s:mdev[n;x];
    (m-k*s;m;m+k*s)  / Returns (lower, middle, upper)
};

/ MACD
macd:{[fast;slow;signal;x]
    macdLine:ema[fast;x]-ema[slow;x];
    signalLine:ema[signal;macdLine];
    histogram:macdLine-signalLine;
    (macdLine;signalLine;histogram)
};

/ Average True Range
atr:{[n;h;l;c]
    prev:prev c;
    tr:max each (h-l;abs h-prev;abs l-prev);
    mavg[n;tr]
};

/ Volume-Weighted Average Price
calcVwap:{[t;s]
    select vwap:size wavg price by sym from t where sym=s
};

/ ============================================================================
/ ALPHA FACTORS
/ ============================================================================

/ Momentum factor
momentum:{[n;x] x%n xprev x - 1}

/ Volatility factor (realized)
realized_vol:{[n;x] 
    sqrt 252 * mvar[n;log x%prev x]
};

/ Price reversal factor
reversal:{[n;x]
    -1 * momentum[n;x]
};

/ Volume surge factor
volume_surge:{[n;x]
    x % mavg[n;x]
};

/ Price-volume correlation
pv_corr:{[n;p;v]
    mcorr[n;p;v]
};

/ Combined alpha signal
alphaSignal:{[prices;volumes]
    mom5:momentum[5;prices];
    mom20:momentum[20;prices];
    vol:realized_vol[20;prices];
    vsurge:volume_surge[20;volumes];
    
    / Z-score normalize
    zscore:{(x-avg x)%dev x};
    
    / Combine factors
    0.4*zscore[mom5] + 0.3*zscore[mom20] - 0.2*zscore[vol] + 0.1*zscore[vsurge]
};

/ ============================================================================
/ LEAD-LAG ANALYSIS
/ ============================================================================

/ Cross-correlation at different lags
xcorr:{[n;lag;x;y]
    xlag:lag xprev x;
    mcorr[n;xlag;y]
};

/ Find optimal lag (maximum correlation)
findLeadLag:{[n;maxLag;x;y]
    lags:neg[maxLag]+til 2*maxLag+1;
    corrs:xcorr[n;;x;y] each lags;
    / Return lag with highest absolute correlation
    lags[corrs?max abs corrs]
};

/ Lead-lag matrix for multiple symbols
leadLagMatrix:{[n;maxLag;rets]
    syms:cols rets;
    pairs:syms cross syms;
    leadLags:{[n;maxLag;rets;pair]
        findLeadLag[n;maxLag;rets pair 0;rets pair 1]
    }[n;maxLag;rets;] each pairs;
    
    / Reshape to matrix
    (count syms) cut leadLags
};

/ ============================================================================
/ PORTFOLIO ANALYTICS
/ ============================================================================

/ Portfolio return
portfolioReturn:{[weights;returns]
    sum weights * returns
};

/ Portfolio volatility
portfolioVol:{[weights;covMatrix]
    sqrt sum sum weights */ covMatrix */ weights
};

/ Sharpe ratio (annualized)
sharpe:{[returns;rf]
    excessRet:avg[returns] - rf%252;
    (sqrt 252) * excessRet % dev returns
};

/ Maximum drawdown
maxDrawdown:{[equity]
    dd:1 - equity % maxs equity;
    max dd
};

/ Calmar ratio
calmar:{[returns]
    eq:prds 1 + returns;
    annRet:(last eq) xexp (252 % count returns) - 1;
    annRet % maxDrawdown eq
};

/ Information ratio
infoRatio:{[strategy;benchmark]
    active:strategy - benchmark;
    (avg active) % dev active
};

/ ============================================================================
/ RISK MANAGEMENT
/ ============================================================================

/ Value at Risk (historical)
varHist:{[conf;returns]
    sorted:asc returns;
    idx:`long$(1-conf)*count sorted;
    sorted idx
};

/ Conditional VaR (Expected Shortfall)
cvarHist:{[conf;returns]
    threshold:varHist[conf;returns];
    avg returns where returns <= threshold
};

/ Rolling VaR
rollingVar:{[n;conf;returns]
    {varHist[y;x]}[;conf] each n mwindow returns
};

/ Position sizing by volatility targeting
volTargetSize:{[targetVol;currentVol;currentPos]
    currentPos * targetVol % currentVol
};

/ ============================================================================
/ DATA PERSISTENCE
/ ============================================================================

/ Save table to disk (splayed)
saveSplayed:{[path;tableName;data]
    (hsym `$path,"/",string[tableName],"/") set .Q.en[hsym `$path] data
};

/ Load splayed table
loadSplayed:{[path;tableName]
    get hsym `$path,"/",string tableName
};

/ Save partitioned by date
savePartitioned:{[path;tableName;data]
    dates:distinct `date$data`time;
    {[path;tableName;data;d]
        dayData:select from data where `date$time=d;
        savePath:`$path,"/",string[d],"/",string tableName;
        (hsym savePath) set dayData
    }[path;tableName;data;] each dates
};

/ ============================================================================
/ IPC - Inter-Process Communication
/ ============================================================================

/ Handler for incoming Python requests
.z.pg:{[x]
    / x is the incoming query
    value x
};

/ Handler for async messages
.z.ps:{[x]
    value x
};

/ Log connections
.z.po:{[x]
    0N!"Client connected: ",string x
};

/ Log disconnections
.z.pc:{[x]
    0N!"Client disconnected: ",string x
};

/ ============================================================================
/ PYTHON INTEGRATION API
/ ============================================================================

/ Get latest quotes for symbols
getQuotes:{[syms]
    select last time, last bid, last ask, last bidsize, last asksize 
    by sym from quote where sym in syms
};

/ Get historical bars
getBars:{[sym_;startDate;endDate;interval]
    $[interval=`daily;
        select from daily where sym=sym_, date>=startDate, date<=endDate;
      interval=`1m;
        select from bar1m where sym=sym_, `date$time>=startDate, `date$time<=endDate;
      / Default to daily
        select from daily where sym=sym_, date>=startDate, date<=endDate]
};

/ Get features for ML model
getFeatures:{[sym_;n]
    data:select from daily where sym=sym_;
    data:n sublist data;  / Last n rows
    
    / Compute features
    data:update 
        returns:close%prev[close]-1,
        log_returns:log close%prev close,
        sma_5:mavg[5;close],
        sma_20:mavg[20;close],
        rsi_14:rsi[14;close],
        vol_20:realized_vol[20;close],
        mom_5:momentum[5;close],
        mom_20:momentum[20;close]
    from data;
    
    / Return as dictionary for Python
    flip data
};

/ Execute trade (from Python)
executeTrade:{[sym_;side_;qty;price]
    orderId:1 + exec max orderId from orders;
    `orders insert (orderId;now[];sym_;side_;`limit;price;qty;0f;`new;`main);
    orderId
};

/ Get positions
getPositions:{[account_]
    select from position where account=account_, time=max time
};

/ ============================================================================
/ STARTUP
/ ============================================================================

/ Initialize sample data (for testing)
initSampleData:{
    / Sample symbols
    syms:`AAPL`MSFT`GOOGL`AMZN`NVDA;
    
    / Generate sample daily data
    n:252;  / 1 year
    dates:2024.01.01+til n;
    
    {[syms;dates;s]
        base:100 + 50 * syms?s;
        returns:1 + 0.02 * (n?1.0) - 0.5;
        prices:base * prds returns;
        
        `daily insert flip `date`sym`open`high`low`close`volume`vwap`trades!
            (dates;
             n#s;
             prices * 1 - 0.005 * n?1.0;  / open
             prices * 1 + 0.01 * n?1.0;   / high
             prices * 1 - 0.01 * n?1.0;   / low
             prices;                       / close
             1000000 + `long$10000000 * n?1.0;  / volume
             prices;                       / vwap
             1000 + `long$10000 * n?1.0)   / trades
    }[syms;dates;] each syms;
    
    0N!"Initialized sample data: ",string[count daily]," rows"
};

/ Main entry point
main:{
    0N!"Elite Quant System - KDB+ Market Data Layer";
    0N!"==========================================";
    
    / Initialize sample data
    initSampleData[];
    
    / Show table counts
    0N!"Tables initialized:";
    0N!"  trade:    ",string count trade;
    0N!"  quote:    ",string count quote;
    0N!"  daily:    ",string count daily;
    0N!"  bar1m:    ",string count bar1m;
    0N!"  position: ",string count position;
    0N!"  orders:   ",string count orders;
    
    0N!"";
    0N!"Listening on port 5001...";
    
    / Start listening for IPC
    system "p 5001"
};

/ Auto-run if executed directly
if[not system "e";main[]]

