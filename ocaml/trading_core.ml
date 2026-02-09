(** 
   Elite Quant System - OCaml Core
   ================================
   Functional programming for trading systems.
   
   OCaml is Jane Street's primary language, providing:
   - Strong static typing catches errors at compile time
   - Functional paradigm for correct, maintainable code
   - Performance close to C
   - Pattern matching for complex logic
   
   Jane Street runs ~$17 billion in daily trading on OCaml.
*)

open Core

(** Side of an order *)
module Side = struct
  type t = Buy | Sell [@@deriving sexp, compare, equal]
  
  let to_string = function
    | Buy -> "BUY"
    | Sell -> "SELL"
  
  let of_string = function
    | "BUY" | "buy" | "B" -> Some Buy
    | "SELL" | "sell" | "S" -> Some Sell
    | _ -> None
  
  let opposite = function
    | Buy -> Sell
    | Sell -> Buy
end

(** Order type *)
module Order_type = struct
  type t = 
    | Market 
    | Limit of float  (* price *)
    | Stop of float   (* trigger price *)
    | Stop_limit of { trigger: float; limit: float }
  [@@deriving sexp, compare]
end

(** Order status *)
module Order_status = struct
  type t =
    | New
    | Pending
    | Partially_filled of { filled: float; remaining: float }
    | Filled
    | Cancelled
    | Rejected of string
  [@@deriving sexp, compare]
end

(** Order representation *)
module Order = struct
  type t = {
    id: int;
    symbol: string;
    side: Side.t;
    order_type: Order_type.t;
    quantity: float;
    status: Order_status.t;
    timestamp: Time_ns.t;
  } [@@deriving sexp, compare, fields]
  
  let create ~symbol ~side ~order_type ~quantity =
    {
      id = Random.int 1_000_000;
      symbol;
      side;
      order_type;
      quantity;
      status = Order_status.New;
      timestamp = Time_ns.now ();
    }
  
  let market_order ~symbol ~side ~quantity =
    create ~symbol ~side ~order_type:Order_type.Market ~quantity
  
  let limit_order ~symbol ~side ~quantity ~price =
    create ~symbol ~side ~order_type:(Order_type.Limit price) ~quantity
end

(** Market data tick *)
module Tick = struct
  type t = {
    symbol: string;
    bid: float;
    ask: float;
    bid_size: float;
    ask_size: float;
    last_price: float;
    last_size: float;
    timestamp: Time_ns.t;
  } [@@deriving sexp, compare, fields]
  
  let mid_price t = (t.bid +. t.ask) /. 2.0
  
  let spread t = t.ask -. t.bid
  
  let spread_bps t = spread t /. mid_price t *. 10000.0
end

(** Position tracking *)
module Position = struct
  type t = {
    symbol: string;
    quantity: float;
    avg_price: float;
    market_value: float;
    unrealized_pnl: float;
    realized_pnl: float;
  } [@@deriving sexp, compare, fields]
  
  let empty symbol = {
    symbol;
    quantity = 0.0;
    avg_price = 0.0;
    market_value = 0.0;
    unrealized_pnl = 0.0;
    realized_pnl = 0.0;
  }
  
  let update t ~side ~quantity ~price =
    match side with
    | Side.Buy ->
      let new_qty = t.quantity +. quantity in
      let new_avg = 
        (t.quantity *. t.avg_price +. quantity *. price) /. new_qty
      in
      { t with quantity = new_qty; avg_price = new_avg }
    | Side.Sell ->
      let realized = quantity *. (price -. t.avg_price) in
      { t with 
        quantity = t.quantity -. quantity;
        realized_pnl = t.realized_pnl +. realized 
      }
  
  let mark_to_market t ~price =
    { t with 
      market_value = t.quantity *. price;
      unrealized_pnl = t.quantity *. (price -. t.avg_price) 
    }
end

(** Risk limits *)
module Risk_limits = struct
  type t = {
    max_position_value: float;
    max_order_value: float;
    max_daily_loss: float;
    max_drawdown_pct: float;
  } [@@deriving sexp, compare]
  
  let default = {
    max_position_value = 1_000_000.0;
    max_order_value = 100_000.0;
    max_daily_loss = 50_000.0;
    max_drawdown_pct = 0.05;
  }
end

(** Risk check result *)
module Risk_check = struct
  type t = 
    | Approved
    | Rejected_position_limit
    | Rejected_order_size
    | Rejected_loss_limit
    | Rejected_drawdown
  [@@deriving sexp, compare]
  
  let to_string = function
    | Approved -> "APPROVED"
    | Rejected_position_limit -> "REJECTED: Position limit exceeded"
    | Rejected_order_size -> "REJECTED: Order size too large"
    | Rejected_loss_limit -> "REJECTED: Daily loss limit exceeded"
    | Rejected_drawdown -> "REJECTED: Drawdown limit exceeded"
end

(** Risk manager *)
module Risk_manager = struct
  type t = {
    limits: Risk_limits.t;
    mutable positions: Position.t String.Map.t;
    mutable daily_pnl: float;
    mutable peak_equity: float;
  }
  
  let create ?(limits = Risk_limits.default) () = {
    limits;
    positions = String.Map.empty;
    daily_pnl = 0.0;
    peak_equity = 0.0;
  }
  
  let check_order t (order : Order.t) ~current_price : Risk_check.t =
    let order_value = order.quantity *. current_price in
    
    (* Check order size *)
    if Float.(order_value > t.limits.max_order_value) then
      Risk_check.Rejected_order_size
    else
      (* Check position limit *)
      let current_qty = 
        match Map.find t.positions order.symbol with
        | Some pos -> pos.quantity
        | None -> 0.0
      in
      let new_qty = 
        match order.side with
        | Side.Buy -> current_qty +. order.quantity
        | Side.Sell -> current_qty -. order.quantity
      in
      let new_value = Float.abs new_qty *. current_price in
      
      if Float.(new_value > t.limits.max_position_value) then
        Risk_check.Rejected_position_limit
      else if Float.(-.t.daily_pnl > t.limits.max_daily_loss) then
        Risk_check.Rejected_loss_limit
      else
        Risk_check.Approved
  
  let update_position t ~symbol ~side ~quantity ~price =
    let pos = 
      Map.find t.positions symbol 
      |> Option.value ~default:(Position.empty symbol)
    in
    let updated = Position.update pos ~side ~quantity ~price in
    t.positions <- Map.set t.positions ~key:symbol ~data:updated
end

(** Portfolio weights *)
module Portfolio = struct
  type t = float String.Map.t
  
  let equal_weight symbols =
    let n = List.length symbols |> Float.of_int in
    let weight = 1.0 /. n in
    List.fold symbols ~init:String.Map.empty ~f:(fun acc sym ->
      Map.set acc ~key:sym ~data:weight
    )
  
  let normalize weights =
    let total = 
      Map.fold weights ~init:0.0 ~f:(fun ~key:_ ~data acc -> acc +. data)
    in
    if Float.(total = 0.0) then weights
    else Map.map weights ~f:(fun w -> w /. total)
  
  let turnover ~current ~target =
    let all_keys = 
      Set.union 
        (Map.key_set current) 
        (Map.key_set target)
    in
    Set.fold all_keys ~init:0.0 ~f:(fun acc key ->
      let cur = Map.find current key |> Option.value ~default:0.0 in
      let tar = Map.find target key |> Option.value ~default:0.0 in
      acc +. Float.abs (tar -. cur)
    )
end

(** Signal generation *)
module Signal = struct
  type t = {
    symbol: string;
    alpha: float;        (* Expected return signal *)
    confidence: float;   (* Signal confidence [0, 1] *)
    timestamp: Time_ns.t;
  } [@@deriving sexp, compare, fields]
  
  let create ~symbol ~alpha ~confidence =
    { symbol; alpha; confidence; timestamp = Time_ns.now () }
  
  (** Combine signals with decay *)
  let combine signals ~decay_half_life =
    let now = Time_ns.now () in
    List.fold signals ~init:String.Map.empty ~f:(fun acc signal ->
      let age = 
        Time_ns.diff now signal.timestamp 
        |> Time_ns.Span.to_sec 
      in
      let decay = Float.exp (-. age *. Float.log 2.0 /. decay_half_life) in
      let weighted_alpha = signal.alpha *. signal.confidence *. decay in
      Map.update acc signal.symbol ~f:(function
        | None -> weighted_alpha
        | Some existing -> existing +. weighted_alpha
      )
    )
end

(** Execution engine *)
module Execution = struct
  type fill = {
    order_id: int;
    symbol: string;
    side: Side.t;
    quantity: float;
    price: float;
    timestamp: Time_ns.t;
  } [@@deriving sexp]
  
  (** Execute market order against current book *)
  let execute_market_order (order : Order.t) (tick : Tick.t) : fill option =
    if not (String.equal order.symbol tick.symbol) then None
    else
      let price = 
        match order.side with
        | Side.Buy -> tick.ask   (* Pay the ask *)
        | Side.Sell -> tick.bid  (* Hit the bid *)
      in
      Some {
        order_id = order.id;
        symbol = order.symbol;
        side = order.side;
        quantity = order.quantity;
        price;
        timestamp = Time_ns.now ();
      }
  
  (** Calculate slippage *)
  let slippage ~expected_price ~actual_price ~side =
    match side with
    | Side.Buy -> actual_price -. expected_price
    | Side.Sell -> expected_price -. actual_price
end

(** Main trading system *)
module Trading_system = struct
  type t = {
    risk_manager: Risk_manager.t;
    mutable current_positions: Position.t String.Map.t;
    mutable order_count: int;
  }
  
  let create ?limits () = {
    risk_manager = Risk_manager.create ?limits ();
    current_positions = String.Map.empty;
    order_count = 0;
  }
  
  let submit_order t order ~current_price =
    match Risk_manager.check_order t.risk_manager order ~current_price with
    | Risk_check.Approved ->
      t.order_count <- t.order_count + 1;
      Ok order.id
    | rejection ->
      Error (Risk_check.to_string rejection)
  
  let process_fill t (fill : Execution.fill) =
    Risk_manager.update_position t.risk_manager 
      ~symbol:fill.symbol 
      ~side:fill.side 
      ~quantity:fill.quantity 
      ~price:fill.price
end

(** Utility: compute returns from prices *)
let compute_returns prices =
  match prices with
  | [] | [_] -> []
  | _ ->
    List.zip_exn 
      (List.tl_exn prices) 
      (List.drop_last_exn prices)
    |> List.map ~f:(fun (curr, prev) -> (curr -. prev) /. prev)

(** Utility: compute volatility *)
let volatility returns =
  let n = List.length returns |> Float.of_int in
  let mean = List.sum (module Float) returns ~f:Fn.id /. n in
  let var = 
    List.sum (module Float) returns ~f:(fun r -> 
      Float.square (r -. mean)
    ) /. n
  in
  Float.sqrt var

(** Utility: compute Sharpe ratio *)
let sharpe_ratio returns ~risk_free_rate =
  let n = List.length returns |> Float.of_int in
  let mean = List.sum (module Float) returns ~f:Fn.id /. n in
  let vol = volatility returns in
  (mean -. risk_free_rate /. 252.0) /. vol *. Float.sqrt 252.0

(* Entry point for testing *)
let () =
  print_endline "Elite Quant System - OCaml Core";
  print_endline "================================";
  
  (* Create trading system *)
  let system = Trading_system.create () in
  
  (* Create sample order *)
  let order = Order.market_order 
    ~symbol:"AAPL" 
    ~side:Side.Buy 
    ~quantity:100.0 
  in
  
  (* Submit order *)
  match Trading_system.submit_order system order ~current_price:150.0 with
  | Ok id -> 
    Printf.printf "Order %d submitted successfully\n" id
  | Error msg -> 
    Printf.printf "Order rejected: %s\n" msg

