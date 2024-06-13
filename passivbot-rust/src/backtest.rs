use std::collections::HashMap;
use std::f64::INFINITY;

struct Config {
    starting_balance: f64,
    long: LongConfig,
    symbols: Vec<String>,
}

struct LongConfig {
    n_positions: usize,
}

#[derive(Clone)]
struct Order {
    price: f64,
    size: f64,
    order_type: OrderType,
}

#[derive(Clone)]
enum OrderType {
    Grid,
    // Add other order types as needed
}

pub struct Fill {
    //
}

pub struct Stat {
    //
}

pub struct Backtest {
    config: Config,
    hlcs: Vec<Vec<[f64; 3]>>,
    noisiness_indices: Vec<Vec<usize>>,
    emas: HashMap<String, [f64; 3]>,
    ema_alphas: HashMap<String, [f64; 3]>,
    fills: Vec<Order>,
    stats: Vec<()>, // Update with appropriate stats type
    balance: f64,
    pnl_cumsum_max: f64,
    pnl_cumsum_running: f64,
    positions: HashMap<String, [f64; 2]>,
    open_orders: OpenOrders,
    trailing_data: HashMap<String, TrailingData>,
    actives: Actives,
    k: usize,
}

struct OpenOrders {
    entry: HashMap<String, Order>,
    close: HashMap<String, Order>,
    unstuck: Order,
}

struct TrailingData {
    min_price_since_open: f64,
    max_price_since_min: f64,
    max_price_since_open: f64,
    min_price_since_max: f64,
}

struct Actives {
    longs: Vec<usize>,
}

impl Backtest {
    fn new(hlcs: Vec<Vec<[f64; 3]>>, noisiness_indices: Vec<Vec<usize>>, config: Config) -> Self {
        let symbols = config.symbols.clone();
        let emas = symbols
            .iter()
            .map(|symbol| (symbol.clone(), [hlcs[0][0][2]; 3]))
            .collect();

        Backtest {
            config,
            hlcs,
            noisiness_indices,
            emas,
            ema_alphas: HashMap::new(),
            fills: Vec::new(),
            stats: Vec::new(),
            balance: 0.0,
            pnl_cumsum_max: 0.0,
            pnl_cumsum_running: 0.0,
            positions: symbols
                .iter()
                .map(|symbol| (symbol.clone(), [0.0, 0.0]))
                .collect(),
            open_orders: OpenOrders {
                entry: HashMap::new(),
                close: HashMap::new(),
                unstuck: Order {
                    price: 0.0,
                    size: 0.0,
                    order_type: OrderType::Grid,
                },
            },
            trailing_data: symbols
                .iter()
                .map(|symbol| {
                    (
                        symbol.clone(),
                        TrailingData {
                            min_price_since_open: 0.0,
                            max_price_since_min: 0.0,
                            max_price_since_open: 0.0,
                            min_price_since_max: INFINITY,
                        },
                    )
                })
                .collect(),
            actives: Actives { longs: Vec::new() },
            k: 0,
        }
    }

    fn prep_emas(&mut self) {
        // Implement prep_emas logic
    }

    fn update_emas(&mut self) {
        // Implement update_emas logic
    }

    fn process_fill(&mut self, close: &Order) {
        // Implement process_fill logic
    }

    fn check_for_fills(&mut self) {
        // Implement check_for_fills logic
    }

    fn update_trailing_data(&mut self) {
        // Implement update_trailing_data logic
    }

    fn update_actives(&mut self) {
        // Implement update_actives logic
    }

    fn update_entry_order(&mut self, symbol: &str) {
        // Implement update_entry_order logic
    }

    fn update_open_orders(&mut self) {
        // Implement update_open_orders logic
    }

    fn update_stats(&mut self) {
        // Implement update_stats logic
    }

    fn run(&mut self) -> (Vec<()>, Vec<Order>) {
        self.prep_emas();
        for k in 1..self.hlcs.len() {
            self.k = k;
            self.check_for_fills();
            self.update_actives();
            self.update_trailing_data();
            self.update_open_orders();
            self.update_stats();
        }
        (self.stats.clone(), self.fills.clone())
    }
}
