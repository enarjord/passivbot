use ndarray::ArrayD;
use std::collections::HashMap;

struct Backtest {
    symbols: Vec<String>,
    emas: HashMap<String, f64>,
    fills: Vec<Fill>,
    stats: Vec<Stat>,
    open_orders: OpenOrders,
    positions: Positions,
}

struct Fill {
    // Define fields for Fill
}

struct Stat {
    // Define fields for Stat
}

struct OpenOrders {
    long: Orders,
    short: Orders,
}

struct Orders {
    entries: HashMap<String, Vec<Order>>,
    closes: HashMap<String, Vec<Order>>,
}

struct Order {
    // Define fields for Order
}

struct Positions {
    long: HashMap<String, (f64, f64)>,
    short: HashMap<String, (f64, f64)>,
}

impl Backtest {
    fn new(symbols: Vec<String>) -> Self {
        Backtest {
            symbols,
            emas: HashMap::new(),
            fills: Vec::new(),
            stats: Vec::new(),
            open_orders: OpenOrders {
                long: Orders {
                    entries: HashMap::new(),
                    closes: HashMap::new(),
                },
                short: Orders {
                    entries: HashMap::new(),
                    closes: HashMap::new(),
                },
            },
            positions: Positions {
                long: HashMap::new(),
                short: HashMap::new(),
            },
        }
    }

    fn backtest(&mut self, hlcs: &ArrayD<f64>) -> (&Vec<Fill>, &Vec<Stat>) {
        self.prep_emas();

        for k in 1..hlcs.len() {
            self.check_for_fills();
            self.fills.extend(self.new_fills());
            self.update_emas();
            self.update_open_orders();
            self.update_stats();
        }

        (&self.fills, &self.stats)
    }

    fn prep_emas(&mut self) {
        // Prepare EMAs
    }

    fn check_for_fills(&mut self) {
        // Check for fills and update positions and open orders
    }

    fn new_fills(&self) -> Vec<Fill> {
        // Return new fills
        Vec::new()
    }

    fn update_emas(&mut self) {
        // Update EMAs
    }

    fn update_open_orders(&mut self) {
        // Update open orders
    }

    fn update_stats(&mut self) {
        // Update stats
    }
}
