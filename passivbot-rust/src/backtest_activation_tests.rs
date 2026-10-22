#[test]
fn side_activation_backtest_blocks_resting_increases_but_fills_held_closes_without_capacity() {
    for side in [LONG, SHORT] {
        let mut data = Array3::from_elem((3, 1, 4), 100.0);
        for k in 0..3 {
            data[[k, 0, HIGH]] = 150.0;
            data[[k, 0, LOW]] = 50.0;
        }
        let btc = Array1::from_elem(3, 20_000.0);
        let mut params = ordinary_ema_fixture_params(3, 1);
        params.side_activation = crate::types::SideActivation {
            long: false,
            short: false,
        };
        params.equity_hard_stop_loss.hsl = None;
        let pair = BotParamsPair::default();
        let mut bt = Backtest::new(
            data.view(),
            btc.view(),
            vec![pair],
            vec![ExchangeParams::default()],
            &params,
        );
        let held = Position {
            size: if side == LONG { 1.0 } else { -1.0 },
            price: 100.0,
        };
        let (positions, orders) = if side == LONG {
            (&mut bt.positions.long, &mut bt.open_orders.long)
        } else {
            (&mut bt.positions.short, &mut bt.open_orders.short)
        };
        positions[0] = held;
        let entry_kind = if side == LONG {
            OrderType::EntryGridNormalLong
        } else {
            OrderType::EntryGridNormalShort
        };
        let close_kind = if side == LONG {
            OrderType::CloseGridLong
        } else {
            OrderType::CloseGridShort
        };
        orders[0].entries.push(BacktestOrder {
            order: Order {
                qty: held.size * 0.25,
                price: 100.0,
                order_type: entry_kind,
            },
            execution_type: orchestrator::ExecutionType::Limit,
        });
        orders[0].closes.push(BacktestOrder {
            order: Order {
                qty: -held.size * 0.5,
                price: 100.0,
                order_type: close_kind,
            },
            execution_type: orchestrator::ExecutionType::Limit,
        });
        assert!(!bt.trading_enabled.long && !bt.trading_enabled.short);
        bt.check_for_fills(1).unwrap();
        assert_eq!(bt.fills.len(), 1);
        assert_eq!(bt.fills[0].order_type, close_kind);
        let position = if side == LONG {
            bt.positions.long[0]
        } else {
            bt.positions.short[0]
        };
        assert_eq!(position.size, held.size * 0.5);
    }
}

#[test]
fn side_activation_backtest_refreshes_cached_native_permission_without_replacing_budgets() {
    let data = Array3::from_elem((3, 1, 4), 100.0);
    let btc = Array1::from_elem(3, 20_000.0);
    let params = ordinary_ema_fixture_params(3, 1);
    let mut pair = BotParamsPair::default();
    pair.long.n_positions = 1;
    pair.long.total_wallet_exposure_limit = 1.5;
    let mut bt = Backtest::new(
        data.view(),
        btc.view(),
        vec![pair],
        vec![ExchangeParams::default()],
        &params,
    );
    let before = bt.get_orchestrator_input_cached(1, None, None);
    bt.orchestrator_input_cache = Some(before.clone());
    bt.backtest_params.side_activation.long = false;
    let after = bt.get_orchestrator_input_cached(2, None, None);
    assert!(before.global.side_activation.long);
    assert!(!after.global.side_activation.long);
    assert_eq!(
        after
            .global
            .global_bot_params
            .long
            .total_wallet_exposure_limit,
        1.5
    );
    assert_eq!(after.global.global_bot_params.long.n_positions, 1);
}

#[test]
fn side_activation_backtest_keeps_held_trailing_state_current_without_entry_capacity() {
    let mut data = Array3::from_elem((3, 1, 4), 100.0);
    data[[1, 0, HIGH]] = 110.0;
    data[[1, 0, LOW]] = 90.0;
    let btc = Array1::from_elem(3, 20_000.0);
    let mut params = ordinary_ema_fixture_params(3, 1);
    params.side_activation = crate::types::SideActivation {
        long: false,
        short: false,
    };
    let mut bt = Backtest::new(
        data.view(),
        btc.view(),
        vec![BotParamsPair::default()],
        vec![ExchangeParams::default()],
        &params,
    );
    bt.positions.long[0] = Position {
        size: 1.0,
        price: 100.0,
    };
    bt.positions.short[0] = Position {
        size: -1.0,
        price: 100.0,
    };
    bt.any_trailing_long = true;
    bt.any_trailing_short = true;
    bt.trailing_enabled[0].long = true;
    bt.trailing_enabled[0].short = true;
    assert!(!bt.trading_enabled.long && !bt.trading_enabled.short);
    bt.update_trailing_prices(1);
    assert_eq!(bt.trailing_prices.long[0].max_since_open, 110.0);
    assert_eq!(bt.trailing_prices.short[0].min_since_open, 90.0);
}
