fn activation_input(kind: StrategyKind, side: PositionSide, held: bool) -> OrchestratorInput {
    let mut symbol = make_basic_symbol(0);
    for params in [&mut symbol.long.bot_params, &mut symbol.short.bot_params] {
        params.entry_initial_qty_pct = 0.1;
        params.entry_grid_double_down_factor = 1.2;
        params.entry_grid_spacing_pct = 0.01;
        params.close_grid_qty_pct = 0.5;
        params.close_trailing_threshold_pct = 0.01;
    }
    for input in [&mut symbol.long, &mut symbol.short] {
        input.mode = Some(TradingMode::Normal);
        let parsed = match kind {
            StrategyKind::TrailingMartingale => {
                StrategyParams::TrailingMartingale(tm_params_for_test(&input.bot_params))
            }
            StrategyKind::EmaAnchor => {
                StrategyParams::EmaAnchor(crate::strategies::EmaAnchorParams {
                    ema_span_0: 10.0,
                    ema_span_1: 20.0,
                    base_qty_pct: 0.1,
                    offset_psize_weight: 0.0,
                    ..Default::default()
                })
            }
        };
        input.parsed_strategy_params = Some(parsed);
    }
    if held {
        let position = match side {
            PositionSide::Long => &mut symbol.long.position,
            PositionSide::Short => &mut symbol.short.position,
        };
        position.size = if side == PositionSide::Long {
            1.0
        } else {
            -1.0
        };
        position.price = 100.0;
    }
    let mut global = make_basic_global();
    global.strategy_kind = kind;
    global.side_activation = crate::types::SideActivation {
        long: side == PositionSide::Long,
        short: side == PositionSide::Short,
    };
    global.global_bot_params = BotParamsPair {
        long: symbol.long.bot_params.clone(),
        short: symbol.short.bot_params.clone(),
    };
    OrchestratorInput {
        timestamp_ms: 1_000_000,
        balance: 1000.0,
        balance_raw: 1000.0,
        global,
        symbols: vec![symbol],
        peek_hints: None,
        forager_hysteresis: None,
    }
}

fn disable_activation(input: &mut OrchestratorInput) {
    input.global.side_activation = crate::types::SideActivation {
        long: false,
        short: false,
    };
}

#[test]
fn side_activation_blocks_flat_forced_normal_and_held_dca_preserving_closes() {
    for kind in [StrategyKind::TrailingMartingale, StrategyKind::EmaAnchor] {
        for side in [PositionSide::Long, PositionSide::Short] {
            for held in [false, true] {
                let mut input = activation_input(kind, side, held);
                let baseline = super::compute_ideal_orders(&input).unwrap();
                assert!(
                    baseline
                        .orders
                        .iter()
                        .any(|o| o.pside == side && o.order_type.is_entry()),
                    "fixture must actually exercise entry/DCA: {kind:?} {side:?} held={held}"
                );
                let baseline_closes: Vec<_> = baseline
                    .orders
                    .iter()
                    .filter(|o| o.pside == side && o.order_type.is_close())
                    .map(|o| (o.qty, o.price, o.order_type))
                    .collect();
                if held {
                    assert!(!baseline_closes.is_empty());
                }
                disable_activation(&mut input);
                let output = super::compute_ideal_orders(&input).unwrap();
                assert!(
                    !output.orders.iter().any(|o| o.order_type.is_entry()),
                    "disabled forced normal leaked an increase: {kind:?} {side:?} held={held}"
                );
                let closes: Vec<_> = output
                    .orders
                    .iter()
                    .filter(|o| o.pside == side && o.order_type.is_close())
                    .map(|o| (o.qty, o.price, o.order_type))
                    .collect();
                assert_eq!(closes, baseline_closes);
            }
        }
    }
}

#[test]
fn side_activation_does_not_become_an_implicit_operational_mode() {
    for kind in [StrategyKind::TrailingMartingale, StrategyKind::EmaAnchor] {
        for side in [PositionSide::Long, PositionSide::Short] {
            for mode in [
                TradingMode::Normal,
                TradingMode::GracefulStop,
                TradingMode::TpOnly,
                TradingMode::Manual,
                TradingMode::Panic,
            ] {
                let mut input = activation_input(kind, side, true);
                disable_activation(&mut input);
                match side {
                    PositionSide::Long => input.symbols[0].long.mode = Some(mode),
                    PositionSide::Short => input.symbols[0].short.mode = Some(mode),
                }
                let output = super::compute_ideal_orders(&input).unwrap();
                assert!(!output.orders.iter().any(|o| o.order_type.is_entry()));
                let closes: Vec<_> = output.orders.iter().filter(|o| o.pside == side).collect();
                if mode == TradingMode::Manual {
                    assert!(closes.is_empty());
                } else {
                    assert!(!closes.is_empty(), "{kind:?} {side:?} {mode:?}");
                }
                assert_eq!(
                    closes.iter().any(|o| matches!(
                        o.order_type,
                        OrderType::ClosePanicLong | OrderType::ClosePanicShort
                    )),
                    mode == TradingMode::Panic
                );
            }
        }
    }
}

#[test]
fn side_activation_transport_requires_exact_booleans_and_both_sides() {
    for value in [
        serde_json::json!({}),
        serde_json::json!({"long":true}),
        serde_json::json!({"long":true,"short":false,"extra":true}),
        serde_json::json!({"long":1,"short":false}),
        serde_json::json!({"long":null,"short":false}),
    ] {
        assert!(serde_json::from_value::<crate::types::SideActivation>(value).is_err());
    }
    let input = activation_input(StrategyKind::EmaAnchor, PositionSide::Long, false);
    let mut value = serde_json::to_value(input).unwrap();
    value["global"]
        .as_object_mut()
        .unwrap()
        .remove("side_activation");
    assert!(serde_json::from_value::<OrchestratorInput>(value).is_err());
}

#[test]
fn side_activation_zero_capacity_preserves_held_closes_and_authored_budget() {
    for kind in [StrategyKind::TrailingMartingale, StrategyKind::EmaAnchor] {
        for side in [PositionSide::Long, PositionSide::Short] {
            for zero_positions in [false, true] {
                for active in [false, true] {
                    let mut input = activation_input(kind, side, true);
                    disable_activation(&mut input);
                    let global = match side {
                        PositionSide::Long => {
                            input.global.side_activation.long = active;
                            &mut input.global.global_bot_params.long
                        }
                        PositionSide::Short => {
                            input.global.side_activation.short = active;
                            &mut input.global.global_bot_params.short
                        }
                    };
                    if zero_positions {
                        global.n_positions = 0;
                    } else {
                        global.total_wallet_exposure_limit = 0.0;
                    }
                    // The real held symbol retains its explicitly configured WEL.
                    // Zero portfolio capacity is not a replacement sizing budget.
                    let policy_before =
                        serde_json::to_value(&input.global.global_bot_params).unwrap();
                    let output = super::compute_ideal_orders(&input).unwrap();
                    assert!(!output.orders.iter().any(|o| o.order_type.is_entry()));
                    assert!(output.orders.iter().any(|o| o.pside == side && o.order_type.is_close()),
                        "zero capacity lost held close: {kind:?} {side:?} n0={zero_positions} active={active}");
                    assert!(!output.orders.iter().any(|o| matches!(
                        o.order_type,
                        OrderType::ClosePanicLong | OrderType::ClosePanicShort
                    )));
                    assert_eq!(
                        serde_json::to_value(&input.global.global_bot_params).unwrap(),
                        policy_before
                    );
                }
            }
        }
    }
}

#[test]
fn side_activation_one_way_does_not_let_a_disabled_forced_side_win() {
    for kind in [StrategyKind::TrailingMartingale, StrategyKind::EmaAnchor] {
        for side in [PositionSide::Long, PositionSide::Short] {
            let mut input = activation_input(kind, side, false);
            input.global.hedge_mode = false;
            // Both sides are forced Normal, but only the explicitly activated
            // side may compete for the flat one-way slot.
            let output = super::compute_ideal_orders(&input).unwrap();
            let entries: Vec<_> = output
                .orders
                .iter()
                .filter(|o| o.order_type.is_entry())
                .collect();
            assert!(!entries.is_empty());
            assert!(entries.iter().all(|o| o.pside == side));
            disable_activation(&mut input);
            assert!(super::compute_ideal_orders(&input)
                .unwrap()
                .orders
                .is_empty());
        }
    }
}

#[test]
fn side_activation_preserves_independently_enabled_risk_reducers() {
    for kind in [StrategyKind::TrailingMartingale, StrategyKind::EmaAnchor] {
        for side in [PositionSide::Long, PositionSide::Short] {
            for risk in ["wel", "twel", "unstuck"] {
                // WEL auto-reduction belongs to the existing TM close owner;
                // activation does not add that reducer to EMA Anchor.
                if kind == StrategyKind::EmaAnchor && risk == "wel" {
                    continue;
                }
                let mut input = activation_input(kind, side, true);
                let symbol_side = match side {
                    PositionSide::Long => &mut input.symbols[0].long,
                    PositionSide::Short => &mut input.symbols[0].short,
                };
                symbol_side.position.size *= 20.0;
                let bot = &mut symbol_side.bot_params;
                bot.risk_wel_enforcer_enabled = risk == "wel";
                bot.risk_wel_enforcer_threshold = 1.0;
                bot.unstuck_enabled = risk == "unstuck";
                bot.unstuck_ema_gating_enabled = false;
                bot.unstuck_close_pct = 0.1;
                bot.unstuck_loss_allowance_pct = 0.1;
                bot.unstuck_threshold = 0.5;
                let global = match side {
                    PositionSide::Long => &mut input.global.global_bot_params.long,
                    PositionSide::Short => &mut input.global.global_bot_params.short,
                };
                global.risk_twel_enforcer_enabled = risk == "twel";
                global.risk_twel_enforcer_threshold = 1.0;
                let expected = match (risk, side) {
                    ("wel", PositionSide::Long) => OrderType::CloseAutoReduceWelLong,
                    ("wel", PositionSide::Short) => OrderType::CloseAutoReduceWelShort,
                    ("twel", PositionSide::Long) => OrderType::CloseAutoReduceTwelLong,
                    ("twel", PositionSide::Short) => OrderType::CloseAutoReduceTwelShort,
                    ("unstuck", PositionSide::Long) => OrderType::CloseUnstuckLong,
                    ("unstuck", PositionSide::Short) => OrderType::CloseUnstuckShort,
                    _ => unreachable!(),
                };
                let baseline = super::compute_ideal_orders(&input).unwrap();
                assert!(
                    baseline.orders.iter().any(|o| o.order_type == expected),
                    "risk fixture did not exercise {risk}: {kind:?} {side:?}"
                );
                disable_activation(&mut input);
                let output = super::compute_ideal_orders(&input).unwrap();
                assert!(
                    output.orders.iter().any(|o| o.order_type == expected),
                    "activation disabled risk reducer {risk}: {kind:?} {side:?}"
                );
                assert!(!output.orders.iter().any(|o| o.order_type.is_entry()));
            }
        }
    }
}

#[test]
fn side_activation_does_not_fabricate_ema_anchor_zero_allocation_close_sizing() {
    for side in [PositionSide::Long, PositionSide::Short] {
        let mut input = activation_input(StrategyKind::EmaAnchor, side, true);
        disable_activation(&mut input);
        let held = match side {
            PositionSide::Long => &mut input.symbols[0].long,
            PositionSide::Short => &mut input.symbols[0].short,
        };
        held.bot_params.wallet_exposure_limit = 0.0;
        held.bot_params.total_wallet_exposure_limit = 0.0;
        held.bot_params.n_positions = 0;
        let global = match side {
            PositionSide::Long => &mut input.global.global_bot_params.long,
            PositionSide::Short => &mut input.global.global_bot_params.short,
        };
        global.total_wallet_exposure_limit = 0.0;
        global.n_positions = 0;
        // EA's existing clip policy returns zero with an actual zero budget.
        // Deactivation must neither invent allocation nor imply panic.
        assert!(super::compute_ideal_orders(&input)
            .unwrap()
            .orders
            .is_empty());
        let held = match side {
            PositionSide::Long => &mut input.symbols[0].long,
            PositionSide::Short => &mut input.symbols[0].short,
        };
        held.mode = Some(TradingMode::Panic);
        let output = super::compute_ideal_orders(&input).unwrap();
        assert_eq!(output.orders.len(), 1);
        assert!(matches!(
            output.orders[0].order_type,
            OrderType::ClosePanicLong | OrderType::ClosePanicShort
        ));
        let held = match side {
            PositionSide::Long => &mut input.symbols[0].long,
            PositionSide::Short => &mut input.symbols[0].short,
        };
        held.mode = Some(TradingMode::Normal);
        let global = match side {
            PositionSide::Long => &mut input.global.global_bot_params.long,
            PositionSide::Short => &mut input.global.global_bot_params.short,
        };
        global.total_wallet_exposure_limit = 0.05;
        global.n_positions = 1;
        global.risk_twel_enforcer_enabled = true;
        global.risk_twel_enforcer_threshold = 1.0;
        let output = super::compute_ideal_orders(&input).unwrap();
        assert!(output.orders.iter().any(|o| matches!(
            o.order_type,
            OrderType::CloseAutoReduceTwelLong | OrderType::CloseAutoReduceTwelShort
        )));
        assert!(!output.orders.iter().any(|o| o.order_type.is_entry()));
    }
}
