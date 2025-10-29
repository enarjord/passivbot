# Realtime snapshot reference

## Sample snapshot with active stop-loss monitoring
```json
{
  "generated_at": "2024-05-04T12:15:30.452817+00:00",
  "accounts": [
    {
      "name": "Binance Futures",
      "balance": 15234.12,
      "daily_realized_pnl": 320.55,
      "positions": [
        {
          "symbol": "BTCUSDT",
          "side": "long",
          "size": 0.25,
          "notional": 16950.25,
          "signed_notional": 16950.25,
          "entry_price": 67800.0,
          "mark_price": 67620.5,
          "liquidation_price": 51250.0,
          "wallet_exposure_pct": 0.45,
          "unrealized_pnl": -450.63,
          "max_drawdown_pct": 0.22,
          "take_profit_price": 70500.0,
          "stop_loss_price": 65500.0
        }
      ],
      "orders": [
        {
          "symbol": "BTCUSDT",
          "side": "sell",
          "order_type": "limit",
          "price": 70500.0,
          "amount": 0.25,
          "remaining": 0.25,
          "status": "open",
          "reduce_only": true,
          "stop_price": null,
          "notional": 17625.0,
          "order_id": "abc123",
          "created_at": "2024-05-04T11:58:12Z"
        }
      ]
    },
    {
      "name": "OKX Futures",
      "balance": 8250.8,
      "daily_realized_pnl": -120.0,
      "positions": [],
      "orders": []
    }
  ],
  "alert_thresholds": {
    "wallet_exposure_pct": 0.65,
    "position_wallet_exposure_pct": 0.25,
    "max_drawdown_pct": 0.25,
    "loss_threshold_pct": -0.08
  },
  "notification_channels": [
    "email:risk-team@example.com",
    "slack:#passivbot-risk-alerts"
  ],
  "account_messages": {
    "OKX Futures": "Authentication for OKX Futures restored"
  },
  "portfolio_stop_loss": {
    "threshold_pct": 12.0,
    "baseline_balance": 25000.0,
    "current_balance": 23484.92,
    "current_drawdown_pct": 0.0606,
    "triggered": false,
    "triggered_at": null,
    "active": true
  },
  "account_stop_losses": {
    "Binance Futures": {
      "threshold_pct": 10.0,
      "baseline_balance": 16000.0,
      "current_balance": 15234.12,
      "current_drawdown_pct": 0.0472,
      "triggered": false,
      "triggered_at": null,
      "active": true
    },
    "OKX Futures": {
      "threshold_pct": 8.0,
      "baseline_balance": 9000.0,
      "current_balance": 8250.8,
      "current_drawdown_pct": 0.0832,
      "triggered": true,
      "triggered_at": "2024-05-04T12:10:07.993201+00:00",
      "active": true
    }
  },
  "performance": {
    "portfolio": {
      "current_balance": 23484.92,
      "latest_snapshot": {
        "date": "2024-05-04",
        "balance": 24010.45,
        "timestamp": "2024-05-04T12:00:00+00:00"
      },
      "daily": {
        "pnl": -215.08,
        "since": "2024-05-03",
        "reference_balance": 23700.0
      },
      "weekly": null,
      "monthly": null
    },
    "accounts": {
      "Binance Futures": {
        "current_balance": 15234.12,
        "latest_snapshot": {
          "date": "2024-05-04",
          "balance": 15800.0,
          "timestamp": "2024-05-04T11:30:00+00:00"
        },
        "daily": null,
        "weekly": null,
        "monthly": null
      },
      "OKX Futures": {
        "current_balance": 8250.8,
        "latest_snapshot": {
          "date": "2024-05-04",
          "balance": 8900.0,
          "timestamp": "2024-05-04T11:50:00+00:00"
        },
        "daily": {
          "pnl": -649.2,
          "since": "2024-05-03",
          "reference_balance": 8900.0
        },
        "weekly": null,
        "monthly": null
      }
    }
  }
}
```

## Sample snapshot after portfolio stop-loss trigger
```json
{
  "generated_at": "2024-05-04T12:45:02.118903+00:00",
  "accounts": [],
  "alert_thresholds": {
    "wallet_exposure_pct": 0.65,
    "position_wallet_exposure_pct": 0.25,
    "max_drawdown_pct": 0.25,
    "loss_threshold_pct": -0.08
  },
  "notification_channels": [],
  "portfolio_stop_loss": {
    "threshold_pct": 12.0,
    "baseline_balance": 25000.0,
    "current_balance": 21000.0,
    "current_drawdown_pct": 0.16,
    "triggered": true,
    "triggered_at": "2024-05-04T12:44:11.532901+00:00",
    "active": true
  }
}
```

## JSON schema for regression tests
The expected payload structure for regression checks is captured in
[`docs/risk_management_realtime_snapshot.schema.json`](risk_management_realtime_snapshot.schema.json).
This draft-07 schema includes reusable definitions for accounts, positions,
orders, stop-loss states, and performance summaries so automated tests can
validate realtime responses.
