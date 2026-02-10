## Quick Guide for Passivbot on Hyperliquid

### Hyperliquid account creation

1. Go to https://app.hyperliquid.xyz/ and create an account by connecting a third party wallet wallet (more secure) or using email (less secure).
2. Click the "Deposit" button to transfer USDC (not USDT, USDC.E or any other stablecoin) via the Arbitrum network to your account's address.
3. Navigate to "More" -> "API" and follow instructions to create an API wallet (agent wallet).

### Passivbot setup

1. If not already installed, install Passivbot. Otherwise, pull latest master branch: `git pull`.
2. Update the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your Hyperliquid wallet address and API wallet to `api-keys.json`:
```json
"hyperliquid_01": {
    "exchange": "hyperliquid",
    "wallet_address": "YOUR_USDC_PUBLIC_WALLET_ADDRESS",
    "private_key": "API_PRIVATE_KEY",
    "is_vault": false
}
```
For the `"private_key"`, use the API wallet created in the API section on Hyperliquid.

Now Passivbot may be run as normal. Note that Hyperliquid has a minimum $10 order size:  
`initial_entry_cost = balance * (total_wallet_exposure_limit / n_positions) * entry_initial_qty_pct`

### Builder Codes (Supporting Passivbot Development)

Passivbot includes a Hyperliquid builder code that adds a small fee (0.02%) to each trade.
This fee goes directly to funding Passivbot development. Builder codes are enabled by default.

**If you use your main wallet private key** in `api-keys.json`, the bot will automatically
approve the builder code on first startup. No extra action needed.

**If you use an agent/API wallet** (recommended for security), you need to approve the builder
code once from your main wallet. Choose one method:

1. **CLI tool**: `python3 src/tools/approve_builder_fee.py` (prompts for main wallet key, hidden input, not stored).
2. **Temporary key swap**: Temporarily put your main wallet private key in `api-keys.json`,
   start the bot once (it will auto-approve), then switch back to your agent key.
3. **Hyperliquid web UI**: Approve via Settings > Approvals on app.hyperliquid.xyz.

Without approval, the bot still works but will show periodic reminders. The builder code can
be revoked at any time. To adjust the fee or disable entirely, edit `broker_codes.hjson`.

#### HyperLiquid with a Vault (CopyTrading-like)
1. In HyperLiquid, navigate to "Vaults" in the top menu and create a new vault.
2. Set the name and description (limited to 250 characters).
3. To find the wallet address of the vault, navigate to "Vaults" again and find your vault in the public vault list.
4. Click on your vault and get the public address of the vault. This address will be your `YOUR_VAULT_PUBLIC_ADDRESS` in `api-keys.json`. Remember to set `"is_vault"` to true.

Update `api-keys.json`:
```json
"hyperliquid_01": {
    "exchange": "hyperliquid",
    "wallet_address": "YOUR_VAULT_PUBLIC_ADDRESS",
    "private_key": "API_PRIVATE_KEY",
    "is_vault": true
}
```

Refer to Hyperliquid's documentation for more details.
