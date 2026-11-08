## Quick Guide for Passivbot on Hyperliquid

### Hyperliquid account creation

1. Go to https://app.hyperliquid.xyz/ and create an account by connecting a third party wallet wallet (more secure) or using email (less secure).
2. Click the "Deposit" button to transfer USDC (not USDT, USDC.E or any other stablecoin) via the Arbitrum network to your account's address.
3. Navigate to "More" -> "API" and follow instructions to create an API wallet (agent wallet).

### Passivbot setup

1. Follow [installation and update guidance](installation.md) for the revision you will run.
2. Add your Hyperliquid wallet address and API wallet to your private `api-keys.json`.
```json
"hyperliquid_01": {
    "exchange": "hyperliquid",
    "wallet_address": "YOUR_USDC_PUBLIC_WALLET_ADDRESS",
    "private_key": "API_PRIVATE_KEY",
    "is_vault": false
}
```
For the `"private_key"`, use the API wallet created in the API section on Hyperliquid.

Set `live.account.user` to the account name in your reviewed v9 config. See
[Running live](live.md) for offline preflight and startup, and
[minimum effective cost](min_effective_cost.md) for how account balance and strategy sizing
affect entry eligibility. For builder-deployed markets, review the
[HIP-3 account and margin requirements](stock_perps.md).

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
