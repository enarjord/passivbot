(() => {
  const state = {
    snapshot: null,
    exchange: null,
    user: null,
    focusSymbol: "",
    recentEvents: [],
    recentTicks: new Map(),
    ws: null,
    reconnectTimer: null,
    snapshotTimer: null,
    lastWsTs: null,
  };

  const qs = new URLSearchParams(window.location.search);
  const els = {
    heroTitle: document.getElementById("hero-title"),
    heroSubtitle: document.getElementById("hero-subtitle"),
    focusSelect: document.getElementById("focus-symbol"),
    refreshButton: document.getElementById("refresh-button"),
    wsStatus: document.getElementById("ws-status"),
    snapshotStatus: document.getElementById("snapshot-status"),
    focusStatus: document.getElementById("focus-status"),
    relayStatus: document.getElementById("relay-status"),
    summaryCards: document.getElementById("summary-cards"),
    focusLabel: document.getElementById("focus-label"),
    focusDetails: document.getElementById("focus-details"),
    positionsMeta: document.getElementById("positions-meta"),
    positionsBody: document.getElementById("positions-body"),
    trailingList: document.getElementById("trailing-list"),
    foragerDetails: document.getElementById("forager-details"),
    unstuckDetails: document.getElementById("unstuck-details"),
    eventsList: document.getElementById("events-list"),
    ticksList: document.getElementById("ticks-list"),
    ordersList: document.getElementById("orders-list"),
    emptyTemplate: document.getElementById("empty-state-template"),
  };

  function fmtNumber(value, digits = 4) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    return Number(value).toFixed(digits);
  }

  function fmtCompact(value, digits = 4) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    const text = Number(value).toFixed(digits).replace(/\.?0+$/, "");
    return text === "-0" ? "0" : text;
  }

  function fmtPctRatio(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    const pct = Number(value) * 100;
    const sign = pct >= 0 ? "+" : "";
    return `${sign}${pct.toFixed(digits)}%`;
  }

  function fmtAgeMs(tsMs) {
    if (!tsMs) return "-";
    const delta = Math.max(0, Date.now() - Number(tsMs));
    if (delta < 1000) return `${Math.round(delta)}ms`;
    const s = delta / 1000;
    if (s < 60) return `${s.toFixed(1)}s`;
    const m = s / 60;
    if (m < 60) return `${m.toFixed(1)}m`;
    return `${(m / 60).toFixed(1)}h`;
  }

  function fmtUptimeMs(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
    let seconds = Math.max(0, Math.floor(Number(value) / 1000));
    const hours = Math.floor(seconds / 3600);
    seconds -= hours * 3600;
    const minutes = Math.floor(seconds / 60);
    seconds -= minutes * 60;
    if (hours) return `${hours}h${String(minutes).padStart(2, "0")}m${String(seconds).padStart(2, "0")}s`;
    if (minutes) return `${minutes}m${String(seconds).padStart(2, "0")}s`;
    return `${seconds}s`;
  }

  function fmtTs(tsMs) {
    if (!tsMs) return "-";
    try {
      return new Date(Number(tsMs)).toLocaleString();
    } catch {
      return String(tsMs);
    }
  }

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function emptyNode() {
    return els.emptyTemplate.content.firstElementChild.cloneNode(true);
  }

  function setChip(el, text, tone = "") {
    el.textContent = text;
    el.classList.remove("is-ok", "is-warn", "is-bad");
    if (tone) el.classList.add(tone);
  }

  function currentPayload() {
    return state.snapshot && state.snapshot.payload ? state.snapshot.payload : null;
  }

  function availableSymbols() {
    const payload = currentPayload();
    if (!payload || !payload.market) return [];
    return Object.keys(payload.market).sort();
  }

  function activePositionRows() {
    const payload = currentPayload();
    if (!payload || !payload.positions) return [];
    const rows = [];
    const openOrders = payload.open_orders || {};
    for (const [symbol, position] of Object.entries(payload.positions)) {
      for (const pside of ["long", "short"]) {
        const side = position && typeof position === "object" ? position[pside] : null;
        if (!side || !Number(side.size)) continue;
        rows.push({
          symbol,
          pside,
          side,
          orderCount: Array.isArray(openOrders[symbol]) ? openOrders[symbol].filter((order) => order.position_side === pside).length : 0,
        });
      }
    }
    rows.sort((a, b) => Math.abs(Number(b.side.wallet_exposure || 0)) - Math.abs(Number(a.side.wallet_exposure || 0)));
    return rows;
  }

  function selectedFocusSymbol() {
    if (state.focusSymbol) return state.focusSymbol;
    const rows = activePositionRows();
    if (rows.length) return rows[0].symbol;
    const symbols = availableSymbols();
    return symbols[0] || "";
  }

  function filteredTrailingRows(focusSymbol) {
    const payload = currentPayload();
    const trailing = payload && payload.trailing ? payload.trailing : {};
    const rows = [];
    for (const [symbol, sides] of Object.entries(trailing || {})) {
      for (const pside of ["long", "short"]) {
        const side = sides && sides[pside];
        if (!side || typeof side !== "object") continue;
        for (const kind of ["entry", "close"]) {
          if (side[kind]) {
            rows.push({ symbol, pside, kind, payload: side[kind] });
          }
        }
      }
    }
    rows.sort((a, b) => {
      const focusBoost = (a.symbol === focusSymbol ? -1 : 0) - (b.symbol === focusSymbol ? -1 : 0);
      if (focusBoost !== 0) return focusBoost;
      return a.symbol.localeCompare(b.symbol) || a.kind.localeCompare(b.kind);
    });
    return rows;
  }

  function filteredEvents(focusSymbol) {
    const events = [...state.recentEvents];
    const focused = focusSymbol ? events.filter((event) => event.symbol === focusSymbol) : [];
    const others = focusSymbol ? events.filter((event) => event.symbol !== focusSymbol) : events;
    const ordered = [...focused, ...others];
    const nonBalance = ordered.filter((event) => event.kind !== "account.balance");
    const balance = ordered.find((event) => event.kind === "account.balance");
    const result = nonBalance.slice(0, 7);
    if (balance) result.push(balance);
    return result.slice(0, 8);
  }

  function filteredTicks(focusSymbol) {
    const items = Array.from(state.recentTicks.entries()).sort((a, b) => Number(b[1].ts || 0) - Number(a[1].ts || 0));
    if (!focusSymbol) return items.slice(0, 8);
    const focused = items.filter(([symbol]) => symbol === focusSymbol);
    const others = items.filter(([symbol]) => symbol !== focusSymbol);
    return [...focused, ...others].slice(0, 8);
  }

  function recentOrders(focusSymbol) {
    const payload = currentPayload();
    const recent = payload && payload.recent ? payload.recent : {};
    const merged = [];
    for (const [key, action] of [["order_executions", "executed"], ["order_cancellations", "canceled"]]) {
      const rows = Array.isArray(recent[key]) ? recent[key] : [];
      for (const entry of rows) {
        merged.push({ ...entry, action });
      }
    }
    merged.sort((a, b) => Number(b.execution_timestamp || 0) - Number(a.execution_timestamp || 0));
    const filtered = focusSymbol ? merged.filter((entry) => entry.symbol === focusSymbol).concat(merged.filter((entry) => entry.symbol !== focusSymbol)) : merged;
    return filtered.slice(0, 8);
  }

  function buildSummaryCards(payload) {
    const account = payload.account || {};
    const health = payload.health || {};
    const hsl = payload.hsl || {};
    const rows = [
      ["Equity", fmtCompact(account.equity, 2)],
      ["Balance", fmtCompact(account.balance_raw, 2)],
      ["Realized", fmtCompact(account.realized_pnl_cumsum && account.realized_pnl_cumsum.current, 2)],
      ["Loop", `${fmtCompact(health.last_loop_duration_ms, 0)} ms`],
      ["Orders", `${fmtCompact(health.orders_placed, 0)} / ${fmtCompact(health.orders_cancelled, 0)}`],
      ["Fills", fmtCompact(health.fills, 0)],
      ["Uptime", fmtUptimeMs(health.uptime_ms)],
      ["HSL", `L ${hsl.long?.tier || "-"} / S ${hsl.short?.tier || "-"}`],
    ];
    els.summaryCards.innerHTML = "";
    for (const [label, value] of rows) {
      const card = document.createElement("article");
      card.className = "stat-card";
      card.innerHTML = `<p class="label">${escapeHtml(label)}</p><p class="value">${escapeHtml(value)}</p>`;
      els.summaryCards.appendChild(card);
    }
  }

  function renderFocus(payload, focusSymbol) {
    els.focusLabel.textContent = focusSymbol || "auto";
    els.focusDetails.innerHTML = "";
    if (!focusSymbol || !payload.market || !payload.market[focusSymbol]) {
      els.focusDetails.appendChild(emptyNode());
      return;
    }
    const market = payload.market[focusSymbol] || {};
    const position = payload.positions?.[focusSymbol] || {};
    const long = position.long || {};
    const short = position.short || {};
    const detailRows = [
      [`Last`, `${fmtCompact(market.last_price)} (${fmtAgeMs(market.last_price_ts_ms)})`],
      [`Tradable`, `${market.tradable} | active=${market.active_symbol}`],
      [`Long`, `${fmtCompact(long.size)} @ ${fmtCompact(long.price)} | WE ${fmtCompact(long.wallet_exposure)} | uPnL ${fmtCompact(long.upnl, 2)}`],
      [`Short`, `${fmtCompact(short.size)} @ ${fmtCompact(short.price)} | WE ${fmtCompact(short.wallet_exposure)} | uPnL ${fmtCompact(short.upnl, 2)}`],
      [`EMA`, `lo ${fmtCompact(market.ema_bands?.long?.lower)} | hi ${fmtCompact(market.ema_bands?.long?.upper)}`],
      [`Approvals`, `L ${market.approved?.long} / S ${market.approved?.short} | ignored L ${market.ignored?.long} / S ${market.ignored?.short}`],
    ];
    for (const [label, value] of detailRows) {
      const row = document.createElement("div");
      row.className = "detail-row";
      row.innerHTML = `<p class="headline"><span>${escapeHtml(label)}</span></p><p class="subline">${escapeHtml(value)}</p>`;
      els.focusDetails.appendChild(row);
    }
  }

  function renderPositions(payload, focusSymbol) {
    const rows = activePositionRows();
    const totalLong = rows.filter((row) => row.pside === "long").reduce((acc, row) => acc + Number(row.side.wallet_exposure || 0), 0);
    const totalShort = rows.filter((row) => row.pside === "short").reduce((acc, row) => acc + Number(row.side.wallet_exposure || 0), 0);
    const twelLong = rows.find((row) => row.pside === "long")?.side.total_wallet_exposure_limit || 0;
    const twelShort = rows.find((row) => row.pside === "short")?.side.total_wallet_exposure_limit || 0;
    els.positionsMeta.textContent = `TWE L ${fmtCompact(totalLong)}/${fmtCompact(twelLong)} | S ${fmtCompact(totalShort)}/${fmtCompact(twelShort)}`;
    els.positionsBody.innerHTML = "";
    if (!rows.length) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td colspan="8"><div class="empty-state">No active positions.</div></td>`;
      els.positionsBody.appendChild(tr);
      return;
    }
    rows.sort((a, b) => {
      const focusBoost = (a.symbol === focusSymbol ? -1 : 0) - (b.symbol === focusSymbol ? -1 : 0);
      if (focusBoost !== 0) return focusBoost;
      return Math.abs(Number(b.side.wallet_exposure || 0)) - Math.abs(Number(a.side.wallet_exposure || 0));
    });
    for (const row of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml(row.symbol.replace(":USDT", "").replace(":USDC", ""))}</td>
        <td><span class="pill ${row.pside === "long" ? "is-ok" : "is-bad"}">${escapeHtml(row.pside)}</span></td>
        <td class="mono">${escapeHtml(`${fmtCompact(row.side.size)} @ ${fmtCompact(row.side.price)}`)}</td>
        <td class="mono">${escapeHtml(fmtCompact(row.side.wallet_exposure))}</td>
        <td class="mono">${escapeHtml(`${fmtPctRatio(row.side.wel_ratio)} / ${fmtPctRatio(row.side.wele_ratio)}`)}</td>
        <td class="mono">${escapeHtml(fmtPctRatio(row.side.price_action_distance))}</td>
        <td class="mono">${escapeHtml(fmtCompact(row.side.upnl, 2))}</td>
        <td class="mono">${escapeHtml(String(row.orderCount))}</td>
      `;
      els.positionsBody.appendChild(tr);
    }
  }

  function renderTrailing(payload, focusSymbol) {
    els.trailingList.innerHTML = "";
    const rows = filteredTrailingRows(focusSymbol);
    if (!rows.length) {
      els.trailingList.appendChild(emptyNode());
      return;
    }
    for (const row of rows) {
      const item = document.createElement("article");
      item.className = "stack-item";
      item.innerHTML = `
        <p class="headline">
          <span>${escapeHtml(row.symbol)} · ${escapeHtml(row.pside)} · ${escapeHtml(row.kind)}</span>
          <span class="pill ${row.payload.status === "triggered" ? "is-ok" : row.payload.status === "waiting_threshold" ? "is-warn" : ""}">${escapeHtml(row.payload.status)}</span>
        </p>
        <p class="subline mono">cur=${escapeHtml(fmtCompact(row.payload.current_price))} thr=${escapeHtml(fmtCompact(row.payload.threshold_price))} ret=${escapeHtml(fmtCompact(row.payload.retracement_price))}</p>
        <p class="minor mono">thr_met=${escapeHtml(String(row.payload.threshold_met))} ret_met=${escapeHtml(String(row.payload.retracement_met))} qty=${escapeHtml(fmtCompact(row.payload.qty))} px=${escapeHtml(fmtCompact(row.payload.price))}</p>
        <p class="minor mono">min_open=${escapeHtml(fmtCompact(row.payload.extrema?.min_since_open))} max_min=${escapeHtml(fmtCompact(row.payload.extrema?.max_since_min))} max_open=${escapeHtml(fmtCompact(row.payload.extrema?.max_since_open))} min_max=${escapeHtml(fmtCompact(row.payload.extrema?.min_since_max))}</p>
      `;
      els.trailingList.appendChild(item);
    }
  }

  function renderForager(payload) {
    const sides = payload.forager || {};
    els.foragerDetails.innerHTML = "";
    for (const pside of ["long", "short"]) {
      const side = sides[pside];
      if (!side || (!side.enabled && (!side.selected_symbols || !side.selected_symbols.length))) continue;
      const item = document.createElement("article");
      item.className = "detail-row";
      item.innerHTML = `
        <p class="headline"><span>${escapeHtml(pside)}</span><span class="pill ${side.enabled ? "is-ok" : "is-warn"}">${side.enabled ? "enabled" : "off"}</span></p>
        <p class="subline">slots ${escapeHtml(String(side.slots?.current || 0))}/${escapeHtml(String(side.slots?.max || 0))} · open ${escapeHtml(String(side.slots?.open || 0))} · next ${escapeHtml(side.next_symbol || "-")}</p>
        <p class="minor">ranking total ${escapeHtml(side.ranking?.top_total?.symbol || "-")} · vol ${escapeHtml(side.ranking?.top_volume?.symbol || "-")} · vola ${escapeHtml(side.ranking?.top_volatility?.symbol || "-")} · ema ${escapeHtml(side.ranking?.top_ema_readiness?.symbol || "-")}</p>
      `;
      els.foragerDetails.appendChild(item);
    }
    if (!els.foragerDetails.children.length) els.foragerDetails.appendChild(emptyNode());
  }

  function renderUnstuck(payload) {
    const unstuck = payload.unstuck || {};
    els.unstuckDetails.innerHTML = "";
    for (const pside of ["long", "short"]) {
      const side = unstuck.sides?.[pside];
      if (!side || (side.status === "disabled" && !side.next_symbol)) continue;
      const item = document.createElement("article");
      item.className = "detail-row";
      item.innerHTML = `
        <p class="headline"><span>${escapeHtml(pside)}</span><span class="pill ${side.status === "ok" ? "is-ok" : side.status === "disabled" ? "" : "is-warn"}">${escapeHtml(side.status || "-")}</span></p>
        <p class="subline">allowance ${escapeHtml(fmtCompact(side.allowance_live, 4))} · next ${escapeHtml(side.next_symbol || "-")}</p>
        <p class="minor">target ${escapeHtml(fmtCompact(side.next_target_price))} · dist ${escapeHtml(fmtPctRatio(side.next_target_distance_ratio))} · ema ${escapeHtml(fmtPctRatio(side.next_unstuck_trigger_distance_ratio))}</p>
      `;
      els.unstuckDetails.appendChild(item);
    }
    if (!els.unstuckDetails.children.length) els.unstuckDetails.appendChild(emptyNode());
  }

  function renderEvents(focusSymbol) {
    els.eventsList.innerHTML = "";
    const rows = filteredEvents(focusSymbol);
    if (!rows.length) {
      els.eventsList.appendChild(emptyNode());
      return;
    }
    for (const event of rows) {
      const item = document.createElement("article");
      item.className = "stack-item";
      item.innerHTML = `
        <p class="headline"><span>${escapeHtml(event.kind)}</span><span>${escapeHtml(fmtAgeMs(event.ts))}</span></p>
        <p class="subline">${escapeHtml(event.symbol || event.pside || "-")}</p>
        <p class="minor mono">${escapeHtml(JSON.stringify(event.payload || {}))}</p>
      `;
      els.eventsList.appendChild(item);
    }
  }

  function renderTicks(payload, focusSymbol) {
    els.ticksList.innerHTML = "";
    const rows = filteredTicks(focusSymbol);
    if (!rows.length) {
      els.ticksList.appendChild(emptyNode());
      return;
    }
    for (const [symbol, tick] of rows) {
      const market = payload.market?.[symbol] || {};
      const item = document.createElement("article");
      item.className = "stack-item";
      item.innerHTML = `
        <p class="headline"><span>${escapeHtml(symbol)}</span><span>${escapeHtml(fmtAgeMs(tick.ts))}</span></p>
        <p class="subline mono">last=${escapeHtml(fmtCompact(tick.payload?.last))} lo=${escapeHtml(fmtCompact(market.ema_bands?.long?.lower))} hi=${escapeHtml(fmtCompact(market.ema_bands?.long?.upper))}</p>
      `;
      els.ticksList.appendChild(item);
    }
  }

  function renderOrders(focusSymbol) {
    els.ordersList.innerHTML = "";
    const rows = recentOrders(focusSymbol);
    if (!rows.length) {
      els.ordersList.appendChild(emptyNode());
      return;
    }
    for (const row of rows) {
      const item = document.createElement("article");
      item.className = "stack-item";
      item.innerHTML = `
        <p class="headline"><span>${escapeHtml(row.action)} · ${escapeHtml(row.symbol || "-")}</span><span>${escapeHtml(fmtTs(row.execution_timestamp))}</span></p>
        <p class="subline mono">${escapeHtml(`${row.position_side || "-"} / ${row.side || "-"} / ${fmtCompact(row.qty)} @ ${fmtCompact(row.price)}`)}</p>
        <p class="minor">${escapeHtml(row.pb_order_type || row.reason || "-")}</p>
      `;
      els.ordersList.appendChild(item);
    }
  }

  function updateFocusOptions() {
    const symbols = availableSymbols();
    const current = state.focusSymbol;
    els.focusSelect.innerHTML = `<option value="">Auto</option>${symbols.map((symbol) => `<option value="${escapeHtml(symbol)}">${escapeHtml(symbol)}</option>`).join("")}`;
    els.focusSelect.value = current && symbols.includes(current) ? current : "";
  }

  function render() {
    const payload = currentPayload();
    if (!payload) return;
    const focusSymbol = selectedFocusSymbol();
    const health = payload.health || {};
    const account = payload.account || {};
    state.exchange = state.snapshot.exchange;
    state.user = state.snapshot.user;
    els.heroTitle.textContent = `${state.exchange} / ${state.user}`;
    els.heroSubtitle.textContent = `Snapshot ${state.snapshot.seq ?? "-"} · ${fmtAgeMs(state.snapshot.ts)} old · equity ${fmtCompact(account.equity, 2)} · loop ${fmtCompact(health.last_loop_duration_ms, 0)} ms`;
    setChip(els.snapshotStatus, `Snapshot: seq ${state.snapshot.seq ?? "-" } · age ${fmtAgeMs(state.snapshot.ts)}`, "is-ok");
    setChip(els.focusStatus, `Focus: ${focusSymbol || "auto"}`, focusSymbol ? "is-ok" : "");
    setChip(els.relayStatus, `Relay: ${window.location.origin}`, "");
    buildSummaryCards(payload);
    renderFocus(payload, focusSymbol);
    renderPositions(payload, focusSymbol);
    renderTrailing(payload, focusSymbol);
    renderForager(payload);
    renderUnstuck(payload);
    renderEvents(focusSymbol);
    renderTicks(payload, focusSymbol);
    renderOrders(focusSymbol);
    updateFocusOptions();
  }

  async function fetchSnapshot() {
    const params = new URLSearchParams();
    if (qs.get("exchange")) params.set("exchange", qs.get("exchange"));
    if (qs.get("user")) params.set("user", qs.get("user"));
    const response = await fetch(`/snapshot?${params.toString()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`snapshot HTTP ${response.status}`);
    state.snapshot = await response.json();
    if (!qs.get("exchange")) qs.set("exchange", state.snapshot.exchange);
    if (!qs.get("user")) qs.set("user", state.snapshot.user);
    render();
  }

  function pushEvent(message) {
    state.recentEvents = [message, ...state.recentEvents.filter((entry) => !(entry.kind === message.kind && entry.ts === message.ts && entry.symbol === message.symbol))].slice(0, 60);
  }

  function pushTick(message) {
    state.recentTicks.set(message.symbol, message);
    const sorted = Array.from(state.recentTicks.entries()).sort((a, b) => Number(b[1].ts || 0) - Number(a[1].ts || 0)).slice(0, 60);
    state.recentTicks = new Map(sorted);
  }

  function connectWs() {
    if (state.ws) state.ws.close();
    const params = new URLSearchParams();
    if (qs.get("exchange")) params.set("exchange", qs.get("exchange"));
    if (qs.get("user")) params.set("user", qs.get("user"));
    const url = new URL(`/ws?${params.toString()}`, window.location.href);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(url);
    state.ws = ws;
    setChip(els.wsStatus, "WS: connecting", "is-warn");
    ws.onopen = () => setChip(els.wsStatus, "WS: live", "is-ok");
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      state.lastWsTs = Date.now();
      if (message.type === "snapshot") {
        state.snapshot = message;
      } else if (message.type === "event") {
        pushEvent(message);
      } else if (message.type === "history" && message.stream === "price_ticks") {
        pushTick(message);
      }
      render();
    };
    ws.onclose = () => {
      setChip(els.wsStatus, "WS: reconnecting", "is-warn");
      if (state.reconnectTimer) window.clearTimeout(state.reconnectTimer);
      state.reconnectTimer = window.setTimeout(connectWs, 1000);
    };
    ws.onerror = () => setChip(els.wsStatus, "WS: error", "is-bad");
  }

  function bindUi() {
    els.focusSelect.addEventListener("change", () => {
      state.focusSymbol = els.focusSelect.value || "";
      render();
    });
    els.refreshButton.addEventListener("click", async () => {
      try {
        await fetchSnapshot();
      } catch (error) {
        setChip(els.snapshotStatus, `Snapshot: ${error.message}`, "is-bad");
      }
    });
  }

  async function start() {
    bindUi();
    try {
      await fetchSnapshot();
      connectWs();
      state.snapshotTimer = window.setInterval(() => {
        fetchSnapshot().catch((error) => setChip(els.snapshotStatus, `Snapshot: ${error.message}`, "is-bad"));
      }, 3000);
    } catch (error) {
      setChip(els.snapshotStatus, `Snapshot: ${error.message}`, "is-bad");
      els.heroTitle.textContent = "Snapshot load failed";
      els.heroSubtitle.textContent = String(error.message || error);
    }
  }

  start();
})();
