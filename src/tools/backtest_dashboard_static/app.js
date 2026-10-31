"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const colors = [
    "var(--blue)",
    "var(--orange)",
    "var(--purple)",
    "var(--green)",
    "var(--red)",
  ];
  const state = {
    tab: "equity",
    offset: 0,
    summary: null,
    equity: null,
    hsl: null,
    requestId: 0,
  };
  const number = (value) =>
    value == null
      ? "—"
      : new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 }).format(
          value,
        );
  const pct = (value) => (value == null ? "—" : (value * 100).toFixed(3) + "%");
  const axisNumber = (value) =>
    new Intl.NumberFormat("en-US", {
      notation: "compact",
      maximumSignificantDigits: 4,
    }).format(value);
  const utc = (value) =>
    new Date(value).toISOString().replace("T", " ").slice(0, 19) + " UTC";
  function element(name, attrs = {}) {
    const el = document.createElementNS(NS, name);
    for (const [key, value] of Object.entries(attrs))
      el.setAttribute(key, value);
    return el;
  }
  function text(parent, x, y, value, anchor = "start") {
    const el = element("text", { x, y, "text-anchor": anchor });
    el.textContent = value;
    parent.append(el);
    return el;
  }
  function query(extra = {}) {
    const q = new URLSearchParams(extra);
    for (const key of ["start", "end"])
      if ($(key).value) q.set(key, String(Date.parse($(key).value + "Z")));
    return q.toString();
  }
  async function request(route, extra) {
    const response = await fetch("/api/" + route + "?" + query(extra));
    if (!response.ok)
      throw new Error((await response.json()).error || "Unable to load result");
    return response.json();
  }
  function chart(
    id,
    series,
    { ylabel = "", scale = "linear", threshold = null, stateAxis = false } = {},
  ) {
    const host = $(id);
    host.replaceChildren();
    const items = series.filter((s) => s.x.length);
    if (!items.length) {
      const p = document.createElement("p");
      p.textContent = "No observations in this interval.";
      host.append(p);
      return;
    }
    const legend = document.createElement("div");
    legend.className = "legend";
    const legendItems = items.map((s, i) => ({
      name: s.name,
      color: s.color || colors[i % colors.length],
    }));
    if (threshold != null)
      legendItems.push({ name: "RED threshold", color: "var(--red)" });
    for (const item of legendItems) {
      const label = document.createElement("span");
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.backgroundColor = item.color;
      label.append(swatch, document.createTextNode(item.name));
      legend.append(label);
    }
    host.append(legend);
    const w = host.clientWidth,
      h = host.clientHeight - legend.offsetHeight - 10,
      m = { left: 74, right: 16, top: 18, bottom: 42 };
    if (w < 150 || h < 100) return;
    const allX = items.flatMap((s) => [s.x[0], s.x[s.x.length - 1]]),
      x0 = Math.min(...allX),
      x1 = Math.max(...allX);
    const values = items.flatMap((s) =>
      s.y.filter((v) => v != null && Number.isFinite(v)),
    );
    if (threshold != null) values.push(threshold);
    if (!values.length) return;
    let ymin = values.reduce((a, b) => Math.min(a, b), Infinity),
      ymax = values.reduce((a, b) => Math.max(a, b), -Infinity);
    const useLog = scale === "log" && ymin > 0;
    if (scale === "log" && !useLog)
      $("status").textContent =
        "Linear scale includes nonpositive observations.";
    const transform = (v) => (useLog ? Math.log(v) : v);
    let lo = transform(ymin),
      hi = transform(ymax),
      pad = (hi - lo || Math.abs(hi) * 0.1 || 1) * 0.08;
    if (stateAxis) {
      lo = -0.15;
      hi = 2.15;
      pad = 0;
    }
    lo -= pad;
    hi += pad;
    const x = (v) =>
      m.left + ((v - x0) / (x1 - x0 || 1)) * (w - m.left - m.right);
    const y = (v) =>
      h - m.bottom - ((transform(v) - lo) / (hi - lo)) * (h - m.top - m.bottom);
    const svg = element("svg", {
      viewBox: `0 0 ${w} ${h}`,
      role: "img",
      "aria-label": ylabel,
    });
    svg.style.height = h + "px";
    host.append(svg);
    const ticks = stateAxis
      ? [0, 1, 2]
      : Array.from({ length: 5 }, (_, i) =>
          useLog
            ? Math.exp(lo + ((hi - lo) * i) / 4)
            : lo + ((hi - lo) * i) / 4,
        );
    for (const value of ticks) {
      const yy = y(value);
      svg.append(
        element("line", {
          x1: m.left,
          x2: w - m.right,
          y1: yy,
          y2: yy,
          class: "grid",
        }),
      );
      text(
        svg,
        m.left - 9,
        yy + 4,
        stateAxis ? ["Normal", "Panic", "Cooldown"][value] : axisNumber(value),
        "end",
      );
    }
    const count = w < 420 ? 3 : 5;
    for (let i = 0; i < count; i++) {
      const value = x0 + ((x1 - x0) * i) / (count - 1),
        xx = x(value);
      text(
        svg,
        xx,
        h - 14,
        new Date(value).toISOString().slice(0, 10),
        i === 0 ? "start" : i === count - 1 ? "end" : "middle",
      );
    }
    text(svg, m.left, m.top - 6, ylabel);
    if (threshold != null)
      svg.append(
        element("line", {
          x1: m.left,
          x2: w - m.right,
          y1: y(threshold),
          y2: y(threshold),
          stroke: "var(--red)",
          "stroke-dasharray": "5 4",
        }),
      );
    items.forEach((s, i) => {
      let path = "",
        previous = null;
      for (let k = 0; k < s.x.length; k++) {
        const value = s.y[k];
        if (value == null || !Number.isFinite(value)) {
          previous = null;
          continue;
        }
        const xx = x(s.x[k]),
          yy = y(value);
        if (s.gaps && s.gaps[k]) previous = null;
        path +=
          previous == null
            ? `M${xx},${yy}`
            : s.step
              ? `H${xx}V${yy}`
              : `L${xx},${yy}`;
        previous = k;
      }
      svg.append(
        element("path", {
          d: path,
          class: "series",
          stroke: s.color || colors[i % colors.length],
        }),
      );
    });
    const guide = element("line", {
      y1: m.top,
      y2: h - m.bottom,
      stroke: "var(--muted)",
      visibility: "hidden",
    });
    svg.append(guide);
    const overlay = element("rect", {
      x: m.left,
      y: m.top,
      width: Math.max(1, w - m.left - m.right),
      height: h - m.top - m.bottom,
      fill: "transparent",
    });
    svg.append(overlay);
    const tooltip = $("tooltip");
    function show(event) {
      const box = svg.getBoundingClientRect(),
        px = Math.max(m.left, Math.min(w - m.right, event.clientX - box.left));
      const time = x0 + ((px - m.left) / (w - m.left - m.right)) * (x1 - x0);
      guide.setAttribute("x1", px);
      guide.setAttribute("x2", px);
      guide.setAttribute("visibility", "visible");
      tooltip.replaceChildren();
      const heading = document.createElement("span");
      heading.textContent = "Nearest saved observations";
      tooltip.append(heading);
      for (const s of items) {
        let a = 0,
          b = s.x.length;
        while (a < b) {
          const k = (a + b) >> 1;
          if (s.x[k] < time) a = k + 1;
          else b = k;
        }
        const i = Math.max(
          0,
          Math.min(
            s.x.length - 1,
            a > 0 &&
              Math.abs(s.x[a - 1] - time) <
                Math.abs((s.x[a] ?? Infinity) - time)
              ? a - 1
              : a,
          ),
        );
        const row = document.createElement("span");
        row.textContent =
          s.name +
          ": " +
          (stateAxis
            ? ["Normal", "Panic", "Cooldown"][s.y[i]]
            : number(s.y[i])) +
          " · " +
          utc(s.x[i]);
        tooltip.append(row);
      }
      tooltip.hidden = false;
      tooltip.style.left =
        Math.max(
          8,
          Math.min(
            event.clientX + 12,
            window.innerWidth - tooltip.offsetWidth - 8,
          ),
        ) + "px";
      tooltip.style.top =
        Math.max(8, event.clientY - tooltip.offsetHeight - 12) + "px";
    }
    overlay.addEventListener("pointermove", show);
    overlay.addEventListener("click", show);
    overlay.addEventListener("pointerleave", () => {
      tooltip.hidden = true;
      guide.setAttribute("visibility", "hidden");
    });
  }
  function drawEquity() {
    const d = state.equity;
    if (!d) return;
    chart(
      "equity-chart",
      [
        { name: "Strategy equity", x: d.timestamp, y: d.strategy_equity },
        { name: "Account balance", x: d.timestamp, y: d.usd_total_balance },
      ],
      { ylabel: "USD", scale: $("scale").value },
    );
    chart(
      "drawdown-chart",
      [
        {
          name: "Drawdown",
          x: d.timestamp,
          y: d.drawdown_fraction.map((v) => (v == null ? null : v * 100)),
          color: "var(--red)",
        },
      ],
      { ylabel: "Drawdown (%)" },
    );
    $("btc-panel").hidden = !$("btc").checked;
    if ($("btc").checked)
      chart(
        "btc-chart",
        [{ name: "Equity in BTC", x: d.timestamp, y: d.btc_total_equity }],
        { ylabel: "BTC", scale: $("scale").value },
      );
  }
  function drawHsl() {
    const view = state.hsl;
    if (!view) {
      $("hsl-chart").replaceChildren();
      $("hsl-policy").textContent = state.summary?.hsl
        ? "Loading native HSL observations…"
        : "HSL is disabled in this result.";
      return;
    }
    const groups = view.scopes,
      series = [];
    if (groups.length === 1) {
      const g = groups[0],
        p = g.policy;
      $("hsl-policy").textContent =
        `${g.name} · threshold ${pct(p.drawdown_threshold_fraction)} · memory ${number(p.memory_ema_span_minutes)} min · EMA ${number(p.drawdown_fraction_ema_span_minutes)} min · cooldown ${number(p.cooldown_minutes_after_red)} min`;
      for (const [key, name] of [
        ["signal_raw_fraction", "Raw drawdown"],
        ["signal_ema_fraction", "Drawdown EMA"],
        ["trigger_signal_fraction", "Native trigger signal"],
      ])
        if (g[key])
          series.push({
            name,
            x: g.timestamp,
            y: g[key].map((v) => v * 100),
            gaps: g.gap_before,
          });
      chart("hsl-chart", series, {
        ylabel: "Drawdown signal (%)",
        threshold: p.drawdown_threshold_fraction * 100,
      });
    } else {
      $("hsl-policy").textContent = view.trace_available
        ? "Active controller distances to RED; negative values exceed the threshold."
        : "Native trace is unavailable in this legacy result.";
      for (const g of groups)
        series.push({
          name: g.name,
          x: g.timestamp,
          y: g.signal_raw_fraction.map(
            (v, i) =>
              (g.policy.drawdown_threshold_fraction -
                (g.trigger_signal_fraction?.[i] ??
                  Math.min(v, g.signal_ema_fraction[i]))) *
              100,
          ),
          gaps: g.gap_before,
        });
      chart("hsl-chart", series, {
        ylabel: "Distance below RED (pp)",
        threshold: 0,
      });
    }
    chart(
      "state-chart",
      groups.map((g) => ({
        name: g.name,
        x: g.state_timestamp,
        y: g.state_action,
        step: true,
      })),
      { ylabel: "Native controller state", stateAxis: true },
    );
  }
  function table(host, headers, rows) {
    const table = document.createElement("table"),
      head = document.createElement("thead"),
      tr = document.createElement("tr");
    headers.forEach((name) => {
      const th = document.createElement("th");
      th.textContent = name;
      tr.append(th);
    });
    head.append(tr);
    table.append(head);
    const body = document.createElement("tbody");
    rows.forEach((values) => {
      const row = document.createElement("tr");
      values.forEach((value) => {
        const cell = document.createElement("td");
        if (value instanceof Node) cell.append(value);
        else cell.textContent = value;
        row.append(cell);
      });
      body.append(row);
    });
    table.append(body);
    host.replaceChildren(table);
  }
  function episodes() {
    const h = state.summary.hsl,
      rows = h && h.execution_episodes;
    if (rows == null) {
      $("episodes").textContent = h
        ? "Per-episode detail is unavailable in this legacy result."
        : "HSL disabled.";
      return;
    }
    if (!rows.length) {
      $("episodes").textContent = "No executed HSL stops.";
      return;
    }
    table(
      $("episodes"),
      [
        "Controller",
        "First fill (UTC)",
        "Loss incl. fees",
        "Loss / account balance",
        "Status",
        "",
      ],
      rows.map((ep) => {
        const name =
          ep.side == null
            ? "portfolio"
            : (ep.coin == null ? "" : h.coins[ep.coin] + ":") +
              ["long", "short"][ep.side];
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "Inspect";
        button.addEventListener("click", () => {
          $("scope").value = name;
          $("start").value = new Date(
            (ep.trigger_timestamp_ms ?? ep.first_fill_timestamp_ms) -
              24 * 3600000,
          )
            .toISOString()
            .slice(0, 16);
          $("end").value =
            ep.finished_timestamp_ms == null
              ? ""
              : new Date(ep.finished_timestamp_ms + 6 * 3600000)
                  .toISOString()
                  .slice(0, 16);
          load();
        });
        return [
          name,
          ep.first_fill_timestamp_ms == null
            ? "—"
            : utc(ep.first_fill_timestamp_ms),
          number(ep.loss_usd),
          pct(ep.loss_fraction),
          ep.finished_timestamp_ms == null ? "Unfinished" : "Finished",
          button,
        ];
      }),
    );
  }
  async function load() {
    const tab = state.tab;
    const requestId = ++state.requestId;
    $("status").textContent = "Loading saved observations…";
    try {
      if (tab === "equity") {
        const data = await request("equity", {
          btc: $("btc").checked ? "1" : "0",
        });
        if (requestId !== state.requestId) return;
        state.equity = data;
        $("status").textContent =
          number(state.equity.observations) +
          " native equity observations in this interval.";
        drawEquity();
      } else if (tab === "hsl") {
        const data = await request("hsl", { scope: $("scope").value });
        if (requestId !== state.requestId) return;
        state.hsl = data;
        $("status").textContent = "Saved native HSL observations.";
        drawHsl();
      } else {
        const data = await request("fills", {
          coin: $("coin").value,
          side: $("side").value,
          offset: String(state.offset),
          limit: "200",
        });
        if (requestId !== state.requestId) return;
        table(
          $("fills-table"),
          [
            "Time",
            "Coin",
            "Order type",
            "Signed quantity",
            "Price",
            "Net PnL incl. fee",
          ],
          data.rows.map((r) => [
            r.timestamp,
            r.coin,
            r.type,
            number(r.qty),
            number(r.price),
            number(r.pnl + r.fee_paid),
          ]),
        );
        $("fill-count").textContent =
          `${data.rows.length ? data.offset + 1 : 0}–${data.offset + data.rows.length} of ${number(data.total)}`;
        $("previous").disabled = state.offset === 0;
        $("next").disabled = state.offset + data.rows.length >= data.total;
        $("status").textContent = "Exact saved fill ledger.";
      }
    } catch (error) {
      if (requestId === state.requestId)
        $("status").textContent = error.message;
    }
  }
  document.querySelectorAll("[data-tab]").forEach((button) =>
    button.addEventListener("click", () => {
      state.tab = button.dataset.tab;
      document
        .querySelectorAll("[data-tab]")
        .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      for (const key of ["equity", "hsl", "fills"])
        $(key + "-panel").hidden = key !== state.tab;
      load();
    }),
  );
  $("apply").addEventListener("click", () => {
    state.offset = 0;
    load();
  });
  $("scale").addEventListener("change", drawEquity);
  $("btc").addEventListener("change", load);
  $("scope").addEventListener("change", load);
  for (const key of ["coin", "side"])
    $(key).addEventListener("change", () => {
      state.offset = 0;
      load();
    });
  $("previous").addEventListener("click", () => {
    state.offset = Math.max(0, state.offset - 200);
    load();
  });
  $("next").addEventListener("click", () => {
    state.offset += 200;
    load();
  });
  new ResizeObserver(() => {
    if (state.tab === "equity") drawEquity();
    else if (state.tab === "hsl") drawHsl();
  }).observe(document.querySelector("main"));
  request("summary")
    .then((summary) => {
      state.summary = summary;
      const a = summary.analysis;
      $("period").textContent =
        summary.dataset.coins.join(", ") +
        " · " +
        a.effective_start_date +
        " → " +
        a.effective_end_date;
      for (const [label, value] of [
        ["CAGR", pct(a.cagr_fraction_strategy_equity)],
        ["Maximum drawdown", pct(a.drawdown_max_fraction_strategy_equity)],
        ["Sharpe", number(a.sharpe_ratio_strategy_equity)],
        ["Executed HSL stops", number(a.hsl_execution_episode_count)],
      ]) {
        const card = document.createElement("div");
        card.className = "metric";
        const caption = document.createElement("span"),
          figure = document.createElement("strong");
        caption.textContent = label;
        figure.textContent = value;
        card.append(caption, figure);
        $("metrics").append(card);
      }
      for (const coin of summary.dataset.coins) {
        const option = document.createElement("option");
        option.value = coin;
        option.textContent = coin;
        $("coin").append(option);
      }
      for (const scope of summary.hsl?.scopes || []) {
        const option = document.createElement("option");
        option.value = scope.name;
        option.textContent = scope.name;
        $("scope").append(option);
      }
      episodes();
      load();
    })
    .catch((error) => {
      $("status").textContent = error.message;
    });
})();
