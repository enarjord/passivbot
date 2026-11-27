"use strict";
// Static fills report. DATA is embedded by fills_report.render_fills_html.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const HOUR = 3600e3, DAY = 864e5, WEEK = 7 * DAY;
  const GLYPH_MODE_MAX_FILLS = 2000;
  const DENSE_FAMILY_LIMIT = 150;
  const MAX_VISIBLE_BARS = 900;
  const RISK = new Set(["unstuck", "auto_reduce", "panic"]);
  const TAKE_PROFIT = new Set(["grid", "trailing", "ema_anchor"]);
  const FAMILY_LABEL = { initial: "initial", grid: "grid", trailing: "trailing", ema_anchor: "EMA anchor", unstuck: "unstuck", auto_reduce: "auto-reduce", panic: "HSL panic" };
  const $ = (id) => document.getElementById(id);

  // ---------- data model
  const coins = DATA.coins;
  const kinds = DATA.types.map((type) => {
    const parts = type.split("_");
    const mechanism = parts.slice(1, -1).join("_");
    const family = ["ema_anchor", "auto_reduce", "initial", "grid", "trailing", "unstuck", "panic"].find((f) => mechanism === f || mechanism.startsWith(f + "_")) || mechanism;
    const action = parts[0];
    const intent = action === "entry" ? "entry" : RISK.has(family) ? "risk" : TAKE_PROFIT.has(family) ? "take_profit" : "other";
    return { type, action, family, mechanism, side: parts[parts.length - 1], intent };
  });
  const F = DATA.fills;
  const fills = F.t.map((t, i) => {
    const k = kinds[F.type[i]];
    return { i, t, coin: coins[F.coin[i]], kind: k, side: k.side, price: F.price[i], qty: F.qty[i], pnl: F.pnl[i] || 0, pf: F.pnl_fraction[i] || 0, psize: F.psize[i], pprice: F.pprice[i], we: F.we[i] };
  });
  const E = DATA.episodes;
  const episodes = E.start_ms.map((t0, i) => ({ i, coin: coins[E.coin[i]], side: E.pside[i], t0, t1: E.end_ms[i], entries: E.entries[i], closes: E.closes[i], pnl: E.pnl[i] || 0, pf: E.pnl_fraction[i] || 0, maxWe: E.max_we[i], exit: E.exit_family[i], finished: E.finished[i], first: E.first_row[i], last: E.last_row[i] }));
  const sidesTraded = [...new Set(fills.map((f) => f.side))];
  const fillsByCoin = {};
  fills.forEach((f) => (fillsByCoin[f.coin] = fillsByCoin[f.coin] || []).push(f));

  // ---------- helpers
  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  const clear = (svg) => { while (svg.firstChild) svg.removeChild(svg.firstChild); };
  const width = (svg) => Math.max(240, svg.getBoundingClientRect().width || svg.parentNode.getBoundingClientRect().width);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const usd = (v) => (v < 0 ? "−" : "") + "$" + Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: Math.abs(v) < 10 ? 2 : 0 });
  const usdShort = (v) => { const a = Math.abs(v); const s = a >= 1e9 ? (a / 1e9).toFixed(1) + "B" : a >= 1e6 ? (a / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M" : a >= 1e3 ? (a / 1e3).toFixed(a >= 1e5 ? 0 : 1) + "k" : a.toFixed(0); return (v < 0 ? "−" : "") + "$" + s; };
  const pct = (fraction, digits = 2) => { const v = fraction * 100; return (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(digits) + "%"; };
  const num = (v) => (v === null || v === undefined ? "–" : Math.abs(v) >= 1000 ? v.toLocaleString("en-US", { maximumFractionDigits: 2 }) : String(+(+v).toPrecision(6)));
  const pad2 = (n) => String(n).padStart(2, "0");
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const dateStr = (t) => { const d = new Date(t); return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`; };
  const timeStr = (t) => { const d = new Date(t); return `${dateStr(t)} ${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`; };
  const shortDate = (t) => { const d = new Date(t); return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`; };
  const dur = (ms) => { const h = ms / HOUR; return h < 1 ? Math.round(h * 60) + "m" : h < 48 ? h.toFixed(1) + "h" : (h / 24).toFixed(1) + "d"; };
  const intervalLabel = (ms) => (ms >= DAY && ms % DAY === 0 ? ms / DAY + "d" : ms >= HOUR && ms % HOUR === 0 ? ms / HOUR + "h" : ms / 60e3 + " min");
  // Iterative extrema: spreading large arrays into Math.min/max overflows the argument limit.
  const minOf = (arr, init = Infinity) => { let m = init; for (const v of arr) if (v < m) m = v; return m; };
  const maxOf = (arr, init = -Infinity) => { let m = init; for (const v of arr) if (v > m) m = v; return m; };
  const sideColor = (s) => (s === "long" ? "var(--long)" : "var(--short)");
  const pnlColor = (v) => (v >= 0 ? "var(--win)" : "var(--loss)");
  function niceTicks(lo, hi, n) {
    if (!(hi > lo)) return [lo];
    const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  function timeTicks(t0, t1, maxN) {
    const span = t1 - t0;
    const steps = [HOUR, 3 * HOUR, 6 * HOUR, 12 * HOUR, DAY, 2 * DAY, WEEK, 14 * DAY, 30 * DAY, 91 * DAY, 182 * DAY, 365 * DAY];
    const step = steps.find((s) => span / s <= maxN) || 365 * DAY;
    const out = [];
    if (step >= 30 * DAY) {
      const months = Math.round(step / (30 * DAY));
      const d = new Date(t0);
      let y = d.getUTCFullYear(), m = d.getUTCMonth() + 1;
      m = Math.ceil(m / months) * months;
      for (;;) { const t = Date.UTC(y + Math.floor(m / 12), m % 12, 1); if (t > t1) break; if (t >= t0) out.push(t); m += months; }
    } else {
      for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) out.push(t);
    }
    const fmt = step >= 365 * DAY ? (t) => String(new Date(t).getUTCFullYear()) : step >= 30 * DAY ? (t) => { const d = new Date(t); return d.getUTCMonth() === 0 ? String(d.getUTCFullYear()) : `${MONTHS[d.getUTCMonth()]} ${String(d.getUTCFullYear()).slice(2)}`; } : step >= DAY ? shortDate : (t) => { const d = new Date(t); return d.getUTCHours() === 0 ? shortDate(t) : `${pad2(d.getUTCHours())}:00`; };
    return out.map((t) => [t, fmt(t)]);
  }
  function drawTimeAxis(svg, x, t0, t1, y, w, maxX = Infinity) {
    const n = Math.max(2, Math.floor(w / 90));
    timeTicks(t0, t1, n).filter(([t]) => x(t) <= maxX).forEach(([t, label]) => el("text", { x: x(t), y, "text-anchor": "middle" }, svg).textContent = label);
  }

  // tooltips with one delegated listener per group
  const tip = $("tip");
  function showTip(ev, html) {
    tip.innerHTML = html;
    tip.style.opacity = 1;
    const r = tip.getBoundingClientRect();
    tip.style.left = Math.min(ev.clientX + 14, innerWidth - r.width - 8) + "px";
    tip.style.top = (ev.clientY + 14 + r.height > innerHeight ? ev.clientY - r.height - 10 : ev.clientY + 14) + "px";
  }
  const hideTip = () => (tip.style.opacity = 0);
  function delegate(group, lookup, onClick) {
    group.addEventListener("mousemove", (ev) => { const k = ev.target.getAttribute && ev.target.getAttribute("data-k"); if (k === null) return hideTip(); const html = lookup(k); if (html) showTip(ev, html); });
    group.addEventListener("mouseleave", hideTip);
    if (onClick) group.addEventListener("click", (ev) => { const k = ev.target.getAttribute && ev.target.getAttribute("data-k"); if (k !== null) onClick(k); });
  }

  function glyph(g, f, x, y, r) {
    const k = f.kind;
    if (k.action === "entry") {
      const h = r * 1.15, up = f.side === "long";
      const pts = up ? `${x},${y - h} ${x - r},${y + h * 0.7} ${x + r},${y + h * 0.7}` : `${x},${y + h} ${x - r},${y - h * 0.7} ${x + r},${y - h * 0.7}`;
      return el("polygon", { points: pts, fill: sideColor(f.side), stroke: "var(--card)", "stroke-width": 1.2 }, g);
    }
    if (k.intent === "risk") {
      const d = r * 1.25;
      return el("polygon", { points: `${x},${y - d} ${x + d},${y} ${x},${y + d} ${x - d},${y}`, fill: pnlColor(f.pnl), stroke: "var(--card)", "stroke-width": 1.2 }, g);
    }
    return el("circle", { cx: x, cy: y, r: r * 0.9, fill: pnlColor(f.pnl), stroke: "var(--card)", "stroke-width": 1.2 }, g);
  }
  const glyphSvg = (inner) => `<svg width="14" height="14" viewBox="-7 -7 14 14" aria-hidden="true">${inner}</svg>`;
  const G = {
    longEntry: glyphSvg('<polygon points="0,-5 -4.5,3.5 4.5,3.5" fill="var(--long)"/>'),
    shortEntry: glyphSvg('<polygon points="0,5 -4.5,-3.5 4.5,-3.5" fill="var(--short)"/>'),
    tpWin: glyphSvg('<circle r="4.2" fill="var(--win)"/>'),
    tpLoss: glyphSvg('<circle r="4.2" fill="var(--loss)"/>'),
    riskWin: glyphSvg('<polygon points="0,-5.5 5.5,0 0,5.5 -5.5,0" fill="var(--win)"/>'),
    riskLoss: glyphSvg('<polygon points="0,-5.5 5.5,0 0,5.5 -5.5,0" fill="var(--loss)"/>'),
  };
  function glyphLegend() {
    const items = [];
    if (sidesTraded.includes("long")) items.push([G.longEntry, "long entry"]);
    if (sidesTraded.includes("short")) items.push([G.shortEntry, "short entry"]);
    items.push([G.tpWin + G.tpLoss, "take-profit close (gain / loss)"]);
    if (kinds.some((k) => k.intent === "risk")) items.push([G.riskWin + G.riskLoss, "risk close: unstuck, auto-reduce, HSL panic"]);
    return items.map(([g, l]) => `<span>${g}${l}</span>`).join("");
  }
  function fillTip(f) {
    const k = f.kind;
    return `<b>${esc(f.coin)} · ${f.side} ${FAMILY_LABEL[k.family] || k.family} ${k.action}</b><br><span class="mono">${esc(k.type)}</span><br>${timeStr(f.t)} UTC<br>price ${num(f.price)} · qty ${num(Math.abs(f.qty))}<br>position ${num(f.psize)} @ ${num(f.pprice)}<br>wallet exposure ${f.we.toFixed(4)}${k.action === "close" ? `<br>PnL <span class="${f.pnl >= 0 ? "pos" : "neg"}">${usd(f.pnl)}</span> (${pct(f.pf, 3)} of balance)` : ""}`;
  }

  // ---------- header
  function renderHeader() {
    $("title").textContent = DATA.title;
    const filt = DATA.filters, notes = [];
    if (filt.coins.length) notes.push("coins " + filt.coins.join(", "));
    if (filt.start_ms !== null || filt.end_ms !== null) notes.push("focused window");
    $("period").textContent = `${dateStr(DATA.start_ms)} → ${dateStr(DATA.end_ms)} · ${coins.length} coin${coins.length === 1 ? "" : "s"} · ${sidesTraded.join(" & ") || "no fills"}${notes.length ? " · " + notes.join(" · ") : ""}`;
    const eq = DATA.equity.equity, closes = fills.filter((f) => f.kind.action === "close");
    const risk = closes.filter((f) => f.kind.intent === "risk").length;
    const days = Math.max(1, (DATA.end_ms - DATA.start_ms) / DAY);
    const holds = episodes.map((e) => e.t1 - e.t0).sort((a, b) => a - b);
    const items = [];
    if (eq.length > 1) items.push([eq[eq.length - 1] / eq[0] >= 10 ? (eq[eq.length - 1] / eq[0]).toFixed(0) + "×" : pct(eq[eq.length - 1] / eq[0] - 1, 1), "equity change"]);
    const focused = filt.coins.length || filt.start_ms !== null || filt.end_ms !== null;
    if (DATA.analysis.drawdown_max_fraction_strategy_equity !== undefined) items.push([(DATA.analysis.drawdown_max_fraction_strategy_equity * 100).toFixed(1) + "%", focused ? "max drawdown (whole run)" : "max drawdown"]);
    items.push([fills.length.toLocaleString(), `fills · ${(fills.length / days).toFixed(1)}/day`]);
    items.push([episodes.length.toLocaleString(), "position episodes"]);
    if (holds.length) items.push([dur(holds[Math.floor(holds.length / 2)]), "median hold"]);
    if (closes.length) items.push([Math.round((risk / closes.length) * 100) + "%", "closes via risk exits"]);
    $("kpis").innerHTML = items.map(([b, s]) => `<div class="kpi"><b>${b}</b><span>${s}</span></div>`).join("");
  }

  // ---------- equity
  function renderEquity() {
    const svg = $("equity"); clear(svg);
    const t = DATA.equity.t, v = DATA.equity.equity;
    if (t.length < 2) { el("text", { x: 0, y: 20 }, svg).textContent = "No equity observations."; return; }
    const w = width(svg), h = +svg.getAttribute("height"), L = 54, R = 8, T = 8, B = 18;
    const finite = v.filter((a) => a !== null);
    const mn = minOf(finite), mx = maxOf(finite), log = mn > 0 && mx / mn > 10;
    const f = log ? Math.log10 : (a) => a;
    const lo = log ? f(mn) : mn - (mx - mn) * 0.05, hi = log ? f(mx) : mx + (mx - mn) * 0.05;
    const t0 = t[0], t1 = t[t.length - 1];
    const x = (a) => L + ((a - t0) / (t1 - t0)) * (w - L - R), y = (a) => T + (1 - (f(a) - lo) / (hi - lo || 1)) * (h - T - B);
    const ticks = log ? Array.from({ length: Math.floor(hi) - Math.ceil(lo) + 1 }, (_, i) => 10 ** (Math.ceil(lo) + i)) : niceTicks(mn, mx, 3);
    if (log && ticks.length < 2) [mn, mx].forEach((a) => ticks.push(a));
    ticks.forEach((a) => { el("line", { class: "gl", x1: L, x2: w - R, y1: y(a), y2: y(a) }, svg); el("text", { x: L - 6, y: y(a) + 3, "text-anchor": "end" }, svg).textContent = usdShort(a); });
    let d = "";
    t.forEach((a, i) => { if (v[i] !== null) d += (d ? "L" : "M") + x(a).toFixed(1) + "," + y(v[i]).toFixed(1); });
    el("path", { d, fill: "none", stroke: "var(--fg)", "stroke-width": 1.5, "stroke-linejoin": "round" }, svg);
    el("text", { class: "title", x: L + 4, y: T + 10 }, svg).textContent = log ? `log scale · ${Math.round(mx / mn)}× range` : "linear scale";
    drawTimeAxis(svg, x, t0, t1, h - 4, w);
    const ch = el("line", { y1: T, y2: h - B, stroke: "var(--ring)", opacity: 0 }, svg);
    const ov = el("rect", { x: L, y: T, width: w - L - R, height: h - T - B, fill: "transparent" }, svg);
    ov.addEventListener("mousemove", (ev) => {
      const r = svg.getBoundingClientRect(), tt = t0 + ((ev.clientX - r.left - L) / (w - L - R)) * (t1 - t0);
      let i = t.findIndex((a) => a >= tt); if (i < 0) i = t.length - 1;
      ch.setAttribute("x1", x(t[i])); ch.setAttribute("x2", x(t[i])); ch.setAttribute("opacity", 1);
      showTip(ev, `<b>${timeStr(t[i])} UTC</b><br>equity ${usd(v[i])}<br>balance ${usd(DATA.equity.balance[i])}`);
    });
    ov.addEventListener("mouseleave", () => { ch.setAttribute("opacity", 0); hideTip(); });
  }

  // ---------- fingerprint
  function renderFingerprint() {
    const svg = $("fingerprint"); clear(svg);
    if (!episodes.length) { el("text", { x: 0, y: 20 }, svg).textContent = "No position episodes."; return; }
    const w = width(svg), h = +svg.getAttribute("height"), L = 52, R = 8, T = 6, B = 18;
    const holds = episodes.map((e) => Math.max(e.t1 - e.t0, 60e3) / HOUR);
    const xmin = Math.min(0.05, minOf(holds)), xmax = Math.max(24 * 30, maxOf(holds) * 1.2);
    const ys = episodes.map((e) => e.pf).sort((a, b) => a - b), q = (p) => ys[Math.floor(p * (ys.length - 1))];
    const m = Math.max(Math.abs(q(0.003)), Math.abs(q(0.997)), 1e-4) * 1.08;
    const x = (hh) => L + ((Math.log10(hh) - Math.log10(xmin)) / (Math.log10(xmax) - Math.log10(xmin))) * (w - L - R);
    const y = (v) => T + (1 - (Math.max(-m, Math.min(m, v)) + m) / (2 * m)) * (h - T - B);
    [[1 / 60, "1m"], [1, "1h"], [24, "1d"], [168, "1w"], [720, "1mo"], [8760, "1y"]].filter(([hh]) => hh >= xmin && hh <= xmax).forEach(([hh, l]) => { el("line", { class: "gl", x1: x(hh), x2: x(hh), y1: T, y2: h - B }, svg); el("text", { x: x(hh), y: h - 4, "text-anchor": "middle" }, svg).textContent = l; });
    el("line", { class: "zero", x1: L, x2: w - R, y1: y(0), y2: y(0) }, svg);
    niceTicks(-m * 100, m * 100, 4).forEach((v) => el("text", { x: L - 6, y: y(v / 100) + 3, "text-anchor": "end" }, svg).textContent = (v > 0 ? "+" : "") + v + "%");
    // Bound the drawn population; extremes are always kept.
    let shown = episodes;
    if (episodes.length > 6000) {
      const keep = new Set();
      [...episodes].sort((a, b) => a.pf - b.pf).slice(0, 300).forEach((e) => keep.add(e.i));
      [...episodes].sort((a, b) => b.pf - a.pf).slice(0, 300).forEach((e) => keep.add(e.i));
      [...episodes].sort((a, b) => b.t1 - b.t0 - (a.t1 - a.t0)).slice(0, 300).forEach((e) => keep.add(e.i));
      const stride = Math.ceil(episodes.length / 5000);
      episodes.forEach((e, i) => { if (i % stride === 0) keep.add(e.i); });
      shown = episodes.filter((e) => keep.has(e.i));
    }
    const g = el("g", { class: "clickable" }, svg);
    shown.forEach((e) => el("circle", { "data-k": e.i, cx: x(Math.max(e.t1 - e.t0, 60e3) / HOUR), cy: y(e.pf), r: shown.length > 2000 ? 2 : 3, fill: sideColor(e.side), opacity: shown.length > 2000 ? 0.45 : 0.7 }, g));
    delegate(g, (k) => episodeTip(episodes[+k]), (k) => selectEpisode(+k));
    $("fp-legend").innerHTML = sidesTraded.map((s) => `<span><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="${sideColor(s)}"/></svg>${s}</span>`).join("") + (shown.length < episodes.length ? `<span>${shown.length.toLocaleString()} of ${episodes.length.toLocaleString()} drawn (all extremes kept)</span>` : "") + `<span>y clipped to the 0.3–99.7% range</span>`;
  }
  function episodeTip(e) {
    return `<b>${esc(e.coin)} ${e.side} episode${e.finished ? "" : " (open at end)"}</b><br>${timeStr(e.t0)} → ${timeStr(e.t1)}<br>held ${dur(e.t1 - e.t0)} · ${e.entries} entries · ${e.closes} closes<br>max wallet exposure ${e.maxWe.toFixed(3)}<br>PnL <span class="${e.pnl >= 0 ? "pos" : "neg"}">${usd(e.pnl)}</span> (${pct(e.pf, 3)})${e.exit ? "<br>mostly " + (FAMILY_LABEL[e.exit] || e.exit) + " exits" : ""}`;
  }

  // ---------- exits
  function renderExits() {
    const box = $("exits"); box.innerHTML = "";
    const groups = {};
    fills.filter((f) => f.kind.action === "close").forEach((f) => {
      const key = f.kind.family + "|" + f.side;
      const g = (groups[key] = groups[key] || { family: f.kind.family, side: f.side, n: 0, wins: 0, gain: 0, loss: 0, usd: 0 });
      g.n++; g.usd += f.pnl;
      if (f.pnl > 0) { g.wins++; g.gain += f.pf; } else g.loss += f.pf;
    });
    const rows = Object.values(groups).sort((a, b) => b.n - a.n);
    if (!rows.length) { box.innerHTML = '<div class="empty">No closes.</div>'; return; }
    const rh = 28, svg = el("svg", { height: rows.length * rh + 22 }, box), w = width(svg);
    const L = Math.min(170, w * 0.36), R = 112, span = w - L - R;
    const ml = maxOf(rows.map((r) => -r.loss), 0), mg = maxOf(rows.map((r) => r.gain), 0), tot = ml + mg || 1;
    const z = L + (span * ml) / tot, sc = (v) => (v / tot) * span;
    rows.forEach((r, i) => {
      const yc = i * rh + rh / 2;
      el("text", { x: 0, y: yc - 1, style: "fill:var(--fg);font-size:11px" }, svg).textContent = `${FAMILY_LABEL[r.family] || r.family} · ${r.side}`;
      el("text", { x: 0, y: yc + 10, style: "font-size:9.5px" }, svg).textContent = `${r.n.toLocaleString()} closes · ${Math.round((r.wins / r.n) * 100)}% win`;
      if (r.loss < 0) el("rect", { x: z + sc(r.loss), y: yc - 6, width: Math.max(1, -sc(r.loss) - 1), height: 12, rx: 2, fill: "var(--loss)" }, svg);
      if (r.gain > 0) el("rect", { x: z + 1, y: yc - 6, width: Math.max(1, sc(r.gain)), height: 12, rx: 2, fill: "var(--win)" }, svg);
      el("text", { x: w - R + 8, y: yc + 3.5, style: "fill:var(--fg)" }, svg).textContent = "net " + pct(r.gain + r.loss, 1);
      const hit = el("rect", { "data-k": i, x: 0, y: yc - rh / 2, width: w, height: rh, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", (ev) => showTip(ev, `<b>${FAMILY_LABEL[r.family] || r.family} closes · ${r.side}</b><br>${r.n.toLocaleString()} fills · ${r.wins.toLocaleString()} winners<br>Σ gains ${pct(r.gain)} · Σ losses ${pct(r.loss)} of balance<br>net ${usd(r.usd)}`));
      hit.addEventListener("mouseleave", hideTip);
    });
    el("line", { class: "zero", x1: z, x2: z, y1: 0, y2: rows.length * rh }, svg);
    el("text", { x: z - 6, y: rows.length * rh + 14, "text-anchor": "end" }, svg).textContent = "← Σ losses";
    el("text", { x: z + 6, y: rows.length * rh + 14 }, svg).textContent = "Σ gains →";
  }

  // ---------- lanes overview
  function renderLanes() {
    const svg = $("lanes"); clear(svg);
    if (!fills.length) { svg.setAttribute("height", 30); el("text", { x: 0, y: 20 }, svg).textContent = "No fills."; return; }
    const w = width(svg), t0 = DATA.start_ms, t1 = Math.max(DATA.end_ms, t0 + HOUR);
    const glyphMode = fills.length <= GLYPH_MODE_MAX_FILLS;
    const bySide = coins.length === 1;
    const split = !bySide && sidesTraded.length > 1;
    const laneKey = (f) => (bySide ? f.side : f.coin);
    const laneTotals = {};
    fills.forEach((f) => (laneTotals[laneKey(f)] = (laneTotals[laneKey(f)] || 0) + f.pf));
    const lanes = Object.keys(laneTotals).sort((a, b) => laneTotals[b] - laneTotals[a]);
    const rowH = glyphMode ? (lanes.length > 12 ? 22 : 30) : lanes.length > 12 ? 14 : 26, T = 2, B = 18, PW = 86;
    const h = T + rowH * lanes.length + B; svg.setAttribute("height", h);
    const L = Math.min(64, Math.max(36, maxOf(lanes.map((l) => l.length)) * 7 + 10)), right = w - PW - 10;
    const x = (t) => L + ((t - t0) / (t1 - t0)) * (right - L);
    const maxTotal = maxOf(lanes.map((l) => Math.abs(laneTotals[l])), 1e-12), z = right + 10 + PW * 0.2, bx = (v) => (v / maxTotal) * PW * 0.42;
    const g = el("g", { class: "clickable" }, svg);
    const tips = {};
    lanes.forEach((lane, li) => {
      const y0 = T + li * rowH, yc = y0 + rowH / 2;
      const coin = bySide ? coins[0] : lane;
      el("rect", { "data-k": "lane:" + coin, x: 0, y: y0, width: w, height: rowH, fill: "transparent" }, g);
      tips["lane:" + coin] = `<b>${esc(lane)}</b><br>${fills.filter((f) => laneKey(f) === lane).length.toLocaleString()} fills · Σ ${pct(laneTotals[lane])} of balance<br>click to open the coin view`;
      el("text", { "data-k": "lane:" + coin, x: L - 6, y: yc + 3.5, "text-anchor": "end", style: "fill:var(--fg);font-size:" + (rowH < 16 ? 9.5 : 11) + "px" }, g).textContent = lane;
      const v = laneTotals[lane];
      el("rect", { x: Math.min(z, z + bx(v)), y: yc - Math.min(5, rowH * 0.3), width: Math.abs(bx(v)), height: Math.min(10, rowH * 0.6), rx: 1.5, fill: pnlColor(v) }, svg);
      el("text", { x: Math.max(z, z + bx(v)) + 4, y: yc + 3.5, style: "font-size:9.5px" }, svg).textContent = pct(v, 1);
      const laneFills = fills.filter((f) => laneKey(f) === lane);
      if (glyphMode) {
        el("line", { class: "gl", x1: L, x2: right, y1: yc, y2: yc }, svg);
        laneFills.forEach((f) => {
          const off = f.kind.action === "entry" ? (f.side === "long" ? -1 : 1) * rowH * 0.18 : split ? (f.side === "long" ? -1 : 1) * rowH * 0.12 : 0;
          const m = glyph(g, f, x(f.t), yc + off, rowH < 26 ? 2.8 : 3.4);
          m.setAttribute("data-k", "f:" + f.i);
        });
      } else {
        const bin = t1 - t0 > 120 * DAY ? WEEK : DAY;
        const halves = split ? ["long", "short"] : [null];
        halves.forEach((side, hi) => {
          const hh = (rowH - 2) / halves.length, yy = y0 + 1 + hi * hh;
          el("rect", { x: L, y: yy, width: right - L, height: hh, fill: "var(--hover)", opacity: 0.6 }, svg);
          const cells = {};
          laneFills.filter((f) => side === null || f.side === side).forEach((f) => { const b = Math.floor((f.t - t0) / bin); const c = (cells[b] = cells[b] || { pf: 0, n: 0 }); c.pf += f.pf; c.n++; });
          Object.entries(cells).forEach(([b, c]) => { const key = `c:${lane}:${side}:${b}`; tips[key] = `<b>${esc(lane)}${side ? " " + side : ""} · ${bin === WEEK ? "week of " : ""}${dateStr(t0 + b * bin)}</b><br>${c.n} fills<br>realized ${pct(c.pf, 3)} of balance`; cells[b].key = key; });
          const mags = Object.values(cells).map((c) => Math.abs(c.pf)).sort((a, b) => a - b);
          const cap = mags[Math.floor(mags.length * 0.95)] || 1e-12;
          const cw = Math.max(1, ((right - L) * bin) / (t1 - t0));
          Object.entries(cells).forEach(([b, c]) => el("rect", { "data-k": c.key, x: x(t0 + b * bin), y: yy, width: Math.max(1, cw - 0.4), height: Math.max(1, hh - (split ? 0.5 : 0)), fill: pnlColor(c.pf), opacity: (0.15 + 0.85 * Math.min(1, Math.abs(c.pf) / cap)).toFixed(2) }, g));
        });
      }
    });
    el("line", { class: "zero", x1: z, x2: z, y1: T, y2: h - B }, svg);
    el("text", { x: z, y: h - 4, "text-anchor": "middle" }, svg).textContent = "Σ % bal";
    drawTimeAxis(svg, x, t0, t1, h - 4, right - L + 60, right - 18);
    delegate(g, (k) => (k.startsWith("f:") ? fillTip(fills[+k.slice(2)]) : tips[k]), (k) => {
      if (k.startsWith("f:")) { const f = fills[+k.slice(2)]; selectCoin(f.coin, [f.t - 3 * DAY, f.t + 3 * DAY]); }
      else if (k.startsWith("lane:")) selectCoin(k.slice(5), null);
      else if (k.startsWith("c:")) { const [, lane, side, b] = k.split(":"); const bin = t1 - t0 > 120 * DAY ? WEEK : DAY; selectCoin(bySide ? coins[0] : lane, [t0 + b * bin, t0 + (+b + 1) * bin]); }
    });
    $("lanes-sub").textContent = glyphMode
      ? `${bySide ? "One lane per side." : "One lane per coin, sorted by realized PnL."} Every fill is a mark. Click a mark or a lane to open it in the coin view.`
      : `${fills.length.toLocaleString()} fills, so each cell sums realized PnL as % of balance per ${t1 - t0 > 120 * DAY ? "week" : "day"}${split ? "; each coin row is split into long (top) and short (bottom)" : ""}. Intensity is capped at the 95th percentile. Click a cell to zoom the coin view to it.`;
    $("lanes-legend").innerHTML = glyphMode ? glyphLegend() : `<span><svg width="58" height="10">${[1, 0.6, 0.3].map((o, i) => `<rect x="${i * 15}" width="14" height="10" fill="var(--loss)" opacity="${o}"/>`).join("")}</svg>loss</span><span><svg width="58" height="10">${[0.3, 0.6, 1].map((o, i) => `<rect x="${i * 15}" width="14" height="10" fill="var(--win)" opacity="${o}"/>`).join("")}</svg>gain</span>`;
  }

  // ---------- coin view state
  const state = { coin: null, range: null, sides: { long: true, short: true }, allMarks: false, episode: null };
  function coinSpan(coin) {
    const cf = fillsByCoin[coin] || [];
    const a = cf.length ? cf[0].t : DATA.start_ms, b = cf.length ? cf[cf.length - 1].t : DATA.end_ms;
    const pad = Math.max((b - a) * 0.02, 6 * HOUR);
    return [Math.max(a - pad, DATA.start_ms - 6 * HOUR), Math.min(b + pad, DATA.end_ms + 6 * HOUR)];
  }
  function selectCoin(coin, range) {
    state.coin = coin; state.range = range; state.episode = null;
    $("coin-select").value = coin;
    renderCoin(); renderLedger();
    $("coin-card").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function selectEpisode(i) {
    const e = episodes[i];
    const pad = Math.max((e.t1 - e.t0) * 0.08, 3 * HOUR);
    state.coin = e.coin; state.range = [e.t0 - pad, e.t1 + pad]; state.episode = i;
    $("coin-select").value = e.coin;
    renderCoin(); renderLedger(); renderEpisodes();
    $("coin-card").scrollIntoView({ behavior: "smooth", block: "start" });
  }
  function decodeBars(tier) {
    if (tier.decoded) return tier.decoded;
    const e = tier.bars, out = new Array(e.dt.length);
    let slot = e.t0 === null ? 0 : e.t0 / tier.interval_ms;
    for (let i = 0; i < e.dt.length; i++) { slot += e.dt[i]; out[i] = [slot * tier.interval_ms, e.o[i], e.h[i], e.l[i], e.c[i]]; }
    return (tier.decoded = out);
  }
  function barsFor(coin, a, b) {
    const src = DATA.candles[coin];
    if (!src) return { interval: null, bars: [] };
    let best = src.overview;
    src.details.forEach((d) => { if (d.start_ms <= a && d.end_ms >= b && d.interval_ms < best.interval_ms) best = d; });
    let bars = decodeBars(best).filter((x) => x[0] + best.interval_ms > a && x[0] <= b);
    let interval = best.interval_ms;
    if (bars.length > MAX_VISIBLE_BARS) {
      const k = Math.ceil(bars.length / MAX_VISIBLE_BARS), merged = [];
      for (let i = 0; i < bars.length; i += k) {
        const chunk = bars.slice(i, i + k);
        merged.push([chunk[0][0], chunk[0][1], maxOf(chunk.map((c) => c[2])), minOf(chunk.map((c) => c[3])), chunk[chunk.length - 1][4]]);
      }
      bars = merged; interval *= k;
    }
    return { interval, bars, native: best.interval_ms };
  }

  function renderCoin() {
    const coin = state.coin;
    const priceSvg = $("price"), flowSvg = $("flow"), weSvg = $("exposure");
    [priceSvg, flowSvg, weSvg].forEach(clear);
    priceCtx = null;
    if (!coin) return;
    const [a, b] = state.range || coinSpan(coin);
    const { interval, bars, native } = barsFor(coin, a, b);
    const cf = (fillsByCoin[coin] || []).filter((f) => state.sides[f.side]);
    const inView = cf.filter((f) => f.t >= a && f.t <= b);
    $("coin-title").textContent = `${coin} · ${shortDate(a)} ${new Date(a).getUTCFullYear()} → ${shortDate(b)} ${new Date(b).getUTCFullYear()}`;
    $("interval").textContent = interval ? `${intervalLabel(interval)} candles${native !== interval ? ` (from ${intervalLabel(native)})` : ""}` : "no candles";
    $("reset").setAttribute("aria-pressed", state.range ? "false" : "true");
    const w = width(priceSvg), L = 58, R = 10;
    const x = (t) => L + ((t - a) / (b - a)) * (w - L - R);

    // price
    const h = +priceSvg.getAttribute("height"), T = 8, B = 18;
    // A bar wider than ~1/12 of the view says nothing about intra-bar paths.
    const tooCoarse = interval && (b - a) / interval < 12;
    if (tooCoarse) bars.length = 0;
    const prices = bars.flatMap((c) => [c[2], c[3]]).concat(inView.map((f) => f.price));
    if (!prices.length) { el("text", { x: L, y: 30 }, priceSvg).textContent = "No candles or fills in this range."; return; }
    let lo = minOf(prices), hi = maxOf(prices); const padY = (hi - lo || hi * 0.01) * 0.06; lo -= padY; hi += padY;
    const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (h - T - B);
    niceTicks(lo, hi, 5).forEach((v) => { el("line", { class: "gl", x1: L, x2: w - R, y1: y(v), y2: y(v) }, priceSvg); el("text", { x: L - 6, y: y(v) + 3, "text-anchor": "end" }, priceSvg).textContent = num(v); });
    if (state.episode !== null) { const e = episodes[state.episode]; if (e.coin === coin) el("rect", { x: x(e.t0), y: T, width: Math.max(1, x(e.t1) - x(e.t0)), height: h - T - B, fill: "var(--hover)" }, priceSvg); }
    const panicAll = inView.filter((f) => f.kind.family === "panic");
    // Trigger lines clarify a few stops; dozens of them only hide the candles.
    const panic = panicAll.length <= 40 ? panicAll : [];
    const bw = Math.max(1, ((w - L - R) / Math.max(1, (b - a) / interval)) * 0.64);
    const cg = el("g", {}, priceSvg);
    bars.forEach(([t, o, hh, ll, c]) => {
      const xc = x(t + interval / 2), up = c >= o, col = up ? "var(--muted)" : "var(--fg)";
      el("line", { x1: xc, x2: xc, y1: y(hh), y2: y(ll), stroke: col, opacity: 0.55 }, cg);
      el("rect", { x: xc - bw / 2, y: y(Math.max(o, c)), width: bw, height: Math.max(0.8, Math.abs(y(o) - y(c))), fill: up ? "var(--card)" : col, stroke: col, "stroke-width": 0.8, opacity: up ? 0.85 : 0.55 }, cg);
    });
    // brush overlay below fills so marks keep their hover
    const ov = el("rect", { x: L, y: T, width: w - L - R, height: h - T - B, fill: "transparent", style: "cursor:crosshair" }, priceSvg);
    // position price per side, clipped to the range
    ["long", "short"].filter((s) => state.sides[s]).forEach((s) => {
      const sd = ((DATA.seed || {})[coin] || {})[s];
      let open = sd ? { t: DATA.start_ms, p: sd[1] } : null, d = "";
      const seg = (t0s, t1s, p) => { const s0 = Math.max(t0s, a), s1 = Math.min(t1s, b); if (s1 > s0 && p > lo && p < hi) d += `M${x(s0).toFixed(1)},${y(p).toFixed(1)}H${x(s1).toFixed(1)}`; };
      (fillsByCoin[coin] || []).filter((f) => f.side === s).forEach((f) => { if (open) seg(open.t, f.t, open.p); open = Math.abs(f.psize) > 0 ? { t: f.t, p: f.pprice } : null; });
      if (open) seg(open.t, b, open.p);
      if (d) el("path", { d, stroke: sideColor(s), "stroke-width": 1.8, "stroke-dasharray": "5 3", fill: "none" }, priceSvg);
    });
    panic.forEach((f) => el("line", { x1: x(f.t), x2: x(f.t), y1: T, y2: h - B, stroke: "var(--loss)", "stroke-dasharray": "2 3", opacity: 0.6 }, priceSvg));
    // level of detail
    const famCount = {};
    inView.forEach((f) => { const k = f.kind.action + ":" + f.kind.family; famCount[k] = (famCount[k] || 0) + 1; });
    const dense = new Set(state.allMarks ? [] : Object.keys(famCount).filter((k) => famCount[k] > DENSE_FAMILY_LIMIT));
    const mg = el("g", {}, priceSvg);
    const r = inView.length > 400 ? 3 : 4.3;
    inView.filter((f) => !dense.has(f.kind.action + ":" + f.kind.family)).forEach((f) => glyph(mg, f, x(f.t), y(f.price), r).setAttribute("data-k", f.i));
    delegate(mg, (k) => fillTip(fills[+k]));
    drawTimeAxis(priceSvg, x, a, b, h - 4, w);
    if (tooCoarse) el("text", { x: L + 6, y: T + 12 }, priceSvg).textContent = `Candles here are ${intervalLabel(interval)}, too coarse for this range. For finer candles, re-run the report with --fills-coins ${coin} --fills-start/--fills-end around this window.`;
    // brush + candle hover; svg-level listeners are attached once in init()
    const brush = el("rect", { y: T, height: h - T - B, fill: "var(--ring)", opacity: 0, "pointer-events": "none" }, priceSvg);
    priceCtx = { a, b, L, R, w, ov, brush, bars, interval, drag: null };
    ov.addEventListener("mousedown", (ev) => { priceCtx.drag = ev.clientX - priceSvg.getBoundingClientRect().left; ev.preventDefault(); });

    // flow panel for collapsed fill types
    flowSvg.style.display = dense.size ? "" : "none";
    if (dense.size) {
      const fh = +flowSvg.getAttribute("height"), FT = 14, FB = 3, mid = FT + (fh - FT - FB) / 2;
      const bins = {}, cMult = (DATA.c_mult || {})[coin] || 1;
      inView.filter((f) => dense.has(f.kind.action + ":" + f.kind.family)).forEach((f) => {
        const k = Math.floor((f.t - a) / interval);
        const o = (bins[k] = bins[k] || { entry: 0, close: 0, n: 0, pnl: 0 });
        o[f.kind.action] += Math.abs(f.qty * f.price) * cMult; o.n++; o.pnl += f.pnl;
      });
      const mx = maxOf(Object.values(bins).map((o) => Math.max(o.entry, o.close)), 1e-12);
      const hasEntry = Object.values(bins).some((o) => o.entry > 0), hasClose = Object.values(bins).some((o) => o.close > 0);
      const base = hasEntry && hasClose ? mid : hasEntry ? fh - FB : FT;
      const scale = (hasEntry && hasClose ? (fh - FT - FB) / 2 : fh - FT - FB) / mx;
      const bwid = Math.max(1, ((w - L - R) * interval) / (b - a) - 1);
      el("line", { class: "zero", x1: L, x2: w - R, y1: base, y2: base }, flowSvg);
      const fg = el("g", {}, flowSvg), tips = {};
      Object.entries(bins).forEach(([k, o]) => {
        const bx = x(a + k * interval) + 0.5;
        tips[k] = `<b>${timeStr(a + k * interval)} UTC · ${intervalLabel(interval)}</b><br>${o.n} collapsed fills<br>entry notional ${usd(o.entry)}<br>close notional ${usd(o.close)}<br>PnL <span class="${o.pnl >= 0 ? "pos" : "neg"}">${usd(o.pnl)}</span>`;
        if (o.entry) el("rect", { "data-k": k, x: bx, y: base - o.entry * scale, width: bwid, height: Math.max(1, o.entry * scale), fill: "var(--ring)" }, fg);
        if (o.close) el("rect", { "data-k": k, x: bx, y: base, width: bwid, height: Math.max(1, o.close * scale), fill: pnlColor(o.pnl) }, fg);
      });
      delegate(fg, (k) => tips[k]);
      el("text", { class: "title", x: L, y: 10 }, flowSvg).textContent = `Collapsed: ${[...dense].map((k) => `${famLabel(k)} (${famCount[k].toLocaleString()})`).join(", ")} · notional per ${intervalLabel(interval)}${hasEntry && hasClose ? " · entries up, closes down" : ""}`;
    }

    // wallet exposure panel
    const wh = +weSvg.getAttribute("height"), WT = 14, WB = 18;
    const series = {};
    ["long", "short"].filter((s) => state.sides[s] && sidesTraded.includes(s)).forEach((s) => {
      const sf = (fillsByCoin[coin] || []).filter((f) => f.side === s);
      const sd = ((DATA.seed || {})[coin] || {})[s];
      let cur = sd ? sd[2] : 0; const pts = [];
      sf.forEach((f) => { if (f.t < a) cur = f.we; });
      pts.push([a, cur]);
      sf.filter((f) => f.t >= a && f.t <= b).forEach((f) => { pts.push([f.t, cur]); cur = f.we; pts.push([f.t, cur]); });
      pts.push([b, cur]);
      series[s] = pts;
    });
    const vals = Object.values(series).flat().map((p) => p[1]);
    const ymax = maxOf(vals.map(Math.abs), 0.01) * 1.12;
    const two = Object.keys(series).length > 1 || vals.some((v) => v < 0) && vals.some((v) => v > 0);
    const wlo = series.short || vals.some((v) => v < 0) ? -ymax : 0, whi = series.long || vals.some((v) => v > 0) ? ymax : 0;
    const wy = (v) => WT + (1 - (v - wlo) / (whi - wlo || 1)) * (wh - WT - WB);
    niceTicks(wlo, whi, two ? 4 : 2).forEach((v) => { el("line", { class: v === 0 ? "zero" : "gl", x1: L, x2: w - R, y1: wy(v), y2: wy(v) }, weSvg); el("text", { x: L - 6, y: wy(v) + 3, "text-anchor": "end" }, weSvg).textContent = +v.toFixed(3); });
    Object.entries(series).forEach(([s, pts]) => {
      let d = `M${x(pts[0][0]).toFixed(1)},${wy(0).toFixed(1)}`; pts.forEach(([t, v]) => (d += `L${x(t).toFixed(1)},${wy(v).toFixed(1)}`)); d += `L${x(b).toFixed(1)},${wy(0).toFixed(1)}Z`;
      el("path", { d, fill: sideColor(s), opacity: 0.18 }, weSvg);
      let q = ""; pts.forEach(([t, v], i) => (q += (i ? "L" : "M") + x(t).toFixed(1) + "," + wy(v).toFixed(1)));
      el("path", { d: q, fill: "none", stroke: sideColor(s), "stroke-width": 1.5 }, weSvg);
    });
    panic.forEach((f) => el("line", { x1: x(f.t), x2: x(f.t), y1: WT, y2: wh - WB, stroke: "var(--loss)", "stroke-dasharray": "2 3", opacity: 0.6 }, weSvg));
    el("text", { class: "title", x: L, y: 10 }, weSvg).textContent = two ? "Wallet exposure (long +, short −)" : "Wallet exposure";
    drawTimeAxis(weSvg, x, a, b, wh - 4, w);

    let legend = glyphLegend();
    ["long", "short"].filter((s) => sidesTraded.includes(s)).forEach((s) => (legend += `<span><svg width="18" height="8"><line x1="0" x2="18" y1="4" y2="4" stroke="${sideColor(s)}" stroke-width="2" stroke-dasharray="5 3"/></svg>${s} position price</span>`));
    if (kinds.some((k) => k.family === "panic")) legend += `<span><svg width="8" height="12"><line x1="4" x2="4" y1="0" y2="12" stroke="var(--loss)" stroke-dasharray="2 3"/></svg>HSL panic close</span>`;
    $("glyph-legend").innerHTML = legend;
    const note = $("episode-note");
    if (state.episode !== null) {
      const e = episodes[state.episode];
      note.hidden = false;
      note.innerHTML = `Episode: ${esc(e.coin)} ${e.side}, ${timeStr(e.t0)} → ${timeStr(e.t1)} UTC${e.finished ? "" : " (still open at the end)"} · held ${dur(e.t1 - e.t0)} · ${e.entries} entries, ${e.closes} closes · max WE ${e.maxWe.toFixed(3)} · PnL <span class="${e.pnl >= 0 ? "pos" : "neg"}">${usd(e.pnl)}</span> (${pct(e.pf, 3)})`;
    } else note.hidden = true;
  }
  let priceCtx = null;
  function attachPriceHandlers() {
    const svg = $("price");
    const px = (ev) => ev.clientX - svg.getBoundingClientRect().left;
    svg.addEventListener("mousemove", (ev) => {
      const c = priceCtx; if (!c) return;
      if (c.drag !== null) { const cur = px(ev); c.brush.setAttribute("x", Math.min(c.drag, cur)); c.brush.setAttribute("width", Math.abs(cur - c.drag)); c.brush.setAttribute("opacity", 0.18); return; }
      if (ev.target !== c.ov) return;
      const tt = c.a + ((px(ev) - c.L) / (c.w - c.L - c.R)) * (c.b - c.a);
      const bar = c.bars.find((x) => x[0] <= tt && x[0] + c.interval > tt);
      if (bar) showTip(ev, `<b>${timeStr(bar[0])} UTC · ${intervalLabel(c.interval)}</b><br>O ${num(bar[1])} H ${num(bar[2])}<br>L ${num(bar[3])} C ${num(bar[4])}`); else hideTip();
    });
    const endDrag = (ev) => {
      const c = priceCtx; if (!c || c.drag === null) return;
      const cur = px(ev), x0 = Math.min(c.drag, cur), x1 = Math.max(c.drag, cur);
      c.drag = null; c.brush.setAttribute("opacity", 0);
      if (x1 - x0 < 6) return;
      const t = (xx) => c.a + ((Math.max(c.L, Math.min(c.w - c.R, xx)) - c.L) / (c.w - c.L - c.R)) * (c.b - c.a);
      state.range = [t(x0), Math.max(t(x1), t(x0) + 2 * 60e3)]; state.episode = null;
      renderCoin(); renderLedger();
    };
    svg.addEventListener("mouseup", endDrag);
    svg.addEventListener("mouseleave", (ev) => { endDrag(ev); hideTip(); });
    svg.addEventListener("dblclick", () => { state.range = null; state.episode = null; renderCoin(); renderLedger(); });
  }
  const famLabel = (k) => { const [action, family] = k.split(":"); return `${FAMILY_LABEL[family] || family} ${action === "entry" ? "entries" : "closes"}`; };

  // ---------- ledger
  function renderLedger() {
    const tb = $("ledger");
    if (!state.coin) { tb.innerHTML = ""; return; }
    const [a, b] = state.range || coinSpan(state.coin);
    let rowsSrc = (fillsByCoin[state.coin] || []).filter((f) => state.sides[f.side] && f.t >= a && f.t <= b);
    if (state.episode !== null) { const e = episodes[state.episode]; rowsSrc = rowsSrc.filter((f) => f.side === e.side && f.i >= e.first && f.i <= e.last); }
    const groups = [];
    rowsSrc.forEach((f) => { const last = groups[groups.length - 1]; if (last && last.type === f.kind.type) last.f.push(f); else groups.push({ type: f.kind.type, f: [f] }); });
    const LIMIT = 400;
    $("ledger-count").textContent = `${rowsSrc.length.toLocaleString()} fills · ${groups.length.toLocaleString()} rows${groups.length > LIMIT ? ` · first ${LIMIT} shown` : ""}`;
    if (!groups.length) { tb.innerHTML = '<tbody><tr><td class="empty">No fills in this range.</td></tr></tbody>'; return; }
    tb.innerHTML = `<thead><tr><th>time (UTC)</th><th>order type</th><th class="n">fills</th><th class="n">price</th><th class="n">Σ qty</th><th class="n">position after</th><th class="n">pos. price</th><th class="n">WE after</th><th class="n">Σ PnL</th><th class="n">Σ % bal</th></tr></thead><tbody>` +
      groups.slice(0, LIMIT).map((g) => {
        const f0 = g.f[0], f1 = g.f[g.f.length - 1], n = g.f.length;
        const pnl = g.f.reduce((s, f) => s + f.pnl, 0), pf = g.f.reduce((s, f) => s + f.pf, 0), qty = g.f.reduce((s, f) => s + Math.abs(f.qty), 0);
        const prices = g.f.map((f) => f.price), p0 = minOf(prices), p1 = maxOf(prices);
        const k = f0.kind;
        return `<tr><td>${timeStr(f0.t)}${n > 1 ? " → " + timeStr(f1.t).slice(11) + (dateStr(f1.t) !== dateStr(f0.t) ? " (" + shortDate(f1.t) + ")" : "") : ""}</td><td>${k.action === "entry" ? (k.side === "long" ? G.longEntry : G.shortEntry) : k.intent === "risk" ? (pnl >= 0 ? G.riskWin : G.riskLoss) : pnl >= 0 ? G.tpWin : G.tpLoss} <span class="mono">${esc(k.type)}</span></td><td class="n">${n}</td><td class="n">${n > 1 && p0 !== p1 ? num(p0) + "–" + num(p1) : num(p0)}</td><td class="n">${num(qty)}</td><td class="n">${num(f1.psize)}</td><td class="n">${num(f1.pprice)}</td><td class="n">${f1.we.toFixed(4)}</td><td class="n ${pnl > 0 ? "pos" : pnl < 0 ? "neg" : ""}">${k.action === "close" ? usd(pnl) : "–"}</td><td class="n">${k.action === "close" ? pct(pf, 3) : "–"}</td></tr>`;
      }).join("") + "</tbody>";
  }

  // ---------- episodes table
  const SORTS = { worst: ["worst PnL %", (a, b) => a.pf - b.pf], best: ["best PnL %", (a, b) => b.pf - a.pf], longest: ["longest", (a, b) => b.t1 - b.t0 - (a.t1 - a.t0)], exposure: ["max WE", (a, b) => b.maxWe - a.maxWe], latest: ["latest", (a, b) => b.t0 - a.t0] };
  const epState = { sort: "worst", limit: 100 };
  function renderEpisodes() {
    $("ep-sorts").innerHTML = Object.entries(SORTS).map(([k, [l]]) => `<button class="chip" data-sort="${k}" aria-pressed="${epState.sort === k}">${l}</button>`).join(" ");
    $("ep-sorts").querySelectorAll("button").forEach((btn) => btn.addEventListener("click", () => { epState.sort = btn.dataset.sort; epState.limit = 100; renderEpisodes(); }));
    const tb = $("episodes");
    if (!episodes.length) { tb.innerHTML = '<tbody><tr><td class="empty">No position episodes.</td></tr></tbody>'; return; }
    const rows = [...episodes].sort(SORTS[epState.sort][1]).slice(0, epState.limit);
    tb.innerHTML = `<thead><tr><th>coin</th><th>side</th><th>opened (UTC)</th><th class="n">held</th><th class="n">entries</th><th class="n">closes</th><th>main exit</th><th class="n">max WE</th><th class="n">PnL</th><th class="n">% bal</th></tr></thead><tbody>` +
      rows.map((e) => `<tr class="clickable${state.episode === e.i ? " sel" : ""}" data-ep="${e.i}"><td>${esc(e.coin)}</td><td>${e.side}</td><td>${timeStr(e.t0)}</td><td class="n">${dur(e.t1 - e.t0)}${e.finished ? "" : " ⋯"}</td><td class="n">${e.entries}</td><td class="n">${e.closes}</td><td>${e.exit ? FAMILY_LABEL[e.exit] || e.exit : "–"}</td><td class="n">${e.maxWe.toFixed(3)}</td><td class="n ${e.pnl >= 0 ? "pos" : "neg"}">${usd(e.pnl)}</td><td class="n">${pct(e.pf, 3)}</td></tr>`).join("") +
      (episodes.length > epState.limit ? `<tr><td colspan="10"><button class="chip" id="ep-more">show more (${(episodes.length - epState.limit).toLocaleString()} remaining)</button></td></tr>` : "") + "</tbody>";
    tb.querySelectorAll("tr[data-ep]").forEach((tr) => tr.addEventListener("click", () => selectEpisode(+tr.dataset.ep)));
    const more = $("ep-more"); if (more) more.addEventListener("click", () => { epState.limit += 200; renderEpisodes(); });
  }

  function renderTaxonomy() {
    const present = (pred) => [...new Set(kinds.filter(pred).map((k) => k.mechanism))].sort().join(", ") || "–";
    $("taxonomy").innerHTML = `<thead><tr><th>shape</th><th>intent</th><th>mechanisms in this run</th><th>color</th></tr></thead><tbody>
      <tr><td>${G.longEntry} ${G.shortEntry}</td><td>entry</td><td>${present((k) => k.intent === "entry")}</td><td>position side</td></tr>
      <tr><td>${G.tpWin} ${G.tpLoss}</td><td>take-profit close</td><td>${present((k) => k.intent === "take_profit" || k.intent === "other")}</td><td>PnL sign</td></tr>
      <tr><td>${G.riskWin} ${G.riskLoss}</td><td>risk close</td><td>${present((k) => k.intent === "risk")}</td><td>PnL sign</td></tr></tbody>`;
  }

  // ---------- wiring
  function init() {
    renderHeader(); renderTaxonomy(); attachPriceHandlers();
    const sel = $("coin-select");
    sel.innerHTML = coins.map((c) => `<option>${esc(c)}</option>`).join("");
    sel.addEventListener("change", () => selectCoin(sel.value, null));
    ["long", "short"].forEach((s) => {
      const btn = $("side-" + s);
      if (!sidesTraded.includes(s)) { btn.hidden = true; return; }
      btn.addEventListener("click", () => { state.sides[s] = !state.sides[s]; btn.setAttribute("aria-pressed", state.sides[s]); renderCoin(); renderLedger(); });
    });
    $("detail-all").addEventListener("click", () => { state.allMarks = !state.allMarks; $("detail-all").setAttribute("aria-pressed", state.allMarks); renderCoin(); });
    $("reset").addEventListener("click", () => { state.range = null; state.episode = null; renderCoin(); renderLedger(); });
    if (coins.length) {
      // Start on the coin with the most fills.
      state.coin = [...coins].sort((p, q) => (fillsByCoin[q] || []).length - (fillsByCoin[p] || []).length)[0];
      sel.value = state.coin;
    } else $("coin-card").hidden = true;
    renderAll();
    let timer = null;
    window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(renderAll, 150); });
  }
  function renderAll() { renderEquity(); renderFingerprint(); renderExits(); renderLanes(); renderCoin(); renderLedger(); renderEpisodes(); }
  init();
})();
