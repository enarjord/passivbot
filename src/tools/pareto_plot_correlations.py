"""Parameter columns and offline correlation explorer for pareto-plot."""

from collections.abc import Sequence

from pareto_analyzer import extract_bot_parameter_values
from pareto_explorer import ParetoCandidate


def build_parameter_columns(candidates: Sequence[ParetoCandidate]) -> list[dict]:
    """Keep numeric bot parameters that can be correlated with a saved metric."""
    if len(candidates) < 3:
        return []
    columns = {}
    for row, candidate in enumerate(candidates):
        for key, value in extract_bot_parameter_values(candidate).items():
            if key not in columns:
                columns[key] = [None] * len(candidates)
            columns[key][row] = value
    parameters = []
    for key, values in sorted(columns.items()):
        finite = [value for value in values if value is not None]
        if len(finite) >= 3 and min(finite) < max(finite):
            parameters.append({"key": key, "values": values})
    return parameters


CORRELATION_STYLES = r"""
.correlation-controls{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0}
.correlation-controls>div{min-width:150px}.correlation-controls>div:first-child{flex:2 1 270px}
.correlation-controls>div:last-child{flex:1 1 170px}
.correlation-controls input{width:100%;padding:7px;font:inherit;color:inherit;background:white;border:1px solid #cbd5e1;border-radius:6px}
.correlation-content{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:18px;align-items:start}
.correlation-table{max-height:410px;overflow:auto;border:1px solid #dce3ed;border-radius:8px}
.correlation-table table{width:100%;border-collapse:separate;border-spacing:0;text-align:left;font-size:13px}
.correlation-table th{position:sticky;top:0;background:#f4f6fa;z-index:1;white-space:nowrap}
.correlation-table th,.correlation-table td{padding:7px 9px;border-bottom:1px solid #e2e8f0}
.correlation-table td:first-child{overflow-wrap:anywhere}
.correlation-table td:not(:first-child){font-variant-numeric:tabular-nums;white-space:nowrap}
.correlation-table button{border:0;background:transparent;color:inherit;text-align:left;padding:4px;width:100%;overflow-wrap:anywhere}
.correlation-table button:hover,.correlation-table button[aria-pressed="true"]{color:#23598a;background:#eaf3fc}
#correlation-pair{margin:2px 0 0;overflow-wrap:anywhere;font-size:13px}
#correlation-plot{height:360px;width:100%}
#correlation-inspected{overflow-wrap:anywhere;padding:6px 9px;border-radius:5px;background:#eef5f7}
@media(max-width:850px){.correlation-content{grid-template-columns:1fr}}
"""


CORRELATION_PANEL = r"""
<h2>Parameter correlations</h2>
<p class="muted">Explore associations between saved bot parameters and a metric across candidates that pass the plot limits. These associations describe the saved candidates and can reflect other parameters that varied alongside them.</p>
<div class="correlation-controls">
  <div><label for="correlation-metric">Backtest metric</label><select id="correlation-metric"></select></div>
  <div><label for="correlation-method">Correlation</label><select id="correlation-method"><option value="pearson">Pearson (linear)</option><option value="spearman">Spearman (rank)</option></select></div>
  <div><label for="correlation-query">Find a parameter</label><input type="search" id="correlation-query" placeholder="e.g. exposure"></div>
</div>
<p class="muted" id="correlation-summary" role="status"></p>
<div class="correlation-content">
  <div class="correlation-table" id="correlation-table" hidden>
    <table aria-label="Parameter correlations ordered by absolute coefficient">
      <thead><tr><th scope="col">Parameter</th><th scope="col" id="correlation-coefficient">Pearson r</th><th scope="col">Pairs</th></tr></thead>
      <tbody id="correlation-results"></tbody>
    </table>
  </div>
  <div id="correlation-chart" hidden>
    <h3 id="correlation-pair"></h3>
    <div id="correlation-plot"></div>
    <p class="muted" id="correlation-inspected" role="status" hidden></p>
    <p id="correlation-error" role="alert" hidden></p>
  </div>
</div>
"""


CORRELATION_SCRIPT = r"""
function completeParameterPairs(parameter, metric, rows) {
    const x = [], y = [], indices = [];
    for (const row of rows) {
        const parameterValue = parameter.values[row], metricValue = metric.values[row];
        if (!Number.isFinite(parameterValue) || !Number.isFinite(metricValue)) continue;
        x.push(parameterValue);
        y.push(metricValue);
        indices.push(row);
    }
    return {x, y, indices};
}
function averageRanks(values) {
    const order = values.map((_, i) => i).sort((a, b) => values[a] - values[b]);
    const ranks = new Array(values.length);
    for (let start = 0; start < order.length;) {
        let end = start + 1;
        while (end < order.length && values[order[end]] === values[order[start]]) end++;
        const rank = (start + end + 1) / 2;
        for (let i = start; i < end; i++) ranks[order[i]] = rank;
        start = end;
    }
    return ranks;
}
function pearsonCorrelation(x, y) {
    // Center around a real observation and scale before taking products to keep
    // large finite values and small changes on a large baseline representable.
    let xScale = 0, yScale = 0;
    for (let i = 0; i < x.length; i++) {
        xScale = Math.max(xScale, Math.abs(x[i]));
        yScale = Math.max(yScale, Math.abs(y[i]));
    }
    if (xScale === 0 || yScale === 0) return null;
    const centered = (value, origin, scale) => {
        const difference = value - origin;
        return Number.isFinite(difference) ? difference / scale : value / scale - origin / scale;
    };
    const xs = x.map(value => centered(value, x[0], xScale));
    const ys = y.map(value => centered(value, y[0], yScale));
    const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
    const xMean = mean(xs), yMean = mean(ys);
    let covariance = 0, xVariance = 0, yVariance = 0;
    for (let i = 0; i < xs.length; i++) {
        const dx = xs[i] - xMean, dy = ys[i] - yMean;
        covariance += dx * dy;
        xVariance += dx * dx;
        yVariance += dy * dy;
    }
    const denominator = Math.sqrt(xVariance) * Math.sqrt(yVariance);
    return denominator ? Math.min(1, Math.max(-1, covariance / denominator)) : null;
}
function parameterMetricCorrelations(parameters, metric, rows, method) {
    const results = [];
    for (const parameter of parameters) {
        const {x, y} = completeParameterPairs(parameter, metric, rows);
        if (x.length < 3) continue;
        const corr = method === "spearman" ?
            pearsonCorrelation(averageRanks(x), averageRanks(y)) : pearsonCorrelation(x, y);
        if (Number.isFinite(corr)) results.push({key:parameter.key, corr, count:x.length});
    }
    return results.sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr) ||
        (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
}
function mountParameterCorrelations(dataset, onCandidateClick) {
    const parameters = new Map((dataset.parameters || []).map(item => [item.key, item]));
    const metrics = new Map(dataset.metrics.filter(item => item.count).map(item => [item.key, item]));
    const metricInput = document.getElementById("correlation-metric");
    const methodInput = document.getElementById("correlation-method");
    const queryInput = document.getElementById("correlation-query");
    const table = document.getElementById("correlation-table");
    const results = document.getElementById("correlation-results");
    const summary = document.getElementById("correlation-summary");
    const chart = document.getElementById("correlation-chart");
    const caption = document.getElementById("correlation-pair");
    const plot = document.getElementById("correlation-plot");
    const inspector = document.getElementById("correlation-inspected");
    const error = document.getElementById("correlation-error");
    for (const group of new Set([...metrics.values()].map(metric => metric.group))) {
        const options = document.createElement("optgroup");
        options.label = group;
        for (const metric of metrics.values()) {
            if (metric.group === group) options.append(new Option(metric.label, metric.key));
        }
        metricInput.append(options);
    }
    metricInput.value = dataset.selected[0];
    let rows = [], rankings = [], parameterKey = null, pressedButton = null;
    let plotted = false, rendering = false, pending = false, inspected = null;

    function renderInspected(indices, x, y) {
        const index = indices.indexOf(inspected);
        inspector.hidden = index < 0;
        if (index >= 0) {
            const label = candidateLabel(dataset.selection, dataset.names[inspected]);
            inspector.textContent = `${label} · Parameter ${formatValue(x[index])} · Metric ${formatValue(y[index])}`;
        }
    }

    async function draw() {
        pending = true;
        if (rendering) return;
        rendering = true;
        try {
            while (pending) {
                pending = false;
                if (!parameterKey || chart.hidden) continue;
                const parameter = parameters.get(parameterKey), metric = metrics.get(metricInput.value);
                const {x, y, indices} = completeParameterPairs(parameter, metric, rows);
                const labels = indices.map(row => escapeHtml(candidateLabel(dataset.selection, dataset.names[row])));
                caption.textContent = parameter.key + " × " + metric.label;
                await Plotly.react(plot, [{
                    type: "scatter", mode: "markers", x, y, customdata: indices, text: labels,
                    marker: {size: 7, color: "#287b91", opacity: 0.8},
                    hovertemplate: "<b>%{text}</b><br>Parameter: %{x:.8g}<br>Metric: %{y:.8g}<extra></extra>",
                }], {
                    margin: {l: 62, r: 20, t: 25, b: 60},
                    paper_bgcolor: "white", plot_bgcolor: "white",
                    xaxis: {title: {text: "Parameter"}, tickformat: ".4~g", automargin: true},
                    yaxis: {title: {text: "Metric"}, tickformat: ".4~g", automargin: true},
                    showlegend: false,
                }, {responsive: true, displaylogo: false, toImageButtonOptions: {format: "png", scale: 2}});
                if (!plotted && onCandidateClick) {
                    plot.on("plotly_click", event => {
                        const row = event.points[0]?.customdata;
                        if (Number.isInteger(row)) {
                            inspected = row;
                            const trace = plot.data[0];
                            renderInspected(trace.customdata, trace.x, trace.y);
                            onCandidateClick(row);
                        }
                    });
                }
                plotted = true;
                renderInspected(indices, x, y);
                error.hidden = true;
            }
        } catch (exception) {
            error.textContent = "Unable to render parameter scatter: " + exception.message;
            error.hidden = false;
        } finally {
            rendering = false;
        }
    }
    function selectPair(key, button) {
        pressedButton?.setAttribute("aria-pressed", "false");
        pressedButton = button;
        button.setAttribute("aria-pressed", "true");
        parameterKey = key;
        draw();
    }
    function renderRankings(redraw = false) {
        const query = queryInput.value.trim().toLowerCase();
        const matches = rankings.filter(item => item.key.toLowerCase().includes(query));
        const shown = matches.slice(0, 20);
        const hadChart = !chart.hidden;
        const previousKey = parameterKey;
        table.hidden = chart.hidden = shown.length === 0;
        results.replaceChildren();
        if (!shown.length) {
            pressedButton = null;
            if (rows.length < 3) summary.textContent = "At least three candidates must pass the plot limits.";
            else if (!parameters.size) summary.textContent = "No numeric bot parameter varies across at least three saved candidates.";
            else if (!rankings.length) summary.textContent = "This metric has no varying parameter pairs with at least three finite observations.";
            else summary.textContent = "No ranked parameters match your search.";
            return;
        }
        const count = rows.length.toLocaleString();
        summary.textContent = `Showing ${shown.length} of ${matches.length} matching parameters, ranked by absolute correlation. Pairs are finite observations out of ${count} candidates passing the limits.`;
        const target = shown.find(item => item.key === parameterKey) || shown[0];
        for (const item of shown) {
            const tr = results.insertRow();
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = item.key;
            const selected = item === target;
            button.setAttribute("aria-pressed", String(selected));
            button.addEventListener("click", () => selectPair(item.key, button));
            tr.insertCell().append(button);
            tr.insertCell().textContent = (item.corr > 0 ? "+" : "") + item.corr.toFixed(3);
            tr.insertCell().textContent = item.count.toLocaleString() + " / " + count;
            if (selected) pressedButton = button;
        }
        parameterKey = target.key;
        if (redraw || !hadChart || previousKey !== parameterKey) draw();
    }
    function recompute() {
        const metric = metrics.get(metricInput.value), method = methodInput.value;
        document.getElementById("correlation-coefficient").textContent =
            method === "spearman" ? "Spearman ρ" : "Pearson r";
        rankings = metric && rows.length >= 3 ?
            parameterMetricCorrelations(parameters.values(), metric, rows, method) : [];
        renderRankings(true);
    }

    metricInput.addEventListener("change", recompute);
    methodInput.addEventListener("change", recompute);
    queryInput.addEventListener("input", () => renderRankings());
    let initialized = false;
    return nextRows => {
        if (initialized && rows.length === nextRows.length && rows.every((row, i) => row === nextRows[i])) return;
        initialized = true;
        rows = nextRows;
        recompute();
    };
}
"""
