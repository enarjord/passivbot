"""Bundled offline page and browser logic for the Pareto plot CLI."""

from tools.pareto_plot_correlations import (
    CORRELATION_PANEL,
    CORRELATION_SCRIPT,
    CORRELATION_STYLES,
)
from tools.pareto_plot_selectors import SELECTOR_SCRIPT
from tools.pareto_plot_tradeoffs import TRADEOFF_CSS, TRADEOFF_HTML, TRADEOFF_SCRIPT

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pareto explorer</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#243449;font:14px system-ui,sans-serif}
header,main{max-width:1650px;margin:auto;padding:0 24px 24px}header{padding-top:23px;padding-bottom:14px;display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap}h1{font-size:29px;letter-spacing:-.03em;margin:0}
p{line-height:1.5;margin:7px 0;color:#5b6c81}.toolbar{display:flex;gap:12px;flex-wrap:wrap;padding:14px 16px}
label{display:block;font-size:12px;font-weight:650;margin-bottom:5px}select,input[type=number],button{font:inherit;border:1px solid #cbd5e1;border-radius:6px;padding:7px;background:white;color:#243449}
select{max-width:100%;width:100%}.axis{flex:1 1 200px;min-width:0}.axis .goal{margin-top:6px;font-size:12px}.mode{width:68px}button{cursor:pointer}button:hover{background:#eef4fa}button:disabled{cursor:default;opacity:.55}
.tabs{display:flex;gap:5px;overflow:auto;margin-bottom:12px}.tabs button{padding:9px 13px;white-space:nowrap;border-color:transparent;background:transparent;font-weight:600}.tabs button[aria-selected=true]{background:#dceef0;color:#145264}.tabs button:hover{background:#e6edf4}
.workspace{display:grid;grid-template-columns:minmax(0,1fr) minmax(245px,285px);gap:14px}.chart,.panel,.limits-sidebar,.selection{background:white;border:1px solid #dce3ed;border-radius:12px;min-width:0}.views{min-width:0}.chart{overflow:hidden}
#plot{width:100%;height:clamp(380px,56vh,650px)}.summary{padding:10px 16px 0;font-weight:650}#ideal{padding:0 16px 8px;color:#675023;font-size:12px;overflow-wrap:anywhere;white-space:pre-line}
.note{padding:0 16px 12px;font-size:12px}.limits-sidebar{padding:15px;align-self:start}h2{font-size:17px;margin:0 0 10px}.limit{border-top:1px solid #e2e8f0;padding:13px 0}.limit-title{overflow-wrap:anywhere;font-size:12px;margin-bottom:8px}.limit-title input{margin-right:7px}.limit-controls{display:grid;grid-template-columns:94px minmax(0,1fr);gap:6px}.limit input[type=range]{width:100%;accent-color:#287b91;margin:10px 0 0}.limit input[type=number]{width:100%;min-width:0}.muted{font-size:12px;color:#5b6c81}.add-filter{display:flex;gap:6px;margin:12px 0}.add-filter select{min-width:0}.hidden,[hidden]{display:none!important}#error{color:#b42318;padding:14px;margin-bottom:10px;background:#fff1ef;border-radius:6px}details{margin-top:12px}summary{cursor:pointer;font-size:13px;font-weight:650}
.ranking-controls{padding:12px 16px;border-block:1px solid #e7ecf2;background:#f8fafc}
.ranking-controls .fields,.weights{display:flex;gap:12px;flex-wrap:wrap;align-items:end}
.ranking-controls select{min-width:200px}.weights input{width:65px}.ranking-controls p{margin:6px 0 0}
.selection{padding:16px;margin-top:14px;overflow-wrap:anywhere}.selection h2{margin-bottom:7px}
.shortlist{border-top:1px solid #e7ecf2;padding:0 16px 14px;overflow-wrap:anywhere}
.link-button{background:transparent;border:0;text-decoration:underline;color:#175d73;padding:0;text-align:left;font:inherit}
#inspected{margin:7px 16px 12px;padding:8px 10px;background:#eef5f7;border-radius:5px;overflow-wrap:anywhere}
.panel{padding:16px}.limit-title{display:flex;gap:7px;align-items:start}.limit-title input{margin:1px 0 0}
.selection h3{font-size:13px;margin:12px 0 6px}.selection summary{font-weight:600}.selection .pick{color:#6239a8}
.table-wrap{overflow:auto;margin-top:8px}.table-wrap table{width:100%;border-collapse:collapse;font-size:12px}
.table-wrap th,.table-wrap td{padding:7px 9px;text-align:left;vertical-align:top;border-bottom:1px solid #e2e8f0}
.table-wrap th{background:#f4f6fa;min-width:70px}.table-wrap td .muted{display:block}.table-wrap td.filename{min-width:160px;font-family:monospace}
@media(max-width:860px){.workspace{grid-template-columns:1fr}header{padding:16px 12px}main{padding:0 12px 12px}}
@media(max-width:520px){.axis{flex-basis:130px}.tabs button{padding:8px}.toolbar,.panel{padding:12px}#plot{height:370px}}
__PANEL_STYLES__
</style><script>__PLOTLY_JS__</script></head>
<body><header><h1>Pareto explorer</h1><span class="muted" id="pool-summary" role="status"></span></header>
<main>
<div class="tabs" role="tablist" aria-label="Explorer views">
<button type="button" role="tab" id="tab-scatter" aria-controls="scatter-panel" aria-selected="true">Scatter</button>
<button type="button" role="tab" id="tab-tradeoffs" aria-controls="tradeoffs-panel" aria-selected="false" tabindex="-1">Metric profiles</button>
<button type="button" role="tab" id="tab-correlations" aria-controls="correlations-panel" aria-selected="false" tabindex="-1">Parameter correlations</button>
</div>
<div id="error" role="alert" hidden></div>
<div class="workspace"><div class="views">
<section class="chart" role="tabpanel" id="scatter-panel" aria-labelledby="tab-scatter">
<div class="toolbar" aria-label="Plot controls">
<div class="mode"><label for="dimension">View</label><select id="dimension"><option value="2">2D</option><option value="3">3D</option></select></div>
<div class="axis"><label for="axis-x">X metric</label><select id="axis-x"></select><select class="goal" id="goal-x" aria-label="X ideal direction"></select></div>
<div class="axis"><label for="axis-y">Y metric</label><select id="axis-y"></select><select class="goal" id="goal-y" aria-label="Y ideal direction"></select></div>
<div class="axis" id="z-controls"><label for="axis-z">Z metric</label><select id="axis-z"></select><select class="goal" id="goal-z" aria-label="Z ideal direction"></select></div>
</div>
<div class="ranking-controls"><div class="fields">
<div><label for="projection-method">Selector on plotted axes</label><select id="projection-method" aria-describedby="projection-help"></select></div>
<div id="projection-weights" class="weights"></div>
<button id="download-view" type="button">Download view JSON</button>
</div><p id="projection-help" class="muted"></p></div>
<div class="summary" id="summary" role="status"></div><div id="plot"></div><div id="ideal"></div>
<div id="inspected" role="status" hidden></div>
<p class="note">Click a candidate to inspect it. The diamond combines the best visible value on each axis. Use the plot toolbar to zoom, reset or export PNG.</p>
<div class="shortlist"><details id="projection-shortlist"><summary>Plotted-axis shortlist</summary><div id="projection-ranking"></div></details></div>
</section>
<section class="panel" role="tabpanel" id="tradeoffs-panel" aria-labelledby="tab-tradeoffs" hidden>__TRADEOFF_PANEL__</section>
<section class="panel" role="tabpanel" id="correlations-panel" aria-labelledby="tab-correlations" hidden>__CORRELATION_PANEL__</section>
</div><aside class="limits-sidebar"><h2>Plot limits</h2><p class="muted">Limits apply across all views. Missing values fail an enabled limit.</p>
<div class="add-filter"><select id="filter-metric" aria-label="Additional limit metric"></select><button id="add-filter">Add</button></div>
<button id="reset-limits">Reset all limits</button><div id="limits"></div>
<details><summary>About saved metrics</summary><p class="muted">Objectives are raw values before penalties. Other metrics use saved aggregates or means. Statistics are listed separately. A scatter coordinate requires a finite value.</p></details>
</aside></div>
<section class="selection" aria-labelledby="selection-title"><h2 id="selection-title">CLI selection</h2><div id="selection"></div></section>
</main><script id="pareto-data" type="application/json">__DATA__</script><script>__APP_SCRIPT__</script></body></html>'''

EXPLORER_SCRIPT = r'''
const finite = value => typeof value === "number" && Number.isFinite(value);
function computeView(dataset, axes, goals, limits, method = "ideal", weights = null) {
    const metrics = new Map(dataset.metrics.map(metric => [metric.key, metric]));
    const columns = axes.map(key => metrics.get(key).values);
    const limitValue = (key, row) => {
        const metric = metrics.get(key), status = metric.statuses?.[row]?.status;
        return status === "positive_infinity" ? Infinity : status === "negative_infinity" ? -Infinity : metric.values[row];
    };
    const active = limits.filter(limit => limit.enabled);
    const indices = [], retained = [];
    let filtered = 0, missing = 0;
    for (let row = 0; row < dataset.names.length; row++) {
        if (!active.every(limit => {
            const value = limitValue(limit.key, row);
            return typeof value === "number" && !Number.isNaN(value) && (limit.mode === "floor" ? value >= limit.value : value <= limit.value);
        })) { filtered++; continue; }
        retained.push(row);
        if (!columns.every(values => finite(values[row]))) { missing++; continue; }
        indices.push(row);
    }
    return {indices, retained, filtered, missing, ...rankProjection(columns, indices, goals, method, weights)};
}
function finiteExtent(values, fallback) {
    if (!values.length) return fallback;
    let min = Infinity, max = -Infinity;
    for (const value of values) {
        min = Math.min(min, value);
        max = Math.max(max, value);
    }
    return {min, max};
}
function axisRange(metric, goal, dimensions) {
    const low = metric.min, high = metric.max;
    const padding = (high - low || Math.abs(low) || 1) * 0.06;
    const range = [low - padding, high + padding];
    return dimensions === 2 && goal === "max" ? range.reverse() : range;
}
const escapeHtml = text => String(text).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const formatValue = value => finite(value) ? Number(value.toPrecision(8)).toString() : "unavailable";
function candidateLabel(record, filename) {
    return record?.candidate_labels?.[filename] ?? filename;
}
function plotSpec(dataset, axes, goals, limits, method = "ideal", weights = null, inspected = null) {
    const metrics = new Map(dataset.metrics.map(metric => [metric.key, metric]));
    const view = computeView(dataset, axes, goals, limits, method, weights), is3d = axes.length === 3;
    const columns = axes.map(key => view.indices.map(row => metrics.get(key).values[row]));
    const extents = axes.map((key, i) => is3d ? metrics.get(key) : finiteExtent(columns[i], metrics.get(key)));
    const hover = axes.map((key, i) => `${escapeHtml(metrics.get(key).label)}: %{${"xyz"[i]}:.8g}`).join("<br>");
    const candidateHover = filename => {
        const label = candidateLabel(dataset.selection, filename);
        return [escapeHtml(label), label === filename ? "" : "<br>File: " + escapeHtml(filename)];
    };
    const points = {type:is3d ? "scatter3d" : "scatter", mode:"markers", name:"Candidates", showlegend:false,
        x:columns[0], y:columns[1], ids:view.indices.map(String),
        customdata:view.indices.map(row => candidateHover(dataset.names[row])),
        hovertemplate:"<b>%{customdata[0]}</b>%{customdata[1]}<br>" + hover + "<extra></extra>",
        marker:{size:is3d ? 5 : 9, opacity:0.85, color:columns[axes.length-1], colorscale:"Viridis",
            cmin:extents.at(-1).min, cmax:extents.at(-1).max,
            line:{width:0.4,color:"white"}}};
    if (is3d) points.z = columns[2];
    const traces = [points];
    if (view.ideal) {
        const ideal = {type:points.type, name:"Theoretical ideal", mode:"markers",
            x:[view.ideal[0]], y:[view.ideal[1]], showlegend:true,
            marker:{symbol:"diamond-open",size:is3d ? 13 : 27,color:"#287b91",line:{color:"#287b91",width:2}},
            hovertemplate:"<b>Theoretical ideal</b><br>" + hover + "<extra></extra>"};
        if (is3d) ideal.z = [view.ideal[2]];
        traces.push(ideal);
    }
    if (view.selected !== null) {
        const values = axes.map(key => metrics.get(key).values[view.selected]);
        const label = method === "ideal" ? "Closest on plotted axes" : PROJECTION_METHODS[method].label + " pick";
        const chosen = {type:points.type, name:label, mode:is3d ? "markers+text" : "markers",
            x:[values[0]], y:[values[1]], showlegend:true,
            customdata:[candidateHover(dataset.names[view.selected])], ids:[String(view.selected)],
            marker:{symbol:is3d ? "circle" : "star",size:is3d ? 6 : 20,color:"#f4ae2b",line:{color:"#704907",width:1.5}},
            hovertemplate:"<b>" + label + ": %{customdata[0]}</b>%{customdata[1]}<br>" + hover + "<extra></extra>"};
        if (is3d) {
            Object.assign(chosen,{z:[values[2]],text:["★"],textposition:"middle center",textfont:{size:32,color:"#e69b12"}});
        }
        traces.push(chosen);
    }
    const pick = dataset.selection?.selected?.file;
    const pickRow = dataset.names.indexOf(pick);
    let pickVisibility = null;
    if (pickRow >= 0) {
        const values = axes.map(key => metrics.get(key).values[pickRow]);
        pickVisibility = !values.every(finite) ? "missing coordinates" :
            view.indices.includes(pickRow) ? "visible" : "outside plot limits";
        if (pickVisibility === "visible") {
            const pickTrace = {type:points.type, name:"Selection pick", mode:"markers", showlegend:true,
                x:[values[0]], y:[values[1]], customdata:[candidateHover(pick)], ids:[String(pickRow)],
                marker:{symbol:"circle-open",size:is3d ? 21 : 37,color:"#673ab7",line:{color:"#673ab7",width:2.5}},
                hovertemplate:"<b>Selection pick: %{customdata[0]}</b>%{customdata[1]}<br>" + hover + "<extra></extra>"};
            if (is3d) pickTrace.z = [values[2]];
            traces.splice(1, 0, pickTrace);
        }
    }
    if (inspected !== null && view.indices.includes(inspected)) {
        const values = axes.map(key => metrics.get(key).values[inspected]);
        const trace = {type:points.type, name:"Inspected candidate", mode:"markers", showlegend:false,
            x:[values[0]], y:[values[1]], ids:[String(inspected)], customdata:[candidateHover(dataset.names[inspected])],
            marker:{symbol:"square-open",size:is3d ? 16 : 25,color:"#146471",line:{color:"#146471",width:2}},
            hovertemplate:"<b>%{customdata[0]}</b>%{customdata[1]}<br>" + hover + "<extra></extra>"};
        if (is3d) trace.z = [values[2]];
        traces.push(trace);
    }
    // Refit both 2D axes on a filter edit, even after manual zoom. Keep the 3D scene stable.
    const activeLimits = limits.filter(limit => limit.enabled).map(({key, mode, value}) => [key, mode, value]);
    const revision = JSON.stringify(is3d ? [axes, goals] : [axes, goals, activeLimits]);
    const layout = {paper_bgcolor:"white",plot_bgcolor:"white",font:{family:"Arial, sans-serif",color:"#243449",size:12},
        margin:{l:75,r:24,t:50,b:80},hoverlabel:{bgcolor:"white"},uirevision:revision,
        legend:{orientation:"h",x:0,y:1.01,xanchor:"left",yanchor:"bottom"},
        annotations:view.indices.length ? [] : [{text:"No points to display",xref:"paper",yref:"paper",x:0.5,y:0.5,showarrow:false}]};
    const axis = (key, index) => {
        const metric = metrics.get(key), goal = goals[index];
        const label = escapeHtml(metric.label).replace(/(.{20,35}_)/g, "$1<br>");
        return {title:{text:label + "<br>(" + (goal === "max" ? "higher is better" : goal === "min" ? "lower is better" : "choose ideal direction") + ")",font:{size:11},standoff:15},
            range:axisRange(extents[index],goal,axes.length),autorange:false,gridcolor:"#e1e7ef",zeroline:false,tickformat:".4~g",automargin:true};
    };
    if (is3d) layout.scene = {xaxis:axis(axes[0],0),yaxis:axis(axes[1],1),zaxis:axis(axes[2],2),
        aspectmode:"cube",dragmode:"orbit",uirevision:revision,camera:{eye:{x:1.65,y:1.65,z:1.2}}};
    else Object.assign(layout,{xaxis:axis(axes[0],0),yaxis:axis(axes[1],1)});
    return {traces,layout,view,pickVisibility};
}
function preserveCamera(layout, previous) {
    if (layout.scene && previous?.scene?.camera && layout.uirevision === previous.uirevision) {
        layout.scene.camera = JSON.parse(JSON.stringify(previous.scene.camera));
    }
    return layout;
}
function appendText(parent, tag, text, className) {
    const node = document.createElement(tag);
    node.textContent = text;
    if (className) node.className = className;
    parent.append(node);
    return node;
}
function appendCandidateLabel(parent, tag, record, filename, className, prefix = "") {
    const label = candidateLabel(record, filename);
    const node = appendText(parent, tag, prefix + label, className);
    if (label !== filename) node.title = filename;
    return node;
}
function detailsSection(parent, title) {
    const details = document.createElement("details");
    parent.append(details);
    appendText(details, "summary", title);
    return details;
}
function reportTable(parent, headings) {
    const wrap = appendText(parent, "div", "", "table-wrap");
    const table = document.createElement("table");
    wrap.append(table);
    const header = table.createTHead().insertRow();
    for (const heading of headings) appendText(header, "th", heading).scope = "col";
    return table.createTBody();
}
function metricValue(values, metric, signed = false) {
    const value = values?.[metric], status = values?.metric_status?.[metric]?.status;
    if (finite(value)) return (signed && value > 0 ? "+" : "") + formatValue(value);
    if (status === "positive_infinity") return "+∞";
    if (status === "negative_infinity") return "−∞";
    return status || "unavailable";
}
function renderShortlist(parent, record) {
    const rows = record.top_candidates, objectives = record.objectives.map(item => item.metric);
    const details = detailsSection(parent, `Ranked shortlist (${rows.length})`);
    const hasMargins = rows.some(row => finite(row.score_margin));
    const hasSecondary = rows.some(row => finite(row.secondary_score));
    const header = ["Rank", "Candidate", hasSecondary ? "Primary score" : "Score"];
    if (hasSecondary) header.push("Tie-break score");
    if (hasMargins) header.push("Gap to pick");
    header.push(...objectives);
    const body = reportTable(details, header);
    for (const entry of rows) {
        const row = body.insertRow();
        appendText(row, "td", entry.rank);
        appendCandidateLabel(row, "td", record, entry.file, "filename");
        appendText(row, "td", formatValue(entry.score));
        if (hasSecondary) appendText(row, "td", formatValue(entry.secondary_score));
        if (hasMargins) appendText(row, "td", finite(entry.score_margin) ? formatValue(entry.score_margin) : "—");
        for (const metric of objectives) {
            const cell = appendText(row, "td", metricValue(entry.objectives, metric));
            if (entry.file !== record.selected.file && entry.objective_deltas) {
                appendText(cell, "span", "Δ " + metricValue(entry.objective_deltas, metric, true), "muted");
            }
        }
    }
    appendText(details, "p", "Δ is the raw candidate value minus the selection pick. Score gaps compare candidates under this selection policy.", "muted");
}
function describeSetting(value) {
    if (Array.isArray(value)) return value.map(describeSetting).join(", ");
    if (value && typeof value === "object") {
        return Object.entries(value).map(([key, item]) => `${key} = ${describeSetting(item)}`).join(" · ");
    }
    return String(value);
}
function renderSelectionPolicy(parent, record) {
    const details = detailsSection(parent, "Selection policy and provenance");
    appendText(details, "p", record.method_description);
    const fields = [
        ["Normalization pool", record.normalization_scope],
        ["Objectives", record.objectives.map(item => {
            const basis = [item.goal, item.reducer, item.scenario].filter(Boolean);
            return `${item.metric} (${basis.join(", ")})`;
        })],
        ["Scenario", record.scenario],
        ["CLI limits", record.applied_limits.length ? record.applied_limits : "None"],
        ["Active Pareto members", record.active_front_count],
        ["Selection record version", record.schema_version],
        ["Candidate set SHA-256", record.candidate_set_sha256],
        ["Selected filename", record.selected.file],
        ["Selected file SHA-256", record.selected.file_sha256],
    ];
    for (const [key, value] of Object.entries(record.settings || {})) {
        if (value != null && (typeof value !== "object" || Object.keys(value).length)) {
            fields.push([key.replaceAll("_", " "), value]);
        }
    }
    for (const [label, value] of fields) {
        if (value != null) appendText(details, "p", `${label}: ${describeSetting(value)}`, "muted");
    }
}
function renderSensitivity(parent, record) {
    const sensitivity = record.sensitivity;
    if (!sensitivity?.checks?.length) return;
    const changed = sensitivity.changed_check_count;
    const total = sensitivity.check_count;
    const details = detailsSection(parent, `Sensitivity (${changed} of ${total} checks change the pick)`);
    const weightStep = sensitivity.weight_transfer_fraction;
    const targetStep = sensitivity.target_step_scales;
    const body = reportTable(details, ["Change tested", "Selection pick"]);
    for (const check of sensitivity.checks) {
        const row = body.insertRow();
        const change = check.kind === "weight_transfer"
            ? `Move ${formatValue(100 * weightStep)} percentage points of weight from ${check.from_metric} to ${check.to_metric}`
            : `Set ${check.metric} target to ${formatValue(check.target)} (${formatValue(targetStep)} scales from the original target)`;
        appendText(row, "td", change);
        appendCandidateLabel(row, "td", record, check.winner, "filename");
    }
}
function renderQualification(parent, record) {
    const evidence = record.qualification;
    if (!evidence) {
        appendText(parent, "p", "Replay evidence: none supplied.", "muted");
        return;
    }
    const candidate = evidence.candidates[record.selected.file_sha256];
    appendText(parent, "p", candidate
        ? "Replay evidence supplied for the selection pick."
        : "Replay evidence for the selection pick is unavailable.", "muted");
    const details = detailsSection(parent, "Replay evidence");
    appendText(details, "p", `Engine: ${evidence.engine} · SHA-256 ${evidence.engine_sha256}`, "muted");
    const comparisons = record.top_candidates.map(row => [row.file, evidence.candidates[row.file_sha256]]);
    if (evidence.incumbent) comparisons.push(["Incumbent", evidence.incumbent]);
    for (const [name, evidenceRow] of comparisons) {
        appendCandidateLabel(details, "h3", record, name);
        if (!evidenceRow) {
            appendText(details, "p", "No replay evidence supplied for this file.", "muted");
            continue;
        }
        appendText(details, "p", `Effective config SHA-256: ${evidenceRow.effective_config_sha256}`, "muted");
        const body = reportTable(details, ["Scenario", "Replay outcome", "Metrics"]);
        for (const [label, scenario] of Object.entries(evidence.scenarios)) {
            const row = body.insertRow();
            const result = Object.hasOwn(evidenceRow.results, label) ? evidenceRow.results[label] : undefined;
            const context = appendText(row, "td", label);
            appendText(context, "span", `${scenario.split} · ${scenario.execution} · ${scenario.start_date} through ${scenario.end_date}`, "muted");
            context.title = `Inputs SHA-256: ${scenario.inputs_sha256}`;
            const outcome = appendText(row, "td", result?.status || "Evidence unavailable");
            if (typeof result?.liquidated === "boolean") {
                appendText(outcome, "span", result.liquidated ? "Liquidated" : "No liquidation", "muted");
            }
            if (result?.reason) appendText(outcome, "span", result.reason, "muted");
            if (result?.effective_start_date && result?.effective_end_date) {
                appendText(outcome, "span", `Effective dates: ${result.effective_start_date} through ${result.effective_end_date}`, "muted");
            }
            if (result?.report) appendText(outcome, "span", `Report: ${result.report}`, "muted");
            const metrics = Object.keys(result?.metrics || {}).filter(key => key !== "metric_status");
            appendText(row, "td", metrics.map(key => `${key} = ${metricValue(result.metrics, key)}`).join(" · ") || "—");
        }
    }
}
function renderSelection(record) {
    const parent = document.getElementById("selection");
    if (!record?.selected) {
        appendText(parent, "p", "CLI selection is unavailable.");
        return;
    }
    appendCandidateLabel(parent, "strong", record, record.selected.file, "pick", "○ Selection pick: ");
    appendText(parent, "p", `${record.method} · ${record.objectives.length} active objectives · ${record.selection_count} ranked · ${record.retained_count} of ${record.loaded_count} retained after CLI limits`);
    appendText(parent, "p", "This ranking is fixed at export. Change selection settings with the CLI and regenerate the page.", "muted");
    renderQualification(parent, record);
    renderShortlist(parent, record);
    renderSensitivity(parent, record);
    renderSelectionPolicy(parent, record);
}
const PROJECTION_SHORTLIST_SIZE = 10;
function renderProjectionRanking(parent, dataset, axes, view, onCandidateClick) {
    parent.replaceChildren();
    if (!view.ranking.length) {
        appendText(parent, "p", "No candidates ranked with the current axes and selector.", "muted");
        return;
    }
    const metrics = new Map(dataset.metrics.map(metric => [metric.key, metric]));
    const method = PROJECTION_METHODS[view.mode];
    const body = reportTable(parent, ["Rank", "Candidate", method.scoreLabel, ...axes]);
    for (const [index, {row, score}] of view.ranking.slice(0, PROJECTION_SHORTLIST_SIZE).entries()) {
        const item = body.insertRow();
        appendText(item, "td", index + 1);
        const cell = item.insertCell();
        const button = appendCandidateLabel(cell, "button", dataset.selection, dataset.names[row], "link-button");
        button.type = "button";
        button.addEventListener("click", () => onCandidateClick(row));
        appendText(item, "td", formatValue(score));
        for (const key of axes) appendText(item, "td", formatValue(metrics.get(key).values[row]));
    }
    appendText(parent, "p", `Showing ${Math.min(PROJECTION_SHORTLIST_SIZE, view.eligible)} of ${view.eligible} ranked candidates. Download the view JSON for the full ranking.`, "muted");
}
function buildViewRecord(dataset, axes, goals, limits, method, weights, view) {
    const metrics = new Map(dataset.metrics.map(metric => [metric.key, metric]));
    const values = row => Object.fromEntries(axes.map(key => [key, metrics.get(key).values[row]]));
    const score = PROJECTION_METHODS[view.mode];
    return {
        schema_version: 1,
        metric_definition_version: dataset.definition_version,
        candidate_set_sha256: dataset.selection?.candidate_set_sha256 ?? null,
        scope: "plotted_axes",
        scenario: dataset.selection?.scenario ?? null,
        cli_limits: dataset.selection?.applied_limits ?? [],
        axes: axes.map((metric, index) => ({metric, goal: goals[index], weight: weights?.[index] ?? null})),
        limits: limits.filter(limit => limit.enabled).map(({key, mode, value}) => ({metric: key, mode, value})),
        method,
        effective_method: view.mode,
        score_label: score.scoreLabel,
        score_goal: score.order,
        ideal: view.ideal && Object.fromEntries(axes.map((key, index) => [key, view.ideal[index]])),
        count_after_limits: view.retained.length,
        missing_coordinates: view.missing,
        cli_pick: dataset.selection?.selected?.file ?? null,
        selected: view.selected === null ? null : dataset.names[view.selected],
        ranking: view.ranking.map(({row, score}, index) => ({rank: index + 1, file: dataset.names[row], score, metrics: values(row)})),
        visible_candidates: view.indices.map(row => dataset.names[row]),
    };
}
function downloadView(record) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2) + "\n"], {type: "application/json"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = "pareto-view.json";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function mountExplorer(dataset) {
    renderSelection(dataset.selection);
    const byKey = new Map(dataset.metrics.map(metric => [metric.key,metric]));
    const available = dataset.metrics.filter(metric => metric.count);
    const selected = dataset.selected.slice();
    if (selected.length === 2) selected.push(available.find(metric => !selected.includes(metric.key))?.key);
    const directions = new Map(dataset.metrics.map(metric => [metric.key,metric.goal]));
    const limits = new Map(), extras = new Set();
    const dimension = document.getElementById("dimension"), axes = ["x","y","z"];
    dimension.value = String(dataset.selected.length);
    dimension.options[1].disabled = available.length < 3;
    const axisSelects = axes.map(axis => document.getElementById(`axis-${axis}`));
    const goalSelects = axes.map(axis => document.getElementById(`goal-${axis}`));
    const activeAxes = () => selected.slice(0,Number(dimension.value));
    const tabs = [...document.querySelectorAll('[role="tab"]')];
    const scatterTab = tabs[0];
    let activeTab = scatterTab;
    const methodSelect = document.getElementById("projection-method");
    for (const [name, {label}] of Object.entries(PROJECTION_METHODS)) methodSelect.append(new Option(label, name));
    const weights = new Map();
    const plot = document.getElementById("plot");
    let inspected = null, rankingView = null, hasClickHandler = false;
    const inspectCandidate = row => {
        inspected = row;
        if (activeTab === scatterTab) schedulePlot();
    };
    const tradeoffs = mountTradeoffs(dataset, fillOptions, inspectCandidate);
    const updateCorrelations = mountParameterCorrelations(dataset, inspectCandidate);
    function fillOptions(select) {
        for (const group of ["Objectives","Metrics","Statistics"]) {
            const optgroup = document.createElement("optgroup"); optgroup.label = group;
            for (const metric of dataset.metrics.filter(item => item.group === group)) {
                const option = new Option(metric.label + (metric.count < dataset.names.length ? ` (${metric.count}/${dataset.names.length})` : ""),metric.key);
                option.disabled = !metric.count; optgroup.append(option);
            }
            if (optgroup.children.length) select.append(optgroup);
        }
    }
    axisSelects.forEach(fillOptions);
    goalSelects.forEach(select => {
        select.append(new Option("Choose ideal direction…",""),new Option("Higher is better","max"),new Option("Lower is better","min"));
    });
    const filterMetric = document.getElementById("filter-metric"); fillOptions(filterMetric);
    function ensureLimit(key) {
        if (!limits.has(key)) {
            const metric = byKey.get(key), mode = directions.get(key) === "min" ? "ceiling" : "floor";
            limits.set(key,{key,mode,value:mode === "floor" ? metric.min : metric.max,enabled:false});
        }
        return limits.get(key);
    }
    function syncAxes() {
        document.getElementById("z-controls").classList.toggle("hidden",dimension.value === "2");
        axisSelects.forEach((select,index) => {
            select.value = selected[index] || "";
            for (const option of select.options) option.disabled = !byKey.get(option.value).count || activeAxes().some((key,i) => i !== index && key === option.value);
            goalSelects[index].value = directions.get(selected[index]) || "";
        });
        renderLimits(); renderProjectionControls(); schedulePlot();
    }
    function renderProjectionControls() {
        const container = document.getElementById("projection-weights");
        container.replaceChildren();
        const method = PROJECTION_METHODS[methodSelect.value];
        document.getElementById("projection-help").textContent = method.description;
        if (!method.weighted) return;
        for (const [index, key] of activeAxes().entries()) {
            const control = document.createElement("div");
            const label = appendText(control, "label", `${axes[index].toUpperCase()} weight`);
            const input = document.createElement("input");
            input.id = `projection-weight-${axes[index]}`;
            input.type = "number";
            input.min = "0";
            input.step = "any";
            input.value = String(weights.get(key) ?? 1);
            label.htmlFor = input.id;
            input.addEventListener("change", () => {
                if (finite(input.valueAsNumber) && input.valueAsNumber >= 0) {
                    weights.set(key, input.valueAsNumber);
                    schedulePlot();
                } else {
                    input.value = String(weights.get(key) ?? 1);
                }
            });
            control.append(input);
            container.append(control);
        }
    }
    const viewWeights = (keys, method) => PROJECTION_METHODS[method].weighted
        ? keys.map(key => weights.get(key) ?? 1) : null;
    function renderLimits() {
        const container = document.getElementById("limits"); container.replaceChildren();
        const keys = new Set([...activeAxes(),...extras,...[...limits.values()].filter(limit => limit.enabled).map(limit => limit.key)]);
        for (const key of keys) {
            const metric = byKey.get(key), state = ensureLimit(key);
            const card = document.createElement("div"); card.className = "limit"; card.dataset.metric = key;
            const title = document.createElement("label"); title.className = "limit-title";
            const enabled = document.createElement("input"); enabled.type = "checkbox"; enabled.checked = state.enabled; enabled.setAttribute("aria-label",`Enable limit for ${metric.label}`);
            title.append(enabled);
            appendText(title, "span", metric.label);
            const controls = document.createElement("div"); controls.className = "limit-controls";
            const mode = document.createElement("select"); mode.setAttribute("aria-label",`Limit direction for ${metric.label}`);
            mode.append(new Option("Floor ≥","floor"),new Option("Ceiling ≤","ceiling")); mode.value = state.mode;
            const numeric = document.createElement("input"); numeric.type = "number"; numeric.step = "any"; numeric.value = String(state.value); numeric.setAttribute("aria-label",`Limit value for ${metric.label}`);
            const slider = document.createElement("input"); slider.type = "range"; slider.min = String(metric.min); slider.max = String(metric.max); slider.step = "any"; slider.value = String(state.value); slider.disabled = metric.min === metric.max; slider.setAttribute("aria-label",`Limit slider for ${metric.label}`);
            const setValue = value => { state.value = value; state.enabled = true; enabled.checked = true; numeric.value = String(value); slider.value = String(value); schedulePlot(); };
            slider.addEventListener("input",() => setValue(slider.valueAsNumber));
            numeric.addEventListener("change",() => {
                if (finite(numeric.valueAsNumber)) setValue(numeric.valueAsNumber);
                else numeric.value = String(state.value);
            });
            mode.addEventListener("change",() => {state.mode = mode.value; schedulePlot();});
            enabled.addEventListener("change",() => {state.enabled = enabled.checked; schedulePlot();});
            const range = document.createElement("div"); range.className = "muted";
            range.textContent = `Saved range: ${formatValue(metric.min)} to ${formatValue(metric.max)}`;
            controls.append(mode,numeric); card.append(title,controls,slider,range); container.append(card);
        }
    }
    let pending = false, rendering = false;
    function schedulePlot() {
        if (pending) return;
        pending = true;
        if (!rendering) requestAnimationFrame(renderPlot);
    }
    async function renderPlot() {
        if (rendering || !pending) return;
        rendering = true;
        try {
            while (pending) {
                pending = false;
                const keys = activeAxes(), goals = keys.map(key => directions.get(key));
                const limitValues = [...limits.values()], method = methodSelect.value;
                const axisWeights = viewWeights(keys, method);
                const currentTab = activeTab;
                const result = currentTab === scatterTab
                    ? plotSpec(dataset, keys, goals, limitValues, method, axisWeights, inspected) : null;
                const view = result?.view ?? computeView(dataset, keys, goals, limitValues, method, axisWeights);
                document.getElementById("pool-summary").textContent = `${view.retained.length.toLocaleString()} of ${dataset.names.length.toLocaleString()} candidates pass plot limits`;
                if (currentTab.id === "tab-tradeoffs") {
                    tradeoffs.update({indices: view.retained, directions, selected: view.selected, inspected});
                    continue;
                }
                if (currentTab.id === "tab-correlations") {
                    updateCorrelations(view.retained);
                    continue;
                }
                await Plotly.react(plot,result.traces,preserveCamera(result.layout,plot.layout),{responsive:true,displaylogo:false,toImageButtonOptions:{format:"png",scale:2}});
                if (!hasClickHandler) {
                    plot.on("plotly_click", event => {
                        const point = event.points.find(item => item.data.ids?.[item.pointNumber] !== undefined);
                        if (point) inspectCandidate(Number(point.data.ids[point.pointNumber]));
                    });
                    hasClickHandler = true;
                }
                document.getElementById("summary").textContent = `${view.indices.length.toLocaleString()} plotted · ${view.filtered} excluded by limits · ${view.missing} missing axis values`;
                const coordinates = values => keys.map((key,i) => `${byKey.get(key).label} = ${formatValue(values[i])}`).join(" · ");
                const lines = [];
                if (result.pickVisibility) {
                    lines.push(`○ CLI selection: ${candidateLabel(dataset.selection, dataset.selection.selected.file)} · ${result.pickVisibility}`);
                }
                if (view.selected !== null) {
                    const fallback = view.mode !== method ? ` (using ${PROJECTION_METHODS[view.mode].label})` : "";
                    lines.push(`★ ${PROJECTION_METHODS[method].label}${fallback}: ${candidateLabel(dataset.selection, dataset.names[view.selected])}`);
                }
                if (view.ideal) {
                    lines.push("◇ Theoretical ideal: " + coordinates(view.ideal));
                } else if (view.indices.length) {
                    lines.push("Choose an ideal direction for each axis to rank the plotted candidates.");
                } else {
                    lines.push("Relax the plot limits or choose different axes.");
                }
                if (view.ideal && view.selected === null) lines.push("Set at least one axis weight above zero.");
                document.getElementById("ideal").textContent = lines.join("\n");
                const inspector = document.getElementById("inspected");
                inspector.hidden = inspected === null;
                if (!inspector.hidden) {
                    const label = candidateLabel(dataset.selection, dataset.names[inspected]);
                    const values = keys.map(key => byKey.get(key).values[inspected]);
                    const status = !view.retained.includes(inspected) ? " · outside plot limits" : "";
                    inspector.textContent = `${label}${status} · ${coordinates(values)}`;
                }
                rankingView = {keys, view};
                if (shortlist.open) renderRanking();
            }
            document.getElementById("error").hidden = true;
        } catch (error) {
            document.getElementById("error").textContent = `Unable to update the view: ${error.message}`;
            document.getElementById("error").hidden = false;
        } finally {
            rendering = false;
            if (pending) requestAnimationFrame(renderPlot);
        }
    }
    const shortlist = document.getElementById("projection-shortlist");
    let renderedRanking = null;
    function renderRanking() {
        if (!rankingView) return;
        const {keys, view} = rankingView;
        // Inspecting a row redraws the scatter. Keep the shortlist DOM and keyboard focus.
        const signature = JSON.stringify([keys, view.mode, view.eligible, view.ranking.slice(0, PROJECTION_SHORTLIST_SIZE)]);
        if (signature === renderedRanking) return;
        renderProjectionRanking(document.getElementById("projection-ranking"), dataset, keys, view, inspectCandidate);
        renderedRanking = signature;
    }
    shortlist.addEventListener("toggle", () => { if (shortlist.open) renderRanking(); });
    methodSelect.addEventListener("change", () => { renderProjectionControls(); schedulePlot(); });
    document.getElementById("download-view").addEventListener("click", () => {
        const keys = activeAxes(), goals = keys.map(key => directions.get(key));
        const method = methodSelect.value, axisWeights = viewWeights(keys, method), limitValues = [...limits.values()];
        const view = computeView(dataset, keys, goals, limitValues, method, axisWeights);
        downloadView(buildViewRecord(dataset, keys, goals, limitValues, method, axisWeights, view));
    });
    function activateTab(tab) {
        if (tab === activeTab) return;
        activeTab = tab;
        for (const button of tabs) {
            const selected = button === tab;
            button.setAttribute("aria-selected", String(selected));
            button.tabIndex = selected ? 0 : -1;
            document.getElementById(button.getAttribute("aria-controls")).hidden = !selected;
        }
        schedulePlot();
        if (tab === scatterTab && plot.layout) Plotly.Plots.resize(plot).catch(error => {
            if (activeTab === scatterTab) {
                document.getElementById("error").textContent = `Unable to resize the plot: ${error.message}`;
                document.getElementById("error").hidden = false;
            }
        });
    }
    tabs.forEach((tab, index) => {
        tab.addEventListener("click", () => activateTab(tab));
        tab.addEventListener("keydown", event => {
            const step = event.key === "ArrowLeft" ? -1 : event.key === "ArrowRight" ? 1 : 0;
            if (!step && event.key !== "Home" && event.key !== "End") return;
            event.preventDefault();
            const next = event.key === "Home" ? tabs[0] : event.key === "End" ? tabs.at(-1)
                : tabs[(index + step + tabs.length) % tabs.length];
            activateTab(next);
            next.focus();
        });
    });
    axisSelects.forEach((select,index) => select.addEventListener("change",() => {selected[index] = select.value; syncAxes();}));
    goalSelects.forEach((select,index) => select.addEventListener("change",() => {directions.set(selected[index],select.value || null); syncAxes();}));
    dimension.addEventListener("change",() => {
        if (dimension.value === "3" && (!selected[2] || selected.slice(0,2).includes(selected[2]))) selected[2] = available.find(metric => !selected.slice(0,2).includes(metric.key)).key;
        syncAxes();
    });
    document.getElementById("add-filter").addEventListener("click",() => {extras.add(filterMetric.value);renderLimits();});
    document.getElementById("reset-limits").addEventListener("click",() => {limits.clear();extras.clear();renderLimits();schedulePlot();});
    syncAxes();
}
if (typeof module !== "undefined") module.exports = {computeView,axisRange,plotSpec,preserveCamera,buildViewRecord};
if (typeof document !== "undefined") mountExplorer(JSON.parse(document.getElementById("pareto-data").textContent));
'''

PAGE = (
    PAGE.replace("__PANEL_STYLES__", TRADEOFF_CSS + CORRELATION_STYLES)
    .replace("__TRADEOFF_PANEL__", TRADEOFF_HTML)
    .replace("__CORRELATION_PANEL__", CORRELATION_PANEL)
)
SCRIPT = (
    '"use strict";\n'
    + SELECTOR_SCRIPT
    + TRADEOFF_SCRIPT
    + CORRELATION_SCRIPT
    + EXPLORER_SCRIPT
)
