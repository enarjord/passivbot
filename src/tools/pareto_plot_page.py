"""Bundled offline page and browser logic for the Pareto plot CLI."""

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pareto trade-off explorer</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#243449;font:14px system-ui,sans-serif}
header{padding:25px 32px 16px}h1{font-size:27px;margin:4px 0 8px}.eyebrow{font-size:11px;letter-spacing:.15em;color:#66788c}
p{line-height:1.5;margin:7px 0;color:#5b6c81}main{padding:0 24px 24px}.toolbar{display:flex;gap:16px;flex-wrap:wrap;padding:18px;background:white;border:1px solid #dce3ed;border-radius:12px}
label{display:block;font-size:12px;font-weight:600;margin-bottom:5px}select,input[type=number],button{font:inherit;border:1px solid #cbd5e1;border-radius:6px;padding:7px;background:white;color:#243449}
select{max-width:100%;width:100%}.axis{flex:1;min-width:220px}.axis .goal{margin-top:7px;font-size:12px}.mode{width:85px}button{cursor:pointer}button:hover{background:#eef4fa}
.workspace{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:16px;margin-top:16px}.chart{background:white;border:1px solid #dce3ed;border-radius:12px;overflow:hidden;min-width:0}
#plot{width:100%;height:650px}.summary{padding:14px 20px 0;min-height:42px;font-weight:600}#ideal{padding:0 20px 12px;color:#805b12;font-size:12px;overflow-wrap:anywhere}
.note{padding:0 20px 16px;font-size:12px}aside{background:#fff;border:1px solid #dce3ed;border-radius:12px;padding:16px;align-self:start}h2{font-size:17px;margin:0 0 10px}.limit{border-top:1px solid #e2e8f0;padding:14px 0}.limit-title{overflow-wrap:anywhere;font-size:12px;margin-bottom:9px}.limit-title input{margin-right:7px}.limit-controls{display:grid;grid-template-columns:95px 1fr;gap:8px}.limit input[type=range]{width:100%;accent-color:#287b91;margin:12px 0 0}.limit input[type=number]{width:100%;min-width:0}.muted{font-size:12px;color:#66788c}.add-filter{display:flex;gap:6px;margin:12px 0}.add-filter select{min-width:0}.hidden{display:none!important}#error{color:#b42318;padding:16px}details{margin-top:12px}summary{cursor:pointer;font-size:12px}
@media(max-width:950px){.workspace{grid-template-columns:1fr}#plot{height:550px}header{padding:20px}main{padding:0 12px 12px}aside{max-height:none}}
</style><script>__PLOTLY_JS__</script></head>
<body><header><div class="eyebrow">PASSIVBOT / PARETO</div><h1>Trade-off explorer</h1>
<p>Choose metrics, adjust limits, and watch the ideal move. Everything runs locally in this file.</p></header>
<main><section class="toolbar" aria-label="Plot controls">
<div class="mode"><label for="dimension">View</label><select id="dimension"><option value="2">2D</option><option value="3">3D</option></select></div>
<div class="axis"><label for="axis-x">X metric</label><select id="axis-x"></select><select class="goal" id="goal-x" aria-label="X ideal direction"></select></div>
<div class="axis"><label for="axis-y">Y metric</label><select id="axis-y"></select><select class="goal" id="goal-y" aria-label="Y ideal direction"></select></div>
<div class="axis" id="z-controls"><label for="axis-z">Z metric</label><select id="axis-z"></select><select class="goal" id="goal-z" aria-label="Z ideal direction"></select></div>
</section><div class="workspace"><section class="chart"><div class="summary" id="summary" role="status"></div><div id="plot"></div><div id="ideal"></div>
<p class="note">★ The ideal combines the best visible value on each axis; it need not be a real candidate. 2D axes point toward a lower-left ideal. Axis ranges stay fixed while filtering so movement is visible. Hover for values; use the toolbar to reset or export PNG.</p>
<p class="note">Saved objectives may include penalties. Other metrics use saved aggregates or means; statistics are listed separately. This is a projection of saved members, not a recomputed Pareto front.</p><div id="error" role="alert" hidden></div></section>
<aside><h2>Limits</h2><p class="muted">Inclusive floors and ceilings. Enabled limits stay active when you switch axes. Missing values fail an enabled limit.</p>
<div class="add-filter"><select id="filter-metric" aria-label="Additional limit metric"></select><button id="add-filter">Add</button></div>
<button id="reset-limits">Reset all limits</button><div id="limits"></div>
<details><summary>Metric values and directions</summary><p class="muted">A metric with missing values omits those candidates only when it is selected as an axis or used by an enabled limit. Choose an ideal direction when the saved metric has no known goal. Changing direction changes the ideal and 2D orientation, not the data.</p></details>
</aside></div></main><script id="pareto-data" type="application/json">__DATA__</script><script>__APP_SCRIPT__</script></body></html>'''

SCRIPT = r'''
"use strict";
const finite = value => typeof value === "number" && Number.isFinite(value);
function computeView(dataset, axes, goals, limits) {
    const columns = new Map(dataset.metrics.map(metric => [metric.key, metric.values]));
    const active = limits.filter(limit => limit.enabled);
    const indices = [];
    let filtered = 0, missing = 0;
    for (let row = 0; row < dataset.names.length; row++) {
        if (!active.every(limit => {
            const value = columns.get(limit.key)[row];
            return finite(value) && (limit.mode === "floor" ? value >= limit.value : value <= limit.value);
        })) { filtered++; continue; }
        if (!axes.every(key => finite(columns.get(key)[row]))) { missing++; continue; }
        indices.push(row);
    }
    let ideal = null;
    if (indices.length && goals.every(goal => goal === "min" || goal === "max")) {
        ideal = axes.map((key, axis) => indices.reduce((best, row) => {
            const value = columns.get(key)[row];
            return goals[axis] === "max" ? Math.max(best, value) : Math.min(best, value);
        }, goals[axis] === "max" ? -Infinity : Infinity));
    }
    return {indices, ideal, filtered, missing};
}
function axisRange(metric, goal, dimensions) {
    const low = metric.min, high = metric.max;
    const padding = (high - low || Math.abs(low) || 1) * 0.06;
    const range = [low - padding, high + padding];
    return dimensions === 2 && goal === "max" ? range.reverse() : range;
}
const escapeHtml = text => String(text).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const formatValue = value => finite(value) ? Number(value.toPrecision(8)).toString() : "unavailable";
function plotSpec(dataset, axes, goals, limits) {
    const metrics = new Map(dataset.metrics.map(metric => [metric.key, metric]));
    const view = computeView(dataset, axes, goals, limits), is3d = axes.length === 3;
    const columns = axes.map(key => view.indices.map(row => metrics.get(key).values[row]));
    const hover = axes.map((key, i) => `${escapeHtml(metrics.get(key).label)}: %{${"xyz"[i]}:.8g}`).join("<br>");
    const points = {type:is3d ? "scatter3d" : "scatter", mode:"markers", name:"Candidates", showlegend:false,
        x:columns[0], y:columns[1], text:view.indices.map(row => escapeHtml(dataset.names[row])),
        hovertemplate:"<b>%{text}</b><br>" + hover + "<extra></extra>",
        marker:{size:is3d ? 5 : 9, opacity:0.85, color:columns[axes.length-1], colorscale:"Viridis",
            cmin:metrics.get(axes.at(-1)).min, cmax:metrics.get(axes.at(-1)).max,
            line:{width:0.4,color:"white"}}};
    if (is3d) points.z = columns[2];
    const traces = [points];
    if (view.ideal) {
        const ideal = {type:points.type, name:"Ideal (best visible values)", mode:is3d ? "markers+text" : "markers",
            x:[view.ideal[0]], y:[view.ideal[1]], showlegend:false,
            marker:{symbol:is3d ? "circle" : "star",size:is3d ? 6 : 20,color:"#f4ae2b",line:{color:"#704907",width:1.5}},
            hovertemplate:"<b>Theoretical ideal</b><br>" + hover + "<extra></extra>"};
        if (is3d) Object.assign(ideal,{z:[view.ideal[2]],text:["★"],textposition:"middle center",textfont:{size:32,color:"#e69b12"}});
        traces.push(ideal);
    }
    const revision = JSON.stringify([axes, goals]);
    const layout = {paper_bgcolor:"white",plot_bgcolor:"white",font:{family:"Arial, sans-serif",color:"#243449",size:12},
        margin:{l:110,r:40,t:35,b:110},hoverlabel:{bgcolor:"white"},uirevision:revision,
        annotations:view.indices.length ? [] : [{text:"No candidates match. Relax the limits or choose other metrics.",xref:"paper",yref:"paper",x:0.5,y:0.5,showarrow:false}]};
    const axis = (key, index) => {
        const metric = metrics.get(key), goal = goals[index];
        const label = escapeHtml(metric.label).replace(/(.{20,35}_)/g, "$1<br>");
        return {title:{text:label + "<br>(" + (goal === "max" ? "higher is better" : goal === "min" ? "lower is better" : "choose ideal direction") + ")",font:{size:11},standoff:15},
            range:axisRange(metric,goal,axes.length),autorange:false,gridcolor:"#e1e7ef",zeroline:false,tickformat:".4~g",automargin:true};
    };
    if (is3d) layout.scene = {xaxis:axis(axes[0],0),yaxis:axis(axes[1],1),zaxis:axis(axes[2],2),
        aspectmode:"cube",dragmode:"orbit",uirevision:revision,camera:{eye:{x:1.65,y:1.65,z:1.2}}};
    else Object.assign(layout,{xaxis:axis(axes[0],0),yaxis:axis(axes[1],1)});
    return {traces,layout,view};
}
function preserveCamera(layout, previous) {
    if (layout.scene && previous?.scene?.camera && layout.uirevision === previous.uirevision) {
        layout.scene.camera = JSON.parse(JSON.stringify(previous.scene.camera));
    }
    return layout;
}
function mountExplorer(dataset) {
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
        renderLimits(); schedulePlot();
    }
    function renderLimits() {
        const container = document.getElementById("limits"); container.replaceChildren();
        const keys = new Set([...activeAxes(),...extras,...[...limits.values()].filter(limit => limit.enabled).map(limit => limit.key)]);
        for (const key of keys) {
            const metric = byKey.get(key), state = ensureLimit(key);
            const card = document.createElement("div"); card.className = "limit"; card.dataset.metric = key;
            const title = document.createElement("label"); title.className = "limit-title";
            const enabled = document.createElement("input"); enabled.type = "checkbox"; enabled.checked = state.enabled; enabled.setAttribute("aria-label",`Enable limit for ${metric.label}`);
            title.append(enabled,document.createTextNode(metric.label));
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
                const result = plotSpec(dataset,keys,goals,[...limits.values()]);
                await Plotly.react("plot",result.traces,preserveCamera(result.layout,document.getElementById("plot").layout),{responsive:true,scrollZoom:true,displaylogo:false,toImageButtonOptions:{format:"png",scale:2}});
                document.getElementById("summary").textContent = `${result.view.indices.length.toLocaleString()} / ${dataset.names.length.toLocaleString()} candidates shown · ${result.view.filtered} excluded by limits · ${result.view.missing} missing axis values`;
                document.getElementById("ideal").textContent = result.view.ideal ? "★ Ideal: " + keys.map((key,i) => `${byKey.get(key).label} = ${formatValue(result.view.ideal[i])}`).join(" · ") : result.view.indices.length ? "Choose an ideal direction for each axis to show the star." : "No ideal: no candidates remain.";
                document.getElementById("error").hidden = true;
            }
        } catch (error) {
            document.getElementById("error").textContent = `Unable to render plot: ${error.message}`;
            document.getElementById("error").hidden = false;
        } finally { rendering = false; }
    }
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
if (typeof module !== "undefined") module.exports = {computeView,axisRange,plotSpec,preserveCamera};
if (typeof document !== "undefined") mountExplorer(JSON.parse(document.getElementById("pareto-data").textContent));
'''
