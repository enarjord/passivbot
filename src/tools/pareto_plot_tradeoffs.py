"""Offline parallel-coordinate profiles and candidate comparison for Pareto plots."""

TRADEOFF_HTML = r'''
<section class="metric-profiles" aria-labelledby="profiles-title">
  <h2 id="profiles-title">Metric profiles</h2>
  <p>Follow a candidate across metrics. Click a point or choose a candidate to compare its saved values with the scatter pick.</p>
  <div class="profiles-controls">
    <div>
      <label for="profile-add-metric">Add a metric</label>
      <select id="profile-add-metric"></select>
    </div>
    <div>
      <label for="profile-inspect-candidate">Inspect candidate</label>
      <select id="profile-inspect-candidate"></select>
    </div>
  </div>
  <div id="profile-metrics" class="profile-metrics" role="group" aria-label="Profile metrics"></div>
  <p id="profile-summary" role="status"></p>
  <p class="muted">Each metric uses finite values within the limits. The best value is at the top when a direction is set. Otherwise the largest value is at the top. Constant values sit in the middle. Missing and infinite values leave gaps.</p>
  <div class="profile-scroll" tabindex="0" role="region" aria-label="Metric profile chart"><div id="profile-plot"></div></div>
  <div id="profile-comparison"></div>
</section>
'''

TRADEOFF_CSS = r'''
.metric-profiles{min-width:0}
.profiles-controls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px 18px;max-width:900px;margin:16px 0 10px}
.profile-metrics{display:flex;flex-wrap:wrap;gap:6px}
.profile-metric{display:inline-flex;align-items:center;gap:6px;background:#f4f6fa;border:1px solid #e2e8f0;border-radius:8px;padding:3px 4px 3px 9px;font-size:12px;overflow-wrap:anywhere;min-width:0;max-width:100%}
.profile-metric button{line-height:1;padding:4px 6px;border-color:transparent;background:transparent}
.profile-metric button:hover{background:#e2e8f0}
.profile-metric button:disabled{opacity:.4;cursor:default}
#profile-summary{margin-top:14px;font-weight:600}
.profile-scroll{overflow-x:auto}
#profile-plot{height:400px}
#profile-comparison{overflow-x:auto;margin-top:10px}
#profile-comparison table{width:100%;border-collapse:collapse;font-size:13px}
#profile-comparison th,#profile-comparison td{padding:9px 10px;text-align:left;vertical-align:top;border-bottom:1px solid #e2e8f0;overflow-wrap:anywhere}
#profile-comparison td:not(:first-child){font-variant-numeric:tabular-nums;white-space:nowrap}
#profile-comparison th{background:#f4f6fa}
.profile-range{color:#66788c;font-size:12px;font-weight:400;display:block}
.profile-filename{display:block;font:12px ui-monospace,monospace;margin-top:3px}
.profile-pick{color:#8b5c06}.profile-inspected{color:#146471}
@media(max-width:580px){.profiles-controls{grid-template-columns:1fr}}
'''

TRADEOFF_SCRIPT = r'''
const MAX_PROFILE_METRICS = 8;

function tradeoffProfiles(metrics, indices, directions) {
    const ranges = metrics.map(metric => ({metric, goal:directions.get(metric.key), low:Infinity, high:-Infinity}));
    const rows = [];
    let complete = 0;
    for (const row of indices) {
        let observed = 0;
        for (const range of ranges) {
            const value = range.metric.values[row];
            if (!finite(value)) continue;
            observed++;
            range.low = Math.min(range.low, value);
            range.high = Math.max(range.high, value);
        }
        if (observed) rows.push(row);
        if (observed === ranges.length) complete++;
    }
    return {ranges, rows, complete};
}

function profilePosition(value, {low, high, goal}) {
    if (!finite(value)) return null;
    if (low === high) return 0.5;
    const span = high - low;
    const scale = Number.isFinite(span) ? 1 : Math.max(Math.abs(low), Math.abs(high));
    const fraction = (value / scale - low / scale) / (high / scale - low / scale);
    return goal === "min" ? 1 - fraction : fraction;
}

function profileTick(label) {
    const parts = label.match(/.{1,23}(?:_|(?=\s|$))|.{1,24}/g) || [];
    return parts.map(escapeHtml).join("<br>");
}

function profileChart(dataset, profiles, selected, inspected) {
    const {ranges, rows} = profiles;
    const type = rows.length * ranges.length > 1800 ? "scattergl" : "scatter";
    function trace(candidates, marker, color, name) {
        const x = [], y = [], customdata = [];
        for (const row of candidates) {
            const label = escapeHtml(candidateLabel(dataset.selection, dataset.names[row]));
            ranges.forEach((range, dimension) => {
                const value = range.metric.values[row];
                x.push(dimension);
                y.push(profilePosition(value, range));
                customdata.push([row, label, escapeHtml(range.metric.label), formatValue(value)]);
            });
            x.push(null);
            y.push(null);
            customdata.push(null);
        }
        return {type, name, mode:"lines+markers", x, y, customdata,
            showlegend:Boolean(name), connectgaps:false, marker:{size:marker, color},
            line:{width:name ? 3 : 1, color},
            hovertemplate:"<b>%{customdata[1]}</b><br>%{customdata[2]} = %{customdata[3]}<extra></extra>"};
    }
    const opacity = Math.min(0.25, 2 / Math.sqrt(rows.length));
    const traces = [trace(rows, 5, `rgba(66, 100, 135, ${opacity})`, "")];
    if (rows.includes(selected)) traces.push(trace([selected], 9, "#b4770d", "Scatter pick"));
    if (rows.includes(inspected) && inspected !== selected) traces.push(trace([inspected], 9, "#146471", "Inspected candidate"));
    const layout = {
        paper_bgcolor:"white", plot_bgcolor:"white", font:{family:"system-ui, sans-serif",color:"#243449",size:12},
        margin:{l:55,r:25,t:48,b:70}, hovermode:"closest", clickmode:"event",
        legend:{orientation:"h",x:0,y:1.015,xanchor:"left",yanchor:"bottom"},
        xaxis:{range:[-0.15,ranges.length - 0.85],tickmode:"array",tickvals:ranges.map((_,i) => i),
            ticktext:ranges.map(({metric}) => profileTick(metric.label)),fixedrange:true,showgrid:true,
            gridcolor:"#bacbd9",zeroline:false,tickfont:{size:11},automargin:true},
        yaxis:{range:[-0.055,1.055],tickmode:"array",tickvals:[0,0.25,0.5,0.75,1],
            tickformat:".0%",fixedrange:true,gridcolor:"#e1e7ef",zeroline:false}
    };
    return {traces, layout};
}

function profileComparison(parent, dataset, profiles, selected, inspected) {
    parent.replaceChildren();
    if (selected == null && inspected == null) return;
    const table = document.createElement("table");
    parent.append(table);
    const header = table.createTHead().insertRow();
    appendText(header, "th", "Metric").scope = "col";
    const addCandidate = (title, row, className) => {
        const cell = appendText(header, "th", title, className);
        cell.scope = "col";
        const filename = dataset.names[row];
        appendCandidateLabel(cell, "span", dataset.selection, filename, "profile-filename");
    };
    if (selected != null) addCandidate("Scatter pick", selected, "profile-pick");
    if (inspected != null) addCandidate("Inspected candidate", inspected, "profile-inspected");
    const compare = selected != null && inspected != null;
    if (compare) appendText(header, "th", "Difference (inspected − pick)").scope = "col";
    const body = table.createTBody();
    for (const {metric, goal, low, high} of profiles.ranges) {
        const row = body.insertRow();
        const heading = appendText(row, "th", metric.label);
        heading.scope = "row";
        const range = finite(low) ? `${formatValue(low)} to ${formatValue(high)}` : "No finite values in range";
        const direction = goal === "min" ? "lower is better" : goal === "max" ? "higher is better" : "no direction";
        appendText(heading, "span", `${range} · ${direction}`, "profile-range");
        const fact = index => metricValue({[metric.key]:metric.values[index],metric_status:{[metric.key]:metric.statuses?.[index]}},metric.key);
        if (selected != null) appendText(row, "td", fact(selected));
        if (inspected != null) appendText(row, "td", fact(inspected));
        if (compare) {
            const pickValue = metric.values[selected], inspectedValue = metric.values[inspected];
            const delta = finite(pickValue) && finite(inspectedValue) ? inspectedValue - pickValue : null;
            appendText(row, "td", finite(delta) ? (delta > 0 ? "+" : "") + formatValue(delta) : "—");
        }
    }
}

function mountTradeoffs(dataset, fillOptions, onInspect) {
    const metrics = new Map(dataset.metrics.map(metric => [metric.key, metric]));
    const preferred = [...(dataset.selection?.objectives || []).map(spec => spec.metric), ...dataset.selected];
    const keys = [...new Set(preferred)].filter(key => metrics.get(key)?.count).slice(0, 5);
    const add = document.getElementById("profile-add-metric");
    const candidate = document.getElementById("profile-inspect-candidate");
    const chips = document.getElementById("profile-metrics");
    const summary = document.getElementById("profile-summary");
    const plot = document.getElementById("profile-plot");
    const comparison = document.getElementById("profile-comparison");
    const config = {responsive:true, displaylogo:false, doubleClick:false,
        modeBarButtonsToRemove:["pan2d","select2d","lasso2d","zoom2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d"],
        toImageButtonOptions:{format:"png",scale:2}};
    fillOptions(add);
    add.prepend(new Option("Choose a metric…", ""));
    let inspected = null, current = null, rendered = null, pending = false, rendering = false, hasPlot = false;

    function renderMetricControls(focusKey) {
        const fragment = document.createDocumentFragment();
        for (const key of keys) {
            const label = metrics.get(key).label;
            const chip = appendText(fragment, "span", label, "profile-metric");
            const remove = appendText(chip, "button", "×");
            remove.type = "button";
            remove.disabled = keys.length <= 2;
            remove.setAttribute("aria-label", `Remove ${label}`);
            remove.addEventListener("click", () => {
                const index = keys.indexOf(key);
                keys.splice(index, 1);
                renderMetricControls(keys[Math.min(index, keys.length - 1)]);
                schedule();
            });
            if (key === focusKey) remove.dataset.focus = "";
        }
        chips.replaceChildren(fragment);
        const next = chips.querySelector("button[data-focus]:not([disabled])");
        if (focusKey) (next || add).focus();
        for (const option of add.options) {
            option.disabled = Boolean(option.value) && (!metrics.get(option.value).count || keys.includes(option.value));
        }
        add.disabled = keys.length >= MAX_PROFILE_METRICS || !Array.from(add.options).some(option => option.value && !option.disabled);
        add.options[0].textContent = keys.length >= MAX_PROFILE_METRICS ? "Eight metrics selected" : "Choose a metric…";
        add.value = "";
    }

    function renderCandidates(indices) {
        const options = document.createDocumentFragment();
        options.append(new Option("Choose a candidate…", ""));
        for (const row of indices) {
            options.append(new Option(candidateLabel(dataset.selection, dataset.names[row]), String(row)));
        }
        if (inspected != null && !indices.includes(inspected)) {
            const label = candidateLabel(dataset.selection, dataset.names[inspected]);
            options.append(new Option(`${label} (outside limits)`, String(inspected)));
        }
        candidate.replaceChildren(options);
        candidate.value = inspected == null ? "" : String(inspected);
    }

    function sameIndices(left, right) {
        return left === right || (left.length === right.length && left.every((row,i) => row === right[i]));
    }

    function schedule() {
        if (pending || !current) return;
        pending = true;
        if (!rendering) requestAnimationFrame(render);
    }

    async function render() {
        pending = false;
        rendering = true;
        try {
            if (!plot.parentElement.clientWidth) return;
            const {indices, directions, selected} = current;
            const active = keys.map(key => metrics.get(key));
            const axes = JSON.stringify(keys.map(key => [key, directions.get(key)]));
            const samePool = rendered && sameIndices(indices, rendered.indices);
            if (samePool && axes === rendered.axes && selected === rendered.selected && inspected === rendered.inspected) {
                if (hasPlot && plot.clientWidth !== rendered.width) {
                    await Plotly.Plots.resize(plot);
                    rendered.width = plot.clientWidth;
                }
                return;
            }
            if (!samePool || inspected !== rendered.inspected) renderCandidates(indices);
            const profiles = tradeoffProfiles(active, indices, directions);
            const snapshot = {axes, indices, selected, inspected};
            const count = value => value.toLocaleString();
            summary.textContent = profiles.rows.length
                ? `${count(profiles.rows.length)} / ${count(indices.length)} candidates shown · ${count(profiles.complete)} have finite values on every axis`
                : "No candidates have finite values for these metrics within the current limits.";
            profileComparison(comparison, dataset, profiles, selected, inspected);
            plot.hidden = !profiles.rows.length;
            if (!plot.hidden) {
                plot.style.minWidth = `${active.length * 125 + 70}px`;
                const {traces, layout} = profileChart(dataset, profiles, selected, inspected);
                await Plotly.react(plot, traces, layout, config);
                if (!hasPlot) {
                    plot.on("plotly_click", event => {
                        const row = event.points?.[0]?.customdata?.[0];
                        if (Number.isInteger(row)) inspect(row);
                    });
                    hasPlot = true;
                }
            }
            rendered = {...snapshot, width:plot.clientWidth};
        } catch (error) {
            plot.hidden = true;
            summary.textContent = `Unable to render metric profiles: ${error.message}`;
            rendered = null;
        } finally {
            rendering = false;
            if (pending) requestAnimationFrame(render);
        }
    }

    function inspect(row) {
        inspected = row;
        onInspect?.(row);
        schedule();
    }
    candidate.addEventListener("change", () => inspect(candidate.value === "" ? null : Number(candidate.value)));
    add.addEventListener("change", () => {
        if (!add.value) return;
        keys.push(add.value);
        renderMetricControls();
        if (add.disabled) chips.lastElementChild?.querySelector("button")?.focus();
        schedule();
    });
    renderMetricControls();
    return {
        update(view) {
            current = view;
            if (Object.hasOwn(view, "inspected")) inspected = view.inspected;
            schedule();
        },
    };
}
'''
