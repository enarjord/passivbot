"""Browser controls for the scoring-objective selection and seed export."""

OBJECTIVE_CONTROLS = r'''
<details class="scoring-objectives" id="scoring-objectives" hidden>
<summary id="scoring-objectives-summary">Scoring objectives for square marker</summary>
<p class="muted">Choose which launch objectives rank the candidates that pass your limits.</p>
<div class="objective-options" id="objective-options" role="group" aria-label="Scoring objectives"></div>
<button type="button" id="restore-objectives">Restore all objectives</button>
</details>
'''

FILTERED_PANEL = r'''
<section class="selection" aria-labelledby="filtered-title">
<h2 id="filtered-title">Scoring pick after limits</h2>
<p class="muted" id="filtered-offline">This is the standalone HTML. Run <code>passivbot tool pareto-plot PATH --open --force</code> for the scoring pick after slider limits and seed export.</p>
<div id="filtered-controls" hidden>
<p id="filtered-status" role="status"></p>
<p class="muted">Uses the CLI selection method from launch and the checked scoring objectives. Plot limits are added to the launch limits.</p>
<button type="button" id="export-seeds">Export filtered seeds</button>
<p id="seed-status" role="status"></p>
<details id="filtered-commands"><summary>Copy commands</summary><div id="command-list"></div></details>
</div>
</section>
'''

FILTERED_STYLES = r'''
.pareto-command{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:6px;margin-top:10px;align-items:center}
.pareto-command label{grid-column:1/-1;margin:0}
.pareto-command input{min-width:0;padding:7px;border:1px solid #cbd5e1;border-radius:6px;font:13px ui-monospace,monospace}
.scoring-objectives{margin:0;padding:10px 16px;border-bottom:1px solid #e7ecf2}
.objective-options{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,290px),1fr));gap:4px 12px;margin:10px 0}
.objective-options label{display:flex;align-items:start;gap:6px;overflow-wrap:anywhere}
.objective-options input{margin:2px 0 0}
'''

FILTERED_SCRIPT = r'''
function objectiveCountLabel(count) {
    return count === 1 ? "1 objective" : `${count} objectives`;
}

function mountFilteredSelection(dataset, onChange) {
    if (!dataset.session) return null;
    document.getElementById("filtered-offline").hidden = true;
    document.getElementById("filtered-controls").hidden = false;
    const status = document.getElementById("filtered-status");
    const seedStatus = document.getElementById("seed-status");
    const exportButton = document.getElementById("export-seeds");
    const commands = document.getElementById("command-list");
    const details = document.getElementById("filtered-commands");
    const initial = dataset.session.initial;
    const available = initial.objectives;
    const options = document.getElementById("objective-options");
    const objectiveSummary = document.getElementById("scoring-objectives-summary");
    const restore = document.getElementById("restore-objectives");
    document.getElementById("scoring-objectives").hidden = false;
    const checkboxes = available.map(metric => {
        const label = appendText(options, "label", "");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.value = metric;
        checkbox.checked = true;
        checkbox.addEventListener("change", onChange);
        label.append(checkbox);
        appendText(label, "span", metric);
        return checkbox;
    });
    restore.addEventListener("click", () => {
        checkboxes.forEach(checkbox => { checkbox.checked = true; });
        onChange();
    });
    let current = initial, limits = [], objectives = available;
    let signature = JSON.stringify([limits, objectives]);
    let timer, running = false, ready = true, exporting = false, exportResult = null;

    function renderCommands() {
        commands.replaceChildren();
        if (!ready || !current.selected) return;
        const entries = [
            ["Select with these objectives and limits", current.selection_command],
            ["Backtest the selected candidate", current.backtest_command],
            ["Start optimization with exported seeds and limits", exportResult?.optimize_command],
        ];
        for (const [index, [label, value]] of entries.entries()) {
            if (!value) continue;
            const field = appendText(commands, "div", "", "pareto-command");
            appendText(field, "label", label).htmlFor = `pareto-command-${index}`;
            const input = document.createElement("input");
            input.id = `pareto-command-${index}`;
            input.value = value;
            input.readOnly = true;
            const button = appendText(field, "button", "Copy");
            button.type = "button";
            button.setAttribute("aria-label", `Copy command: ${label}`);
            button.addEventListener("click", async () => {
                try {
                    await navigator.clipboard.writeText(value);
                    button.textContent = "Copied";
                } catch {
                    input.focus();
                    input.select();
                }
            });
            field.insertBefore(input, button);
        }
    }

    function refresh() {
        exportButton.disabled = !ready || !current.selected || exporting;
        details.hidden = !ready || !current.selected;
        if (ready && current.selected) {
            const label = candidateLabel(dataset.selection, current.selected);
            const method = current.method === "ideal" ? "Euclidean ideal" : current.method;
            status.textContent = `${label} · ${method} · ${objectiveCountLabel(current.objective_count)} · ${current.retained.length.toLocaleString()} retained`;
        } else {
            status.textContent = ready ? current.error : "Updating scoring pick…";
        }
        renderCommands();
    }

    async function request(exportSeeds) {
        const response = await fetch(dataset.session.endpoint, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({limits, objectives, export: exportSeeds}),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "The local selection request failed.");
        return result;
    }

    async function applySelection() {
        if (running) return;
        running = true;
        const requestSignature = signature;
        try {
            const result = limits.length || objectives.length !== available.length ? await request(false) : initial;
            if (requestSignature !== signature) return;
            current = result;
            ready = true;
        } catch (error) {
            if (requestSignature !== signature) return;
            current = {retained: [], selected: null, error: error.message};
            ready = true;
        } finally {
            running = false;
            if (requestSignature === signature) {
                refresh();
                onChange();
            } else if (!timer) {
                timer = setTimeout(runTimer, 0);
            }
        }
    }

    function runTimer() {
        timer = undefined;
        applySelection();
    }

    exportButton.addEventListener("click", async () => {
        const requestSignature = signature;
        exporting = true;
        exportButton.disabled = true;
        seedStatus.textContent = "Exporting seeds…";
        try {
            const result = await request(true);
            seedStatus.textContent = `Exported ${result.retained.length.toLocaleString()} seeds to ${result.seed_directory}`;
            if (requestSignature === signature) {
                exportResult = result;
                details.open = true;
                renderCommands();
            }
        } catch (error) {
            seedStatus.textContent = error.message;
        } finally {
            exporting = false;
            exportButton.disabled = !ready || !current.selected;
        }
    });

    function updateObjectiveControls() {
        objectiveSummary.textContent = `Scoring objectives for square marker (${objectives.length} of ${available.length})`;
        restore.disabled = objectives.length === available.length;
        checkboxes.forEach(checkbox => { checkbox.disabled = checkbox.checked && objectives.length === 1; });
    }
    updateObjectiveControls();
    refresh();
    return {
        get current() { return current; },
        get pending() { return !ready; },
        get ready() { return ready && !current.error; },
        update(plotLimits) {
            const next = plotLimits.filter(limit => limit.enabled).map(({key, mode, value}) => ({key, mode, value}));
            const nextObjectives = checkboxes.filter(checkbox => checkbox.checked).map(checkbox => checkbox.value);
            const nextSignature = JSON.stringify([next, nextObjectives]);
            if (nextSignature === signature) return;
            signature = nextSignature;
            limits = next;
            objectives = nextObjectives;
            updateObjectiveControls();
            ready = false;
            exportResult = null;
            if (!exporting) seedStatus.textContent = "";
            refresh();
            clearTimeout(timer);
            timer = setTimeout(runTimer, 150);
        },
    };
}
'''
