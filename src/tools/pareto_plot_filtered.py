"""Browser controls for the all-objective selection and seed export."""

FILTERED_PANEL = r'''
<section class="selection" aria-labelledby="filtered-title">
<h2 id="filtered-title">All-objective selection after limits</h2>
<p class="muted" id="filtered-offline">This is the standalone HTML. Run <code>passivbot tool pareto-plot PATH --open --force</code> for the all-objective pick after slider limits and seed export.</p>
<div id="filtered-controls" hidden>
<p id="filtered-status" role="status"></p>
<p class="muted">Uses the CLI selection method and objectives from launch. Plot limits are added to the launch limits.</p>
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
'''

FILTERED_SCRIPT = r'''
function mountFilteredSelection(dataset, onChange) {
    if (!dataset.session) return null;
    document.getElementById("filtered-offline").hidden = true;
    document.getElementById("filtered-controls").hidden = false;
    const status = document.getElementById("filtered-status");
    const seedStatus = document.getElementById("seed-status");
    const exportButton = document.getElementById("export-seeds");
    const commands = document.getElementById("command-list");
    const details = document.getElementById("filtered-commands");
    let current = dataset.session.initial, limits = [], signature = "[]";
    let timer, running = false, ready = true, exporting = false, exportResult = null;

    function renderCommands() {
        commands.replaceChildren();
        if (!ready || !current.selected) return;
        const entries = [
            ["Select with these limits", current.selection_command],
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
            status.textContent = `${label} · ${method} · ${current.objective_count} objectives · ${current.retained.length.toLocaleString()} retained`;
        } else {
            status.textContent = ready ? current.error : "Applying CLI limits…";
        }
        renderCommands();
    }

    async function request(requestLimits, exportSeeds) {
        const response = await fetch(dataset.session.endpoint, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({limits: requestLimits, export: exportSeeds}),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "The local selection request failed.");
        return result;
    }

    async function applyLimits() {
        if (running) return;
        running = true;
        const requestSignature = signature;
        try {
            const result = limits.length ? await request(limits, false) : dataset.session.initial;
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
        applyLimits();
    }

    exportButton.addEventListener("click", async () => {
        const requestSignature = signature;
        exporting = true;
        exportButton.disabled = true;
        seedStatus.textContent = "Exporting seeds…";
        try {
            const result = await request(limits, true);
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

    refresh();
    return {
        get current() { return current; },
        get pending() { return !ready; },
        get ready() { return ready && !current.error; },
        update(plotLimits) {
            const next = plotLimits.filter(limit => limit.enabled).map(({key, mode, value}) => ({key, mode, value}));
            const nextSignature = JSON.stringify(next);
            if (nextSignature === signature) return;
            signature = nextSignature;
            limits = next;
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
