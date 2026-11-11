"""Selectors for the offline viewer's current metric projection."""

SELECTOR_SCRIPT = r"""
const PROJECTION_METHODS = {
    ideal: {
        label: "Euclidean ideal", scoreLabel: "Distance to ideal", order: "min", weighted: true,
        description: "Minimize the weighted Euclidean distance to the best visible value on each axis.",
    },
    knee: {
        label: "Knee point", scoreLabel: "Distance toward ideal", order: "max", weighted: false,
        description: "Among visible non-dominated points, prefer the greatest bend toward the ideal from the affine span of the axis extremes. Uses maximin when no such bend is detected.",
    },
    maximin: {
        label: "Maximin", scoreLabel: "Worst-axis utility", order: "max", weighted: false,
        description: "Maximize the worst normalized axis utility, breaking ties by total utility.",
    },
    utility: {
        label: "Weighted utility", scoreLabel: "Weighted utility", order: "max", weighted: true,
        description: "Maximize the weighted average of normalized axis utilities.",
    },
    lexicographic: {
        label: "Axis priority", scoreLabel: "X-axis utility", order: "max", weighted: false,
        description: "Choose the best X value, then use Y and Z in order to break ties.",
    },
};

function compareCoordinates(a, b) {
    for (let axis = 0; axis < a.length; axis++) {
        if (a[axis] !== b[axis]) return a[axis] > b[axis] ? -1 : 1;
    }
    return 0;
}

function projectionFront(rows) {
    // A descending lexicographic sweep leaves only Y and Z to check for dominance.
    // A prefix maximum over descending Y ranks makes the 3D sweep O(n log n).
    const ordered = [...rows].sort((a, b) => compareCoordinates(a.coordinates, b.coordinates));
    const yValues = [...new Set(ordered.map(item => item.coordinates[1] ?? 0))].sort((a, b) => b - a);
    const yRanks = new Map(yValues.map((value, rank) => [value, rank + 1]));
    const maxZ = new Float64Array(yValues.length + 1).fill(-Infinity);
    const front = [];
    for (let start = 0; start < ordered.length;) {
        const coordinates = ordered[start].coordinates;
        let end = start + 1;
        while (end < ordered.length && compareCoordinates(coordinates, ordered[end].coordinates) === 0) end++;
        const yRank = yRanks.get(coordinates[1] ?? 0);
        const z = coordinates[2] ?? 0;
        let dominated = false;
        for (let index = yRank; index > 0; index -= index & -index) {
            if (maxZ[index] >= z) { dominated = true; break; }
        }
        if (!dominated) {
            // Equal coordinates represent distinct candidates and do not dominate one another.
            for (let index = start; index < end; index++) front.push(ordered[index]);
            for (let index = yRank; index < maxZ.length; index += index & -index) {
                maxZ[index] = Math.max(maxZ[index], z);
            }
        }
        start = end;
    }
    return front.sort((a, b) => a.position - b.position);
}

const projectionDot = (a, b) => a.reduce((sum, value, axis) => sum + value * b[axis], 0);

function kneeScores(front, varyingAxes) {
    if (front.length < 3 || varyingAxes.length < 2) return null;
    const anchors = varyingAxes.map(axis => front.reduce((best, item) =>
        item.coordinates[axis] > best.coordinates[axis] ? item : best));
    const origin = anchors[0].utility;
    const basis = [];
    const orthogonal = vector => {
        for (const direction of basis) {
            const projection = projectionDot(vector, direction);
            vector = vector.map((value, axis) => value - projection * direction[axis]);
        }
        return vector;
    };
    for (const anchor of anchors.slice(1)) {
        const vector = orthogonal(anchor.utility.map((value, axis) => value - origin[axis]));
        const norm = Math.hypot(...vector);
        if (norm > 1e-12) basis.push(vector.map(value => value / norm));
    }
    if (!basis.length) return null;

    const toIdeal = orthogonal(origin.map(value => 1 - value));
    const norm = Math.hypot(...toIdeal);
    if (norm <= 1e-12) return null;

    const direction = toIdeal.map(value => value / norm);
    const scores = front.map(item => projectionDot(
        item.utility.map((value, axis) => value - origin[axis]), direction));
    return scores.some(score => score > 1e-12) ? scores : null;
}

function rankProjection(columns, indices, goals, method = "ideal", weights = null) {
    if (!Object.hasOwn(PROJECTION_METHODS, method)) throw new Error(`Unknown projection selector: ${method}`);
    const empty = {selected: null, ranking: [], ideal: null, eligible: 0, mode: method, score: null};
    if (!indices.length || !goals.every(goal => goal === "min" || goal === "max")) return empty;
    const extrema = columns.map((column, axis) => {
        let low = Infinity, high = -Infinity;
        for (const row of indices) {
            low = Math.min(low, column[row]);
            high = Math.max(high, column[row]);
        }
        const span = high - low;
        const best = goals[axis] === "max" ? high : low;
        // Halving each operand before subtraction handles the full finite number range.
        const scale = Number.isFinite(span) ? 1 : 2;
        const scaledSpan = high / scale - low / scale;
        const deficit = span <= 1e-15 ? () => 0 : value => Math.abs(value / scale - best / scale) / scaledSpan;
        return {best, span, deficit};
    });
    const ideal = extrema.map(axis => axis.best);
    const rows = indices.map((row, position) => {
        const deficits = extrema.map(({deficit}, axis) => deficit(columns[axis][row]));
        return {
            row, position, deficits, utility: deficits.map(value => 1 - value),
            coordinates: columns.map((column, axis) => goals[axis] === "max" ? column[row] : -column[row]),
        };
    });

    let mode = method;
    let eligibleRows = rows;
    let scores;
    let scoreScale = score => score;
    if (method === "knee") {
        eligibleRows = projectionFront(rows);
        const varyingAxes = extrema.flatMap(({span}, axis) => span > 1e-15 ? [axis] : []);
        scores = kneeScores(eligibleRows, varyingAxes);
        if (!scores) mode = "maximin";
    }
    if (mode === "ideal" || mode === "utility") {
        const relative = weights ?? columns.map(() => 1);
        if (relative.length !== columns.length ||
            relative.some(value => !Number.isFinite(value) || value < 0)) return {...empty, ideal};
        const largest = Math.max(...relative);
        if (!(largest > 0)) return {...empty, ideal};
        const safeWeights = relative.map(value => value / largest);
        const total = safeWeights.reduce((sum, weight) => sum + weight, 0);
        scores = mode === "ideal"
            ? rows.map(item => projectionDot(item.deficits.map(value => value * value), safeWeights))
            : rows.map(item => projectionDot(item.utility, safeWeights));
        // Scale the scores after ordering to preserve ties in the unweighted default.
        scoreScale = mode === "ideal" ? score => Math.sqrt(score / total) : score => score / total;
    }
    if (mode === "maximin") scores = eligibleRows.map(item => Math.min(...item.utility));
    if (mode === "lexicographic") scores = eligibleRows.map(item => item.utility[0]);
    const entries = eligibleRows.map((item, index) => ({item, score: scores[index]}));
    if (mode === "lexicographic") {
        entries.sort((a, b) => compareCoordinates(a.item.coordinates, b.item.coordinates));
    } else if (mode === "maximin") {
        const total = item => item.utility.reduce((sum, value) => sum + value, 0);
        entries.sort((a, b) => b.score - a.score || total(b.item) - total(a.item));
    } else if (PROJECTION_METHODS[mode].order === "min") {
        entries.sort((a, b) => a.score - b.score);
    } else {
        entries.sort((a, b) => b.score - a.score);
    }
    const ranking = entries.map(({item, score}) => ({row: item.row, score: scoreScale(score)}));
    return {selected: ranking[0].row, ranking, ideal, eligible: ranking.length, mode, score: ranking[0].score};
}
"""
