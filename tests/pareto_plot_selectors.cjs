const assert = require("node:assert/strict");
const {rankProjection, projectionFront, PROJECTION_METHODS} = require(process.argv[2]);

const columns = [[0, 0.35, 0.66, 1, 0], [1, 0.99, 0.66, 0, 0]];
const indices = [0, 1, 2, 3, 4];
const goals = ["max", "max"];
const rank = (method, weights) => rankProjection(columns, indices, goals, method, weights);
assert.equal(rank("ideal").selected, 2);
assert.equal(rank("ideal", [9, 1]).selected, 3);
assert.equal(rank("utility", [1, 99]).selected, 0);
assert.equal(rank("maximin").selected, 2);
assert.equal(rank("lexicographic").selected, 3);
assert.equal(rank("lexicographic").score, 1);

const knee = rank("knee");
assert.equal(knee.selected, 1);
assert.equal(knee.mode, "knee");
assert.equal(knee.eligible, 4);
assert.equal(knee.ranking.some(item => item.row === 4), false);
assert.deepEqual(knee.ideal, [1, 1]);
assert.ok(knee.score > 0);
for (const [method, {order}] of Object.entries(PROJECTION_METHODS)) {
    const result = rank(method);
    assert.equal(result.score, result.ranking[0].score);
    if (method === "lexicographic") continue;
    const scores = result.ranking.map(entry => entry.score);
    assert.deepEqual(scores, [...scores].sort((a, b) => order === "min" ? a - b : b - a));
}

const concave = rankProjection([[0, 0.4, 1], [1, 0.4, 0]], [0, 1, 2], goals, "knee");
assert.equal(concave.mode, "maximin");
assert.equal(concave.selected, 1);
assert.equal(concave.score, 0.4);

const tilted = rankProjection([[0, 0.35, 0.66, 1], [0, 0.01, 0.34, 1]], [0, 1, 2, 3], ["max", "min"], "knee");
assert.equal(tilted.selected, 1);
assert.deepEqual(tilted.ideal, [1, 0]);

const xyz = [
    [1, 0, 0, 0.6, 0.59, 0.6],
    [0, 1, 0, 0.6, 0.6, 0.6],
    [0, 0, 1, 0.6, 0.6, 0.6],
];
const three = rankProjection(xyz, [0, 1, 2, 5, 3, 4], ["max", "max", "max"], "knee");
assert.equal(three.mode, "knee");
assert.equal(three.selected, 5);
assert.deepEqual(three.ranking.slice(0, 2).map(entry => entry.row), [5, 3]);
assert.equal(three.eligible, 5);

const tie = rankProjection([[1, 2], [2, 1]], [1, 0], goals);
assert.deepEqual(tie.ranking.map(entry => entry.row), [1, 0]);
assert.equal(tie.score, Math.sqrt(0.5));
const constant = rankProjection([[1, 1, 1], [1, 2, 3]], [0, 1, 2], goals, "knee");
assert.equal(constant.mode, "maximin");
assert.equal(constant.selected, 2);
assert.equal(constant.eligible, 1);
assert.equal(constant.score, 1);
assert.equal(rankProjection(columns, indices, [null, "max"]).selected, null);
assert.equal(rankProjection(columns, [], goals).selected, null);
assert.equal(rank("utility", [0, 0]).selected, null);

const huge = Number.MAX_VALUE;
const wide = rankProjection([[-huge, 0, huge], [huge, 0, -huge]], [0, 1, 2], goals);
assert.equal(wide.selected, 1);
assert.equal(wide.score, 0.5);

// Compare the skyline sweep with the definition of strict Pareto dominance.
let state = 19;
for (const dims of [1, 2, 3]) {
    for (let pass = 0; pass < 12; pass++) {
        const points = Array.from({length: 80}, (_, position) => ({
            position,
            coordinates: Array.from({length: dims}, () => {
                state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
                return (state >>> 16) % 9;
            }),
        }));
        const expected = points.filter(a => !points.some(b =>
            b.coordinates.every((value, axis) => value >= a.coordinates[axis]) &&
            b.coordinates.some((value, axis) => value > a.coordinates[axis])));
        assert.deepEqual(projectionFront(points), expected);
    }
}
