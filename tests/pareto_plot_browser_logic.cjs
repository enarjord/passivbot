const assert = require('node:assert/strict');
const {computeView, axisRange, plotSpec, preserveCamera, buildViewRecord} = require(process.argv[2]);
const metric = (key,values,goal) => ({key,label:key,values,goal,min:Math.min(...values.filter(v=>v!==null)),max:Math.max(...values.filter(v=>v!==null))});
const data = {names:['a','b','c','d'],metrics:[metric('adg',[0.01,0.02,0.03,0.04],'max'),metric('risk',[0.1,0.4,0.2,null],'min'),metric('third',[3,2,1,9],'max'),metric('extra',[10,null,30,40],null)]};
const bound = (key,mode,value) => ({key,mode,value,enabled:true});
const view = limits => computeView(data,['adg','risk'],['max','min'],limits);
const membership = ({indices,retained,ideal,selected,filtered,missing}) => ({indices,retained,ideal,selected,filtered,missing});
assert.deepEqual(membership(view([])),{indices:[0,1,2],retained:[0,1,2,3],ideal:[0.03,0.1],selected:2,filtered:0,missing:1});
assert.deepEqual(view([bound('adg','floor',0.02)]).ideal,[0.03,0.2]);
assert.deepEqual(view([bound('risk','ceiling',0.15)]).ideal,[0.01,0.1]);
assert.deepEqual(view([bound('adg','floor',0.02),bound('risk','ceiling',0.2)]).indices,[2]);
assert.deepEqual(membership(view([bound('extra','floor',20)])),{indices:[2],retained:[2,3],ideal:[0.03,0.2],selected:2,filtered:2,missing:1});
assert.deepEqual(view([{...bound('extra','floor',20),enabled:false}]).indices,[0,1,2]);
assert.equal(view([bound('adg','floor',1)]).ideal,null);
assert.equal(computeView(data,['adg','risk'],[null,'min'],[]).ideal,null);
assert.deepEqual(computeView(data,['adg','risk','third'],['max','min','max'],[]).ideal,[0.03,0.1,3]);
for(const x of ['min','max']) for(const y of ['min','max']) {
    const result=plotSpec(data,['adg','risk'],[x,y],[]);
    assert.equal(result.layout.xaxis.range[0]>result.layout.xaxis.range[1],x==='max');
    assert.equal(result.layout.yaxis.range[0]>result.layout.yaxis.range[1],y==='max');
    assert.equal(result.traces[1].marker.symbol,'diamond-open');
    assert.equal(result.traces[2].marker.symbol,'star');
    assert.deepEqual(result.traces[2].x,[data.metrics[0].values[result.view.selected]]);
    assert.deepEqual(result.traces[2].y,[data.metrics[1].values[result.view.selected]]);
    const filtered=plotSpec(data,['adg','risk'],[x,y],[bound('adg','floor',0.02)]);
    assert.notDeepEqual(result.layout.xaxis.range,filtered.layout.xaxis.range);
    assert.notDeepEqual(result.layout.yaxis.range,filtered.layout.yaxis.range);
    assert.notEqual(result.layout.uirevision,filtered.layout.uirevision);
    const sameFilters=plotSpec(data,['adg','risk'],[x,y],[bound('adg','floor',0.02)],'knee');
    assert.equal(filtered.layout.uirevision,sameFilters.layout.uirevision);
}
const three=plotSpec(data,['adg','risk','third'],['max','min','max'],[]);
assert.equal(three.traces[2].text[0],'★');
assert.equal(three.traces[1].marker.symbol,'diamond-open');
assert.equal(three.view.selected,0);
assert.deepEqual(three.traces[2].z,[3]);
assert.equal(three.traces[0].type,'scatter3d');
assert.deepEqual(three.traces[1].z,[3]);
assert.ok(three.layout.scene.xaxis.range[0]<three.layout.scene.xaxis.range[1]);
const filteredThree=plotSpec(data,['adg','risk','third'],['max','min','max'],[bound('extra','floor',20)]);
assert.deepEqual(three.layout.scene,filteredThree.layout.scene);
const empty=plotSpec(data,['adg','risk'],['max','min'],[bound('adg','floor',1)]);
assert.equal(empty.traces.length,1);
assert.deepEqual(empty.traces[0].x,[]);
assert.ok(empty.layout.annotations[0].text.includes('No points'));
assert.deepEqual(axisRange({min:0,max:0},'min',2),[-0.06,0.06]);
console.log('Browser logic passed');

const camera = {eye:{x:2,y:1,z:3},up:{x:0.3,y:0.4,z:0.5}};
const previous = {uirevision:'same',scene:{camera}};
const retained = preserveCamera({uirevision:'same',scene:{camera:{eye:{x:1}}}},previous);
assert.deepEqual(retained.scene.camera,camera);
assert.notEqual(retained.scene.camera,camera);
assert.deepEqual(preserveCamera({uirevision:'changed',scene:{camera:{eye:{x:1}}}},previous).scene.camera,{eye:{x:1}});

// Normalization must prevent a large-unit metric from dominating selection.
const scaled = {names:['a','b','c'],metrics:[metric('x',[0,0.006,0.01],'max'),metric('y',[0,400,1000],'min')]};
assert.equal(computeView(scaled,['x','y'],['max','min'],[]).selected,1);
// Recompute ranges as well as the ideal after limits, including off-axis limits.
assert.equal(view([bound('risk','ceiling',0.15)]).selected,0);
assert.equal(view([bound('extra','floor',20)]).selected,2);
assert.equal(view([bound('adg','floor',1)]).selected,null);
assert.equal(computeView(data,['adg','risk'],[null,'min'],[]).selected,null);
const tie = {names:['first','second'],metrics:[metric('x',[0,1],'max'),metric('y',[0,1],'min'),metric('constant',[5,5],'min')]};
assert.equal(computeView(tie,['x','y'],['max','min'],[]).selected,0);
assert.equal(computeView(tie,['x','constant'],['max','min'],[]).selected,1);
assert.equal(computeView(tie,['constant'],['min'],[]).selected,0);
const singleton=plotSpec(tie,['x','y'],['max','min'],[bound('x','floor',1)]);
assert.equal(singleton.view.selected,1);
assert.deepEqual(singleton.traces[1].x,singleton.traces[2].x);
assert.deepEqual(singleton.traces[1].y,singleton.traces[2].y);
assert.notEqual(singleton.traces[1].marker.symbol,singleton.traces[2].marker.symbol);

const rescaled = {names:['a','b','c','d'],metrics:[metric('x',[0,6,8,10],'max'),metric('y',[0,4,7,10],'min')]};
assert.equal(computeView(rescaled,['x','y'],['max','min'],[]).selected,1);
assert.equal(computeView(rescaled,['x','y'],['max','min'],[bound('x','floor',6)]).selected,2);
const filteredAxes=plotSpec(rescaled,['x','y'],['max','min'],[bound('x','floor',6)]);
assert.deepEqual(filteredAxes.layout.xaxis.range,axisRange({min:6,max:10},'max',2));
assert.deepEqual(filteredAxes.layout.yaxis.range,axisRange({min:4,max:10},'min',2));
assert.equal(filteredAxes.traces[0].marker.cmin,4);

// Machine-readable exports preserve the plotted settings and full ranking.
const exportLimits=[bound('x','floor',6),{...bound('y','ceiling',5),enabled:false}];
const exportAxes=['x','y'], exportGoals=['max','min'];
const exportView=computeView(rescaled,exportAxes,exportGoals,exportLimits,'ideal',[0,1]);
const record=buildViewRecord(rescaled,exportAxes,exportGoals,exportLimits,'ideal',[0,1],exportView);
assert.deepEqual(record.axes,[{metric:'x',goal:'max',weight:0},{metric:'y',goal:'min',weight:1}]);
assert.deepEqual(record.limits,[{metric:'x',mode:'floor',value:6}]);
assert.equal(record.selected,'b');
assert.deepEqual(record.ranking.map(item=>item.file),['b','c','d']);
assert.deepEqual(record.ranking[0].metrics,{x:6,y:4});

// Status-encoded infinite raw values obey off-axis bounds but have no axis coordinate.
const statusData = {names:['positive','negative','undefined'],metrics:[
    metric('x',[1,2,3],'max'),metric('y',[3,2,1],'min'),
    {...metric('raw',[null,null,null],null),statuses:[{status:'positive_infinity'},{status:'negative_infinity'},{status:'undefined'}]}
]};
assert.deepEqual(computeView(statusData,['x','y'],['max','min'],[bound('raw','floor',0)]).indices,[0]);
assert.deepEqual(computeView(statusData,['x','y'],['max','min'],[bound('raw','ceiling',0)]).indices,[1]);
assert.deepEqual(computeView(statusData,['x','raw'],['max','min'],[]).indices,[]);

// The ranking pick is supplied by Python. Axis settings and plot filters only affect its visibility.
const ranked = {...data, selection:{selected:{file:'b'}}};
for (const axes of [['adg','risk'],['adg','risk','third']]) {
    const goals = axes.map(key => data.metrics.find(item => item.key === key).goal);
    const result = plotSpec(ranked,axes,goals,[]);
    const pick = result.traces.find(trace => trace.name === 'Selection pick');
    const projection = result.traces.find(trace => trace.name === 'Closest on plotted axes');
    assert.equal(result.pickVisibility,'visible');
    assert.equal(pick.marker.symbol,'circle-open');
    assert.deepEqual(pick.customdata[0],['b','']);
    assert.notEqual(projection.customdata[0][0],pick.customdata[0][0]);
    assert.equal(axes.length === 3,Array.isArray(pick.z));
}
const hiddenPick = plotSpec(ranked,['adg','risk'],['max','min'],[bound('adg','floor',0.025)]);
assert.equal(hiddenPick.pickVisibility,'outside plot limits');
assert.equal(hiddenPick.traces.some(trace => trace.name === 'Selection pick'),false);
const missingPick = plotSpec(ranked,['adg','extra'],['max','max'],[]);
assert.equal(missingPick.pickVisibility,'missing coordinates');
const unspecifiedDirection = plotSpec(ranked,['adg','risk'],[null,'min'],[]);
assert.equal(unspecifiedDirection.pickVisibility,'visible');
assert.deepEqual(unspecifiedDirection.traces.find(trace => trace.name === 'Selection pick').customdata[0],['b','']);
assert.equal(unspecifiedDirection.traces.some(trace => trace.name === 'Closest on plotted axes'),false);
const overlappingPick = plotSpec({...tie,selection:{selected:{file:'second'}}},['x','y'],['max','min'],[bound('x','floor',1)]);
assert.deepEqual(overlappingPick.traces.slice(1).map(trace => trace.marker.symbol),['circle-open','diamond-open','star']);
for (const trace of overlappingPick.traces.slice(1)) {
    assert.deepEqual(trace.x,[1]);
    assert.deepEqual(trace.y,[1]);
}

// Plot labels can abbreviate a legacy filename while the marker matches the saved file.
const legacyFile = "a".repeat(64) + ".json";
const shortLabel = "a".repeat(24) + ".json";
const legacy = {...data, names:['a',legacyFile,'c','d'], selection:{
    selected:{file:legacyFile}, candidate_labels:{[legacyFile]:shortLabel},
}};
for (const axes of [['adg','risk'],['adg','risk','third']]) {
    const goals = axes.map(key => data.metrics.find(item => item.key === key).goal);
    const result = plotSpec(legacy,axes,goals,[]);
    const pick = result.traces.find(trace => trace.name === 'Selection pick');
    assert.equal(result.pickVisibility,'visible');
    assert.deepEqual(pick.x,[data.metrics[0].values[1]]);
    assert.deepEqual(pick.customdata[0],[shortLabel,"<br>File: " + legacyFile]);
    assert.deepEqual(result.traces[0].customdata[1],pick.customdata[0]);
}

// The served viewer uses CLI-retained rows even when a chart column has a
// different reducer or a candidate has no coordinate on the current axes.
const cliFilter = {retained:[1, 3], selected:'b'};
const browserLimits = [bound('adg', 'floor', 0.025)];
const cliProjection = plotSpec(ranked, ['adg','risk'], ['max','min'], browserLimits, 'ideal', null, null, cliFilter);
assert.deepEqual(cliProjection.view.indices, [1]);
assert.deepEqual(cliProjection.view.retained, [1, 3]);
assert.equal(cliProjection.filteredPickVisibility, 'visible');
assert.equal(cliProjection.traces.find(trace => trace.name === 'All-objective pick after limits').marker.symbol, 'square-open');
const cliProjection3D = plotSpec(ranked, ['adg','risk','third'], ['max','min','max'], browserLimits, 'knee', null, null, cliFilter);
assert.deepEqual(cliProjection3D.traces.find(trace => trace.name === 'All-objective pick after limits').z, [2]);
const missingCliPick = plotSpec(ranked, ['adg','risk'], ['max','min'], browserLimits, 'ideal', null, null, {...cliFilter, selected:'d'});
assert.equal(missingCliPick.filteredPickVisibility, 'missing coordinates');
