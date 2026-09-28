const assert = require('node:assert/strict');
const {computeView, axisRange, plotSpec, preserveCamera} = require(process.argv[2]);
const metric = (key,values,goal) => ({key,label:key,values,goal,min:Math.min(...values.filter(v=>v!==null)),max:Math.max(...values.filter(v=>v!==null))});
const data = {names:['a','b','c','d'],metrics:[metric('adg',[0.01,0.02,0.03,0.04],'max'),metric('risk',[0.1,0.4,0.2,null],'min'),metric('third',[3,2,1,9],'max'),metric('extra',[10,null,30,40],null)]};
const bound = (key,mode,value) => ({key,mode,value,enabled:true});
const view = limits => computeView(data,['adg','risk'],['max','min'],limits);
assert.deepEqual(view([]),{indices:[0,1,2],ideal:[0.03,0.1],selected:2,filtered:0,missing:1});
assert.deepEqual(view([bound('adg','floor',0.02)]).ideal,[0.03,0.2]);
assert.deepEqual(view([bound('risk','ceiling',0.15)]).ideal,[0.01,0.1]);
assert.deepEqual(view([bound('adg','floor',0.02),bound('risk','ceiling',0.2)]).indices,[2]);
assert.deepEqual(view([bound('extra','floor',20)]),{indices:[2],ideal:[0.03,0.2],selected:2,filtered:2,missing:1});
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
    assert.deepEqual(result.layout.xaxis.range,filtered.layout.xaxis.range);
    assert.equal(result.layout.uirevision,filtered.layout.uirevision);
}
const three=plotSpec(data,['adg','risk','third'],['max','min','max'],[]);
assert.equal(three.traces[2].text[0],'★');
assert.equal(three.traces[1].marker.symbol,'diamond-open');
assert.equal(three.view.selected,0);
assert.deepEqual(three.traces[2].z,[3]);
assert.equal(three.traces[0].type,'scatter3d');
assert.deepEqual(three.traces[1].z,[3]);
assert.ok(three.layout.scene.xaxis.range[0]<three.layout.scene.xaxis.range[1]);
const empty=plotSpec(data,['adg','risk'],['max','min'],[bound('adg','floor',1)]);
assert.equal(empty.traces.length,1);
assert.deepEqual(empty.traces[0].x,[]);
assert.ok(empty.layout.annotations[0].text.includes('No candidates'));
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
