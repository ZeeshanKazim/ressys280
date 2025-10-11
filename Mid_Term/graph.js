// graph.js — lightweight co-vis graph + Personalized PageRank re-rank (optional)

let ITEM_NEIGHBORS = null; // itemId -> Map(neiId -> weight)

function buildCoVisGraph(maxUsers = 5000) {
  // Build only once
  if (ITEM_NEIGHBORS) return ITEM_NEIGHBORS;
  ITEM_NEIGHBORS = new Map();
  const users = [...window.DATASET.USERS.entries()].slice(0, maxUsers);
  for (const [,arr] of users) {
    const items = arr.map(x=>x.item);
    for (let a=0;a<items.length;a++){
      for (let b=a+1;b<items.length;b++){
        const i = items[a], j = items[b];
        if (!ITEM_NEIGHBORS.has(i)) ITEM_NEIGHBORS.set(i,new Map());
        if (!ITEM_NEIGHBORS.has(j)) ITEM_NEIGHBORS.set(j,new Map());
        ITEM_NEIGHBORS.get(i).set(j, (ITEM_NEIGHBORS.get(i).get(j)||0)+1);
        ITEM_NEIGHBORS.get(j).set(i, (ITEM_NEIGHBORS.get(j).get(i)||0)+1);
      }
    }
  }
  // normalize by degree
  for (const [i, m] of ITEM_NEIGHBORS.entries()) {
    let sum = 0; for (const v of m.values()) sum+=v;
    if (sum>0) for (const [j,v] of m.entries()) m.set(j, v/sum);
  }
  return ITEM_NEIGHBORS;
}

function personalizedPageRank(seeds, alpha=0.15, iters=20) {
  buildCoVisGraph();
  const itemToIdx = new Map(window.DATASET.revItem.map((id,i)=>[id,i]));
  const N = window.DATASET.revItem.length;
  const r = new Float32Array(N);
  const p = new Float32Array(N);
  if (seeds.length===0) return r;
  seeds.forEach(id=>{ const j=itemToIdx.get(id); if (j!=null) p[j]+=1; });
  const pSum = p.reduce((a,b)=>a+b,0)||1; for(let i=0;i<N;i++) p[i]/=pSum;
  // power iterations on sparse graph
  for (let t=0;t<iters;t++){
    const rnext = new Float32Array(N);
    for (const [i,nei] of ITEM_NEIGHBORS.entries()){
      const ii = itemToIdx.get(i); if (ii==null) continue;
      for (const [j,w] of nei.entries()){
        const jj = itemToIdx.get(j); if (jj==null) continue;
        rnext[jj] += (1-alpha) * (r[ii] || p[ii]) * w; // start with p as initial mass
      }
    }
    for (let k=0;k<N;k++) r[k] = rnext[k] + alpha * p[k];
  }
  return r;
}

// Re-rank helper: take original recs [{itemId,score,...}], mix with PPR
function reRankWithPPR(recs, seedItems, alpha=0.15, iters=10) {
  const ppr = personalizedPageRank(seedItems, alpha, iters);
  const itemToIdx = new Map(window.DATASET.revItem.map((id,i)=>[id,i]));
  const mixed = recs.map(r=>{
    const j = itemToIdx.get(r.itemId);
    const boost = j!=null ? ppr[j] : 0;
    return {...r, score: r.score + 0.5*boost};
  });
  mixed.sort((a,b)=>b.score-a.score);
  return mixed;
}

window.reRankWithPPR = reRankWithPPR;
