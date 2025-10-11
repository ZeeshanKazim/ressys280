/* graph.js — simple co-vis graph and Personalized PageRank
   Build an item-item graph: edges weighted by co-occurrence within same user.
   Use it to optionally re-rank recommender scores with PPR from the user’s history.
*/

let GRAPH = null; // Map itemIdx -> Map neighborIdx -> weight
let GRAPH_N = 0;

function ensureGraph(INTERACTIONS){
  if (GRAPH) return GRAPH;
  GRAPH = new Map();
  // Build by user windows (small memory footprint)
  const byUser = new Map();
  for(const x of INTERACTIONS){
    if(!byUser.has(x.userId)) byUser.set(x.userId, []);
    byUser.get(x.userId).push(x.itemId);
  }
  for(const arr of byUser.values()){
    // use set to reduce self duplicates
    const uniq = Array.from(new Set(arr));
    for(let a=0;a<uniq.length;a++){
      for(let b=a+1;b<uniq.length;b++){
        addEdge(uniq[a], uniq[b], 1);
      }
    }
  }
  GRAPH_N = new Set([].concat(...Array.from(GRAPH.keys()))).size;
  return GRAPH;

  function addEdge(itemAId, itemBId, w){
    const A = window.item2idx.get(itemAId);
    const B = window.item2idx.get(itemBId);
    if (A==null || B==null || A===B) return;
    if(!GRAPH.has(A)) GRAPH.set(A, new Map());
    if(!GRAPH.has(B)) GRAPH.set(B, new Map());
    GRAPH.get(A).set(B, (GRAPH.get(A).get(B)||0) + w);
    GRAPH.get(B).set(A, (GRAPH.get(B).get(A)||0) + w);
  }
}

/* Personalized PageRank using power iteration.
   seeds: array of itemIds (history)
   returns Map idx->score
*/
function personalizedPageRank(seeds, alpha=0.15, iters=30){
  if(!GRAPH) return new Map();
  const M = window.idx2item.length;
  const v = new Float32Array(M); // personalization
  const seedIdx = seeds.map(id=>window.item2idx.get(id)).filter(i=>i!=null);
  if (!seedIdx.length) return new Map();
  const mass = 1/seedIdx.length;
  for(const s of seedIdx){ v[s]=mass; }

  // degree
  const deg = new Float32Array(M);
  for(const [i,nb] of GRAPH){
    let sum=0; for(const [,w] of nb) sum+=w;
    deg[i]=sum||1;
  }

  // iterate: r_{t+1} = alpha*v + (1-alpha)*P^T r_t
  let r = v.slice();
  for(let t=0;t<iters;t++){
    const r2 = new Float32Array(M);
    for(const [i,nb] of GRAPH){
      const ri = r[i];
      if(ri===0) continue;
      for(const [j,w] of nb){
        r2[j] += (1-alpha) * ri * (w/deg[i]);
      }
    }
    for(let k=0;k<M;k++) r2[k] += alpha * v[k];
    r = r2;
  }
  const out = new Map();
  for(let i=0;i<M;i++) if(r[i]>0) out.set(i, r[i]);
  return out;
}

/* Combine base scores with PPR: score' = (1-l)*score + l*norm(ppr) */
function rerankWithGraph(base, ppr, lambda=0.3){
  if(!base.length) return base;
  const maxBase = Math.max(...base.map(x=>x.score)) || 1;
  const maxPPR = Math.max(1e-9, ...Array.from(ppr.values()));
  const m = new Map(base.map(x=>[x.idx, x.score/maxBase]));
  const out = base.map(x=>{
    const s = (1-lambda)*(m.get(x.idx)||0) + lambda*((ppr.get(x.idx)||0)/maxPPR);
    return {idx:x.idx, score:s};
  });
  out.sort((a,b)=>b.score-a.score);
  return out;
}

// expose
window.ensureGraph = ensureGraph;
window.personalizedPageRank = personalizedPageRank;
window.rerankWithGraph = rerankWithGraph;
