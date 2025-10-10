/* graph.js — Build item–item co‑vis graph, PageRank & Personalized PR */

(function(){
  const GraphUtils = {
    buildCoVisPairs(train, positivesByUser, userToIdx, itemToIdx){
      // gather per-user positive item sets from train (rating>=4 or all if no rating)
      const byUser = new Map();
      for(const r of train){
        if(!userToIdx.has(r.userId) || !itemToIdx.has(r.itemId)) continue;
        const ok = (r.rating==null) ? true : (r.rating>=4);
        if(!ok) continue;
        if(!byUser.has(r.userId)) byUser.set(r.userId, []);
        byUser.get(r.userId).push(itemToIdx.get(r.itemId));
      }
      // build co-visit pairs (upper triangle)
      const pairs = [];
      for(const arr of byUser.values()){
        const uniq = Array.from(new Set(arr));
        const L = Math.min(uniq.length, 80); // cap per-user to limit O(n^2)
        for(let i=0;i<L;i++){
          for(let j=i+1;j<L;j++){
            pairs.push([uniq[i], uniq[j]]);
          }
        }
      }
      return pairs;
    },

    buildGraph(pairs, numItems){
      // adjacency with weights (undirected -> add both directions)
      const adj = new Array(numItems);
      for(let i=0;i<numItems;i++) adj[i] = new Map();
      for(const [a,b] of pairs){
        adj[a].set(b, 1+(adj[a].get(b)||0));
        adj[b].set(a, 1+(adj[b].get(a)||0));
      }
      // row-normalize to transition matrix components
      const outW = new Array(numItems);
      for(let i=0;i<numItems;i++){
        let sum=0; adj[i].forEach(v=>sum+=v);
        const row = new Map();
        adj[i].forEach((w,j)=> row.set(j, sum? (w/sum) : 0));
        outW[i]=row;
      }
      return { N:numItems, rows: outW };
    },

    personalizedPageRank(graph, {seeds=[], d=0.85, maxIters=50, tol=1e-8}={}){
      const N = graph.N; if(N===0) return [];
      const pr = new Float64Array(N);
      const next = new Float64Array(N);
      // teleport vector: if no seeds, uniform
      const tele = new Float64Array(N);
      if(seeds.length>0){ for(const s of seeds){ if(s>=0 && s<N) tele[s]=1; } }
      const teleSum = seeds.length>0 ? seeds.length : N;
      for(let i=0;i<N;i++) tele[i] = tele[i]/teleSum;

      for(let i=0;i<N;i++) pr[i] = 1/N;

      for(let it=0; it<maxIters; it++){
        for(let i=0;i<N;i++) next[i]=0;
        for(let i=0;i<N;i++){
          const row = graph.rows[i]; const p = d*pr[i];
          if(row.size===0) continue;
          row.forEach((w,j)=>{ next[j]+= p*w; });
        }
        const invN = (1-d);
        for(let j=0;j<N;j++) next[j] += invN * tele[j];

        let diff=0; for(let k=0;k<N;k++){ diff += Math.abs(next[k]-pr[k]); pr[k]=next[k]; }
        if(diff<tol) break;
      }
      return Array.from(pr);
    }
  };

  window.GraphUtils = GraphUtils;
})();
