/* graph.js — simple Personalized PageRank on user–item bipartite graph */
(function(global){
  function personalizedPageRankForUser(uId, user2items, item2users, opts={}){
    const alpha = opts.alpha ?? 0.15, iters=opts.iters ?? 20;
    // Build index maps on the fly
    const items = new Set();
    (user2items.get(uId)||[]).forEach(r=>items.add(r.i));
    for(const it of items){
      for(const v of (item2users.get(it)||[])) (user2items.get(v)||[]).forEach(r=>items.add(r.i));
    }
    const nodes = new Map(); // id->idx
    const rev = [];
    let k=0;
    nodes.set(`U:${uId}`, k); rev[k]=`U:${uId}`; k++;
    for(const it of items){ nodes.set(`I:${it}`, k); rev[k]=`I:${it}`; k++; }
    const n = k;

    // Build row-stochastic transition
    const rows = Array.from({length:n}, ()=>[]);
    function addEdge(a,b){ rows[a].push(b); }
    // U -> items rated
    const hist = user2items.get(uId)||[];
    for(const r of hist){ addEdge(nodes.get(`U:${uId}`), nodes.get(`I:${r.i}`)); }
    // Items -> their users (including uId); then to their items
    for(const it of items){
      const itsUsers = item2users.get(it)||new Set();
      for(const v of itsUsers){
        const vHist = user2items.get(v)||[];
        for(const r of vHist){ if (items.has(r.i)) addEdge(nodes.get(`I:${it}`), nodes.get(`I:${r.i}`)); }
      }
    }
    // normalize
    const P = Array.from({length:n}, ()=>new Float32Array(n));
    for(let i=0;i<n;i++){
      const outs = rows[i];
      if(!outs.length){ P[i][i]=1; continue; }
      const w = 1/outs.length;
      outs.forEach(j=>{ P[i][j]+=w; });
    }

    // power iterations
    let x = new Float32Array(n); x[0]=1; // start at U:uId
    let tmp = new Float32Array(n);
    for(let t=0;t<iters;t++){
      tmp.fill(0);
      for(let i=0;i<n;i++){
        const xi = x[i];
        if(xi===0) continue;
        const row = P[i];
        for(let j=0;j<n;j++){ const pij=row[j]; if(pij) tmp[j]+=xi*pij; }
      }
      // teleport
      for(let j=0;j<n;j++) tmp[j] = alpha*(j===0?1:0) + (1-alpha)*tmp[j];
      x = tmp.slice();
    }
    // collect item scores
    const res = new Map();
    for(let j=0;j<n;j++){
      const tag = rev[j];
      if(tag.startsWith('I:')) res.set(parseInt(tag.slice(2),10), x[j]);
    }
    return res;
  }

  global.personalizedPageRankForUser = personalizedPageRankForUser;
})(window);
