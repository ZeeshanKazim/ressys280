/* ppr.js */
function personalizedPageRank(userId, user2items, item2users, {alpha=0.15, iters=20}={}){
  // Build small ego graph around the user for speed.
  const uStr = String(userId);
  const items = new Set((user2items.get(userId)||[]).map(r=>r.i));
  const neighUsers = new Set();
  for (const it of items){
    for (const u of (item2users.get(it)||[])) neighUsers.add(u);
  }
  const nodes = new Set([uStr]);
  items.forEach(i=>nodes.add('i:'+i));
  neighUsers.forEach(u=>nodes.add('u:'+u));

  // adjacency
  const out = new Map();
  const push = (a,b)=>{ if(!out.has(a)) out.set(a, new Set()); out.get(a).add(b); };

  for (const it of items){
    const iNode = 'i:'+it;
    push(uStr, iNode); push(iNode, uStr);
    for (const u of (item2users.get(it)||[])){ const uN='u:'+u; push(iNode,uN); push(uN,iNode); }
  }

  const id = new Map(); let idx=0;
  for (const n of nodes) id.set(n, idx++);
  const N = idx;
  const p = new Float32Array(N); p[id.get(uStr)] = 1;
  let x = Float32Array.from(p);

  for (let t=0;t<iters;t++){
    const y = new Float32Array(N);
    for (const [a,nb] of out){
      const ia = id.get(a); const deg = nb.size || 1;
      for (const b of nb){ y[id.get(b)] += (1-alpha) * (x[ia]/deg); }
    }
    for (let i=0;i<N;i++) y[i] += alpha * p[i];
    x = y;
  }

  const scores = new Map();
  for (const it of items) scores.set(it, 0); // exclude seen later
  for (const [node,ix] of id){
    if (node.startsWith('i:')) scores.set(parseInt(node.slice(2),10), x[ix]);
  }
  return scores;
}
window.PPR = { personalizedPageRank };
