/* ---- BEGIN PATCH (app.js) --------------------------------------------- */
/* Helper: try several paths, with cache-busting so GH Pages picks updates */
async function fetchTextTry(paths) {
  for (const p of paths) {
    try {
      const r = await fetch(p + (p.includes('?') ? '' : `?v=${Date.now()}`));
      if (r.ok) return await r.text();
    } catch { /* try next */ }
  }
  throw new Error(`Could not fetch any of: ${paths.join(', ')}`);
}

/* Replace your existing loadData() with this version */
async function loadData(){
  await tf.ready();
  try{
    // Try ./data/... first (your repo layout), then fall back to ./...
    const itemTxt = await fetchTextTry(['./data/u.item', './u.item', 'u.item', 'data/u.item']);
    const dataTxt = await fetchTextTry(['./data/u.data', './u.data', 'u.data', 'data/u.data']);

    // Reset containers
    S.items.clear(); S.itemIds.clear();
    S.interactions.length = 0; S.users.clear();

    // ---- parse u.item ----
    for (const line of itemTxt.split('\n').filter(Boolean)) {
      const parts = line.split('|'); if (parts.length < 2) continue;
      const id = +parts[0];
      let title = parts[1];
      let year = null;
      const m = title.match(/\((\d{4})\)\s*$/);
      if (m) { year = +m[1]; title = title.replace(/\s*\(\d{4}\)\s*$/, ''); }

      // genres start at col 5; MovieLens-100K has 19 flags incl. "Unknown"
      const flags = parts.slice(5).map(x => x === '1' ? 1 : 0);
      const use = (flags.length >= 19) ? flags.slice(1) : flags; // drop "Unknown"
      const g = new Int8Array(18);
      for (let i = 0; i < Math.min(18, use.length); i++) g[i] = use[i];

      S.items.set(id, { title, year, genres: g });
      S.itemIds.add(id);
    }

    // ---- parse u.data ----
    for (const line of dataTxt.split('\n').filter(Boolean)) {
      const [u,i,r,t] = line.split('\t');
      const userId = +u, itemId = +i, rating = +r, ts = +t;
      if (Number.isFinite(userId) && Number.isFinite(itemId)) {
        S.interactions.push({ userId, itemId, rating, ts });
        S.users.add(userId);
      }
    }

    // ---- indexers ----
    const userIds = [...S.users].sort((a,b)=>a-b);
    const itemIds = [...S.itemIds].sort((a,b)=>a-b);
    S.idx2userId = userIds;
    S.idx2itemId = itemIds;
    S.userId2idx = new Map(userIds.map((u,idx)=>[u,idx]));
    S.itemId2idx = new Map(itemIds.map((i,idx)=>[i,idx]));

    // ---- per-user sets & top-rated cache (for Test view) ----
    S.userToRated.clear(); S.userTopRated.clear();
    for (const {userId,itemId} of S.interactions) {
      if (!S.userToRated.has(userId)) S.userToRated.set(userId, new Set());
      S.userToRated.get(userId).add(itemId);
    }
    const byUser = new Map();
    for (const r of S.interactions) {
      if (!byUser.has(r.userId)) byUser.set(r.userId, []);
      byUser.get(r.userId).push(r);
    }
    for (const [uid, arr] of byUser) {
      arr.sort((a,b)=> (b.rating - a.rating) || (b.ts - a.ts));
      const top = arr.slice(0, 60).map(x => ({
        itemId: x.itemId, rating: x.rating, ts: x.ts,
        title: S.items.get(x.itemId)?.title ?? String(x.itemId),
        year:  S.items.get(x.itemId)?.year  ?? ''
      }));
      S.userTopRated.set(uid, top);
    }

    // Enable buttons & show status
    $('btn-train').classList.remove('secondary');
    $('btn-test').classList.remove('secondary');
    setStatus(`data loaded — users: ${userIds.length}, items: ${itemIds.length}, interactions: ${S.interactions.length}`);
  } catch (e) {
    console.error(e);
    setStatus('fetch failed. Make sure files are at ./data/u.item and ./data/u.data (or next to index.html).');
  }
}
/* ---- END PATCH --------------------------------------------------------- */
