// Global data
let movies = [];
let ratings = [];

/** Load MovieLens-style files (same folder) */
async function loadData() {
  try {
    const itemResp = await fetch('u.item');
    if (!itemResp.ok) throw new Error('Failed to load u.item');
    parseItemData(await itemResp.text());

    const dataResp = await fetch('u.data');
    if (!dataResp.ok) throw new Error('Failed to load u.data');
    parseRatingData(await dataResp.text());
  } catch (err) {
    console.error(err);
    const el = document.getElementById('result');
    if (el) el.textContent = 'Error: make sure u.item and u.data are next to this page.';
    throw err;
  }
}

/** Parse u.item into {id,title,genres[]} */
function parseItemData(text) {
  // MovieLens 100K has 19 flags (Unknown..Western)
  const genreNames19 = [
    'Unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama',
    'Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'
  ];
  movies = [];
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    const parts = line.split('|');
    if (parts.length < 5 + 19) continue;
    const id = parseInt(parts[0],10);
    const title = parts[1];
    const flags = parts.slice(-19);
    const genres = [];
    for (let i=0;i<flags.length;i++){
      if (flags[i]==='1' && genreNames19[i] !== 'Unknown') genres.push(genreNames19[i]);
    }
    movies.push({ id, title, genres });
  }
}

/** Parse u.data into rating rows */
function parseRatingData(text) {
  ratings = [];
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    const [u,i,r,t] = line.split('\t');
    ratings.push({ userId:+u, itemId:+i, rating:+r, timestamp:+t });
  }
}
