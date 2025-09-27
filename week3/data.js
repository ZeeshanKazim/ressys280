/* data.js
   PURPOSE (readable review):
   - Load MovieLens 100K files (u.item, u.data) from the SAME folder as this page.
   - Parse movies and ratings into memory.
   - Expose (for script.js): movies[], ratings[], numUsers, numMovies.
   - Design choice: be tolerant to both 18- and 19-genre-flag variants of u.item.
*/

let movies = [];   // [{ id, title, genres: string[] }]
let ratings = [];  // [{ userId, itemId, rating, timestamp }]

// For quick inspection and UI
let numUsers = 0;  // number of distinct users in u.data
let numMovies = 0; // number of parsed movie rows from u.item

/**
 * loadData()
 * 1) fetch u.item → parseItemData
 * 2) fetch u.data → parseRatingData
 * 3) compute distinct user count + movie count for display
 * If files are missing, show a friendly message in #result and throw.
 */
async function loadData() {
  try {
    const itemResp = await fetch('u.item');
    if (!itemResp.ok) throw new Error('Failed to load u.item');
    parseItemData(await itemResp.text());

    const dataResp = await fetch('u.data');
    if (!dataResp.ok) throw new Error('Failed to load u.data');
    parseRatingData(await dataResp.text());

    // counts for UI
    const userSet = new Set(ratings.map(r => r.userId));
    numUsers = userSet.size;
    numMovies = movies.length;
  } catch (err) {
    console.error(err);
    const res = document.getElementById('result');
    if (res) {
      res.textContent = 'Error: make sure u.item and u.data are in the same folder as this page.';
    }
    throw err;
  }
}

/**
 * parseItemData(text)
 * MovieLens 100K (classic) has 19 genre flags per row including "Unknown".
 * Some copies ship with 18 (no "Unknown").
 * We support BOTH by detecting how many trailing fields exist after column 4.
 * 
 * Row format (pipe-separated):
 *   movieId | title | release date | video release date | IMDb URL | [flags...]
 */
function parseItemData(text) {
  const genres19 = [
    'Unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama',
    'Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'
  ];
  const genres18 = genres19.slice(1); // drop "Unknown" if absent

  movies = [];
  const lines = text.split('\n').filter(l => l.trim().length);
  for (const line of lines) {
    const parts = line.split('|');
    if (parts.length < 5) continue;

    const id = parseInt(parts[0], 10);
    const title = parts[1];

    // Tail fields contain genre flags. Detect 18 vs 19.
    const tail = parts.slice(5);
    const is19 = tail.length >= 19;
    const flags = is19 ? tail.slice(0,19) : tail.slice(0,18);
    const names = is19 ? genres19 : genres18;

    const genres = [];
    for (let i = 0; i < flags.length; i++) {
      if (flags[i] === '1') {
        const g = names[i];
        // ignore "Unknown" when present
        if (g && g !== 'Unknown') genres.push(g);
      }
    }
    movies.push({ id, title, genres });
  }
}

/**
 * parseRatingData(text)
 * u.data = tab-separated rows: userId \t itemId \t rating \t timestamp
 */
function parseRatingData(text) {
  ratings = [];
  const lines = text.split('\n').filter(l => l.trim().length);
  for (const line of lines) {
    const [userId, itemId, rating, timestamp] = line.split('\t');
    ratings.push({
      userId: parseInt(userId, 10),
      itemId: parseInt(itemId, 10),
      rating: parseInt(rating, 10),
      timestamp: parseInt(timestamp, 10)
    });
  }
}
