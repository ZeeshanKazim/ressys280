/* data.js
   - Load MovieLens 100K files (u.item, u.data) from this folder
   - Parse movies and ratings
   - Expose: movies[], ratings[], numUsers, numMovies
*/

let movies = [];   // [{ id, title, genres: [] }]
let ratings = [];  // [{ userId, itemId, rating, timestamp }]

let numUsers = 0;  // distinct users in u.data
let numMovies = 0; // movies parsed from u.item

async function loadData() {
  try {
    const itemResp = await fetch('u.item');
    if (!itemResp.ok) throw new Error('Failed to load u.item');
    parseItemData(await itemResp.text());

    const dataResp = await fetch('u.data');
    if (!dataResp.ok) throw new Error('Failed to load u.data');
    parseRatingData(await dataResp.text());

    const userSet = new Set(ratings.map(r => r.userId));
    numUsers = userSet.size;
    numMovies = movies.length;
  } catch (err) {
    console.error(err);
    const res = document.getElementById('result');
    if (res) res.textContent = 'Error: make sure u.item and u.data are in the same folder as this page.';
    throw err;
  }
}

function parseItemData(text) {
  const genres19 = [
    'Unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama',
    'Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'
  ];
  const genres18 = genres19.slice(1);

  movies = [];
  const lines = text.split('\n').filter(l => l.trim().length);
  for (const line of lines) {
    const parts = line.split('|');
    if (parts.length < 5) continue;

    const id = parseInt(parts[0], 10);
    const title = parts[1];

    const tail = parts.slice(5);
    const is19 = tail.length >= 19;
    theFlags = is19 ? tail.slice(0,19) : tail.slice(0,18);
    const names = is19 ? genres19 : genres18;

    const genres = [];
    for (let i = 0; i < theFlags.length; i++) {
      if (theFlags[i] === '1') {
        const g = names[i];
        if (g && g !== 'Unknown') genres.push(g);
      }
    }
    movies.push({ id, title, genres });
  }
}

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
