// Global arrays
let movies = [];
let ratings = [];

/** Load MovieLens-style files in the same folder */
async function loadData(){
  try{
    const uItem = await fetch('u.item');
    if (!uItem.ok) throw new Error('Failed to load u.item');
    parseItemData(await uItem.text());

    const uData = await fetch('u.data');
    if (!uData.ok) throw new Error('Failed to load u.data');
    parseRatingData(await uData.text());
  }catch(err){
    console.error(err);
    const el = document.getElementById('result');
    if (el) el.textContent = 'Error: put u.item and u.data in this folder.';
    throw err;
  }
}

/** Robust parser: supports 18 **or** 19 genre flags */
function parseItemData(text){
  const genres19 = [
    'Unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama',
    'Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'
  ];
  const genres18 = genres19.slice(1);

  movies = [];
  for (const line of text.split('\n')){
    if (!line.trim()) continue;
    const parts = line.split('|');
    if (parts.length < 5) continue;

    const id = parseInt(parts[0],10);
    const title = parts[1];

    const tail = parts.slice(5);
    const is19 = tail.length >= 19;
    const flags = is19 ? tail.slice(0,19) : tail.slice(0,18);
    const names = is19 ? genres19 : genres18;

    const g = [];
    for (let i=0;i<flags.length;i++){
      if (flags[i] === '1'){
        const name = names[i];
        if (name && name !== 'Unknown') g.push(name);
      }
    }
    movies.push({ id, title, genres: g });
  }
}

function parseRatingData(text){
  ratings = [];
  for (const line of text.split('\n')){
    if (!line.trim()) continue;
    const [u,i,r,t] = line.split('\t');
    ratings.push({ userId:+u, itemId:+i, rating:+r, timestamp:+t });
  }
}
