// Global variables to store movie and rating data
let movies = [];
let ratings = [];

// Sample movie data as fallback
const sampleMovies = [
    { id: 1, title: "Toy Story (1995)", genres: ["Animation", "Children's", "Comedy"] },
    { id: 2, title: "Jumanji (1995)", genres: ["Adventure", "Children's", "Fantasy"] },
    { id: 3, title: "Grumpier Old Men (1995)", genres: ["Comedy", "Romance"] },
    { id: 4, title: "Waiting to Exhale (1995)", genres: ["Comedy", "Drama", "Romance"] },
    { id: 5, title: "Father of the Bride Part II (1995)", genres: ["Comedy"] },
    { id: 6, title: "Heat (1995)", genres: ["Action", "Crime", "Thriller"] },
    { id: 7, title: "Sabrina (1995)", genres: ["Comedy", "Romance"] },
    { id: 8, title: "Tom and Huck (1995)", genres: ["Adventure", "Children's"] },
    { id: 9, title: "Sudden Death (1995)", genres: ["Action"] },
    { id: 10, title: "GoldenEye (1995)", genres: ["Action", "Adventure", "Thriller"] },
    { id: 11, title: "American President, The (1995)", genres: ["Comedy", "Drama", "Romance"] },
    { id: 12, title: "Dracula: Dead and Loving It (1995)", genres: ["Comedy", "Horror"] },
    { id: 13, title: "Balto (1995)", genres: ["Animation", "Children's"] },
    { id: 14, title: "Nixon (1995)", genres: ["Drama"] },
    { id: 15, title: "Cutthroat Island (1995)", genres: ["Action", "Adventure", "Romance"] }
];

// Sample rating data as fallback
const sampleRatings = [
    { userId: 1, itemId: 1, rating: 5, timestamp: 887431883 },
    { userId: 1, itemId: 2, rating: 3, timestamp: 875693118 },
    { userId: 1, itemId: 3, rating: 4, timestamp: 888717495 },
    { userId: 2, itemId: 1, rating: 4, timestamp: 878542960 },
    { userId: 2, itemId: 4, rating: 5, timestamp: 886397596 },
    { userId: 2, itemId: 5, rating: 3, timestamp: 884182806 },
    { userId: 3, itemId: 6, rating: 5, timestamp: 881251949 },
    { userId: 3, itemId: 7, rating: 4, timestamp: 876862836 },
    { userId: 3, itemId: 8, rating: 3, timestamp: 878542960 }
];

/**
 * Load and parse data from local files with fallback to sample data
 * @returns {Promise} Promise that resolves when data is loaded
 */
async function loadData() {
    try {
        // Try to load movie data from u.item
        const moviesResponse = await fetch('u.item');
        if (moviesResponse.ok) {
            const moviesText = await moviesResponse.text();
            parseItemData(moviesText);
            console.log('Loaded movie data from u.item');
        } else {
            throw new Error('u.item not found');
        }
        
        // Try to load rating data from u.data
        const ratingsResponse = await fetch('u.data');
        if (ratingsResponse.ok) {
            const ratingsText = await ratingsResponse.text();
            parseRatingData(ratingsText);
            console.log('Loaded rating data from u.data');
        } else {
            throw new Error('u.data not found');
        }
        
    } catch (error) {
        console.log('Using fallback data:', error.message);
        // Use sample data as fallback
        movies = sampleMovies;
        ratings = sampleRatings;
        
        // Show message about fallback data
        document.getElementById('data-status').classList.remove('hidden');
    }
}

/**
 * Parse movie data from u.item file
 * @param {string} text - Raw text data from u.item
 */
function parseItemData(text) {
    const genreNames = [
        "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ];
    
    const lines = text.split('\n').filter(line => line.trim() !== '');
    
    movies = lines.map(line => {
        const parts = line.split('|');
        if (parts.length < 5) return null;
        
        const movieId = parseInt(parts[0]);
        const title = parts[1];
        
        // Extract genre information (binary flags for 18 genres)
        const genres = [];
        for (let i = 0; i < 18; i++) {
            const genreIndex = 5 + i;
            if (parts.length > genreIndex && parts[genreIndex] === '1') {
                genres.push(genreNames[i]);
            }
        }
        
        return {
            id: movieId,
            title: title,
            genres: genres
        };
    }).filter(movie => movie !== null);
}

/**
 * Parse rating data from u.data file
 * @param {string} text - Raw text data from u.data
 */
function parseRatingData(text) {
    const lines = text.split('\n').filter(line => line.trim() !== '');
    
    ratings = lines.map(line => {
        const parts = line.split('\t');
        if (parts.length < 4) return null;
        
        return {
            userId: parseInt(parts[0]),
            itemId: parseInt(parts[1]),
            rating: parseInt(parts[2]),
            timestamp: parseInt(parts[3])
        };
    }).filter(rating => rating !== null);
}
