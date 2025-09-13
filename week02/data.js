// Global variables to store movie and rating data
let movies = [];
let ratings = [];

/**
 * Load and parse data from local files
 * @returns {Promise} Promise that resolves when data is loaded
 */
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error('Failed to load movie data');
        }
        const moviesText = await moviesResponse.text();
        parseItemData(moviesText);
        
        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error('Failed to load rating data');
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);
    } catch (error) {
        console.error('Error loading data:', error);
        document.getElementById('result').textContent = 
            'Error loading data. Please check if u.item and u.data files are available.';
        throw error;
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
