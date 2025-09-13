/**
 * Initialize application when window loads
 */
window.onload = async function() {
    try {
        // Wait for data to load
        await loadData();
        
        // Populate the movie dropdown
        populateMoviesDropdown();
        
        // Update status message
        document.getElementById('result').textContent = 
            'Data loaded. Please select a movie.';
            
        // Add event listener to the recommendation button
        document.getElementById('recommend-btn').addEventListener('click', getRecommendations);
    } catch (error) {
        console.error('Error initializing application:', error);
    }
};

/**
 * Populate the movie dropdown with available movies
 */
function populateMoviesDropdown() {
    const movieSelect = document.getElementById('movie-select');
    
    // Clear existing options except the first one
    while (movieSelect.options.length > 1) {
        movieSelect.remove(1);
    }
    
    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) => {
        return a.title.localeCompare(b.title);
    });
    
    // Add movies to dropdown
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        movieSelect.appendChild(option);
    });
}

/**
 * Convert genre array to binary feature vector
 * @param {Array} genres - Array of genre names
 * @param {Array} allGenres - All possible genre names
 * @returns {Array} Binary feature vector
 */
function getFeatureVector(genres, allGenres) {
    return allGenres.map(genre => genres.includes(genre) ? 1 : 0);
}

/**
 * Calculate dot product of two vectors
 * @param {Array} vectorA - First vector
 * @param {Array} vectorB - Second vector
 * @returns {number} Dot product
 */
function dotProduct(vectorA, vectorB) {
    return vectorA.reduce((sum, val, i) => sum + val * vectorB[i], 0);
}

/**
 * Calculate magnitude of a vector
 * @param {Array} vector - Input vector
 * @returns {number} Magnitude
 */
function magnitude(vector) {
    return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
}

/**
 * Calculate cosine similarity between two vectors
 * @param {Array} vectorA - First vector
 * @param {Array} vectorB - Second vector
 * @returns {number} Cosine similarity
 */
function calculateCosineSimilarity(vectorA, vectorB) {
    const dot = dotProduct(vectorA, vectorB);
    const magA = magnitude(vectorA);
    const magB = magnitude(vectorB);
    
    if (magA === 0 || magB === 0) return 0;
    return dot / (magA * magB);
}

/**
 * Get recommendations based on the selected movie
 */
function getRecommendations() {
    // Step 1: Get user input
    const movieSelect = document.getElementById('movie-select');
    const selectedMovieId = parseInt(movieSelect.value);
    const resultElement = document.getElementById('result');
    
    if (!selectedMovieId) {
        resultElement.textContent = 'Please select a movie first.';
        return;
    }
    
    // Step 2: Find liked movie
    const likedMovie = movies.find(movie => movie.id === selectedMovieId);
    if (!likedMovie) {
        resultElement.textContent = 'Selected movie not found.';
        return;
    }
    
    // Define all possible genres
    const allGenres = [
        "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ];
    
    // Step 3: Prepare for similarity calculation
    const likedMovieVector = getFeatureVector(likedMovie.genres, allGenres);
    const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);
    
    // Step 4: Calculate similarity scores using Cosine Similarity
    const scoredMovies = candidateMovies.map(candidateMovie => {
        const candidateVector = getFeatureVector(candidateMovie.genres, allGenres);
        const score = calculateCosineSimilarity(likedMovieVector, candidateVector);
        
        return {
            ...candidateMovie,
            score: score
        };
    });
    
    // Step 5: Sort by score (descending)
    scoredMovies.sort((a, b) => b.score - a.score);
    
    // Step 6: Select top recommendations
    const topRecommendations = scoredMovies.slice(0, 5); // Show 5 recommendations
    
    // Step 7: Display results
    if (topRecommendations.length > 0) {
        let html = `<p>Because you liked "<strong>${likedMovie.title}</strong>", we recommend:</p>`;
        html += '<ul class="recommendation-list">';
        
        topRecommendations.forEach(movie => {
            html += `<li>${movie.title} <em>(similarity: ${movie.score.toFixed(3)})</em></li>`;
        });
        
        html += '</ul>';
        html += '<p><small>Using Cosine Similarity instead of Jaccard Similarity</small></p>';
        resultElement.innerHTML = html;
    } else {
        resultElement.textContent = 'No recommendations found.';
    }
}
