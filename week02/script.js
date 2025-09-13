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
 * Calculate Jaccard similarity between two sets
 * @param {Set} setA - First set
 * @param {Set} setB - Second set
 * @returns {number} Jaccard similarity coefficient
 */
function calculateJaccardSimilarity(setA, setB) {
    const intersection = new Set([...setA].filter(x => setB.has(x)));
    const union = new Set([...setA, ...setB]);
    
    return union.size === 0 ? 0 : intersection.size / union.size;
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
    
    // Step 3: Prepare for similarity calculation
    const likedMovieGenres = new Set(likedMovie.genres);
    const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);
    
    // Step 4: Calculate similarity scores
    const scoredMovies = candidateMovies.map(candidateMovie => {
        const candidateGenres = new Set(candidateMovie.genres);
        const score = calculateJaccardSimilarity(likedMovieGenres, candidateGenres);
        
        return {
            ...candidateMovie,
            score: score
        };
    });
    
    // Step 5: Sort by score (descending)
    scoredMovies.sort((a, b) => b.score - a.score);
    
    // Step 6: Select top recommendations
    const topRecommendations = scoredMovies.slice(0, 2);
    
    // Step 7: Display results
    if (topRecommendations.length > 0) {
        const recommendationTitles = topRecommendations.map(movie => movie.title);
        resultElement.innerHTML = `Because you liked '<strong>${likedMovie.title}</strong>', we recommend: <strong>${recommendationTitles.join('</strong>, <strong>')}</strong>`;
    } else {
        resultElement.textContent = 'No recommendations found.';
    }
}
