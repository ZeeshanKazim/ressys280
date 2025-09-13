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
            'Data loaded. Please select a movie to get recommendations.';
            
        // Add event listener to the recommendation button
        document.getElementById('recommend-btn').addEventListener('click', getRecommendations);
        
        // Add event listener to the hero CTA button
        document.querySelector('.cta-button').addEventListener('click', function() {
            document.querySelector('.recommender-section').scrollIntoView({ behavior: 'smooth' });
        });
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
 * Calculate cosine similarity between two vectors
 * @param {Array} vectorA - First vector
 * @param {Array} vectorB - Second vector
 * @returns {number} Cosine similarity value
 */
function calculateCosineSimilarity(vectorA, vectorB) {
    // Calculate dot product
    let dotProduct = 0;
    for (let i = 0; i < vectorA.length; i++) {
        dotProduct += vectorA[i] * vectorB[i];
    }
    
    // Calculate magnitudes
    let magnitudeA = 0;
    let magnitudeB = 0;
    for (let i = 0; i < vectorA.length; i++) {
        magnitudeA += vectorA[i] * vectorA[i];
        magnitudeB += vectorB[i] * vectorB[i];
    }
    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);
    
    // Avoid division by zero
    if (magnitudeA === 0 || magnitudeB === 0) {
        return 0;
    }
    
    // Return cosine similarity
    return dotProduct / (magnitudeA * magnitudeB);
}

/**
 * Get recommendations based on the selected movie
 */
function getRecommendations() {
    // Step 1: Get user input
    const movieSelect = document.getElementById('movie-select');
    const selectedMovieId = parseInt(movieSelect.value);
    const resultElement = document.getElementById('result');
    const movieCardsContainer = document.getElementById('movie-cards');
    
    if (!selectedMovieId) {
        resultElement.textContent = 'Please select a movie first.';
        movieCardsContainer.innerHTML = '';
        return;
    }
    
    // Step 2: Find liked movie
    const likedMovie = movies.find(movie => movie.id === selectedMovieId);
    if (!likedMovie) {
        resultElement.textContent = 'Selected movie not found.';
        movieCardsContainer.innerHTML = '';
        return;
    }
    
    // Step 3: Prepare for similarity calculation
    const candidateMovies = movies.filter(movie => movie.id !== likedMovie.id);
    
    // Step 4: Calculate similarity scores using cosine similarity
    const scoredMovies = candidateMovies.map(candidateMovie => {
        const score = calculateCosineSimilarity(likedMovie.genreVector, candidateMovie.genreVector);
        
        return {
            ...candidateMovie,
            score: score
        };
    });
    
    // Step 5: Sort by score (descending)
    scoredMovies.sort((a, b) => b.score - a.score);
    
    // Step 6: Select top recommendations
    const topRecommendations = scoredMovies.slice(0, 6);
    
    // Step 7: Display results
    if (topRecommendations.length > 0) {
        resultElement.innerHTML = `Because you liked <strong>"${likedMovie.title}"</strong>, we recommend:`;
        
        // Create movie cards
        movieCardsContainer.innerHTML = '';
        topRecommendations.forEach(movie => {
            const movieCard = document.createElement('div');
            movieCard.className = 'movie-card';
            
            movieCard.innerHTML = `
                <div class="movie-poster">
                    <i class="fas fa-film fa-3x"></i>
                </div>
                <div class="movie-info">
                    <div class="movie-title">${movie.title}</div>
                    <div class="movie-genres">${movie.genres.join(', ')}</div>
                    <div class="movie-similarity">Similarity: ${(movie.score * 100).toFixed(1)}%</div>
                </div>
            `;
            
            movieCardsContainer.appendChild(movieCard);
        });
    } else {
        resultElement.textContent = 'No recommendations found.';
        movieCardsContainer.innerHTML = '';
    }
}
