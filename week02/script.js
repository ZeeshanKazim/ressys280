// Track if data is loaded
let isDataLoaded = false;

/**
 * Initialize application when window loads
 */
window.onload = async function() {
    const resultElement = document.getElementById('result');
    const recommendBtn = document.getElementById('recommend-btn');
    
    // Disable button until data is loaded
    recommendBtn.disabled = true;
    resultElement.textContent = 'Loading data, please wait...';
    
    try {
        // Wait for data to load
        await loadData();
        
        // Enable button and update status
        recommendBtn.disabled = false;
        resultElement.textContent = 'Data loaded. Please select a movie.';
        isDataLoaded = true;
        
        // Populate the movie dropdown
        populateMoviesDropdown();
            
        // Add event listener to the recommendation button
        recommendBtn.addEventListener('click', getRecommendations);
    } catch (error) {
        console.error('Error initializing application:', error);
        resultElement.textContent = 'Error loading data. Please check the console for details.';
        resultElement.classList.add('error');
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
    if (!isDataLoaded) {
        document.getElementById('result').textContent = 'Data not loaded yet. Please wait.';
        return;
    }
    
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
    const topRecommendations = scoredMovies.slice(0, 5);
    
    // Step 7: Display results
    if (topRecommendations.length > 0) {
        let html = `<p>Because you liked "<strong>${likedMovie.title}</strong>", we recommend:</p>`;
        html += '<ul class="recommendation-list">';
        
        topRecommendations.forEach(movie => {
            html += `<li>${movie.title} <em>(similarity: ${movie.score.toFixed(2)})</em></li>`;
        });
        
        html += '</ul>';
        resultElement.innerHTML = html;
    } else {
        resultElement.textContent = 'No recommendations found.';
    }
}
