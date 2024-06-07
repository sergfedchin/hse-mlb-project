function predictRevenue() {
    var title = document.getElementById('movie-title').value;
    var releaseYear = document.getElementById('movie-release-year').value;
    var runtime = document.getElementById('movie-runtime').value;
    var budget = document.getElementById('movie-budget').value;
    var genres = document.getElementById('movie-genres').value;
    var productionCountries = document.getElementById('movie-production-countries').value;
    var originalLanguage = document.getElementById('movie-original-language').value;
    var popularity = document.getElementById('movie-popularity').value;
    var overview = document.getElementById('movie-overview').value;

    $.ajax({
        type: 'POST',
        url: '/predict',
        contentType: 'application/json',
        data: JSON.stringify({ 'title': title, 
                               'release_year': releaseYear, 
                               'runtime': runtime, 
                               'budget': budget, 
                               'genres': genres, 
                               'production_countries': productionCountries, 
                               'original_language': originalLanguage, 
                               'popularity': popularity, 
                               'overview': overview}),
        success: function (data) {
            var modelPrediction = document.getElementById('model-prediction');
            modelPrediction.innerHTML = "Predicted revenue: " + data.revenue;
            modelPrediction.style.display = 'block';
        },
        error: function(xhr, status, error) {
            var errorMessage = xhr.responseJSON.error;
            alert(errorMessage);
            // window.location.reload();
        }
    });
}