// static/js/app.js
$(document).ready(function() {
       // Commenting out the AJAX part to test tab switching
    $('#location_form').submit(function(e) {
        e.preventDefault();  // Prevent the default form submission
        
        $.ajax({
            url: '/location_data',
            type: 'POST',
            data: $(this).serialize(),  // Serialize form data for the request
            success: function(response) {
                // Assuming the response contains a plot_url to display
                $('#plot-container').html('<img src="data:image/png;base64,' + response.plot_url + '" alt="Plot">');
            },
            error: function(error) {
                console.error("Error fetching data: ", error);
            }
        });
    });
    
});
