document.getElementById("predictionForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const formData = {
        soil_pH: document.getElementById("soil_pH").value,
        moisture: document.getElementById("moisture").value,
        nitrogen: document.getElementById("nitrogen").value,
        phosphorus: document.getElementById("phosphorus").value,
        potassium: document.getElementById("potassium").value,
        temperature: document.getElementById("temperature").value,
        rainfall: document.getElementById("rainfall").value
    };

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerHTML = `
            <p>Soil Condition: <strong>${data.soil_condition}</strong></p>
            <p>Recommended Crop: <strong>${data.crop_recommendation}</strong></p>
        `;
    })
    .catch(error => console.error("Error:", error));
});
