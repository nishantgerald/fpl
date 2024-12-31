async function showPlayerStats(playerId) {
    const response = await fetch(`/player-stats/${playerId}`);
    const data = await response.json();
    if (response.ok) {
        const modalContent = `
            <div class="player-header">
                <img src="${data.photo}" alt="${data.name}" class="player-image">
                <div class="player-details">
                    <span class="player-position ${data.position.toLowerCase()}">${data.position}</span>
                    <h2 class="player-name">${data.name}</h2>
                    <span class="player-team">${data.team}</span>
                </div>
            </div>
            <div class="player-stats">
                <div class="stat">
                    <span class="stat-label">Price</span>
                    <span class="stat-value">£${data.price}m</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Form</span>
                    <span class="stat-value">${data.form}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Points</span>
                    <span class="stat-value">${data.total_points}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Bonus</span>
                    <span class="stat-value">${data.bonus || 0}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Selected by</span>
                    <span class="stat-value">${data.selected_by_percent}%</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Status</span>
                    <span class="stat-value">${data.status}</span>
                </div>
            </div>
        `;
        document.getElementById("modal-body").innerHTML = modalContent;
        document.getElementById("modal").style.display = "block";
    } else {
        alert("Failed to fetch player stats.");
    }
}

function closeModal() {
    document.getElementById("modal").style.display = "none";
}