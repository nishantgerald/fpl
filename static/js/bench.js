async function showPlayerStats(playerId) {
    try {
        // Fetch player data from the API using the player ID
        const response = await fetch(`/api/players?player_id=${playerId}`);
        if (!response.ok) {
            alert("Failed to fetch player stats.");
            return;
        }

        const responseData = await response.json();

        // Ensure the response contains a data array with at least one player
        if (!responseData || !Array.isArray(responseData.data) || responseData.data.length === 0) {
            console.error("Player data is incomplete or undefined", responseData);
            alert("Error: Unable to show player stats. Data is incomplete.");
            return;
        }

        // Extract the player data (the first and only entry in the data array)
        const player = responseData.data[0];

        // Safely access player data properties
        // Safely access player data properties
        const positionClassMap = {
            GKP: "gkp", // Goalkeeper
            DEF: "def", // Defender
            MID: "mid", // Midfielder
            FWD: "fwd"  // Forward
        };

        // Determine the CSS class for the position
        const positionClass = positionClassMap[player.position] || "unknown"; // Fallback to "unknown" if position is not found

        const modalContent = `
    <div class="player-header">
        <img src="${player.photo || "#"}" alt="${player.name || "Unknown Player"}" class="player-image">
        <div class="player-details">
            <span class="player-position ${positionClass}">${player.position || "N/A"}</span>
            <h2 class="player-name">${player.name || "Unknown"}</h2>
            <span class="player-team">${player.team || "N/A"}</span>
        </div>
    </div>
    <div class="player-stats">
        <div class="stat">
            <span class="stat-label">Price</span>
            <span class="stat-value">£${player.price || "N/A"}m</span>
        </div>
        <div class="stat">
            <span class="stat-label">Form</span>
            <span class="stat-value">${player.form || "N/A"}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Total Points</span>
            <span class="stat-value">${player.total_points || "N/A"}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Next 3 FDR</span>
            <span class="stat-value">${player.next_3_fdr || "N/A"}</span>
        </div>
        <div class="stat">
            <span class="stat-label">Selected By</span>
            <span class="stat-value">${player.selected_by_percent || "N/A"}%</span>
        </div>
        <div class="stat">
            <span class="stat-label">Status</span>
            <span class="stat-value">${player.status || "N/A"}</span>
        </div>
        <div class="stat">
            <span class="stat-label">FCPS</span>
            <span class="stat-value">${player.fcps || "N/A"}</span>
        </div>
        <div class="stat">
            <span class="stat-label">ICT Index</span>
            <span class="stat-value">${player.ict_index || "N/A"}</span>
        </div>
    </div>
`;

        // Update the modal content
        document.getElementById("modal-body").innerHTML = modalContent;

        // Display the modal
        document.getElementById("modal").style.display = "block";
    } catch (error) {
        console.error("An error occurred while fetching player stats:", error);
        alert("An unexpected error occurred. Please try again later.");
    }
}

function closeModal() {
    document.getElementById("modal").style.display = "none";
}