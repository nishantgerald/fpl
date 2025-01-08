$(document).ready(function () {
    let initializing = true; // Flag to distinguish initialization phase

    const table = $("#playersTable").DataTable({
        ajax: "/api/players",
        columns: [
            { data: "name" },
            { data: "team" },
            { data: "position" },
            { data: "price" },
            { data: "total_points" },
            { data: "form" },
            { data: "selected_by_percent" },
            { data: "status" }
        ],
        initComplete: function () {
            // Dynamically populate the team filter dropdown
            $.ajax({
                url: "/api/players",
                method: "GET",
                success: function (data) {
                    const teams = [...new Set(data.data.map(player => player.team))].sort();
                    const teamFilter = $("#teamFilter");
                    teams.forEach(team => {
                        teamFilter.append(new Option(team, team));
                    });

                    // Initialize filters from URL
                    initializeFiltersFromURL();
                },
                error: function () {
                    console.error("Failed to fetch team data");
                }
            });
        }
    });

    function applyFilters() {
        const team = $("#teamFilter").val();
        const position = $("#positionFilter").val();
        const minPrice = $("#minPrice").val();
        const maxPrice = $("#maxPrice").val();
        const minPoints = $("#minPoints").val();
        const maxPoints = $("#maxPoints").val();
        const minForm = $("#minForm").val();
        const maxForm = $("#maxForm").val();
        const minSelectedBy = $("#minSelectedBy").val();
        const maxSelectedBy = $("#maxSelectedBy").val();
        const status = $("#statusFilter").val();
    
        // Preserve existing URL parameters
        const params = new URLSearchParams(window.location.search);
    
        // Update parameters based on current filter values
        if (team) params.set("team", team); else params.delete("team");
        if (position) params.set("position", position); else params.delete("position");
        if (minPrice) params.set("price_min", minPrice); else params.delete("price_min");
        if (maxPrice) params.set("price_max", maxPrice); else params.delete("price_max");
        if (minPoints) params.set("points_min", minPoints); else params.delete("points_min");
        if (maxPoints) params.set("points_max", maxPoints); else params.delete("points_max");
        if (minForm) params.set("form_min", minForm); else params.delete("form_min");
        if (maxForm) params.set("form_max", maxForm); else params.delete("form_max");
        if (minSelectedBy) params.set("selected_min", minSelectedBy); else params.delete("selected_min");
        if (maxSelectedBy) params.set("selected_max", maxSelectedBy); else params.delete("selected_max");
        if (status) params.set("status", status); else params.delete("status");
    
        // Update the URL only if not initializing
        if (!initializing) {
            const newURL = `${window.location.pathname}?${params.toString()}`;
            window.history.replaceState({}, "", newURL);
        }
    
        // Use DataTables' built-in filter
        $.fn.dataTable.ext.search = [];
        $.fn.dataTable.ext.search.push(function (settings, data, dataIndex) {
            const dataTeam = data[1]; // Team column
            const dataPosition = data[2]; // Position column
            const dataPrice = parseFloat(data[3]); // Price column
            const dataPoints = parseFloat(data[4]); // Total Points column
            const dataForm = parseFloat(data[5]); // Form column
            const dataSelectedBy = parseFloat(data[6]); // Selected By (%) column
            const dataStatus = data[7]; // Status column
    
            return (
                (!team || dataTeam === team) &&
                (!position || dataPosition === position) &&
                (!minPrice || dataPrice >= parseFloat(minPrice)) &&
                (!maxPrice || dataPrice <= parseFloat(maxPrice)) &&
                (!minPoints || dataPoints >= parseFloat(minPoints)) &&
                (!maxPoints || dataPoints <= parseFloat(maxPoints)) &&
                (!minForm || dataForm >= parseFloat(minForm)) &&
                (!maxForm || dataForm <= parseFloat(maxForm)) &&
                (!minSelectedBy || dataSelectedBy >= parseFloat(minSelectedBy)) &&
                (!maxSelectedBy || dataSelectedBy <= parseFloat(maxSelectedBy)) &&
                (!status || dataStatus === status)
            );
        });
    
        // Redraw the table to apply the filter
        table.draw();
    }

    function initializeFiltersFromURL() {
        const params = new URLSearchParams(window.location.search);

        const team = params.get("team");
        const position = params.get("position");
        const minPrice = params.get("price_min");
        const maxPrice = params.get("price_max");
        const minPoints = params.get("points_min");
        const maxPoints = params.get("points_max");
        const minForm = params.get("form_min");
        const maxForm = params.get("form_max");
        const minSelectedBy = params.get("selected_min");
        const maxSelectedBy = params.get("selected_max");
        const status = params.get("status");

        if (team) $("#teamFilter").val(team);
        if (position) $("#positionFilter").val(position);
        if (minPrice) $("#minPrice").val(minPrice);
        if (maxPrice) $("#maxPrice").val(maxPrice);
        if (minPoints) $("#minPoints").val(minPoints);
        if (maxPoints) $("#maxPoints").val(maxPoints);
        if (minForm) $("#minForm").val(minForm);
        if (maxForm) $("#maxForm").val(maxForm);
        if (minSelectedBy) $("#minSelectedBy").val(minSelectedBy);
        if (maxSelectedBy) $("#maxSelectedBy").val(maxSelectedBy);
        if (status) $("#statusFilter").val(status);

        // Apply the filters and redraw the table
        applyFilters();

        // Mark initialization as complete
        initializing = false;
    }

    // Bind filter inputs
    $("#teamFilter, #positionFilter, #minPrice, #maxPrice, #minPoints, #maxPoints, #minForm, #maxForm, #minSelectedBy, #maxSelectedBy, #statusFilter").on("input change", function () {
        applyFilters();
    });

    // Initial application of filters
    applyFilters();
});