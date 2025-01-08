$(document).ready(function () {
    let initializing = true; // Flag to distinguish initialization phase

    const table = $("#playersTable").DataTable({
        ajax: "/api/players",
        columns: [
            { data: "name", title: "Name" },
            { data: "team", title: "Team" },
            { data: "position", title: "Position" },
            { data: "price", title: "Price (£)" },
            { data: "total_points", title: "Total Points" },
            { data: "form", title: "Form" },
            { data: "selected_by_percent", title: "Selected By (%)" },
            { data: "status", title: "Status" },
            { data: "next_3_fdr", title: "Next 3 FDR" }
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
        const minPrice = parseFloat($("#minPrice").val());
        const maxPrice = parseFloat($("#maxPrice").val());
        const minPoints = parseFloat($("#minPoints").val());
        const maxPoints = parseFloat($("#maxPoints").val());
        const minForm = parseFloat($("#minForm").val());
        const maxForm = parseFloat($("#maxForm").val());
        const minSelectedBy = parseFloat($("#minSelectedBy").val());
        const maxSelectedBy = parseFloat($("#maxSelectedBy").val());
        const status = $("#statusFilter").val();

        // New: Get min/max FDR values
        const minFDR = parseFloat($("#minFDR").val());
        const maxFDR = parseFloat($("#maxFDR").val());

        // Preserve existing URL parameters
        const params = new URLSearchParams(window.location.search);

        // Update parameters based on current filter values
        if (team) params.set("team", team); else params.delete("team");
        if (position) params.set("position", position); else params.delete("position");
        if (!isNaN(minPrice)) params.set("price_min", minPrice); else params.delete("price_min");
        if (!isNaN(maxPrice)) params.set("price_max", maxPrice); else params.delete("price_max");
        if (!isNaN(minPoints)) params.set("points_min", minPoints); else params.delete("points_min");
        if (!isNaN(maxPoints)) params.set("points_max", maxPoints); else params.delete("points_max");
        if (!isNaN(minForm)) params.set("form_min", minForm); else params.delete("form_min");
        if (!isNaN(maxForm)) params.set("form_max", maxForm); else params.delete("form_max");
        if (!isNaN(minSelectedBy)) params.set("selected_min", minSelectedBy); else params.delete("selected_min");
        if (!isNaN(maxSelectedBy)) params.set("selected_max", maxSelectedBy); else params.delete("selected_max");
        if (!isNaN(minFDR)) params.set("fdr_min", minFDR); else params.delete("fdr_min"); // New
        if (!isNaN(maxFDR)) params.set("fdr_max", maxFDR); else params.delete("fdr_max"); // New
        if (status) params.set("status", status); else params.delete("status");

        // Update the URL only if not initializing
        if (!initializing) {
            const newURL = `${window.location.pathname}?${params.toString()}`;
            window.history.replaceState({}, "", newURL);
        }

        // Use DataTables' built-in filter
        $.fn.dataTable.ext.search = []; // Clear existing filters
        $.fn.dataTable.ext.search.push(function (settings, data) {
            const dataTeam = data[1]; // Team column
            const dataPosition = data[2]; // Position column
            const dataPrice = parseFloat(data[3]); // Price column
            const dataPoints = parseFloat(data[4]); // Total Points column
            const dataForm = parseFloat(data[5]); // Form column
            const dataSelectedBy = parseFloat(data[6]); // Selected By (%) column
            const dataStatus = data[7]; // Status column
            const dataFDR = parseFloat(data[8]); // Next 3 FDR column (New)

            return (
                (!team || dataTeam === team) &&
                (!position || dataPosition === position) &&
                (isNaN(minPrice) || dataPrice >= minPrice) &&
                (isNaN(maxPrice) || dataPrice <= maxPrice) &&
                (isNaN(minPoints) || dataPoints >= minPoints) &&
                (isNaN(maxPoints) || dataPoints <= maxPoints) &&
                (isNaN(minForm) || dataForm >= minForm) &&
                (isNaN(maxForm) || dataForm <= maxForm) &&
                (isNaN(minSelectedBy) || dataSelectedBy >= minSelectedBy) &&
                (isNaN(maxSelectedBy) || dataSelectedBy <= maxSelectedBy) &&
                (isNaN(minFDR) || dataFDR >= minFDR) && // New
                (isNaN(maxFDR) || dataFDR <= maxFDR) && // New
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
        const minFDR = params.get("fdr_min"); // New
        const maxFDR = params.get("fdr_max"); // New
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
        if (minFDR) $("#minFDR").val(minFDR); // New
        if (maxFDR) $("#maxFDR").val(maxFDR); // New
        if (status) $("#statusFilter").val(status);

        // Apply the filters and redraw the table
        applyFilters();

        // Mark initialization as complete
        initializing = false;
    }

    // Bind filter inputs
    $("#teamFilter, #positionFilter, #minPrice, #maxPrice, #minPoints, #maxPoints, #minForm, #maxForm, #minSelectedBy, #maxSelectedBy, #statusFilter, #minFDR, #maxFDR").on("input change", function () {
        applyFilters();
    });

    // Initial application of filters
    applyFilters();
});