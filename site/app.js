(function () {
  const statusNode = document.getElementById("data-status");
  const rowsNode = document.getElementById("landscape-data-rows");

  function formatNumber(value, digits) {
    const number = Number(value);
    if (Math.abs(number) < 5e-10) {
      return "0.000000";
    }
    return number.toFixed(digits);
  }

  function formatScientific(value) {
    return Number(value).toExponential(2);
  }

  function labelFamily(name) {
    return name
      .split("_")
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  }

  function render(data) {
    const rows = data.rows || [];
    const maxGap = Math.max(...rows.map((row) => Math.abs(Number(row.witness_certificate_gap))));
    const families = Array.from(new Set(rows.map((row) => row.family))).sort();
    statusNode.textContent = `${rows.length} SDP rows across ${families.length} families; maximum dual-witness gap ${formatScientific(maxGap)}.`;
    rowsNode.innerHTML = rows
      .map(
        (row) => `
          <tr>
            <td>${labelFamily(row.family)}</td>
            <td>${Number(row.parameter).toFixed(2)}</td>
            <td>${formatNumber(row.generalized_robustness, 6)}</td>
            <td>${formatScientific(row.witness_certificate_gap)}</td>
            <td>${formatScientific(row.equality_residual_fro)}</td>
          </tr>
        `
      )
      .join("");
  }

  fetch("data/switch_robustness_landscape.json")
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then(render)
    .catch((error) => {
      statusNode.textContent = `Landscape data could not be loaded: ${error.message}.`;
    });
})();
