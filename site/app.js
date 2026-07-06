(function () {
  const statusNode = document.getElementById("data-status");
  const rowsNode = document.getElementById("landscape-data-rows");
  const fullStatusNode = document.getElementById("full-data-status");
  const fullRowsNode = document.getElementById("full-data-rows");
  const fullGridNode = document.getElementById("full-grid-count");
  const fullApplicabilityNode = document.getElementById("full-applicability-count");
  const fullMaxFprNode = document.getElementById("full-max-fpr");

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

  function formatPercent(value) {
    return `${(100 * Number(value)).toFixed(1)}%`;
  }

  function labelFamily(name) {
    return name
      .split("_")
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");
  }

  function renderLandscape(data) {
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

  function renderFullTomography(data) {
    const rows = (data.rows || []).slice().sort((left, right) => (
      Number(left.lambda_true) - Number(right.lambda_true)
      || Number(left.n_total) - Number(right.n_total)
      || Number(left.visibility) - Number(right.visibility)
    ));
    if (rows.length === 0 || !fullStatusNode || !fullRowsNode) {
      return;
    }

    const maxPower = Math.max(...rows.map((row) => Number(row.power)));
    const maxFalsePositive = Math.max(
      ...rows.map((row) => Number(row.false_positive_miscalibrated))
    );
    const applicabilityPassed = rows.filter((row) => row.applicability_passed).length;
    const best = rows.reduce((winner, row) => (
      Number(row.power) > Number(winner.power) ? row : winner
    ), rows[0]);

    if (fullGridNode) {
      fullGridNode.textContent = String(rows.length);
    }
    if (fullApplicabilityNode) {
      fullApplicabilityNode.textContent = `${applicabilityPassed}/${rows.length}`;
    }
    if (fullMaxFprNode) {
      fullMaxFprNode.textContent = formatPercent(maxFalsePositive);
    }

    fullStatusNode.textContent =
      `${rows.length} finite-count rows loaded; best empirical LR power ` +
      `${formatPercent(maxPower)} at lambda=${Number(best.lambda_true).toFixed(2)}, ` +
      `N=${Number(best.n_total).toFixed(0)}, visibility=${Number(best.visibility).toFixed(2)}. ` +
      `Maximum miscalibrated false-positive rate: ${formatPercent(maxFalsePositive)}.`;

    fullRowsNode.innerHTML = rows
      .map(
        (row) => `
          <tr>
            <td>${Number(row.lambda_true).toFixed(2)}</td>
            <td>${Number(row.n_total).toFixed(0)}</td>
            <td>${Number(row.visibility).toFixed(2)}</td>
            <td>${formatPercent(row.power)}</td>
            <td>${formatPercent(row.false_positive_miscalibrated)}</td>
            <td>${Number(row.lambda_hat_mean).toFixed(4)}</td>
            <td>${row.applicability_passed ? "yes" : "review"}</td>
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
    .then(renderLandscape)
    .catch((error) => {
      statusNode.textContent = `Landscape data could not be loaded: ${error.message}.`;
    });

  fetch("data/full_tomography/full_tomography_report.json")
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then(renderFullTomography)
    .catch((error) => {
      if (fullStatusNode) {
        fullStatusNode.textContent = `Full tomography data could not be loaded: ${error.message}.`;
      }
    });
})();
