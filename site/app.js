(function () {
  const statusNode = document.getElementById("data-status");
  const rowsNode = document.getElementById("landscape-data-rows");
  const fullStatusNode = document.getElementById("full-data-status");
  const fullRowsNode = document.getElementById("full-data-rows");
  const fullGridNode = document.getElementById("full-grid-count");
  const fullApplicabilityNode = document.getElementById("full-applicability-count");
  const fullMaxFprNode = document.getElementById("full-max-fpr");
  const certStatusNode = document.getElementById("cert-data-status");
  const certRowsNode = document.getElementById("cert-data-rows");
  const certRgNode = document.getElementById("cert-rg");
  const certFamilyCountNode = document.getElementById("cert-family-count");
  const certAffineCountNode = document.getElementById("cert-affine-count");
  const certBoundCountNode = document.getElementById("cert-bound-count");
  const finiteStatusNode = document.getElementById("finite-data-status");
  const finiteScalarsNode = document.getElementById("finite-scalars");
  const finiteScalingNode = document.getElementById("finite-scaling");
  const finiteKrelFpNode = document.getElementById("finite-krel-fp");
  const boundsStatusNode = document.getElementById("bounds-data-status");
  const boundsRowsNode = document.getElementById("bounds-data-rows");
  const boundsWidthNode = document.getElementById("bounds-width");
  const externalStatusNode = document.getElementById("external-data-status");
  const externalCountsNode = document.getElementById("external-counts");
  const externalSNode = document.getElementById("external-s");
  const externalHashNode = document.getElementById("external-hash");

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

  function renderRegion(region) {
    if (!Array.isArray(region) || region.length !== 2) {
      return "not reported";
    }
    return `[${Number(region[0]).toFixed(3)}, ${Number(region[1]).toFixed(3)}]`;
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

  function renderCertifiedWitness(data) {
    if (!certStatusNode || !certRowsNode) {
      return;
    }
    const families = data.families || {};
    const entries = Object.entries(families);
    if (entries.length === 0) {
      return;
    }

    const affineCount = entries.filter(([, family]) => family.is_affine).length;
    const boundCount = entries.filter(([, family]) => family.lower_bound_holds_on_grid).length;
    const reference = Number(data.reference_witness && data.reference_witness.R_g);

    if (certRgNode) {
      certRgNode.textContent = formatNumber(reference, 6);
    }
    if (certFamilyCountNode) {
      certFamilyCountNode.textContent = String(entries.length);
    }
    if (certAffineCountNode) {
      certAffineCountNode.textContent = `${affineCount}/${entries.length}`;
    }
    if (certBoundCountNode) {
      certBoundCountNode.textContent = `${boundCount}/${entries.length}`;
    }

    certStatusNode.textContent =
      `${entries.length} certified families loaded; ${affineCount} affine ` +
      `single-scalar families and ${boundCount} lower-bound verification grids passed.`;

    certRowsNode.innerHTML = entries
      .map(([name, family]) => `
        <tr>
          <td>${labelFamily(name)}</td>
          <td>${family.is_affine ? "affine / one-sided" : "nonlinear / two-sided"}</td>
          <td>${renderRegion(family.certified_nonseparable_region)}</td>
          <td>${formatNumber(family.reference_value, 6)}</td>
          <td>${family.lower_bound_holds_on_grid ? "passed" : "review"}</td>
        </tr>
      `)
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

  function renderFiniteCount(data) {
    if (!finiteStatusNode) {
      return;
    }
    const scaling = data.one_over_N_scaling || {};
    const fp = data.false_positive_under_calibrated_drift || {};
    const maxKrelFp = Math.max(...(fp.fp_rate_krel || [0]).map(Number));
    const rawFp = Math.max(...(fp.fp_rate_raw_witness || [0]).map(Number));
    const ratios = scaling.empirical_over_analytic_ratio || [];

    if (finiteScalarsNode) {
      finiteScalarsNode.textContent = String(data.scalars_measured || 1);
    }
    if (finiteScalingNode) {
      finiteScalingNode.textContent = scaling.scaling_confirmed ? "pass" : "review";
    }
    if (finiteKrelFpNode) {
      finiteKrelFpNode.textContent = formatPercent(maxKrelFp);
    }

    finiteStatusNode.textContent =
      `${data.scalars_measured || 1} scalar measured against ` +
      `${Number(data.process_free_parameters || 4096).toFixed(0)} process entries; ` +
      `1/N empirical/analytic ratios ${ratios.map((x) => Number(x).toFixed(2)).join(", ")}. ` +
      `K_rel max false positive ${formatPercent(maxKrelFp)} versus raw witness ${formatPercent(rawFp)}.`;
  }

  function renderCertifiedBounds(data) {
    if (!boundsStatusNode || !boundsRowsNode) {
      return;
    }
    const interval = data.R_g_interval || {};
    if (boundsWidthNode) {
      boundsWidthNode.textContent = formatScientific(interval.width || 0);
    }
    boundsStatusNode.textContent =
      `Certified interval [${formatNumber(interval.lower || 0, 12)}, ` +
      `${formatNumber(interval.upper || 0, 12)}], width ${formatScientific(interval.width || 0)}.`;

    boundsRowsNode.innerHTML = (data.solver_table || [])
      .map((row) => `
        <tr>
          <td>${row.solver}</td>
          <td>${row.available ? "yes" : "no"}</td>
          <td>${row.status}</td>
          <td>${row.R_g_lower === null ? "-" : formatNumber(row.R_g_lower, 12)}</td>
          <td>${row.R_g_upper === null ? "-" : formatNumber(row.R_g_upper, 12)}</td>
          <td>${row.interval_width === null ? "-" : formatScientific(row.interval_width)}</td>
        </tr>
      `)
      .join("");
  }

  function renderExternalData(data) {
    if (!externalStatusNode) {
      return;
    }
    if (externalCountsNode) {
      externalCountsNode.textContent = `${(Number(data.total_counts) / 1e6).toFixed(1)}M`;
    }
    if (externalSNode) {
      externalSNode.textContent = Number(data.s_experiment_recomputed).toFixed(6);
    }
    if (externalHashNode) {
      externalHashNode.textContent = String(data.raw_sha256 || "").slice(0, 8);
    }
    externalStatusNode.textContent =
      `${Number(data.total_counts).toLocaleString()} public experimental counts verified; ` +
      `S_experiment=${Number(data.s_experiment_recomputed).toFixed(12)}, ` +
      `multinomial-only z=${Number(data.statistical_z_score).toFixed(1)}. ` +
      "This remains a semi-DI public-data pilot, not a DeltaW/K_rel tomography claim.";
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

  fetch("data/certified_witness/certified_witness_landscape.json")
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then(renderCertifiedWitness)
    .catch((error) => {
      if (certStatusNode) {
        certStatusNode.textContent = `Certified witness data could not be loaded: ${error.message}.`;
      }
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

  fetch("data/finite_count/finite_count_report.json")
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then(renderFiniteCount)
    .catch((error) => {
      if (finiteStatusNode) {
        finiteStatusNode.textContent = `Finite-count data could not be loaded: ${error.message}.`;
      }
    });

  fetch("data/certified_witness/certified_bounds_report.json")
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then(renderCertifiedBounds)
    .catch((error) => {
      if (boundsStatusNode) {
        boundsStatusNode.textContent = `Certified bounds data could not be loaded: ${error.message}.`;
      }
    });

  fetch("data/external/cao2023_sdi_report.json")
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then(renderExternalData)
    .catch((error) => {
      if (externalStatusNode) {
        externalStatusNode.textContent = `External public-data report could not be loaded: ${error.message}.`;
      }
    });
})();
