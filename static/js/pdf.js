/* ══════════════════════════════════════════════════════════
   pdf.js — PDF Report Download
══════════════════════════════════════════════════════════ */

async function downloadPDF() {
  if (!window.lastPrediction) {
    showToast('Run a prediction first!', 'error');
    return;
  }

  const user = getUser();
  const pred = window.lastPrediction;

  showLoading('Generating PDF report…');
  try {
    const res = await fetch('/api/report/pdf', {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({
        probability:     pred.probability,
        tier_label:      pred.tier_label,
        tier_key:        pred.tier_key,
        tier_color:      pred.tier_color,
        recommendations: pred.recommendations,
        inputs:          pred.inputs,
        derived:         pred.derived,
        patient_mode:    pred.patient_mode,
        patient_name:    user.name || 'Anonymous',
        gauge_b64:       pred.gauge_b64,
        shap_chart_b64:  pred.shap_chart_b64,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      showToast('PDF error: ' + (err.error || 'Unknown problem'), 'error');
      return;
    }

    // Trigger browser download
    const blob     = await res.blob();
    const url      = URL.createObjectURL(blob);
    const a        = document.createElement('a');
    a.href         = url;
    a.download     = `CardioClarity_Report_${new Date().toISOString().slice(0,10)}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast('✅ PDF report downloaded!', 'success');
  } catch (e) {
    showToast('Download failed: ' + e.message, 'error');
  } finally {
    hideLoading();
  }
}
