const API_BASE = "";

/* Upload */
export async function uploadFiles(formData) {
    const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData
    });

    if (!res.ok) throw new Error("Upload failed");
    return res.json();
}

/* Radiogenomics (optional for now) */
export async function runRadiogenomics(caseId) {
    const res = await fetch(`${API_BASE}/radiogenomics`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            upload_id: caseId,
            patient_id: caseId
        })
    });

    if (!res.ok) throw new Error("Radiogenomics failed");
    return res.json();
}