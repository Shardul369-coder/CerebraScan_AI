window.addEventListener("DOMContentLoaded", () => {
// Simple fetch version (no ES module import)
async function uploadFiles(formData) {
    const res = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    if (!res.ok) throw new Error("Upload failed");
    return res.json();
}

let currentCaseId = null;
let compositionChart = null;
let volumeChart = null;

let netMesh = null;
let edemaMesh = null;
let etMesh = null;
let brainReference = null;

let scene = null;
let camera = null;
let renderer = null;
let controls = null;
let raycaster = null;
let mouse = null;
let hoverInfo = null;

// Loaders
let plyLoader = null;
let gltfLoader = null;

/* ==============================
   Clear Dashboard (Empty Initial State)
============================== */

function clearDashboard() {
    console.log('Clearing dashboard to empty state');
    
    // Clear KPI cards
    const wholeVolumeEl = document.getElementById("wholeVolume");
    if (wholeVolumeEl) {
        wholeVolumeEl.innerHTML = '— <span style="font-size:0.9rem; color:#94a3b8;">cm³</span>';
    }
    
    const coreVolumeEl = document.getElementById("coreVolume");
    if (coreVolumeEl) {
        coreVolumeEl.innerHTML = '— <span style="font-size:0.9rem; color:#94a3b8;">cm³</span>';
    }
    
    const edemaVolumeEl = document.getElementById("edemaVolume");
    if (edemaVolumeEl) {
        edemaVolumeEl.innerHTML = '— <span style="font-size:0.9rem; color:#94a3b8;">cm³</span>';
    }
    
    const etVolumeEl = document.getElementById("etVolume");
    if (etVolumeEl) {
        etVolumeEl.innerHTML = '— <span style="font-size:0.9rem; color:#94a3b8;">cm³</span>';
    }
    
    // Clear percentage displays
    const netPercentEl = document.getElementById("netPercent");
    const edemaPercentEl = document.getElementById("edemaPercent");
    const etPercentEl = document.getElementById("etPercent");
    const totalVolumeEl = document.getElementById("totalVolume");
    const coreVolumeSmallEl = document.getElementById("coreVolumeSmall");

    if (netPercentEl) netPercentEl.textContent = '—';
    if (edemaPercentEl) edemaPercentEl.textContent = '—';
    if (etPercentEl) etPercentEl.textContent = '—';
    if (totalVolumeEl) totalVolumeEl.textContent = '—';
    if (coreVolumeSmallEl) coreVolumeSmallEl.textContent = '—';
    
    // Destroy existing charts if they exist
    if (window.compositionChart) {
        window.compositionChart.destroy();
        window.compositionChart = null;
    }
    if (window.volumeChart) {
        window.volumeChart.destroy();
        window.volumeChart = null;
    }
    
    // Clear any stored volumes
    window.volumes = null;
    
    console.log('Dashboard cleared - ready for new data');
}

/* ==============================
   Navigation
============================== */

const tabs = document.querySelectorAll(".sidebar li");
const panels = document.querySelectorAll(".panel");

tabs.forEach(tab => {
    tab.addEventListener("click", () => {

        tabs.forEach(t => t.classList.remove("active"));
        tab.classList.add("active");

        panels.forEach(p => p.classList.remove("active"));

        const target = document.getElementById(`panel-${tab.dataset.tab}`);
        if (target) target.classList.add("active");
    });
});

/* ==============================
   Upload
============================== */

const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");

uploadBtn?.addEventListener("click", async () => {

    if (!fileInput?.files.length) {
        uploadStatus.innerText = "No file selected.";
        return;
    }

    const formData = new FormData();
    
    // Handle both single file (zip) and multiple files
    if (fileInput.files.length === 1 && fileInput.files[0].name.endsWith('.zip')) {
        // If it's a single zip file
        formData.append("file", fileInput.files[0]);
    } else {
        // If it's multiple individual files
        for (let i = 0; i < fileInput.files.length; i++) {
            formData.append("files", fileInput.files[i]);
        }
    }

    uploadStatus.innerText = "Uploading...";

    try {
        const result = await uploadFiles(formData);
        currentCaseId = result.case_id;
        window.currentCaseId = result.case_id;
        uploadStatus.innerText = `Upload successful: ${currentCaseId}`;
    } catch (error) {
        console.error("Upload error:", error);
        uploadStatus.innerText = "Upload failed";
    }
});

/* ==============================
   Segmentation + Logs
============================== */

const segBtn = document.getElementById("segmentBtn");
const terminal = document.getElementById("terminal");

function streamLogs(caseId) {

    terminal.innerText = "";

    const eventSource = new EventSource(`/logs/${caseId}`);

    eventSource.onmessage = async (event) => {

        terminal.innerText += event.data + "\n";
        terminal.scrollTop = terminal.scrollHeight;

        if (event.data.includes("Completed successfully.")) {

            eventSource.close();

            const res = await fetch(`/results/${caseId}/status`);
            const data = await res.json();

            if (data.volumetrics) {
                updateDashboardWithVolumetrics(data.volumetrics);
                
                // Store for 3D viewer
                window.volumes = data.volumetrics;
                
                // Switch to dashboard view
                document.querySelector('[data-tab="dashboard"]')?.click();
                
                // Show success message
                if (uploadStatus) {
                    uploadStatus.innerHTML = '<span style="color:#10b981;">✓ Pipeline completed successfully!</span>';
                }
            }
        }
    };

    eventSource.onerror = () => {
        eventSource.close();
    };
}

segBtn?.addEventListener("click", async () => {

    if (!currentCaseId) {
        terminal.innerText = "Upload MRI first.";
        return;
    }

    terminal.innerText = "Starting pipeline...\n";

    const response = await fetch("/analyze/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            upload_id: currentCaseId,
            patient_id: currentCaseId
        })
    });

    if (response.ok) {
        streamLogs(currentCaseId);
    } else {
        terminal.innerText += "Failed to start pipeline.";
    }
});

/* ==============================
   Download
============================== */

const downloadBtn = document.getElementById("downloadBtn");

downloadBtn?.addEventListener("click", () => {
    if (currentCaseId) {
        window.location.href = `/download/${currentCaseId}`;
    }
});

/* ==============================
   Load Dashboard Data from Your JSON Structure
============================== */

async function loadDashboardData(patientId) {
    try {
        // First, try to get the latest results from your backend
        const uploadId = currentCaseId || patientId;
        
        // Try to fetch from the results endpoint
        const response = await fetch(`/results/${uploadId}/status`);
        
        if (response.ok) {
            const data = await response.json();
            
            // Your data structure from pipeline.py includes volumetrics
            if (data.volumetrics) {
                updateDashboardWithVolumetrics(data.volumetrics);
                return data;
            }
        }
        
        // If no data from backend, return null (don't load sample)
        console.log('No data available - dashboard remains empty');
        return null;
        
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        return null;
    }
}

/* ==============================
   Dashboard Charts - Enhanced Version
============================== */

function updateDashboardWithVolumetrics(volumes) {
    if (!volumes) return;

    // Store volumes globally for 3D viewer hover
    window.volumes = volumes;

    // Update all KPI cards with your actual data
    document.getElementById("wholeVolume").innerHTML = 
        `<strong>${volumes.Whole_Tumor_cm3?.toFixed(2) || 0}</strong> <span style="font-size:0.9rem; color:#94a3b8;">cm³</span>`;
    
    document.getElementById("coreVolume").innerHTML = 
        `<strong>${volumes.Tumor_Core_cm3?.toFixed(2) || 0}</strong> <span style="font-size:0.9rem; color:#94a3b8;">cm³</span>`;
    
    // Check if edemaVolume element exists
    const edemaElement = document.getElementById("edemaVolume");
    if (edemaElement) {
        edemaElement.innerHTML = 
            `<strong>${volumes.Edema_cm3?.toFixed(2) || 0}</strong> <span style="font-size:0.9rem; color:#94a3b8;">cm³</span>`;
    }
    
    // Check if etVolume element exists
    const etElement = document.getElementById("etVolume");
    if (etElement) {
        etElement.innerHTML = 
            `<strong>${volumes.ET_cm3?.toFixed(2) || 0}</strong> <span style="font-size:0.9rem; color:#94a3b8;">cm³</span>`;
    }

    // Destroy existing charts if they exist
    if (compositionChart) compositionChart.destroy();
    if (volumeChart) volumeChart.destroy();

    // Create Composition Chart (Donut) - Using percentages
    const ctx1 = document.getElementById("compositionChart")?.getContext("2d");
    if (ctx1) {
        compositionChart = new Chart(ctx1, {
            type: "doughnut",
            data: {
                labels: ["NET (Necrotic)", "Edema", "ET (Enhancing)"],
                datasets: [{
                    data: [
                        volumes.NET_percent || 0,
                        volumes.Edema_percent || 0,
                        volumes.ET_percent || 0
                    ],
                    backgroundColor: ["#3b82f6", "#10b981", "#f59e0b"],
                    borderColor: "#1f2937",
                    borderWidth: 2,
                    hoverOffset: 15
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { 
                            color: '#e2e8f0',
                            font: { size: 11 }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.raw.toFixed(2)}%`;
                            }
                        }
                    }
                },
                cutout: '65%'
            }
        });
    }

    // Create Volume Chart (Bar) - Using cm³ values
    const ctx2 = document.getElementById("volumeChart")?.getContext("2d");
    if (ctx2) {
        volumeChart = new Chart(ctx2, {
            type: "bar",
            data: {
                labels: ["NET", "Edema", "ET"],
                datasets: [{
                    label: "Volume (cm³)",
                    data: [
                        volumes.NET_cm3 || 0,
                        volumes.Edema_cm3 || 0,
                        volumes.ET_cm3 || 0
                    ],
                    backgroundColor: ["#3b82f6", "#10b981", "#f59e0b"],
                    borderRadius: 6,
                    barPercentage: 0.7
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.raw.toFixed(2)} cm³`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: '#334155' },
                        ticks: { 
                            color: '#94a3b8',
                            callback: function(value) {
                                return value + ' cm³';
                            }
                        }
                    },
                    x: {
                        ticks: { color: '#e2e8f0' }
                    }
                }
            }
        });
    }

    // Update percentage display elements if they exist
    const netPercent = document.getElementById("netPercent");
    const edemaPercent = document.getElementById("edemaPercent");
    const etPercent = document.getElementById("etPercent");
    const totalVolume = document.getElementById("totalVolume");
    const coreVolumeSmall = document.getElementById("coreVolumeSmall");

    if (netPercent) netPercent.textContent = volumes.NET_percent?.toFixed(2) + '%' || '0%';
    if (edemaPercent) edemaPercent.textContent = volumes.Edema_percent?.toFixed(2) + '%' || '0%';
    if (etPercent) etPercent.textContent = volumes.ET_percent?.toFixed(2) + '%' || '0%';
    if (totalVolume) totalVolume.textContent = volumes.Whole_Tumor_cm3?.toFixed(2) + ' cm³' || '0 cm³';
    if (coreVolumeSmall) coreVolumeSmall.textContent = volumes.Tumor_Core_cm3?.toFixed(2) + ' cm³' || '0 cm³';

    // Update 3D viewer volumes
    update3DViewerVolumes(volumes);
}

function update3DViewerVolumes(volumes) {
    // Update the hover info with actual volumes
    const hoverInfoEl = document.querySelector('#hover-coordinates');
    if (hoverInfoEl) {
        hoverInfoEl.setAttribute('data-net-volume', volumes.NET_cm3);
        hoverInfoEl.setAttribute('data-edema-volume', volumes.Edema_cm3);
        hoverInfoEl.setAttribute('data-et-volume', volumes.ET_cm3);
    }
}

/* ==============================
   Create Brain Reference (Semi-transparent)
============================== */

function createBrainReference() {
    // Create a semi-transparent brain-shaped reference
    const brainGroup = new THREE.Group();
    
    // Material for brain - semi-transparent
    const brainMat = new THREE.MeshPhongMaterial({
        color: 0x88aaff,
        transparent: true,
        opacity: 0.15,
        emissive: 0x112233,
        wireframe: false,
        side: THREE.DoubleSide,
        depthWrite: false  // Prevents z-fighting with tumors
    });
    
    // Material for brain stem - slightly different
    const stemMat = new THREE.MeshPhongMaterial({
        color: 0x7799cc,
        transparent: true,
        opacity: 0.12,
        emissive: 0x112233,
        side: THREE.DoubleSide,
        depthWrite: false
    });
    
    // Create two hemispheres (left and right)
    const leftHemisphere = new THREE.Mesh(
        new THREE.SphereGeometry(80, 64, 32),
        brainMat
    );
    leftHemisphere.position.set(-30, 10, 0);
    leftHemisphere.scale.set(1.0, 0.9, 0.8);
    brainGroup.add(leftHemisphere);
    
    const rightHemisphere = new THREE.Mesh(
        new THREE.SphereGeometry(80, 64, 32),
        brainMat
    );
    rightHemisphere.position.set(30, 10, 0);
    rightHemisphere.scale.set(1.0, 0.9, 0.8);
    brainGroup.add(rightHemisphere);
    
    // Add cerebellum (back)
    const cerebellum = new THREE.Mesh(
        new THREE.SphereGeometry(50, 48, 24),
        brainMat
    );
    cerebellum.position.set(0, -20, -50);
    cerebellum.scale.set(1.2, 0.6, 0.8);
    brainGroup.add(cerebellum);
    
    // Add brain stem
    const stem = new THREE.Mesh(
        new THREE.CylinderGeometry(20, 25, 70, 16),
        stemMat
    );
    stem.position.set(0, -40, -20);
    stem.rotation.x = 0.2;
    stem.rotation.z = 0.1;
    brainGroup.add(stem);
    
    // Add corpus callosum (connecting part)
    const connector = new THREE.Mesh(
        new THREE.TorusGeometry(30, 10, 16, 32, Math.PI),
        brainMat
    );
    connector.position.set(0, 20, 10);
    connector.rotation.y = Math.PI / 2;
    connector.rotation.x = 0.2;
    connector.scale.set(0.8, 0.6, 0.5);
    brainGroup.add(connector);
    
    return brainGroup;
}

/* ==============================
   3D VIEWER (Enhanced with Brain Reference)
============================== */

function initViewer(caseId) {

    const container = document.getElementById("viewer3d");
    if (!container) return;

    // Clear previous scene
    container.innerHTML = "";
    
    // Add hover info div
    hoverInfo = document.createElement('div');
    hoverInfo.style.position = 'absolute';
    hoverInfo.style.background = 'rgba(15, 23, 42, 0.95)';
    hoverInfo.style.color = '#f8fafc';
    hoverInfo.style.padding = '12px 16px';
    hoverInfo.style.borderRadius = '8px';
    hoverInfo.style.fontSize = '14px';
    hoverInfo.style.pointerEvents = 'none';
    hoverInfo.style.display = 'none';
    hoverInfo.style.zIndex = '1000';
    hoverInfo.style.border = '1px solid #3b82f6';
    hoverInfo.style.boxShadow = '0 10px 25px -5px rgba(0,0,0,0.5)';
    hoverInfo.style.backdropFilter = 'blur(8px)';
    container.style.position = 'relative';
    container.appendChild(hoverInfo);
    
    // Reset mesh references
    netMesh = null;
    edemaMesh = null;
    etMesh = null;
    brainReference = null;

    // Setup scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1120);

    // Setup camera
    camera = new THREE.PerspectiveCamera(
        60,
        container.clientWidth / container.clientHeight,
        0.1,
        5000
    );
    camera.position.set(200, 100, 300);
    camera.lookAt(0, 0, 0);

    // Setup renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x0b1120);
    container.appendChild(renderer.domElement);

    // Setup OrbitControls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.rotateSpeed = 0.8;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.8;
    controls.maxDistance = 800;
    controls.minDistance = 100;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;
    controls.enableZoom = true;
    controls.enablePan = true;
    controls.enableRotate = true;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404060);
    scene.add(ambientLight);

    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight1.position.set(1, 2, 1);
    scene.add(directionalLight1);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight2.position.set(-1, 0.5, -1);
    scene.add(directionalLight2);

    // Add back light
    const backLight = new THREE.DirectionalLight(0x8866aa, 0.4);
    backLight.position.set(0, 0, -2);
    scene.add(backLight);

    // Add hemisphere light
    const hemiLight = new THREE.HemisphereLight(0x445566, 0x221133, 0.6);
    scene.add(hemiLight);

    // Add a subtle grid for reference
    const gridHelper = new THREE.GridHelper(500, 20, 0x3b82f6, 0x1e293b);
    gridHelper.position.y = -60;
    gridHelper.material.opacity = 0.15;
    gridHelper.material.transparent = true;
    scene.add(gridHelper);

    // Create and add semi-transparent brain reference
    brainReference = createBrainReference();
    scene.add(brainReference);
    window.brainReference = brainReference;

    // Initialize loaders
    plyLoader = new THREE.PLYLoader();
    gltfLoader = new THREE.GLTFLoader();

    // Setup raycaster for hover detection
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // Track loaded meshes
    let loadedCount = 0;
    let errorCount = 0;
    const totalMeshes = 3;

    // Store mesh info for hover
    const meshInfo = {};

    function centerModel(model) {
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        
        model.position.sub(center);
        
        // Log size for debugging
        console.log(`${model.name || 'Model'} size:`, size);
        
        return size;
    }

    function checkAllLoaded() {
        if (loadedCount === totalMeshes) {
            console.log(`All meshes loaded. Success: ${totalMeshes - errorCount}, Errors: ${errorCount}`);
            setupOpacityControls();
            
            // If all meshes failed, show a message
            if (errorCount === totalMeshes) {
                const errorDiv = document.createElement('div');
                errorDiv.style.color = 'red';
                errorDiv.style.padding = '20px';
                errorDiv.style.textAlign = 'center';
                errorDiv.innerText = 'Failed to load 3D models. Check console for details.';
                container.appendChild(errorDiv);
            }
        }
    }

    function loadMesh(region, colorHex, assignVar) {
        console.log(`Attempting to load ${region} mesh as PLY...`);
        
        // Try PLY first
        plyLoader.load(
            `/mesh/${caseId}/${region}`,
            (geometry) => {
                console.log(`✅ Loaded ${region} as PLY`);
                
                geometry.computeVertexNormals();
                geometry.computeBoundingBox();
                geometry.center();
                
                const material = new THREE.MeshStandardMaterial({
                    color: colorHex,
                    transparent: true,
                    opacity: 1,
                    side: THREE.DoubleSide,
                    emissive: 0x000000,
                    roughness: 0.3,
                    metalness: 0.1
                });
                
                const mesh = new THREE.Mesh(geometry, material);
                mesh.userData = { 
                    type: region,
                    color: colorHex,
                    volume: calculateVolume(geometry)
                };
                scene.add(mesh);
                
                if (assignVar === "net") netMesh = mesh;
                if (assignVar === "edema") edemaMesh = mesh;
                if (assignVar === "et") etMesh = mesh;
                
                meshInfo[region] = mesh;
                
                loadedCount++;
                checkAllLoaded();
            },
            (xhr) => {
                console.log(`${region} PLY: ${Math.round(xhr.loaded / xhr.total * 100)}% loaded`);
            },
            (plyError) => {
                console.error(`❌ PLY failed for ${region}, trying GLTF as fallback:`, plyError);
                
                // Fallback to GLTF loader
                gltfLoader.load(
                    `/mesh/${caseId}/${region}`,
                    (gltf) => {
                        console.log(`✅ Loaded ${region} as GLTF (fallback)`);
                        const model = gltf.scene;
                        
                        model.traverse((child) => {
                            if (child.isMesh) {
                                child.material = new THREE.MeshStandardMaterial({
                                    color: colorHex,
                                    transparent: true,
                                    opacity: 1,
                                    side: THREE.DoubleSide,
                                    emissive: 0x000000,
                                    roughness: 0.3,
                                    metalness: 0.1
                                });
                                
                                child.userData = {
                                    type: region,
                                    color: colorHex
                                };
                                
                                if (assignVar === "net") netMesh = child;
                                if (assignVar === "edema") edemaMesh = child;
                                if (assignVar === "et") etMesh = child;
                                
                                meshInfo[region] = child;
                            }
                        });

                        scene.add(model);
                        centerModel(model);
                        loadedCount++;
                        checkAllLoaded();
                    },
                    (xhr) => {
                        console.log(`${region} GLTF: ${Math.round(xhr.loaded / xhr.total * 100)}% loaded`);
                    },
                    (gltfError) => {
                        console.error(`❌ Both PLY and GLTF failed for ${region}:`, gltfError);
                        errorCount++;
                        loadedCount++;
                        checkAllLoaded();
                    }
                );
            }
        );
    }

    // Helper function to calculate approximate volume from geometry
    function calculateVolume(geometry) {
        if (!geometry.boundingBox) geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        const size = new THREE.Vector3();
        box.getSize(size);
        return (size.x * size.y * size.z).toFixed(1);
    }

    // Load all three mesh types
    loadMesh("NET", 0x3b82f6, "net");     // Blue
    loadMesh("Edema", 0x10b981, "edema"); // Green
    loadMesh("ET", 0xf59e0b, "et");       // Yellow/Orange

    // Mouse move handler for hover effects
    function onMouseMove(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);

        const meshes = [netMesh, edemaMesh, etMesh].filter(m => m !== null);
        
        if (meshes.length === 0) return;

        const intersects = raycaster.intersectObjects(meshes, true);

        // Reset all materials
        meshes.forEach(m => {
            if (m && m.material) {
                if (Array.isArray(m.material)) {
                    m.material.forEach(mat => mat.emissive?.setHex(0x000000));
                } else {
                    m.material.emissive?.setHex(0x000000);
                }
            }
        });

        if (intersects.length > 0) {
            const intersect = intersects[0];
            const mesh = intersect.object;
            const point = intersect.point;
            
            // Get mesh type and volume from your data
            let regionName = 'Unknown';
            let volume = 0;
            let colorHex = '#ffffff';
            let voxels = 0;
            let mm3 = 0;
            
            if (mesh === netMesh || (mesh.parent === netMesh)) {
                regionName = 'NET (Necrotic Core)';
                colorHex = '#3b82f6';
                volume = window.volumes?.NET_cm3 || 0;
                voxels = window.volumes?.NET_voxels || 0;
                mm3 = window.volumes?.NET_mm3 || 0;
            } else if (mesh === edemaMesh || (mesh.parent === edemaMesh)) {
                regionName = 'Edema';
                colorHex = '#10b981';
                volume = window.volumes?.Edema_cm3 || 0;
                voxels = window.volumes?.Edema_voxels || 0;
                mm3 = window.volumes?.Edema_mm3 || 0;
            } else if (mesh === etMesh || (mesh.parent === etMesh)) {
                regionName = 'ET (Enhancing Tumor)';
                colorHex = '#f59e0b';
                volume = window.volumes?.ET_cm3 || 0;
                voxels = window.volumes?.ET_voxels || 0;
                mm3 = window.volumes?.ET_mm3 || 0;
            }
            
            // Show enhanced hover info with all your data
            hoverInfo.style.display = 'block';
            hoverInfo.style.left = (event.clientX - rect.left + 15) + 'px';
            hoverInfo.style.top = (event.clientY - rect.top + 15) + 'px';
            
            hoverInfo.innerHTML = `
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                    <div style="width:16px; height:16px; background:${colorHex}; border-radius:4px;"></div>
                    <strong style="color:${colorHex}; font-size:1.1rem;">${regionName}</strong>
                </div>
                <div style="display:grid; grid-template-columns:90px 1fr; gap:6px;">
                    <span style="color:#94a3b8;">Position:</span>
                    <span style="color:#f8fafc; font-family:monospace;">(${point.x.toFixed(1)}, ${point.y.toFixed(1)}, ${point.z.toFixed(1)})</span>
                    
                    <span style="color:#94a3b8;">Volume:</span>
                    <span style="color:#f8fafc; font-weight:500;">${volume.toFixed(2)} cm³</span>
                    
                    <span style="color:#94a3b8;">Voxels:</span>
                    <span style="color:#f8fafc;">${voxels.toLocaleString()}</span>
                    
                    <span style="color:#94a3b8;">Volume mm³:</span>
                    <span style="color:#f8fafc;">${(mm3).toFixed(0)} mm³</span>
                    
                    ${window.volumes ? `
                    <span style="color:#94a3b8;">% of Tumor:</span>
                    <span style="color:#f8fafc;">${regionName === 'NET (Necrotic Core)' ? window.volumes.NET_percent : 
                                                       regionName === 'Edema' ? window.volumes.Edema_percent : 
                                                       window.volumes.ET_percent}%</span>
                    ` : ''}
                </div>
            `;
            
            // Highlight hovered mesh
            if (mesh.material) {
                if (Array.isArray(mesh.material)) {
                    mesh.material.forEach(mat => mat.emissive?.setHex(0x333333));
                } else {
                    mesh.material.emissive?.setHex(0x333333);
                }
            }
        } else {
            hoverInfo.style.display = 'none';
        }
    }

    // Add mouse move listener
    renderer.domElement.addEventListener('mousemove', onMouseMove);

    // Animation
    function animate() {
        requestAnimationFrame(animate);

        // Update controls
        if (controls) {
            controls.update();
        }

        if (renderer && scene && camera) {
            renderer.render(scene, camera);
        }
    }

    animate();

    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);
}

function onWindowResize() {
    const container = document.getElementById("viewer3d");
    if (!container || !camera || !renderer || !controls) return;

    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

/* ==============================
   Opacity Controls (with Brain)
============================== */

function setupOpacityControls() {
    const netSlider = document.getElementById("netOpacity");
    const edemaSlider = document.getElementById("edemaOpacity");
    const etSlider = document.getElementById("etOpacity");
    const brainSlider = document.getElementById("brainOpacity");

    if (netSlider) {
        netSlider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            if (netMesh) {
                if (Array.isArray(netMesh.material)) {
                    netMesh.material.forEach(m => m.opacity = val);
                } else {
                    netMesh.material.opacity = val;
                }
            }
        });
    }

    if (edemaSlider) {
        edemaSlider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            if (edemaMesh) {
                if (Array.isArray(edemaMesh.material)) {
                    edemaMesh.material.forEach(m => m.opacity = val);
                } else {
                    edemaMesh.material.opacity = val;
                }
            }
        });
    }

    if (etSlider) {
        etSlider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            if (etMesh) {
                if (Array.isArray(etMesh.material)) {
                    etMesh.material.forEach(m => m.opacity = val);
                } else {
                    etMesh.material.opacity = val;
                }
            }
        });
    }

    // Brain opacity control
    if (brainSlider) {
        brainSlider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            
            // Update brain reference if it exists
            if (window.brainReference) {
                window.brainReference.traverse((child) => {
                    if (child.isMesh) {
                        child.material.opacity = val;
                    }
                });
            }
        });
        
        // Set initial value
        brainSlider.value = 0.15;
    }
}

/* ==============================
   Viewer Tab
============================== */

document.querySelector('[data-tab="viewer"]')?.addEventListener("click", () => {
    if (currentCaseId) {
        // Small delay to ensure tab is visible
        setTimeout(() => {
            initViewer(currentCaseId);
        }, 100);
    } else {
        alert("Please upload and segment an MRI first.");
    }
});

/* ==============================
   Initialize - Clear Dashboard on Load
============================== */

// Clear dashboard on initial load (empty state)
clearDashboard();

console.log('CerebraScan AI - Dashboard ready (empty), upload MRI to begin');

/* ==============================
   Clinical Report Generation
============================== */

(function setupReportButton() {
    console.log('Setting up report button');

    const reportBtn = document.getElementById("generateReportBtn");
    const reportOutput = document.getElementById("reportOutput");
    const reportStatus = document.getElementById("reportStatus");

    console.log('Report button found:', !!reportBtn);
    
    if (reportBtn) {
        // Simple direct click handler - no cloning
        reportBtn.onclick = async function(event) {
            event.preventDefault();
            console.log('Report button clicked');
            
            // Access currentCaseId from the global scope
            // If currentCaseId is declared with let/const in another closure, 
            // we need to access it through window
            const caseId = window.currentCaseId || currentCaseId;
            console.log('Case ID:', caseId);
            
            if (!caseId) {
                if (reportStatus) {
                    reportStatus.innerHTML = '<span style="color: #f59e0b;">⚠ Please upload and segment an MRI first.</span>';
                }
                return;
            }
            
            // Show loading
            if (reportStatus) {
                reportStatus.innerHTML = '<span style="color: #3b82f6;"><i class="fas fa-spinner fa-spin"></i> Generating report...</span>';
            }
            
            if (reportOutput) {
                reportOutput.innerHTML = `
                    <div style="text-align: center; padding: 30px;">
                        <i class="fas fa-spinner fa-spin" style="font-size: 2rem; color: #3b82f6;"></i>
                        <p style="color: #94a3b8; margin-top: 10px;">Generating clinical report...</p>
                    </div>
                `;
            }
            
            try {
                console.log('Fetching report...');
                const response = await fetch(`/generate-report/${caseId}`);
                
                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}`);
                }
                
                const data = await response.json();
                console.log('Report received:', data);

                if (data.status === 'error') {
                    throw new Error(data.detail || 'Report generation failed on the server.');
                }

                // Get the report text
                const reportText = data.report || 'No report content';
                
                // Simple display - just show the text with line breaks
                if (reportOutput) {
                    reportOutput.innerHTML = `
                        <div style="background: #0f172a; border-radius: 12px; padding: 20px;">
                            <div style="border-bottom: 1px solid #334155; padding-bottom: 10px; margin-bottom: 15px;">
                                <h3 style="color: #3b82f6; margin: 0;">
                                    <i class="fas fa-file-medical"></i> Clinical Report
                                </h3>
                                <small style="color: #6b7280;">${new Date().toLocaleString()}</small>
                            </div>
                            <div style="color: #e2e8f0; line-height: 1.6; white-space: pre-wrap; font-family: monospace;">
                                ${reportText.replace(/\n/g, '<br>')}
                            </div>
                        </div>
                    `;
                }
                
                if (reportStatus) {
                    reportStatus.innerHTML = '<span style="color: #10b981;">✅ Report generated</span>';
                }
                
            } catch (error) {
                console.error('Error:', error);
                if (reportStatus) {
                    reportStatus.innerHTML = `<span style="color: #ef4444;">❌ Error: ${error.message}</span>`;
                }
                if (reportOutput) {
                    reportOutput.innerHTML = `
                        <div style="color: #ef4444; padding: 20px; text-align: center;">
                            <i class="fas fa-exclamation-circle"></i>
                            <p>Failed to generate report</p>
                            <small>${error.message}</small>
                        </div>
                    `;
                }
            }
        };
        
        console.log('Report button handler attached');
    } else {
        console.error('Report button not found!');
    }
})();
});