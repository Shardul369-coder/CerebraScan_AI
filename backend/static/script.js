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
                updateDashboard(data.volumetrics);
            }

            document.querySelector('[data-tab="dashboard"]')?.click();
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
   Dashboard Charts
============================== */

function updateDashboard(volumes) {

    if (!volumes) return;

    document.getElementById("wholeVolume").innerText =
        volumes.Whole_Tumor_cm3 + " cm³";

    document.getElementById("coreVolume").innerText =
        volumes.Tumor_Core_cm3 + " cm³";

    if (compositionChart) compositionChart.destroy();
    if (volumeChart) volumeChart.destroy();

    const ctx1 = document.getElementById("compositionChart")?.getContext("2d");
    const ctx2 = document.getElementById("volumeChart")?.getContext("2d");

    if (!ctx1 || !ctx2) return;

    compositionChart = new Chart(ctx1, {
        type: "doughnut",
        data: {
            labels: ["NET", "Edema", "ET"],
            datasets: [{
                data: [
                    volumes.NET_percent,
                    volumes.Edema_percent,
                    volumes.ET_percent
                ],
                backgroundColor: ["#3b82f6", "#22c55e", "#facc15"]
            }]
        },
        options: { responsive: true }
    });

    volumeChart = new Chart(ctx2, {
        type: "bar",
        data: {
            labels: ["NET", "Edema", "ET"],
            datasets: [{
                label: "Volume (cm³)",
                data: [
                    volumes.NET_cm3,
                    volumes.Edema_cm3,
                    volumes.ET_cm3
                ],
                backgroundColor: ["#3b82f6", "#22c55e", "#facc15"]
            }]
        },
        options: {
            responsive: true,
            scales: { y: { beginAtZero: true } }
        }
    });
}

/* ==============================
   3D VIEWER (Enhanced with Controls)
============================== */

function initViewer(caseId) {

    const container = document.getElementById("viewer3d");
    if (!container) return;

    // Clear previous scene
    container.innerHTML = "";
    
    // Add hover info div
    hoverInfo = document.createElement('div');
    hoverInfo.style.position = 'absolute';
    hoverInfo.style.background = 'rgba(0,0,0,0.8)';
    hoverInfo.style.color = 'white';
    hoverInfo.style.padding = '8px 12px';
    hoverInfo.style.borderRadius = '4px';
    hoverInfo.style.fontSize = '14px';
    hoverInfo.style.pointerEvents = 'none';
    hoverInfo.style.display = 'none';
    hoverInfo.style.zIndex = '1000';
    container.style.position = 'relative';
    container.appendChild(hoverInfo);
    
    // Reset mesh references
    netMesh = null;
    edemaMesh = null;
    etMesh = null;

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
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Setup OrbitControls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.rotateSpeed = 1.0;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.8;
    controls.maxDistance = 800;
    controls.minDistance = 50;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.0;
    controls.enableZoom = true;
    controls.enablePan = true;
    controls.enableRotate = true;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight1.position.set(1, 2, 1);
    scene.add(directionalLight1);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight2.position.set(-1, 0.5, -1);
    scene.add(directionalLight2);

    // Add a subtle grid for reference
    const gridHelper = new THREE.GridHelper(500, 20, 0x3b82f6, 0x1e293b);
    gridHelper.position.y = -50;
    scene.add(gridHelper);

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
        
        // Try PLY first (since your files are PLY format)
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
        // This is a simplified volume calculation
        // For more accurate volume, you'd need to compute from the mesh
        if (!geometry.boundingBox) geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        const size = new THREE.Vector3();
        box.getSize(size);
        return (size.x * size.y * size.z).toFixed(1);
    }

    // Load all three mesh types
    loadMesh("NET", 0x3b82f6, "net");     // Blue
    loadMesh("Edema", 0x22c55e, "edema"); // Green
    loadMesh("ET", 0xfacc15, "et");       // Yellow

    // Mouse move handler for hover effects
    function onMouseMove(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);

        const meshes = [netMesh, edemaMesh, etMesh].filter(m => m !== null);
        
        if (meshes.length === 0) return;

        const intersects = raycaster.intersectObjects(meshes, true);

        if (intersects.length > 0) {
            const intersect = intersects[0];
            const mesh = intersect.object;
            const point = intersect.point;
            
            // Show hover info
            hoverInfo.style.display = 'block';
            hoverInfo.style.left = (event.clientX - rect.left + 10) + 'px';
            hoverInfo.style.top = (event.clientY - rect.top + 10) + 'px';
            
            let regionName = mesh.userData?.type || 'Unknown';
            let colorHex = mesh.userData?.color || 0xffffff;
            let colorName = '';
            
            switch(regionName) {
                case 'NET': colorName = 'Necrotic Core'; break;
                case 'Edema': colorName = 'Edema'; break;
                case 'ET': colorName = 'Enhancing Tumor'; break;
                default: colorName = regionName;
            }
            
            hoverInfo.innerHTML = `
                <strong style="color: #${colorHex.toString(16).padStart(6, '0')}">${colorName}</strong><br>
                Position: (${point.x.toFixed(1)}, ${point.y.toFixed(1)}, ${point.z.toFixed(1)})<br>
                Volume: ${mesh.userData?.volume || 'N/A'} cm³
            `;
            
            // Highlight hovered mesh
            meshes.forEach(m => {
                if (m) m.material.emissive.setHex(0x000000);
            });
            if (mesh.material) {
                mesh.material.emissive.setHex(0x333333);
            }
        } else {
            hoverInfo.style.display = 'none';
            // Reset highlights
            meshes.forEach(m => {
                if (m) m.material.emissive.setHex(0x000000);
            });
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
   Opacity Controls
============================== */

function setupOpacityControls() {
    const netSlider = document.getElementById("netOpacity");
    const edemaSlider = document.getElementById("edemaOpacity");
    const etSlider = document.getElementById("etOpacity");

    if (netSlider) {
        netSlider.addEventListener("input", (e) => {
            if (netMesh) netMesh.material.opacity = parseFloat(e.target.value);
        });
    }

    if (edemaSlider) {
        edemaSlider.addEventListener("input", (e) => {
            if (edemaMesh) edemaMesh.material.opacity = parseFloat(e.target.value);
        });
    }

    if (etSlider) {
        etSlider.addEventListener("input", (e) => {
            if (etMesh) etMesh.material.opacity = parseFloat(e.target.value);
        });
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

});