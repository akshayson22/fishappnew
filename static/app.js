
// Client-side logic: toggling camera, capture, cropping, sending to server
let stream = null;
const video = document.getElementById('video');
const hiddenCanvas = document.getElementById('hiddenCanvas');
const cropCanvas = document.getElementById('cropCanvas');
const ctxHidden = hiddenCanvas.getContext('2d');
const ctxCrop = cropCanvas.getContext('2d');
const modal = document.getElementById('cropModal');

let cropping = false;
let cropRect = null;
let originalImg = null;
let loadedImage = null; // New variable to store the loaded image object
let imageAspectRatio = 1;
let imageOffsetX = 0;
let imageOffsetY = 0;
let lastProcessedImage = null;

function showWarning(msg, id='warning') {
    const element = document.getElementById(id);
    
    if (id === 'analysisWarnings') {
        // Show/hide warnings container based on content
        if (msg && msg.length > 0) {
            element.style.display = 'block';
            // Clear previous warnings
            element.innerHTML = '';
            
            // Create individual warning items
            const warnings = Array.isArray(msg) ? msg : [msg];
            warnings.forEach(warning => {
                const warningItem = document.createElement('div');
                warningItem.className = 'warning-item';
                warningItem.textContent = warning;
                element.appendChild(warningItem);
            });
        } else {
            element.style.display = 'none';
            element.innerHTML = '';
        }
    } else {
        // For regular warning element
        if (msg) {
            element.textContent = msg;
            element.style.display = 'block';
        } else {
            element.textContent = '';
            element.style.display = 'none';
        }
    }
}

function clearWarnings() {
    showWarning('', 'warning');
    showWarning([], 'analysisWarnings');
}

// Event listeners for UI buttons
document.getElementById('startCam').addEventListener('click', toggleCamera);
document.getElementById('captureBtn').addEventListener('click', captureImage);
document.getElementById('openFile').addEventListener('click', () => {
    document.getElementById('fileInput').click();
});
document.getElementById('fileInput').addEventListener('change', handleFileSelect);
document.getElementById('useCrop').addEventListener('click', handleCrop);
document.getElementById('useFullImage').addEventListener('click', useFullImage);
document.getElementById('resetCrop').addEventListener('click', resetCrop);
document.getElementById('savePdf').addEventListener('click', savePdf);
document.getElementById('recropBtn').addEventListener('click', recropImage);

// Modal event listeners
document.querySelector('.close').addEventListener('click', () => {
    modal.style.display = 'none';
});

window.addEventListener('click', (event) => {
    if (event.target === modal) {
        modal.style.display = 'none';
    }
});

function toggleCamera() {
    const cameraBtn = document.getElementById('startCam');
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
        document.getElementById('cameraBox').style.display = 'none';
        document.getElementById('captureBtn').disabled = true;
        
        // Change button back to "Toggle Camera" with camera icon
        cameraBtn.innerHTML = '<i class="fas fa-camera"></i> Toggle Camera';
    } else {
        navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
            .then(s => {
                stream = s;
                video.srcObject = s;
                video.play();
                document.getElementById('cameraBox').style.display = 'block';
                document.getElementById('captureBtn').disabled = false;
                
                // Change button to "Stop Camera" with stop icon
                cameraBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Camera';
            })
            .catch(err => {
                console.error("Error accessing camera: ", err);
                showWarning("Error: Could not access camera. " + err.message);
            });
    }
}

function captureImage() {
    if (!stream) {
        showWarning("Camera not active.");
        return;
    }

    // Set hidden canvas dimensions to match video stream
    hiddenCanvas.width = video.videoWidth;
    hiddenCanvas.height = video.videoHeight;
    ctxHidden.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
    
    // Convert to base64 for display and processing
    const dataUrl = hiddenCanvas.toDataURL('image/png');
    originalImg = dataUrl;
    lastProcessedImage = dataUrl;
    
    // Stop camera and show cropper
    toggleCamera();
    showCropper(dataUrl);
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        const dataUrl = e.target.result;
        originalImg = dataUrl;
        lastProcessedImage = dataUrl;
        showCropper(dataUrl);
    };
    reader.readAsDataURL(file);
}

function showCropper(dataUrl) {
    clearWarnings();
    modal.style.display = 'block';
    
    const img = new Image();
    img.onload = () => {
        // Calculate aspect ratio
        imageAspectRatio = img.width / img.height;
        
        // Set canvas dimensions based on container and image aspect ratio
        const container = document.querySelector('.crop-container');
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;
        
        let canvasWidth, canvasHeight;
        
        if (containerWidth / containerHeight > imageAspectRatio) {
            // Container is wider than image aspect ratio
            canvasHeight = containerHeight;
            canvasWidth = canvasHeight * imageAspectRatio;
        } else {
            // Container is taller than image aspect ratio
            canvasWidth = containerWidth;
            canvasHeight = canvasWidth / imageAspectRatio;
        }
        
        cropCanvas.width = canvasWidth;
        cropCanvas.height = canvasHeight;
        
        // Calculate image offset to center it
        imageOffsetX = (containerWidth - canvasWidth) / 2;
        imageOffsetY = (containerHeight - canvasHeight) / 2;
        
        // Store the loaded image object
        loadedImage = img;
        
        // Draw image on canvas
        ctxCrop.drawImage(loadedImage, 0, 0, canvasWidth, canvasHeight);
        
        // Reset cropping state
        cropping = false;
        cropRect = null;
        setupCroppingEvents();
    };
    img.src = dataUrl;
}

function resetCrop() {
    cropping = false;
    cropRect = null;
    
    // Redraw the image without the crop rectangle
    if (loadedImage) {
        ctxCrop.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
        ctxCrop.drawImage(loadedImage, 0, 0, cropCanvas.width, cropCanvas.height);
    }
}

function useFullImage() {
    // Send the full image for processing
    sendForProcessing(originalImg);
    modal.style.display = 'none';
}

function setupCroppingEvents() {
    let startX, startY;
    
    const startCrop = (e) => {
        const rect = cropCanvas.getBoundingClientRect();
        // Calculate click position relative to the canvas, not the container
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;
        
        // Only start cropping if clicking on the image area
        if (startX >= 0 && startX <= cropCanvas.width && 
            startY >= 0 && startY <= cropCanvas.height) {
            cropRect = { x: startX, y: startY, w: 0, h: 0 };
            cropping = true;
        }
    };

    const drawCrop = (e) => {
        if (!cropping || !loadedImage) return; // Add check for loadedImage
        const rect = cropCanvas.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;
        
        // Constrain to canvas boundaries
        const constrainedX = Math.max(0, Math.min(currentX, cropCanvas.width));
        const constrainedY = Math.max(0, Math.min(currentY, cropCanvas.height));
        
        ctxCrop.clearRect(0, 0, cropCanvas.width, cropCanvas.height);
        
        // Use the pre-loaded image instead of creating a new one
        ctxCrop.drawImage(loadedImage, 0, 0, cropCanvas.width, cropCanvas.height);

        cropRect.w = constrainedX - startX;
        cropRect.h = constrainedY - startY;
        
        // Draw crop rectangle with red border
        ctxCrop.strokeStyle = 'rgba(255, 0, 0, 0.8)';
        ctxCrop.lineWidth = 2;
        ctxCrop.strokeRect(cropRect.x, cropRect.y, cropRect.w, cropRect.h);
    };
    
    const endCrop = (e) => {
        if (!cropping) return;
        cropping = false;
        if (cropRect.w < 0) {
            cropRect.x += cropRect.w;
            cropRect.w = -cropRect.w;
        }
        if (cropRect.h < 0) {
            cropRect.y += cropRect.h;
            cropRect.h = -cropRect.h;
        }
    };
    
    // Add both mouse and touch events
    cropCanvas.addEventListener('mousedown', startCrop);
    cropCanvas.addEventListener('mousemove', drawCrop);
    cropCanvas.addEventListener('mouseup', endCrop);
    
    cropCanvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        startCrop(e.touches[0]);
    });
    cropCanvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        drawCrop(e.touches[0]);
    });
    cropCanvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        endCrop(e.changedTouches[0]);
    });
}

function handleCrop() {
    if (!cropRect || cropRect.w === 0 || cropRect.h === 0) {
        showWarning("Please select an area by dragging on the image, or click 'Use Entire Image'.");
        return;
    }
    
    // Get the cropped image data from the canvas
    if (!loadedImage) {
        showWarning("Image not loaded correctly. Please try again.");
        return;
    }
    
    // Create a new canvas to get the cropped portion
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = cropRect.w;
    tempCanvas.height = cropRect.h;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Calculate scale factor between displayed image and original
    const scaleX = loadedImage.naturalWidth / cropCanvas.width;
    const scaleY = loadedImage.naturalHeight / cropCanvas.height;
    
    tempCtx.drawImage(
        loadedImage, 
        cropRect.x * scaleX, 
        cropRect.y * scaleY, 
        cropRect.w * scaleX, 
        cropRect.h * scaleY, 
        0, 0, 
        tempCanvas.width, 
        tempCanvas.height
    );
    
    const croppedDataUrl = tempCanvas.toDataURL('image/png');
    
    // Send to server for processing
    sendForProcessing(croppedDataUrl);
    modal.style.display = 'none';
}

function recropImage() {
    if (!lastProcessedImage) {
        showWarning("No image available to recrop. Please capture or load an image first.");
        return;
    }
    
    // Show the crop modal again with the original image
    originalImg = lastProcessedImage;
    showCropper(lastProcessedImage);
}

function sendForProcessing(imageData) {
    const tvbn = document.getElementById('tvbn').value;
    const mass = document.getElementById('mass').value;
    const headspace = document.getElementById('headspace').value;
    const temp = document.getElementById('temp').value;
    
    const data = {
        image: imageData,
        tvbn_limit: tvbn,
        fish_mass: mass,
        headspace: headspace,
        temp: temp,
    };

    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showWarning(data.error, 'analysisWarnings');
            return;
        }
        
        // Update the UI with results
        updateResults(data);
    })
    .catch((error) => {
        console.error('Error:', error);
        showWarning(['Error during processing. Please try again.'], 'analysisWarnings');
    });
}

function updateResults(data) {
    // Update images
    document.getElementById('img_original').src = data.images.original;
    document.getElementById('img_black').src = data.images.black;
    document.getElementById('img_white').src = data.images.white;
    document.getElementById('img_target').src = data.images.target;
    
    // Update reference values
    document.getElementById('rgb_label').textContent = data.results.rgb;
    document.getElementById('hsv_label').textContent = data.results.hsv;
    document.getElementById('chroma_label').textContent = data.results.chroma;
    document.getElementById('hue_label').textContent = data.results.hue;
    
    // Update corrected values
    document.getElementById('corrected_rgb_label').textContent = data.results.corrected_rgb;
    document.getElementById('corrected_v_label').textContent = data.results.corrected_v;
    document.getElementById('corrected_chroma_label').textContent = data.results.corrected_chroma;
    document.getElementById('corrected_hue_label').textContent = data.results.corrected_hue;
    
    // Update freshness parameters
    document.getElementById('ph_label').textContent = data.results.ph;
    document.getElementById('tvbn_label').textContent = data.results.tvbn;
    document.getElementById('ammonia_label').textContent = data.results.ammonia;
    document.getElementById('shelf_life_label').textContent = data.results.shelf_life;
    
    // Update warnings
    if (data.results.warnings && data.results.warnings.length > 0) {
        showWarning(data.results.warnings, 'analysisWarnings');
    } else {
        showWarning([], 'analysisWarnings');
    }
}

function savePdf() {
    fetch('/save_pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.blob();
    })
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'analysis_report.pdf';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    })
    .catch(error => {
        console.error('Error saving PDF:', error);
        showWarning(["Error saving PDF. Please try again."], 'analysisWarnings');
    });
}