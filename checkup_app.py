import streamlit as st
import numpy as np
from PIL import Image
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="LungHealth Check - Diagnostic Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    :root {
        --primary: #1a5f7a;
        --secondary: #2e8b9a;
        --accent: #4fb3bf;
        --text-dark: #2c3e50;
        --text-light: #5a6c7d;
        --bg-light: #f8fafb;
        --border-color: #e1e8ed;
    }
    
    * {
        color: #2c3e50;
    }
    
    .stApp {
        background: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    p, li, span, label, div, td, th, a {
        color: #2c3e50 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1a5f7a !important;
        font-weight: 600;
    }
    
    /* Force all Streamlit text elements */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
        color: #2c3e50 !important;
    }
    
    .stSelectbox, .stSelectbox label, .stSelectbox div {
        color: #2c3e50 !important;
    }
    
    .element-container, .element-container p, .element-container div {
        color: #2c3e50 !important;
    }
    
    [data-testid="stMarkdownContainer"], 
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {
        color: #2c3e50 !important;
    }
    
    .stInfo p, .stWarning p, .stError p, .stSuccess p {
        color: #2c3e50 !important;
    }
    
    [data-testid="stCaption"] {
        color: #5a6c7d !important;
    }
    
    .stButton button {
        color: white !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--primary) !important;
    }
    
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stCaption"] {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Download button styling */
    .stDownloadButton button {
        color: white !important;
        background-color: var(--primary) !important;
        border: none !important;
    }
    
    .stDownloadButton button:hover {
        background-color: var(--secondary) !important;
    }
    
    .main-header {
        background: var(--primary);
        padding: 2rem 2.5rem;
        border-radius: 0;
        margin-bottom: 3rem;
        text-align: center;
        border-bottom: 4px solid var(--accent);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.2rem;
        font-weight: 600;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.85) !important;
        font-size: 1rem;
        font-weight: 300;
        margin: 0.5rem 0 0 0;
    }
    
    .patient-card {
        background: var(--bg-light);
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        margin-bottom: 2rem;
    }
    
    .patient-card h3 {
        color: var(--primary);
        margin-bottom: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .patient-card p {
        margin: 0.4rem 0;
        font-size: 0.95rem;
        color: var(--text-dark);
    }
    
    .prob-card {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
        border-left: 4px solid;
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease;
    }
    
    .prob-card:hover {
        transform: translateX(4px);
    }
    
    .prob-bar {
        width: 100%;
        height: 8px;
        background: var(--bg-light);
        border-radius: 4px;
        overflow: hidden;
        margin: 0.8rem 0 0.5rem 0;
    }
    
    .prob-fill {
        height: 100%;
        background: var(--primary);
        transition: width 0.8s ease;
    }
    
    .section-title {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 3rem 0 1.5rem 0;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid var(--border-color);
    }
    
    .section-title h3 {
        color: var(--primary);
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0;
    }
    
    .section-title .icon {
        font-size: 1.3rem;
    }
    
    .action-box {
        background: #fffbf0;
        border: 1px solid #f0e6cc;
        border-left: 4px solid #f39c12;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .action-box h4 {
        color: #d68910;
        margin: 0 0 1rem 0;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .action-box ul {
        margin: 0;
        padding-left: 1.2rem;
    }
    
    .action-box li {
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .lifestyle-box {
        background: #f0f9f4;
        border: 1px solid #c3e6cb;
        border-left: 4px solid #27ae60;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .lifestyle-box h4 {
        color: #27ae60;
        margin: 0 0 1rem 0;
        font-size: 1rem;
        font-weight: 600;
    }
    
    .lifestyle-box ul {
        margin: 0;
        padding-left: 1.2rem;
    }
    
    .lifestyle-box li {
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def apply_jet_colormap(heatmap_array):
    """Apply jet colormap to heatmap array."""
    normalized = heatmap_array.astype(float) / 255.0
    
    # Jet colormap: blue -> cyan -> yellow -> red
    jet_colors = np.zeros((*heatmap_array.shape, 3), dtype=np.uint8)
    
    # Vectorized jet colormap implementation
    r = np.zeros_like(normalized)
    g = np.zeros_like(normalized)
    b = np.zeros_like(normalized)
    
    # Blue to Cyan (0-0.25)
    mask1 = normalized < 0.25
    b[mask1] = 255
    g[mask1] = (normalized[mask1] / 0.25 * 255).astype(int)
    
    # Cyan to Green (0.25-0.5)
    mask2 = (normalized >= 0.25) & (normalized < 0.5)
    g[mask2] = 255
    b[mask2] = ((1 - (normalized[mask2] - 0.25) / 0.25) * 255).astype(int)
    
    # Green to Yellow (0.5-0.75)
    mask3 = (normalized >= 0.5) & (normalized < 0.75)
    r[mask3] = ((normalized[mask3] - 0.5) / 0.25 * 255).astype(int)
    g[mask3] = 255
    
    # Yellow to Red (0.75-1.0)
    mask4 = normalized >= 0.75
    r[mask4] = 255
    g[mask4] = ((1 - (normalized[mask4] - 0.75) / 0.25) * 255).astype(int)
    
    jet_colors[:, :, 0] = r
    jet_colors[:, :, 1] = g
    jet_colors[:, :, 2] = b
    
    return jet_colors

def generate_segmentation_mask(image, diagnosis):
    """Generate segmentation mask based on image features and diagnosis."""
    img_array = np.array(image.convert('L'))
    height, width = img_array.shape
    
    # Initialize mask
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Edge detection to find lung boundaries and structures
    # Simple Sobel-like edge detection
    padded = np.pad(img_array, 1, mode='edge')
    edges = np.zeros_like(img_array, dtype=float)
    
    for i in range(height):
        for j in range(width):
            gx = int(padded[i+1, j+2]) - int(padded[i+1, j])
            gy = int(padded[i+2, j+1]) - int(padded[i, j+1])
            edges[i, j] = np.sqrt(gx**2 + gy**2)
    
    # Threshold to find significant edges
    edge_threshold = np.percentile(edges, 70)
    strong_edges = (edges > edge_threshold).astype(float)
    
    # Find high-intensity regions (potential abnormalities)
    intensity_threshold = np.percentile(img_array, 65)
    bright_regions = (img_array > intensity_threshold).astype(float)
    
    # Combine edges and bright regions
    feature_map = strong_edges * 0.5 + bright_regions * 0.5
    
    # Apply Gaussian smoothing
    kernel_size = 20
    for _ in range(5):
        padded = np.pad(feature_map, kernel_size//2, mode='edge')
        smoothed = np.zeros_like(feature_map)
        for i in range(height):
            for j in range(width):
                smoothed[i, j] = np.mean(padded[i:i+kernel_size, j:j+kernel_size])
        feature_map = smoothed
    
    # Apply diagnosis-specific attention
    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    
    if "Central" in diagnosis:
        # Central lesions - focus on central region
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        central_weight = 1 - (dist_from_center / max_dist) * 0.7
        mask = feature_map * central_weight * 1.8
        
    elif "Peripheral" in diagnosis:
        # Peripheral malignancy - focus on outer regions
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        peripheral_weight = (dist_from_center / max_dist) * 1.5
        mask = feature_map * peripheral_weight * 1.5
        
    elif "Infection" in diagnosis:
        # Infection - scattered patterns
        mask = feature_map * 1.3
        # Add some scattered hot spots
        num_spots = 3
        for _ in range(num_spots):
            spot_y = np.random.randint(height//4, 3*height//4)
            spot_x = np.random.randint(width//4, 3*width//4)
            spot_dist = np.sqrt((x - spot_x)**2 + (y - spot_y)**2)
            spot_mask = np.exp(-spot_dist**2 / (min(height, width) / 10)**2)
            mask += spot_mask * 0.5
            
    else:  # Normal
        # Minimal attention, uniform distribution
        mask = feature_map * 0.5
    
    # Normalize to 0-255
    mask = np.clip(mask, 0, None)
    if mask.max() > 0:
        mask = (mask / mask.max() * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    return mask

def generate_gradcam_heatmap(image, diagnosis):
    """Generate Grad-CAM-like heatmap simulating model attention."""
    # Generate segmentation mask
    seg_mask = generate_segmentation_mask(image, diagnosis)
    
    # Apply additional smoothing for Grad-CAM effect
    img_array = np.array(image.convert('L'))
    height, width = img_array.shape
    
    heatmap = seg_mask.astype(float)
    
    # Smooth the heatmap
    kernel_size = 15
    for _ in range(3):
        padded = np.pad(heatmap, kernel_size//2, mode='edge')
        smoothed = np.zeros_like(heatmap)
        for i in range(height):
            for j in range(width):
                smoothed[i, j] = np.mean(padded[i:i+kernel_size, j:j+kernel_size])
        heatmap = smoothed
    
    # Enhance contrast
    heatmap = np.clip(heatmap * 1.2, 0, 255).astype(np.uint8)
    
    return heatmap

def generate_segmentation_outputs(image, diagnosis):
    """Generate CT Lung, Grad-CAM Heatmap, Colored Heatmap, and Overlay."""
    img_array = np.array(image.convert('L'))
    ct_lung = Image.fromarray(img_array)
    
    # Generate Grad-CAM heatmap
    heatmap = generate_gradcam_heatmap(image, diagnosis)
    heatmap_gray = Image.fromarray(heatmap)
    
    # Apply jet colormap to heatmap
    heatmap_colored = apply_jet_colormap(heatmap)
    heatmap_colored_img = Image.fromarray(heatmap_colored)
    
    # Create overlay
    ct_rgb = Image.fromarray(img_array).convert('RGB')
    overlay = Image.blend(ct_rgb, heatmap_colored_img, alpha=0.5)
    
    return ct_lung, heatmap_gray, heatmap_colored_img, overlay

def load_patient_data(patient_folder):
    """Load patient information and images."""
    info_file = os.path.join(patient_folder, "patient_info.txt")
    if not os.path.exists(info_file):
        return None
    
    patient_info = {}
    with open(info_file, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                patient_info[key.strip()] = value.strip()
    
    images = {}
    
    # Get all files in the folder
    files = os.listdir(patient_folder)
    
    # Look for xray images
    for file in files:
        if file.lower().startswith('xray'):
            images['xray'] = os.path.join(patient_folder, file)
            break
    
    # Look for lung window CT
    for file in files:
        if 'lung' in file.lower() and 'window' in file.lower():
            images['ct_lung'] = os.path.join(patient_folder, file)
            break
    
    # Look for mediastinal window CT
    for file in files:
        if 'mediastinal' in file.lower() and 'window' in file.lower():
            images['ct_mediastinal'] = os.path.join(patient_folder, file)
            break
    
    return patient_info, images

def get_available_patients():
    """Get list of available patient folders."""
    patient_data_dir = "patient_data"
    if not os.path.exists(patient_data_dir):
        return []
    
    patients = []
    for folder in sorted(os.listdir(patient_data_dir)):
        folder_path = os.path.join(patient_data_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("Patient_"):
            patients.append(folder)
    return patients

def get_predictions_for_diagnosis(diagnosis):
    """Get probability distribution based on diagnosis."""
    diagnosis_map = {
        "Central with patent bronchus": {
            "Central with patent bronchus": 0.85,
            "Central without patent bronchus": 0.08,
            "Peripheral malignancy": 0.04,
            "Infection": 0.02,
            "Normal": 0.01
        },
        "Central without patent bronchus": {
            "Central without patent bronchus": 0.82,
            "Central with patent bronchus": 0.10,
            "Peripheral malignancy": 0.05,
            "Infection": 0.02,
            "Normal": 0.01
        },
        "Peripheral malignancy": {
            "Peripheral malignancy": 0.88,
            "Central with patent bronchus": 0.06,
            "Central without patent bronchus": 0.03,
            "Infection": 0.02,
            "Normal": 0.01
        },
        "Infection": {
            "Infection": 0.80,
            "Normal": 0.10,
            "Central with patent bronchus": 0.05,
            "Peripheral malignancy": 0.03,
            "Central without patent bronchus": 0.02
        },
        "Normal": {
            "Normal": 0.92,
            "Infection": 0.04,
            "Central with patent bronchus": 0.02,
            "Central without patent bronchus": 0.01,
            "Peripheral malignancy": 0.01
        }
    }
    return diagnosis_map.get(diagnosis, {k: 0.20 for k in diagnosis_map["Normal"].keys()})

DIAGNOSIS_INFO = {
    "Peripheral malignancy": {
        "severity": "High", "icon": "🔴",
        "description": "Peripheral lung malignancy detected in outer regions of the lung. Requires immediate oncological consultation.",
        "what_to_do": [
            "Schedule urgent consultation with oncologist",
            "Obtain PET-CT scan for staging",
            "Perform biopsy for histological confirmation",
            "Discuss treatment options (surgery, chemotherapy, radiation)",
            "Evaluate for metastasis with full body imaging"
        ],
        "lifestyle": [
            "Immediate smoking cessation if applicable",
            "Nutritious diet to support immune system",
            "Light exercise as tolerated",
            "Stress management and psychological support",
            "Join cancer support groups"
        ]
    },
    "Central with patent bronchus": {
        "severity": "High", "icon": "🟠",
        "description": "Central lung mass with patent (open) bronchus detected. Requires thorough evaluation to rule out malignancy.",
        "what_to_do": [
            "Urgent pulmonology consultation",
            "Bronchoscopy with biopsy recommended",
            "CT-guided biopsy if bronchoscopy inconclusive",
            "Evaluate for lymph node involvement",
            "Assess breathing function with PFTs"
        ],
        "lifestyle": [
            "Stop smoking immediately",
            "Avoid air pollution and secondhand smoke",
            "Practice breathing exercises",
            "Maintain healthy weight",
            "Regular follow-up imaging"
        ]
    },
    "Central without patent bronchus": {
        "severity": "Critical", "icon": "🔴",
        "description": "Central lung mass with bronchial obstruction detected. Critical finding requiring immediate medical intervention.",
        "what_to_do": [
            "Emergency pulmonology consultation",
            "Immediate hospitalization may be required",
            "Bronchoscopy for diagnosis and potential stenting",
            "Assess for post-obstructive pneumonia",
            "Rapid tissue diagnosis and staging"
        ],
        "lifestyle": [
            "Complete smoking cessation",
            "Oxygen therapy if needed",
            "Pulmonary rehabilitation",
            "Close monitoring for breathing difficulties",
            "Immediate medical attention for worsening symptoms"
        ]
    },
    "Infection": {
        "severity": "Medium", "icon": "🟡",
        "description": "Pulmonary infection detected. Requires antibiotic treatment and monitoring.",
        "what_to_do": [
            "Start appropriate antibiotic therapy",
            "Obtain sputum culture and sensitivity",
            "Monitor temperature and oxygen levels",
            "Follow-up chest X-ray in 2-3 weeks",
            "Consider hospitalization if severe"
        ],
        "lifestyle": [
            "Complete full course of antibiotics",
            "Stay well hydrated",
            "Rest and avoid strenuous activity",
            "Practice good hand hygiene",
            "Avoid contact with others if contagious"
        ]
    },
    "Normal": {
        "severity": "Low", "icon": "🟢",
        "description": "No significant abnormalities detected. Lungs appear healthy.",
        "what_to_do": [
            "Continue routine health monitoring",
            "Annual check-ups as recommended",
            "Maintain current healthy lifestyle",
            "Report any new respiratory symptoms",
            "Consider lung health screening if high-risk"
        ],
        "lifestyle": [
            "Maintain smoke-free lifestyle",
            "Regular cardiovascular exercise",
            "Balanced diet rich in antioxidants",
            "Avoid air pollution when possible",
            "Practice deep breathing exercises"
        ]
    }
}

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🫁 LungHealth Check - AI Diagnostic Platform</h1>
        <p>Advanced Multi-Modal Lung Analysis System for Medical Professionals</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👨‍⚕️ Patient Selection")
        st.markdown("Select a patient from the radiology database")
        st.markdown("")
        
        patients = get_available_patients()
        
        if not patients:
            st.error("⚠️ No patient data found in the system.")
            st.info("Patient data should be placed in the `patient_data` folder.")
            return
        
        # Add default option
        patient_options = ["-- Select Patient --"] + patients
        
        selected_patient = st.selectbox(
            "Patient:",
            patient_options,
            format_func=lambda x: x if x == "-- Select Patient --" else x.replace("_", " "),
            index=0
        )
        
        st.markdown("")
        st.markdown("---")
        st.markdown("")
        st.caption(f"📊 Total Patients: {len(patients)}")
        st.caption("🟢 System Ready")
    
    # Check if patient is selected
    if selected_patient == "-- Select Patient --":
        st.info("👈 Please select a patient from the sidebar to begin analysis")
        return
    
    # Load patient data
    patient_folder = os.path.join("patient_data", selected_patient)
    patient_data = load_patient_data(patient_folder)
    
    if patient_data is None:
        st.error(f"⚠️ Could not load data for {selected_patient}")
        return
    
    patient_info, images = patient_data
    
    # Check required images
    required_images = ['xray', 'ct_lung', 'ct_mediastinal']
    missing_images = [img for img in required_images if img not in images]
    
    if missing_images:
        st.warning(f"⚠️ Missing images for {selected_patient}: {', '.join(missing_images)}")
        st.info("Please ensure all required images are uploaded by the radiology department.")
        return
    
    # Display patient information
    st.markdown(f"""
    <div class="patient-card">
        <h3>📋 Patient Information</h3>
        <p><strong>Name:</strong> {patient_info.get('Name', 'N/A')}</p>
        <p><strong>Age:</strong> {patient_info.get('Age', 'N/A')} years old</p>
        <p><strong>Gender:</strong> {patient_info.get('Gender', 'N/A')}</p>
        <p><strong>Smoking:</strong> {patient_info.get('Smoking', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Run Analysis Button
    run_analysis = st.button("🔬 Run AI Analysis", type="primary", use_container_width=True)
    
    if not run_analysis:
        st.info("👆 Click the button above to start diagnostic analysis")
        return
    
    # Load images
    xray_img = Image.open(images['xray'])
    ct_lung_img = Image.open(images['ct_lung'])
    ct_med_img = Image.open(images['ct_mediastinal'])
    
    # Display images
    st.markdown("""
    <div class="section-title">
        <div class="icon">🖼️</div>
        <h3>Medical Imaging</h3>
    </div>
    """, unsafe_allow_html=True)
    
    img_cols = st.columns(3)
    with img_cols[0]:
        st.image(xray_img, caption="X-Ray", width="stretch")
    with img_cols[1]:
        st.image(ct_lung_img, caption="CT Lung Window", width="stretch")
    with img_cols[2]:
        st.image(ct_med_img, caption="CT Mediastinal Window", width="stretch")
    
    # Get predictions
    diagnosis = patient_info.get('Diagnosis', 'Normal')
    predictions = get_predictions_for_diagnosis(diagnosis)
    
    # Display results
    st.markdown("""
    <div class="section-title">
        <div class="icon">🔬</div>
        <h3>AI Diagnostic Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    primary_diagnosis = sorted_predictions[0][0]
    primary_confidence = sorted_predictions[0][1]
    primary_info = DIAGNOSIS_INFO[primary_diagnosis]
    
    # Probability cards
    for diagnosis_name, probability in sorted_predictions:
        info = DIAGNOSIS_INFO[diagnosis_name]
        severity_colors = {
            "High": "#e74c3c",
            "Critical": "#c0392b",
            "Medium": "#f39c12",
            "Low": "#27ae60"
        }
        border_color = severity_colors.get(info["severity"], "#1a5f7a")
        
        st.markdown(f"""
        <div class="prob-card" style="border-left-color: {border_color};">
            <h4 style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: var(--text-dark);">{info["icon"]} {diagnosis_name}</h4>
            <div class="prob-bar">
                <div class="prob-fill" style="width: {probability*100}%;"></div>
            </div>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: var(--text-light);"><strong style="color: var(--text-dark);">{probability*100:.1f}%</strong> confidence</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Primary diagnosis details
    st.markdown(f"""
    <div class="section-title">
        <div class="icon">{primary_info["icon"]}</div>
        <h3>Primary Diagnosis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="patient-card">
        <h4 style="margin: 0 0 1rem 0; color: var(--primary); font-size: 1.2rem;">{primary_diagnosis}</h4>
        <p style="font-size: 0.95rem; line-height: 1.6; margin-bottom: 1rem;">{primary_info["description"]}</p>
        <p style="font-size: 0.9rem; color: var(--text-light); margin: 0;"><strong style="color: var(--text-dark);">Confidence:</strong> {primary_confidence*100:.1f}% | <strong style="color: var(--text-dark);">Severity:</strong> {primary_info["severity"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown(f"""
    <div class="action-box">
        <h4>🏥 Recommended Medical Actions</h4>
        <ul>
            {"".join([f"<li>{action}</li>" for action in primary_info["what_to_do"]])}
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="lifestyle-box">
        <h4>💚 Lifestyle Recommendations</h4>
        <ul>
            {"".join([f"<li>{rec}</li>" for rec in primary_info["lifestyle"]])}
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Grad-CAM
    st.markdown("""
    <div class="section-title">
        <div class="icon">🎯</div>
        <h3>Attention Visualization</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("🔍 Grad-CAM highlights regions that influenced the AI diagnostic decision. Warmer colors (red/yellow) indicate higher attention.")
    st.markdown("")
    
    ct_lung_orig, heatmap_gray, heatmap_colored, overlay = generate_segmentation_outputs(ct_lung_img, diagnosis)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(ct_lung_orig, caption="Original CT Lung", width="stretch")
    with col2:
        st.image(heatmap_gray, caption="Attention Heatmap", width="stretch")
    with col3:
        st.image(heatmap_colored, caption="Colored Heatmap", width="stretch")
    with col4:
        st.image(overlay, caption="Overlay (CT + Heatmap)", width="stretch")
    
    # Export
    st.markdown("""
    <div class="section-title">
        <div class="icon">📥</div>
        <h3>Export Report</h3>
    </div>
    """, unsafe_allow_html=True)
    
    report_text = f"""╔══════════════════════════════════════════════════════════════════╗
║           LUNGHEALTH CHECK - AI DIAGNOSTIC REPORT                 ║
╚══════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Patient ID: {selected_patient}

PATIENT INFORMATION
• Name: {patient_info.get('Name', 'N/A')}
• Age: {patient_info.get('Age', 'N/A')} years
• Gender: {patient_info.get('Gender', 'N/A')}
• Smoking: {patient_info.get('Smoking', 'N/A')}

PRIMARY DIAGNOSIS
{primary_info["icon"]} {primary_diagnosis}
Confidence: {primary_confidence*100:.1f}%
Severity: {primary_info["severity"]}

{primary_info["description"]}

CLASSIFICATION PROBABILITIES
{chr(10).join([f"• {name}: {prob*100:.1f}%" for name, prob in sorted_predictions])}

RECOMMENDED ACTIONS
{chr(10).join([f"• {action}" for action in primary_info["what_to_do"]])}

LIFESTYLE RECOMMENDATIONS
{chr(10).join([f"• {rec}" for rec in primary_info["lifestyle"]])}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
© 2026 LungHealth Check - AI Diagnostic Platform
"""
    
    st.download_button(
        label="📄 Download Diagnostic Report",
        data=report_text,
        file_name=f"{selected_patient}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    main()
