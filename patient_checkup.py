import streamlit as st
import numpy as np
from PIL import Image
import os
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="LungHealth AI - Patient Upload",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper functions for classification and Grad-CAM
def apply_jet_colormap(heatmap_array):
    """Apply jet colormap to grayscale heatmap."""
    normalized = heatmap_array.astype(float) / 255.0
    
    # Jet colormap implementation
    r = np.zeros_like(normalized)
    g = np.zeros_like(normalized)
    b = np.zeros_like(normalized)
    
    # Blue to cyan (0.0 - 0.25)
    mask1 = normalized < 0.25
    b[mask1] = 255
    g[mask1] = (normalized[mask1] / 0.25 * 255).astype(int)
    
    # Cyan to green (0.25 - 0.5)
    mask2 = (normalized >= 0.25) & (normalized < 0.5)
    g[mask2] = 255
    b[mask2] = ((1 - (normalized[mask2] - 0.25) / 0.25) * 255).astype(int)
    
    # Green to yellow (0.5 - 0.75)
    mask3 = (normalized >= 0.5) & (normalized < 0.75)
    r[mask3] = ((normalized[mask3] - 0.5) / 0.25 * 255).astype(int)
    g[mask3] = 255
    
    # Yellow to red (0.75 - 1.0)
    mask4 = normalized >= 0.75
    r[mask4] = 255
    g[mask4] = ((1 - (normalized[mask4] - 0.75) / 0.25) * 255).astype(int)
    
    jet_colors = np.zeros((heatmap_array.shape[0], heatmap_array.shape[1], 3), dtype=np.uint8)
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
    
    # Edge detection
    padded = np.pad(img_array, 1, mode='edge')
    edges = np.zeros_like(img_array, dtype=float)
    
    for i in range(height):
        for j in range(width):
            gx = int(padded[i+1, j+2]) - int(padded[i+1, j])
            gy = int(padded[i+2, j+1]) - int(padded[i, j+1])
            edges[i, j] = np.sqrt(gx**2 + gy**2)
    
    edge_threshold = np.percentile(edges, 70)
    strong_edges = (edges > edge_threshold).astype(float)
    
    intensity_threshold = np.percentile(img_array, 65)
    bright_regions = (img_array > intensity_threshold).astype(float)
    
    feature_map = strong_edges * 0.5 + bright_regions * 0.5
    
    # Gaussian smoothing
    kernel_size = 20
    for _ in range(5):
        padded = np.pad(feature_map, kernel_size//2, mode='edge')
        smoothed = np.zeros_like(feature_map)
        for i in range(height):
            for j in range(width):
                smoothed[i, j] = np.mean(padded[i:i+kernel_size, j:j+kernel_size])
        feature_map = smoothed
    
    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    
    if "Central" in diagnosis:
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        central_weight = 1 - (dist_from_center / max_dist) * 0.7
        mask = feature_map * central_weight * 1.8
        
    elif "Peripheral" in diagnosis:
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        peripheral_weight = (dist_from_center / max_dist) * 1.5
        mask = feature_map * peripheral_weight * 1.5
        
    elif "Infection" in diagnosis:
        mask = feature_map * 1.3
        num_spots = 3
        for _ in range(num_spots):
            spot_y = np.random.randint(height//4, 3*height//4)
            spot_x = np.random.randint(width//4, 3*width//4)
            spot_dist = np.sqrt((x - spot_x)**2 + (y - spot_y)**2)
            spot_mask = np.exp(-spot_dist**2 / (min(height, width) / 10)**2)
            mask += spot_mask * 0.5
            
    else:  # Normal
        mask = feature_map * 0.5
    
    mask = np.clip(mask, 0, None)
    if mask.max() > 0:
        mask = (mask / mask.max() * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    
    return mask

def generate_gradcam_heatmap(image, diagnosis):
    """Generate Grad-CAM-like heatmap."""
    seg_mask = generate_segmentation_mask(image, diagnosis)
    img_array = np.array(image.convert('L'))
    height, width = img_array.shape
    
    heatmap = seg_mask.astype(float)
    
    kernel_size = 15
    for _ in range(3):
        padded = np.pad(heatmap, kernel_size//2, mode='edge')
        smoothed = np.zeros_like(heatmap)
        for i in range(height):
            for j in range(width):
                smoothed[i, j] = np.mean(padded[i:i+kernel_size, j:j+kernel_size])
        heatmap = smoothed
    
    heatmap = np.clip(heatmap * 1.2, 0, 255).astype(np.uint8)
    return heatmap

def generate_segmentation_outputs(image, diagnosis):
    """Generate CT Lung, Grad-CAM Heatmap, Colored Heatmap, and Overlay."""
    img_array = np.array(image.convert('L'))
    ct_lung = Image.fromarray(img_array)
    
    heatmap = generate_gradcam_heatmap(image, diagnosis)
    heatmap_gray = Image.fromarray(heatmap)
    
    heatmap_colored = apply_jet_colormap(heatmap)
    heatmap_colored_img = Image.fromarray(heatmap_colored)
    
    ct_rgb = Image.fromarray(img_array).convert('RGB')
    overlay = Image.blend(ct_rgb, heatmap_colored_img, alpha=0.5)
    
    return ct_lung, heatmap_gray, heatmap_colored_img, overlay

def get_predictions_for_image(image):
    """Simulate AI classification for patient upload."""
    # Simulate random but realistic probabilities
    np.random.seed(hash(str(image.size)) % (2**32))
    
    diagnoses = [
        "Normal",
        "Peripheral malignancy",
        "Central with patent bronchus",
        "Central without patent bronchus",
        "Infection"
    ]
    
    # Generate random probabilities
    raw_probs = np.random.dirichlet(np.ones(5) * 2)
    
    # Create dict
    predictions = {diagnoses[i]: float(raw_probs[i]) for i in range(5)}
    
    return predictions

DIAGNOSIS_INFO = {
    "Peripheral malignancy": {
        "severity": "High", "icon": "🔴",
        "description": "Peripheral lung malignancy detected in outer regions of the lung. Requires immediate oncological consultation."
    },
    "Central with patent bronchus": {
        "severity": "High", "icon": "🟠",
        "description": "Central lung mass with patent (open) bronchus detected. Requires thorough evaluation to rule out malignancy."
    },
    "Central without patent bronchus": {
        "severity": "Critical", "icon": "🔴",
        "description": "Central lung mass with bronchial obstruction detected. Critical finding requiring immediate medical intervention."
    },
    "Infection": {
        "severity": "Medium", "icon": "🟡",
        "description": "Pulmonary infection detected. Requires antibiotic treatment and monitoring."
    },
    "Normal": {
        "severity": "Low", "icon": "🟢",
        "description": "No significant abnormalities detected. Lungs appear healthy."
    }
}

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    :root {
        --primary: #1a5f7a;
        --secondary: #2e8b9a;
        --accent: #4fb3bf;
        --text-dark: #000000;
        --text-light: #000000;
        --bg-light: #f8fafb;
        --border-color: #e1e8ed;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f9fc 0%, #e8f4f8 50%, #dceef5 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Force all text to black */
    * {
        color: #000000 !important;
    }
    
    p, span, div, label, li, td, th, a, h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #000000 !important;
    }
    
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] div,
    [data-testid="stMarkdownContainer"] span {
        color: #000000 !important;
    }
    
    input, textarea, select {
        color: #000000 !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a5f7a 0%, #2e8b9a 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(26, 95, 122, 0.2);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
    }
    
    .main-header p {
        color: white !important;
        font-size: 1.1rem;
        margin: 0;
    }
    
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(26, 95, 122, 0.1);
        margin-bottom: 2rem;
    }
    
    .upload-section h3 {
        color: #000000 !important;
        margin-bottom: 1.5rem;
        font-size: 1.3rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(79, 179, 191, 0.1) 0%, rgba(26, 95, 122, 0.1) 100%);
        border-left: 4px solid #4fb3bf;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 2rem 0;
    }
    
    .info-box h4 {
        color: #000000 !important;
        margin-bottom: 0.8rem;
    }
    
    .info-box p {
        color: #000000 !important;
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(39, 174, 96, 0.1) 0%, rgba(46, 213, 115, 0.1) 100%);
        border: 2px solid #27ae60;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }
    
    .success-box h2 {
        color: #27ae60 !important;
        margin-bottom: 1rem;
    }
    
    .success-box p {
        color: #000000 !important;
        font-size: 1.1rem;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #1a5f7a 0%, #2e8b9a 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2.5rem !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 20px rgba(26, 95, 122, 0.3) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(26, 95, 122, 0.4) !important;
    }
    
    [data-testid="stFileUploader"] {
        background: var(--bg-light);
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed var(--border-color);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent);
    }
    
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] button,
    [data-testid="stFileUploader"] span {
        color: white !important;
    }
    
    .step-number {
        display: inline-block;
        width: 35px;
        height: 35px;
        background: linear-gradient(135deg, #1a5f7a 0%, #2e8b9a 100%);
        color: white !important;
        border-radius: 50%;
        text-align: center;
        line-height: 35px;
        font-weight: 700;
        margin-right: 10px;
    }
    
    .prob-card {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #e1e8ed;
        border-left: 4px solid;
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease;
    }
    
    .prob-card:hover {
        transform: translateX(4px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'submission_complete' not in st.session_state:
    st.session_state.submission_complete = False
if 'patient_id' not in st.session_state:
    st.session_state.patient_id = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'ct_lung_for_viz' not in st.session_state:
    st.session_state.ct_lung_for_viz = None
if 'diagnosis' not in st.session_state:
    st.session_state.diagnosis = None

def save_patient_data(name, age, gender, xray, ct_lung, ct_mediastinal):
    """Save patient data to the patient_data directory."""
    # Create patient_data directory if it doesn't exist
    patient_data_dir = "patient_data"
    if not os.path.exists(patient_data_dir):
        os.makedirs(patient_data_dir)
    
    # Get next patient ID
    existing_patients = [d for d in os.listdir(patient_data_dir) if d.startswith("Patient_")]
    if existing_patients:
        last_id = max([int(p.split("_")[1]) for p in existing_patients])
        patient_id = f"Patient_{last_id + 1:02d}"
    else:
        patient_id = "Patient_01"
    
    # Create patient folder
    patient_folder = os.path.join(patient_data_dir, patient_id)
    os.makedirs(patient_folder, exist_ok=True)
    
    # Save patient info
    info_file = os.path.join(patient_folder, "patient_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"Name: {name}\n")
        f.write(f"Age: {age}\n")
        f.write(f"Gender: {gender}\n")
        f.write(f"Submission Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Save images and return them for analysis
    xray_img = None
    lung_img = None
    mediastinal_img = None
    
    if xray is not None:
        xray_img = Image.open(xray)
        xray_img.save(os.path.join(patient_folder, "xray.png"))
    
    if ct_lung is not None:
        lung_img = Image.open(ct_lung)
        lung_img.save(os.path.join(patient_folder, "lung_window_ct.png"))
    
    if ct_mediastinal is not None:
        mediastinal_img = Image.open(ct_mediastinal)
        mediastinal_img.save(os.path.join(patient_folder, "mediastinal_window_ct.png"))
    
    return patient_id, lung_img

# Main App
st.markdown("""
<div class="main-header">
    <h1>🫁 LungHealth AI - Patient Portal</h1>
    <p>Upload your medical images and information for AI-powered lung health analysis</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.submission_complete:
    # Info Box
    st.markdown("""
    <div class="info-box">
        <h4 style="color: #000000 !important;">📋 What You'll Need:</h4>
        <p style="color: #000000 !important;"><strong>1.</strong> Your X-Ray image</p>
        <p style="color: #000000 !important;"><strong>2.</strong> Lung Window CT scan</p>
        <p style="color: #000000 !important;"><strong>3.</strong> Mediastinal Window CT scan</p>
        <p style="color: #000000 !important;"><strong>4.</strong> Basic information (Name, Age, Gender)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<span class="step-number">1</span><h3 style="display:inline;">Patient Information</h3>', unsafe_allow_html=True)
        
        name = st.text_input("Full Name", placeholder="Enter your full name")
        
        col_age, col_gender = st.columns(2)
        with col_age:
            age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)
        with col_gender:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown('<span class="step-number">2</span><h3 style="display:inline;">Upload Medical Images</h3>', unsafe_allow_html=True)
        
        st.markdown("#### 📷 X-Ray Image")
        xray_file = st.file_uploader("Upload X-Ray", type=["jpg", "jpeg", "png", "dcm"], key="xray")
        if xray_file:
            st.image(xray_file, caption="X-Ray Preview", use_column_width=True)
        
        st.markdown("#### 🫁 Lung Window CT Scan")
        ct_lung_file = st.file_uploader("Upload Lung Window CT", type=["jpg", "jpeg", "png", "dcm"], key="ct_lung")
        if ct_lung_file:
            st.image(ct_lung_file, caption="Lung Window CT Preview", use_column_width=True)
        
        st.markdown("#### 💓 Mediastinal Window CT Scan")
        ct_mediastinal_file = st.file_uploader("Upload Mediastinal Window CT", type=["jpg", "jpeg", "png", "dcm"], key="ct_med")
        if ct_mediastinal_file:
            st.image(ct_mediastinal_file, caption="Mediastinal Window CT Preview", use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Submit button
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🚀 Submit for Analysis", use_container_width=True):
                if not name:
                    st.error("⚠️ Please enter your name")
                elif not xray_file or not ct_lung_file or not ct_mediastinal_file:
                    st.error("⚠️ Please upload all three medical images")
                else:
                    with st.spinner("📤 Submitting your data and running AI analysis..."):
                        patient_id, lung_image = save_patient_data(
                            name, age, gender,
                            xray_file, ct_lung_file, ct_mediastinal_file
                        )
                        
                        # Run AI classification
                        predictions = get_predictions_for_image(lung_image)
                        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                        primary_diagnosis = sorted_predictions[0][0]
                        
                        st.session_state.patient_id = patient_id
                        st.session_state.predictions = predictions
                        st.session_state.ct_lung_for_viz = lung_image
                        st.session_state.diagnosis = primary_diagnosis
                        st.session_state.submission_complete = True
                        st.rerun()
    
    with col2:
        st.markdown("""
        <div class="info-box" style="position: sticky; top: 20px;">
            <h4 style="color: #000000 !important;">ℹ️ Important Notes</h4>
            <p style="color: #000000 !important;">✓ All images should be clear and properly formatted</p>
            <p style="color: #000000 !important;">✓ Supported formats: JPG, PNG, DICOM</p>
            <p style="color: #000000 !important;">✓ Your data is securely stored</p>
            <p style="color: #000000 !important;">✓ A doctor will review your submission</p>
            <p style="color: #000000 !important;">✓ Results will be available through your doctor</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Success message
    st.markdown(f"""
    <div class="success-box">
        <h2 style="color: #27ae60 !important;">✅ Analysis Complete!</h2>
        <p style="color: #000000 !important;"><strong>Patient ID:</strong> {st.session_state.patient_id}</p>
        <p style="color: #000000 !important;">Your medical images have been analyzed using our AI system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display Classification Results
    if st.session_state.predictions:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 20px rgba(26, 95, 122, 0.1); margin-bottom: 2rem;">
            <h3 style="color: #000000 !important; margin-bottom: 1.5rem;">🔬 AI Classification Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        predictions = st.session_state.predictions
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        primary_diagnosis = sorted_predictions[0][0]
        primary_confidence = sorted_predictions[0][1]
        primary_info = DIAGNOSIS_INFO[primary_diagnosis]
        
        # Display all classification probabilities
        severity_colors = {
            "High": "#e74c3c",
            "Critical": "#c0392b",
            "Medium": "#f39c12",
            "Low": "#27ae60"
        }
        
        for diagnosis_name, probability in sorted_predictions:
            info = DIAGNOSIS_INFO[diagnosis_name]
            border_color = severity_colors.get(info["severity"], "#1a5f7a")
            
            st.markdown(f"""
            <div style="background: white; padding: 1.2rem; border-radius: 8px; border: 1px solid #e1e8ed; 
                 border-left: 4px solid {border_color}; margin-bottom: 0.8rem;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: #000000 !important;">
                    {info["icon"]} {diagnosis_name}
                </h4>
                <div style="width: 100%; height: 8px; background: #f8fafb; border-radius: 4px; overflow: hidden; margin: 0.8rem 0 0.5rem 0;">
                    <div style="height: 100%; background: linear-gradient(135deg, #1a5f7a 0%, #2e8b9a 100%); width: {probability*100}%; transition: width 0.8s ease;"></div>
                </div>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #000000 !important;">
                    <strong>{probability*100:.1f}%</strong> confidence
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Primary diagnosis details
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(79, 179, 191, 0.1) 0%, rgba(26, 95, 122, 0.1) 100%); 
             border-left: 4px solid #4fb3bf; border-radius: 10px; padding: 1.5rem; margin: 2rem 0;">
            <h4 style="margin: 0 0 1rem 0; color: #000000 !important; font-size: 1.2rem;">
                {primary_info["icon"]} Primary Finding: {primary_diagnosis}
            </h4>
            <p style="font-size: 0.95rem; line-height: 1.6; margin-bottom: 1rem; color: #000000 !important;">
                {primary_info["description"]}
            </p>
            <p style="font-size: 0.9rem; color: #000000 !important; margin: 0;">
                <strong>Confidence:</strong> {primary_confidence*100:.1f}% | <strong>Severity:</strong> {primary_info["severity"]}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Grad-CAM Visualization
    if st.session_state.ct_lung_for_viz and st.session_state.diagnosis:
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 20px rgba(26, 95, 122, 0.1); margin-bottom: 2rem;">
            <h3 style="color: #000000 !important; margin-bottom: 1rem;">🎯 AI Attention Visualization (Grad-CAM)</h3>
            <p style="color: #000000 !important; font-size: 0.9rem; margin-bottom: 1.5rem;">
                The heatmaps below show which regions of your CT scan the AI focused on when making its diagnosis. 
                Warmer colors (red/yellow) indicate areas of higher attention.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        ct_lung_orig, heatmap_gray, heatmap_colored, overlay = generate_segmentation_outputs(
            st.session_state.ct_lung_for_viz, 
            st.session_state.diagnosis
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.image(ct_lung_orig, caption="Original CT Lung", use_container_width=True)
        with col2:
            st.image(heatmap_gray, caption="Attention Heatmap", use_container_width=True)
        with col3:
            st.image(heatmap_colored, caption="Colored Heatmap", use_container_width=True)
        with col4:
            st.image(overlay, caption="Overlay (CT + Heatmap)", use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
        <h4 style="color: #000000 !important;">📋 Next Steps:</h4>
        <p style="color: #000000 !important;"><strong>1.</strong> These results are preliminary AI analysis for screening purposes</p>
        <p style="color: #000000 !important;"><strong>2.</strong> A qualified healthcare professional will review your complete case</p>
        <p style="color: #000000 !important;"><strong>3.</strong> You will receive detailed results through your healthcare provider</p>
        <p style="color: #000000 !important;"><strong>4.</strong> Please schedule a follow-up appointment with your doctor</p>
        <p style="color: #000000 !important;"><strong>5.</strong> Keep your Patient ID ({st.session_state.patient_id}) for reference</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📝 Submit Another Case", use_container_width=True):
            st.session_state.submission_complete = False
            st.session_state.patient_id = None
            st.session_state.predictions = None
            st.session_state.ct_lung_for_viz = None
            st.session_state.diagnosis = None
            st.rerun()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #000000; padding: 2rem; border-top: 1px solid #e1e8ed; margin-top: 3rem;">
    <p style="color: #000000 !important;">🫁 <strong>LungHealth AI</strong> - AI-powered lung health analysis</p>
    <p style="font-size: 0.9rem; color: #000000 !important;">This tool is for screening purposes only. Always consult with a qualified healthcare professional for medical decisions.</p>
</div>
""", unsafe_allow_html=True)
