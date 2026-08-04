let currentPage = 'home';
let patientData = {};
let predictionResult = null;
let chart = null;

// Page navigation functionality
function showPage(pageId) {
    // Hide all pages
    const pages = document.querySelectorAll('.page');
    pages.forEach(page => {
        page.classList.remove('active');
    });
    
    // Remove active class from all nav buttons
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected page
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
    }
    
    // Add active class to corresponding nav button (only for main nav pages)
    const pageIndex = ['home', 'main', 'result'].indexOf(pageId);
    if (pageIndex !== -1 && navButtons[pageIndex]) {
        navButtons[pageIndex].classList.add('active');
    }
    
    // Update current page
    currentPage = pageId;
}

// Form submission handler
async function handleFormSubmission(event) {
    event.preventDefault();
    
    const form = event.target;
    const loading = document.getElementById('loading');
    const errorMessage = document.getElementById('errorMessage');
    const submitButton = form.querySelector('button[type="submit"]');
    
    errorMessage.textContent = '';
    errorMessage.style.display = 'none';
    
    loading.style.display = 'block';
    submitButton.disabled = true;
    submitButton.textContent = 'Processing...';
    
    try {
        // Validate form
        if (!validateForm(form)) {
            throw new Error('Please fill in all required fields correctly.');
        }
        
        const formData = new FormData(form);

        patientData = Object.fromEntries(formData.entries());
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            predictionResult = result;
            displayResults(result);
            showPage('result');
        } else {
            throw new Error(result.error || 'Prediction failed');
        }
        
    } catch (error) {
        console.error('Prediction error:', error);
        errorMessage.textContent = error.message;
        errorMessage.style.display = 'block';
        errorMessage.style.color = '#e74c3c';
        errorMessage.style.textAlign = 'center';
        errorMessage.style.padding = '15px';
        errorMessage.style.backgroundColor = '#ffeaea';
        errorMessage.style.border = '1px solid #e74c3c';
        errorMessage.style.borderRadius = '8px';
        errorMessage.style.marginBottom = '20px';
    } finally {
        // Hide loading spinner
        loading.style.display = 'none';
        submitButton.disabled = false;
        submitButton.textContent = 'Predict CKD Risk';
    }
}

// Form validation
function validateForm(form) {
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            field.style.borderColor = '#e74c3c';
            isValid = false;
        } else {
            field.style.borderColor = '#e1e8ed';
        }
    });
    
    return isValid;
}

// Display prediction results with sidebar patient summary
function displayResults(result) {
    const resultContent = document.getElementById('resultContent');
    
    // Handle different result formats
    let isPositive, confidence, engineeredFeatures = {}, currentPatientData;
    
    if (result.prediction) {
        isPositive = result.prediction === 'ckd';
        confidence = (result.confidence * 100).toFixed(1);
        engineeredFeatures = result.engineered_features || {};
        currentPatientData = result.patient_data || patientData;
    } else {
        isPositive = result.hasCKD || predictionResult?.hasCKD;
        confidence = ((result.confidence || predictionResult?.confidence || 0) * 100).toFixed(1);
        currentPatientData = patientData;
    }
    
    // Main result content
    let resultHTML = '';
    
    if (isPositive) {
        resultHTML = `
            <div class="result-card">
                <h2>⚠️ High Risk of CKD Detected</h2>
                <p style="font-size: 1.2em; margin-bottom: 15px;">
                    The analysis indicates a high probability of Chronic Kidney Disease. Immediate medical consultation is recommended.
                </p>
                <p style="font-size: 1.1em;">
                    Confidence Level: <strong>${confidence}%</strong>
                </p>
            </div>
        `;
    } else {
        resultHTML = `
            <div class="result-card negative">
                <h2>✅ Low Risk of CKD</h2>
                <p style="font-size: 1.2em; margin-bottom: 15px;">
                    The analysis suggests a low probability of Chronic Kidney Disease. Continue regular health monitoring.
                </p>
                <p style="font-size: 1.1em;">
                    Confidence Level: <strong>${confidence}%</strong>
                </p>
            </div>
        `;
    }
    
    // Adding the engineered features as risk factors
    if (Object.keys(engineeredFeatures).length > 0) {
        let riskFactorsHTML = '<div class="next-steps" style="margin-top: 25px; border-left-color: #e74c3c;"><h3>⚠️ Clinical Risk Assessment</h3><ul>';
        
        // eGFR
        if (engineeredFeatures.eGFR !== undefined) {
            const egfr = engineeredFeatures.eGFR;
            if (egfr < 60) {
                riskFactorsHTML += `<li><strong>Reduced Kidney Function:</strong> eGFR = ${egfr.toFixed(1)} mL/min/1.73m² (Normal: >90)</li>`;
            } else if (egfr < 90) {
                riskFactorsHTML += `<li><strong>Mildly Reduced eGFR:</strong> ${egfr.toFixed(1)} mL/min/1.73m² (Consider monitoring, normal: >90)</li>`;
            } else {
                riskFactorsHTML += `<li><strong>Normal Kidney Function:</strong> eGFR = ${egfr.toFixed(1)} mL/min/1.73m² (Normal: >90)</li>`;
            }
        }
        
        // Comorbidity score
        if (engineeredFeatures.comorb_score !== undefined) {
            const score = engineeredFeatures.comorb_score;
            if (score >= 2) {
                riskFactorsHTML += `<li><strong>High Comorbidity Burden:</strong> ${score}/3 major conditions present (Hypertension, Diabetes, CAD)</li>`;
            } else if (score === 1) {
                riskFactorsHTML += `<li><strong>Moderate Risk:</strong> ${score}/3 major comorbid condition present</li>`;
            } else {
                riskFactorsHTML += `<li><strong>No Major Comorbidities:</strong> Low cardiovascular risk profile</li>`;
            }
        }
        
        // Anemia severity
        if (engineeredFeatures.anemia_severity !== undefined) {
            const severity = engineeredFeatures.anemia_severity;
            if (severity > 0.5) {
                riskFactorsHTML += `<li><strong>Severe Anemia:</strong> Multiple blood parameters below normal (Score: ${severity.toFixed(2)}, <0.2 is normal)</li>`;
            } else if (severity > 0.2) {
                riskFactorsHTML += `<li><strong>Mild to Moderate Anemia:</strong> Some blood parameters reduced (Score: ${severity.toFixed(2)}, <0.2 is normal)</li>`;
            } else {
                riskFactorsHTML += `<li><strong>Normal Blood Parameters:</strong> No significant anemia detected (Score: ${severity.toFixed(2)}, <0.2 is normal)</li>`;
            }
        }
        
        // Kidney function score
        if (engineeredFeatures.kidney_func_score !== undefined) {
            const kfScore = engineeredFeatures.kidney_func_score;
            if (kfScore > 1.0) {
                riskFactorsHTML += `<li><strong>Significant Kidney Dysfunction:</strong> Multiple kidney markers abnormal (Score: ${kfScore.toFixed(2)}, <0.3 is normal)</li>`;
            } else if (kfScore > 0.3) {
                riskFactorsHTML += `<li><strong>Mild Kidney Impairment:</strong> Some kidney markers elevated (Score: ${kfScore.toFixed(2)}, <0.3 is normal)</li>`;
            } else {
                riskFactorsHTML += `<li><strong>Normal Kidney Markers:</strong> Kidney function parameters within range (Score: ${kfScore.toFixed(2)}, <0.3 is normal)</li>`;
            }
        }
        
        // Symptom severity
        if (engineeredFeatures.symptom_severity !== undefined) {
            const symScore = engineeredFeatures.symptom_severity;
            if (symScore >= 2) {
                riskFactorsHTML += `<li><strong>High Symptom Burden:</strong> ${symScore}/3 CKD-related symptoms present</li>`;
            } else if (symScore === 1) {
                riskFactorsHTML += `<li><strong>Mild Symptoms:</strong> ${symScore}/3 CKD-related symptom present</li>`;
            } else {
                riskFactorsHTML += `<li><strong>Asymptomatic:</strong> No major CKD symptoms reported</li>`;
            }
        }
        
        riskFactorsHTML += '</ul></div>';
        resultHTML += riskFactorsHTML;
    }
    
    // Next steps
    if (isPositive) {
        resultHTML += `
            <div class="next-steps">
                <h3>Immediate Action Required</h3>
                <ul>
                    <li>Schedule an urgent appointment with a nephrologist</li>
                    <li>Conduct additional kidney function tests (proteinuria)</li>
                    <li>Review and adjust current medications</li>
                    <li>Implement dietary restrictions (low sodium, protein management)</li>
                    <li>Monitor blood pressure and blood sugar levels daily</li>
                    <li>Consider lifestyle modifications (exercise, weight management)</li>
                </ul>
            </div>
        `;
    } else {
        resultHTML += `
            <div class="next-steps">
                <h3>🌟 Recommended Next Steps</h3>
                <ul>
                    <li>Continue regular health check-ups every 6-12 months</li>
                    <li>Maintain a healthy diet and regular exercise routine</li>
                    <li>Monitor blood pressure and maintain healthy levels</li>
                    <li>Stay hydrated and avoid excessive protein intake</li>
                    <li>Avoid nephrotoxic medications without medical supervision</li>
                    <li>Report any symptoms like swelling, fatigue, or changes in urination</li>
                </ul>
            </div>
        `;
    }
    
    // Disclaimer and action buttons
    resultHTML += `
        <div style="background: rgba(255, 243, 205, 0.1); border: 1px solid rgba(255, 234, 167, 0.3); border-radius: 8px; padding: 20px; margin-top: 25px;">
            <h4 style="color: #FFD54F; margin-bottom: 10px;">⚠️ Important Disclaimer</h4>
            <p style="color: #FFD54F; font-size: 0.95em; line-height: 1.6; margin: 0;">
                This prediction is based on machine learning analysis and should not replace professional medical advice. 
                Always consult with qualified healthcare professionals for proper diagnosis and treatment decisions.
            </p>
        </div>
        
        <div class="action-buttons">
            <button class="secondary-btn" onclick="resetApplication()">
                New Prediction
            </button>
            <button class="exit-btn" onclick="exitApplication()">
                Exit System
            </button>
        </div>
    `;
    
    // Patient summary for sidebar
    const labels = {
        age: 'Age',
        bp: 'Blood Pressure',
        sg: 'Specific Gravity',
        al: 'Albumin Level',
        su: 'Sugar Level',
        rbc: 'RBC in Urine',
        pc: 'Pus Cells',
        pcc: 'Pus Cell Clumps',
        ba: 'Bacteria',
        bgr: 'Blood Glucose',
        bu: 'Blood Urea',
        sc: 'Serum Creatinine',
        sod: 'Sodium',
        pot: 'Potassium',
        hemo: 'Hemoglobin',
        pcv: 'PCV',
        wc: 'WBC Count',
        rc: 'RBC Count',
        htn: 'Hypertension',
        dm: 'Diabetes',
        cad: 'CAD',
        appet: 'Appetite',
        pe: 'Pedal Edema',
        ane: 'Anemia'
    };
    
    const units = {
        age: ' years',
        bp: ' mm/Hg',
        bgr: ' mg/dL',
        bu: ' mg/dL',
        sc: ' mg/dL',
        sod: ' mEq/L',
        pot: ' mEq/L',
        hemo: ' g/dL',
        pcv: '%',
        wc: ' cells/cumm',
        rc: ' millions/cmm'
    };
    
    const sections = {
        'Patient Information': ['age', 'bp'],
        'Urine Tests': ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba'],
        'Blood Tests': ['bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'],
        'Medical History': ['htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    };
    
    let summaryHTML = `<h3>Patient Summary</h3>`;
    
    for (const [sectionTitle, keys] of Object.entries(sections)) {
        summaryHTML += `<h4>${sectionTitle}</h4><div class="sidebar-summary-grid">`;
    
        keys.forEach(key => {
            const label = labels[key] || key;
            const value = (key === 'al' || key === 'su') && currentPatientData[key] === 0 ? 0 : currentPatientData[key] || 'N/A';
            summaryHTML += `
                <div class="sidebar-summary-item">
                    <span class="sidebar-summary-label">${label}</span>
                    <span class="sidebar-summary-value">${value}${units[key] || ''}</span>
                </div>
            `;
        });
    
        summaryHTML += `</div>`;
    }
    
    // Update DOM
    resultContent.innerHTML = resultHTML;

    // If not already created, add the sidebar dynamically
    if (!document.querySelector('.patient-sidebar')) {
        const sidebar = document.createElement('div');
        sidebar.className = 'patient-sidebar';
        sidebar.id = 'patientSummary';
        document.querySelector('.results-container').appendChild(sidebar);
    }

    // Update the content inside it
    const patientSidebar = document.getElementById('patientSummary');
    patientSidebar.innerHTML = summaryHTML;

    // Update grid layout dynamically
    document.querySelector('.results-container').style.display = 'grid';
    document.querySelector('.results-container').style.gridTemplateColumns = '1fr 350px';
}

// Helper function to get variable labels
function getVariableLabel(variable) {
    const labels = {
        'sc': 'Serum Creatinine (mg/dL)',
        'bu': 'Blood Urea (mg/dL)',
        'hemo': 'Hemoglobin (gms)',
        'eGFR': 'eGFR (mL/min/1.73m²)',
        'anemia_severity': 'Anemia Severity'
    };
    return labels[variable] || variable;
}

// Reset application (add patient sidebar reset)
function resetApplication() {
    // Reset form
    const form = document.getElementById('predictionForm');
    if (form) {
        form.reset();
        
        const fields = form.querySelectorAll('input, select');
        fields.forEach(field => {
            field.style.borderColor = '#e1e8ed';
        });
    }
    
    // Clear error messages
    const errorMessage = document.getElementById('errorMessage');
    if (errorMessage) {
        errorMessage.style.display = 'none';
        errorMessage.textContent = '';
    }
    
    // Reset global variables
    patientData = {};
    predictionResult = null;
    
    // Reset result content and sidebar
    const resultContent = document.getElementById('resultContent');
    const patientSummary = document.getElementById('patientSummary');
    
    if (resultContent) {
        resultContent.innerHTML = `
            <p style="text-align: center; color: #B0BEC5; font-size: 1.1em">
                Please complete the prediction form first to see results here.
            </p>
        `;
    }
    
    if (patientSummary) {
        patientSummary.innerHTML = '';
    }

    const sidebar = document.querySelector('.patient-sidebar');
    if (sidebar) sidebar.remove();

    document.querySelector('.results-container').style.display = '';
    document.querySelector('.results-container').style.gridTemplateColumns = '';

    // Go back to main page
    showPage('main');
}

function exitApplication() {
    showPage('thankyou');
}

// Form validation helpers
function validateNumericalInput(input, min, max) {
    const value = parseFloat(input.value);
    if (isNaN(value) || value < min || value > max) {
        input.style.borderColor = '#e74c3c';
        return false;
    }
    input.style.borderColor = '#27ae60';
    return true;
}

// Smooth scrolling functionality
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Handle form submission with smooth scrolling
function handleFormSubmissionWithScroll(event) {
    scrollToTop();
    handleFormSubmission(event);
}


let charts = {}; // Store all chart instances

async function initializeChart(chartType, canvasId, options = {}) {    
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`[CHART] Canvas element '${canvasId}' not found!`);
        return;
    }
    
    try {
        // Fetch the data from Flask backend
        const response = await fetch(`/get_chart_data/${chartType}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Check if data has error
        if (data.error) {
            throw new Error(`Server error: ${data.error}`);
        }
        
        // Validate data structure
        if (!Array.isArray(data) || data.length === 0) {
            throw new Error(`Invalid data format: expected non-empty array, got ${typeof data} with length ${data.length}`);
        }
        
        // Check if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.error('[CHART] Chart.js library not found!');
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#fff';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Chart library not found.', canvas.width/2, canvas.height/2);
            return;
        }
        
        const ctx = canvas.getContext('2d');
        
        // Process data based on chart type
        let chartConfig;
        
        switch (chartType) {
            case 'sc_egfr':
                chartConfig = createScatterConfig(data, 'sc', 'eGFR', 
                    'Serum Creatinine vs. eGFR by CKD Status',
                    'Serum Creatinine (mg/dL)', 
                    'eGFR (mL/min/1.73m²)', 
                    options);
                break;
                
            case 'bu_anemia':
                chartConfig = createScatterConfig(data, 'bu', 'anemia_severity',
                    'Blood Urea vs. Anemia Severity by CKD Status',
                    'Blood Urea (mg/dL)',
                    'Anemia Severity',
                    options);
                break;
                
            case 'pairplot':
                // For pairplot, we need to specify which variables to plot
                const xVar = options.xVar || 'sc';
                const yVar = options.yVar || 'bu';
                const xLabel = options.xLabel || getVariableLabel(xVar);
                const yLabel = options.yLabel || getVariableLabel(yVar);
                
                chartConfig = createScatterConfig(data, xVar, yVar,
                    `${xLabel} vs ${yLabel}`,
                    xLabel,
                    yLabel,
                    options);
                break;
                
            default:
                throw new Error(`Unsupported chart type: ${chartType}`);
        }
        
        // Destroy existing chart if it exists
        if (charts[canvasId]) {
            charts[canvasId].destroy();
        }
        
        // Create the chart
        charts[canvasId] = new Chart(ctx, chartConfig);
        
    } catch (error) {
        console.error(`[CHART] Error during ${chartType} chart initialization:`, {
            message: error.message,
            stack: error.stack,
            timestamp: new Date().toISOString()
        });
        
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#fff';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`${chartType} Chart Error: ${error.message}`, canvas.width/2, canvas.height/2);
    }
}

// Helper function to create scatter plot configuration
function createScatterConfig(data, xVar, yVar, title, xLabel, yLabel, options = {}) {
    // Filter and process data for CKD and non-CKD patients
    const ckdData = data.filter(item => {
        const isCkd = item.classification === 'ckd' || item.classification === 'CKD' || item.classification === 1;
        const hasValidData = item[xVar] !== null && item[yVar] !== null && 
                           !isNaN(item[xVar]) && !isNaN(item[yVar]);
        return isCkd && hasValidData;
    }).map(item => ({x: parseFloat(item[xVar]), y: parseFloat(item[yVar])}));
    
    const nonCkdData = data.filter(item => {
        const isNonCkd = item.classification === 'notckd' || item.classification === 'NOTCKD' || item.classification === 0;
        const hasValidData = item[xVar] !== null && item[yVar] !== null && 
                           !isNaN(item[xVar]) && !isNaN(item[yVar]);
        return isNonCkd && hasValidData;
    }).map(item => ({x: parseFloat(item[xVar]), y: parseFloat(item[yVar])}));
    
    
    // Calculate dynamic scales with fallback values
    const allXValues = [...ckdData.map(d => d.x), ...nonCkdData.map(d => d.x)];
    const allYValues = [...ckdData.map(d => d.y), ...nonCkdData.map(d => d.y)];

    // Handle empty data cases with fallback ranges
    let xMin, xMax, yMin, yMax;

    if (allXValues.length === 0) {
        xMin = 0;
        xMax = 10;
    } else {
        xMin = Math.min(...allXValues);
        xMax = Math.max(...allXValues);
        // Handle case where min equals max
        if (xMin === xMax) {
            xMin = xMin - 1;
            xMax = xMax + 1;
        }
    }

    if (allYValues.length === 0) {
        yMin = 0;
        yMax = 100;
    } else {
        yMin = Math.min(...allYValues);
        yMax = Math.max(...allYValues);
        // Handle case where min equals max
        if (yMin === yMax) {
            yMin = yMin - 5;
            yMax = yMax + 5;
        }
    }

    // Calculate padding based on data range with fallbacks
    const xRange = Math.abs(xMax - xMin);
    const yRange = Math.abs(yMax - yMin);
    const xPadding = options.xPadding || Math.max(xRange * 0.1, 0.5);
    const yPadding = options.yPadding || Math.max(yRange * 0.1, 5);
    
    // Create chart configuration
    const chartData = {
        datasets: [{
            label: 'CKD Patients',
            data: ckdData,
            backgroundColor: 'rgba(231, 76, 60, 0.6)',
            borderColor: 'rgba(231, 76, 60, 1)',
            pointRadius: options.pointRadius || 5,
            pointHoverRadius: options.pointHoverRadius || 7
        }, {
            label: 'Non-CKD Patients',
            data: nonCkdData,
            backgroundColor: 'rgba(52, 152, 219, 0.6)',
            borderColor: 'rgba(52, 152, 219, 1)',
            pointRadius: options.pointRadius || 5,
            pointHoverRadius: options.pointHoverRadius || 7
        }]
    };
    
    const config = {
        type: 'scatter',
        data: chartData,
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: title,
                    color: '#fff',
                    font: {
                        size: options.titleSize || 16
                    }
                },
                legend: {
                    labels: {
                        color: '#fff'
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    min: Math.max(xMin - xPadding, -1000), // Prevent negative infinity
                    max: Math.min(xMax + xPadding, 1000),  // Prevent positive infinity
                    title: {
                        display: true,
                        text: xLabel,
                        color: '#fff'
                    },
                    ticks: {
                        color: '#fff'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    min: Math.max(yMin - yPadding, -1000),
                    max: Math.min(yMax + yPadding, 1000),
                    title: {
                        display: true,
                        text: yLabel,
                        color: '#fff'
                    },
                    ticks: {
                        color: '#fff'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'point'
            }
        }
    };
    
    return config;
}


async function initializeHomeChart() {
    return await initializeChart('sc_egfr', 'homePerformanceChart');
}

async function initializeBuAnemiaChart() {
    return await initializeChart('bu_anemia', 'buAnemiaChart');
}

async function initializePairplotCharts() {
    const pairs = [
        {canvasId: 'scHemoChart', xVar: 'sc', yVar: 'hemo'},
        {canvasId: 'buHemoChart', xVar: 'bu', yVar: 'hemo'}
    ];
    
    try {
        for (const pair of pairs) {
            await initializeChart('pairplot', pair.canvasId, {
                xVar: pair.xVar,
                yVar: pair.yVar,
                pointRadius: 4,
                pointHoverRadius: 6,
                titleSize: 14
            });
        }
    } catch (error) {
        console.error('[CHART] Error during pairplot initialization:', error);
    }
}


// DOM Content Loaded Event Handler
document.addEventListener('DOMContentLoaded', function() {
    // Initialize application
    showPage('home');
    
    // Set up form submission handler
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', handleFormSubmission);
    }
    
    // Initialize chart if about page is active
    if (document.getElementById('home') && document.getElementById('home').classList.contains('active')) {
        initializeHomeChart();
        initializeBuAnemiaChart();
        initializePairplotCharts();
    }
    
    // Set up real-time validation for numerical inputs
    const numericalInputs = document.querySelectorAll('input[type="number"]');
    numericalInputs.forEach(input => {
        input.addEventListener('blur', function() {
            const min = parseFloat(this.getAttribute('min'));
            const max = parseFloat(this.getAttribute('max'));
            validateNumericalInput(this, min, max);
        });
        
        input.addEventListener('input', function() {
            // Reset border color when user starts typing
            this.style.borderColor = '#e1e8ed';
        });
    });
});