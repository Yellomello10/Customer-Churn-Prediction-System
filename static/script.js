document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    
    // UI State Elements
    const placeholderState = document.getElementById('results-placeholder');
    const loadingState = document.getElementById('results-loading');
    const successState = document.getElementById('results-success');
    
    // Result Output Elements
    const gaugeProgress = document.querySelector('.gauge-progress');
    const gaugePercent = document.getElementById('gauge-percent');
    const riskBadge = document.getElementById('risk-tier-badge');
    const resStatus = document.getElementById('res-status');
    const resProbability = document.getElementById('res-probability');
    const insightsList = document.getElementById('insights-list');

    // Handle Form Submit
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Switch UI to loading state
        showState(loadingState);
        submitBtn.disabled = true;
        
        // Parse Form Data
        const formData = new FormData(form);
        const payload = {};
        
        // Extract features
        formData.forEach((value, key) => {
            // Convert numeric values
            if (key === 'tenure' || key === 'SeniorCitizen') {
                payload[key] = parseInt(value, 10);
            } else if (key === 'MonthlyCharges' || key === 'TotalCharges') {
                payload[key] = parseFloat(value);
            } else {
                payload[key] = value;
            }
        });
        
        try {
            // Post payload to backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                throw new Error('Prediction request failed on backend server.');
            }
            
            const result = await response.json();
            
            // Render Results
            renderResults(result);
            showState(successState);
            
        } catch (error) {
            console.error('Error:', error);
            alert(`An error occurred: ${error.message}`);
            showState(placeholderState);
        } finally {
            submitBtn.disabled = false;
        }
    });

    // Helper to toggle active visibility of result panel sections
    function showState(activeStateElement) {
        [placeholderState, loadingState, successState].forEach(el => {
            el.classList.remove('active');
        });
        activeStateElement.classList.add('active');
    }

    // Render results back onto UI widgets
    function renderResults(data) {
        const prob = data.churn_probability;
        const probPct = Math.round(prob * 100);
        
        // 1. Update Gauge Circle
        // Circumference of r=70 circle is ~440. 
        // offset = 440 - (pct * 440)
        const offset = 440 - (prob * 440);
        gaugeProgress.style.strokeDashoffset = offset;
        
        // Update text labels
        gaugePercent.textContent = `${probPct}%`;
        resProbability.textContent = `${(prob * 100).toFixed(2)}%`;
        
        if (data.churn_prediction === 1) {
            resStatus.textContent = 'Predicted to Churn (Yes)';
            resStatus.style.color = '#F87171'; // soft coral/red
        } else {
            resStatus.textContent = 'Predicted Active (No)';
            resStatus.style.color = '#2DD4BF'; // soft teal
        }
        
        // 2. Risk Badge Color classing
        riskBadge.textContent = data.risk_tier.toUpperCase();
        riskBadge.className = 'risk-badge'; // Reset classes
        
        if (data.risk_tier === 'High') {
            riskBadge.classList.add('badge-high');
            gaugeProgress.style.stroke = '#E63946'; // Red
        } else if (data.risk_tier === 'Medium') {
            riskBadge.classList.add('badge-medium');
            gaugeProgress.style.stroke = '#D97706'; // Gold
        } else {
            riskBadge.classList.add('badge-low');
            gaugeProgress.style.stroke = '#0D9488'; // Teal
        }
        
        // 3. Populate Actionable Insights
        insightsList.innerHTML = '';
        data.insights.forEach(insight => {
            const card = document.createElement('div');
            card.className = 'insight-card';
            card.textContent = insight;
            insightsList.appendChild(card);
        });
    }
});
