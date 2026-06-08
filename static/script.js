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

    // Simulator Elements
    const simContract = document.getElementById('sim-contract');
    const simPayment = document.getElementById('sim-payment');
    const simDiscount = document.getElementById('sim-discount');
    const simDiscountVal = document.getElementById('sim-discount-val');
    const simGaugePercent = document.getElementById('sim-gauge-percent');
    const simGaugeFill = document.querySelector('.sim-gauge-fill');
    const simRiskBadge = document.getElementById('sim-risk-badge');
    const simDiffText = document.getElementById('sim-diff-text');

    // Global variables to track baseline prediction state
    let baselinePayload = null;
    let baselineProbability = 0.0;

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
            
            // Save state for simulation baseline
            baselinePayload = { ...payload };
            baselineProbability = result.churn_probability;

            // Reset simulator UI controls to default baseline states
            simContract.value = 'current';
            simPayment.value = 'current';
            simDiscount.value = 0;
            simDiscountVal.textContent = '0%';

            // Render Results
            renderResults(result);
            renderSimulationResults(result.churn_probability, result.risk_tier, 0);
            showState(successState);
            
        } catch (error) {
            console.error('Error:', error);
            alert(`An error occurred: ${error.message}`);
            showState(placeholderState);
        } finally {
            submitBtn.disabled = false;
        }
    });

    // Handle simulator control interactions
    [simContract, simPayment].forEach(control => {
        control.addEventListener('change', runSimulation);
    });

    simDiscount.addEventListener('input', () => {
        simDiscountVal.textContent = `${simDiscount.value}%`;
    });
    simDiscount.addEventListener('change', runSimulation);

    async function runSimulation() {
        if (!baselinePayload) return;

        // Build simulated payload
        const simulatedPayload = { ...baselinePayload };

        if (simContract.value !== 'current') {
            simulatedPayload.Contract = simContract.value;
        }

        if (simPayment.value !== 'current') {
            simulatedPayload.PaymentMethod = simPayment.value;
        }

        const discountPct = parseFloat(simDiscount.value) / 100.0;
        if (discountPct > 0) {
            simulatedPayload.MonthlyCharges = baselinePayload.MonthlyCharges * (1.0 - discountPct);
        }

        try {
            // Dynamic check text during api request
            simDiffText.textContent = 'Recalculating simulation...';
            simDiffText.className = 'sim-diff-text text-neutral-glow';

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(simulatedPayload)
            });

            if (!response.ok) {
                throw new Error('Simulated prediction call failed.');
            }

            const data = await response.json();
            const simProb = data.churn_probability;
            const simRisk = data.risk_tier;

            renderSimulationResults(simProb, simRisk, simProb - baselineProbability);

        } catch (error) {
            console.error('Simulation Error:', error);
            simDiffText.textContent = 'Failed to evaluate simulated scenario.';
            simDiffText.className = 'sim-diff-text text-neutral-glow';
        }
    }

    // Render Simulation widgets
    function renderSimulationResults(prob, risk, diff) {
        const probPct = Math.round(prob * 100);
        
        // 1. Update Sim Circle (Circumference ~264, offset = 264 - (prob * 264))
        const offset = 264 - (prob * 264);
        simGaugeFill.style.strokeDashoffset = offset;
        simGaugePercent.textContent = `${probPct}%`;

        // 2. Class simulated badge
        simRiskBadge.textContent = risk.toUpperCase();
        simRiskBadge.className = 'sim-badge';

        if (risk === 'High') {
            simRiskBadge.classList.add('badge-high');
            simGaugeFill.style.stroke = 'var(--accent-red-border)';
        } else if (risk === 'Medium') {
            simRiskBadge.classList.add('badge-medium');
            simGaugeFill.style.stroke = 'var(--accent-yellow-border)';
        } else {
            simRiskBadge.classList.add('badge-low');
            simGaugeFill.style.stroke = 'var(--accent-green-border)';
        }

        // 3. Difference stats text
        if (diff < -0.005) {
            const pctDrop = Math.abs(diff) * 100;
            simDiffText.innerHTML = `<strong>Outcome:</strong> Risk score decreased by <span class="text-success-flat">${pctDrop.toFixed(1)}%</span>. Churn probability fell from ${Math.round(baselineProbability*100)}% down to ${probPct}%.`;
            simDiffText.className = 'sim-diff-text text-success-flat';
        } else if (diff > 0.005) {
            const pctRise = diff * 100;
            simDiffText.innerHTML = `<strong>Caution:</strong> Simulated configuration increases churn risk by ${pctRise.toFixed(1)}% (new score: ${probPct}%).`;
            simDiffText.className = 'sim-diff-text text-neutral-glow';
        } else {
            simDiffText.textContent = 'Baseline scenario. Slide or toggle values to check improvements.';
            simDiffText.className = 'sim-diff-text text-neutral-glow';
        }
    }

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
            resStatus.textContent = 'High Risk of Churn';
            resStatus.style.color = 'var(--accent-red-border)';
        } else {
            resStatus.textContent = 'Low Risk / Active Account';
            resStatus.style.color = 'var(--accent-green-border)';
        }
        
        // 2. Risk Badge Color classing
        riskBadge.textContent = data.risk_tier.toUpperCase();
        riskBadge.className = 'risk-badge'; // Reset classes
        
        if (data.risk_tier === 'High') {
            riskBadge.classList.add('badge-high');
            gaugeProgress.style.stroke = 'var(--accent-red-border)';
        } else if (data.risk_tier === 'Medium') {
            riskBadge.classList.add('badge-medium');
            gaugeProgress.style.stroke = 'var(--accent-yellow-border)';
        } else {
            riskBadge.classList.add('badge-low');
            gaugeProgress.style.stroke = 'var(--accent-green-border)';
        }
        
        // 3. Populate Actionable Insights as prioritized CRM tickets
        insightsList.innerHTML = '';
        data.insights.forEach(insight => {
            const card = document.createElement('div');
            card.className = 'insight-card';
            
            // Determine priority tag
            let priority = 'Low Priority';
            if (insight.startsWith('Contract') || insight.startsWith('Autopay')) {
                priority = 'High Priority';
            } else if (insight.startsWith('Fiber') || insight.startsWith('High Bill')) {
                priority = 'Medium Priority';
            }
            
            const tag = document.createElement('span');
            tag.className = 'insight-tag';
            tag.textContent = priority;
            
            // Apply styling colors to tags
            if (priority === 'High Priority') {
                tag.style.backgroundColor = 'var(--accent-red)';
                tag.style.borderColor = 'var(--border-dark)';
                tag.style.color = 'var(--text-primary)';
            } else if (priority === 'Medium Priority') {
                tag.style.backgroundColor = 'var(--accent-yellow)';
                tag.style.borderColor = 'var(--border-dark)';
                tag.style.color = 'var(--text-primary)';
            } else {
                tag.style.backgroundColor = 'var(--accent-green)';
                tag.style.borderColor = 'var(--border-dark)';
                tag.style.color = 'var(--text-primary)';
            }
            
            const textNode = document.createElement('span');
            textNode.textContent = insight;
            
            card.appendChild(tag);
            card.appendChild(textNode);
            insightsList.appendChild(card);
        });
    }
});
