/* ==========================================================================
   CinePulse Hollywood — Cinema & Gold Dark Mode Interactive Script
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // Preset reviews dataset
    const PRESETS = {
        positive1: "This movie is an absolute masterpiece of modern cinema! Brilliant direction, breathtaking visual effects, and a mesmerizing original score that keeps you hooked from start to finish. Highly recommended!",
        negative1: "Complete waste of time and money. The script makes zero sense, the character development is non-existent, and the acting is completely wooden. Easily one of the worst films of the decade.",
        mixed: "The movie has stunning action sequences and incredible cinematography, but the pacing drags terribly in the second act and the dialogue feels forced at times. A decent one-time watch.",
        positive2: "An unbelievable adrenaline rush! Superb storytelling paired with spectacular character performances. The climax had everyone in the theater cheering.",
        negative2: "Incredibly boring and dull. I fell asleep halfway through. Uninspired plot, terrible sound editing, and utterly predictable twist."
    };

    // Elements
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    
    const reviewInput = document.getElementById("review-input");
    const charCounter = document.getElementById("char-counter");
    const analyzeBtn = document.getElementById("analyze-btn");
    const clearBtn = document.getElementById("clear-btn");
    const presetBtns = document.querySelectorAll(".preset-btn");

    const placeholderView = document.getElementById("placeholder-view");
    const resultView = document.getElementById("result-view");
    const sentimentHero = document.getElementById("sentiment-hero");
    const sentimentIcon = document.getElementById("sentiment-icon");
    const sentimentResultText = document.getElementById("sentiment-result-text");
    const sentimentSubtext = document.getElementById("sentiment-subtext");
    const confidenceBadge = document.getElementById("confidence-badge");
    const gaugeFill = document.getElementById("gauge-fill");
    
    const posScoreVal = document.getElementById("pos-score-val");
    const negScoreVal = document.getElementById("neg-score-val");
    const confidenceVal = document.getElementById("confidence-val");
    const tokenPillsContainer = document.getElementById("token-pills-container");

    const batchInput = document.getElementById("batch-input");
    const batchAnalyzeBtn = document.getElementById("batch-analyze-btn");
    const batchResultsWrapper = document.getElementById("batch-results-wrapper");
    const batchTotalNum = document.getElementById("batch-total-num");
    const batchPosNum = document.getElementById("batch-pos-num");
    const batchNegNum = document.getElementById("batch-neg-num");
    const batchTableBody = document.getElementById("batch-table-body");

    // Pipeline steps
    const step1Code = document.getElementById("step1-code");
    const step2Code = document.getElementById("step2-code");
    const step3Code = document.getElementById("step3-code");
    const step4Code = document.getElementById("step4-code");

    // Check API Status on load
    checkApiHealth();

    // Tab Switching
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            
            btn.classList.add("active");
            const targetTab = btn.getAttribute("data-tab");
            document.getElementById(targetTab).classList.add("active");
        });
    });

    // Character Counter
    reviewInput.addEventListener("input", () => {
        charCounter.textContent = `${reviewInput.value.length} characters`;
    });

    // Clear Button
    clearBtn.addEventListener("click", () => {
        reviewInput.value = "";
        charCounter.textContent = "0 characters";
        placeholderView.classList.remove("hidden");
        resultView.classList.add("hidden");
        confidenceBadge.textContent = "Awaiting Input";
        confidenceBadge.className = "badge-gold";
    });

    // Preset Buttons
    presetBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const key = btn.getAttribute("data-preset");
            if (PRESETS[key]) {
                reviewInput.value = PRESETS[key];
                charCounter.textContent = `${reviewInput.value.length} characters`;
                runSinglePrediction(PRESETS[key]);
            }
        });
    });

    // Analyze Button Click
    analyzeBtn.addEventListener("click", () => {
        const text = reviewInput.value.trim();
        if (!text) {
            alert("Please enter a movie review script first!");
            return;
        }
        runSinglePrediction(text);
    });

    // Batch Analyze Click
    batchAnalyzeBtn.addEventListener("click", () => {
        const raw = batchInput.value.trim();
        if (!raw) {
            alert("Please enter at least one review script for batch screening.");
            return;
        }
        const reviews = raw.split("\n").map(r => r.trim()).filter(r => r.length > 0);
        runBatchPrediction(reviews);
    });

    // API Health Check
    async function checkApiHealth() {
        try {
            const res = await fetch("/api/health");
            const data = await res.json();
            if (data.status === "ok") {
                console.log("CinePulse Hollywood API connected smoothly!");
            }
        } catch (e) {
            console.warn("Backend server connection pending, using fallback engine.");
        }
    }

    // Run Single Review Prediction
    async function runSinglePrediction(text) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Evaluating Verdict...`;

        try {
            let data;
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });
            
            if (res.ok) {
                data = await res.json();
            } else {
                data = fallbackPredict(text);
            }

            renderPredictionResult(data);
            updatePipelineStepper(data);

        } catch (err) {
            console.warn("Network error, running fallback predictor:", err);
            const data = fallbackPredict(text);
            renderPredictionResult(data);
            updatePipelineStepper(data);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Evaluate Verdict`;
        }
    }

    // Render Prediction Output UI
    function renderPredictionResult(data) {
        placeholderView.classList.add("hidden");
        resultView.classList.remove("hidden");

        const isPositive = data.sentiment === "Positive";
        
        // Sentiment Hero Styling
        sentimentHero.className = `sentiment-hero ${isPositive ? 'positive' : 'negative'}`;
        sentimentIcon.className = `fa-solid ${isPositive ? 'fa-trophy' : 'fa-skull-crossbones'}`;
        sentimentResultText.textContent = isPositive ? "CRITICS PICK (POSITIVE)" : "BOX OFFICE FLOP (NEGATIVE)";
        sentimentSubtext.textContent = isPositive ? "Highly Recommended Cinematic Masterpiece" : "Negative Critical Response & Poor Review Rating";

        // Gauge Fill %
        const posPercent = data.positive_score;
        gaugeFill.style.width = `${posPercent}%`;

        // Scores
        posScoreVal.textContent = `${data.positive_score}%`;
        negScoreVal.textContent = `${data.negative_score}%`;
        confidenceVal.textContent = `${data.confidence}%`;

        // Badge
        confidenceBadge.textContent = `${data.confidence}% Confidence`;
        confidenceBadge.className = `badge-gold ${isPositive ? 'pos' : 'neg'}`;

        // Token Pills
        tokenPillsContainer.innerHTML = "";
        const matchedWords = data.matched_vocab_words || [];
        if (matchedWords.length === 0) {
            tokenPillsContainer.innerHTML = `<span style="color:var(--text-muted); font-size:0.8rem;">No exact 5000-vocabulary words matched</span>`;
        } else {
            matchedWords.forEach(w => {
                const pill = document.createElement("span");
                pill.className = "token-pill";
                pill.textContent = w;
                tokenPillsContainer.appendChild(pill);
            });
        }
    }

    // Update Preprocessing Pipeline Inspector
    function updatePipelineStepper(data) {
        if (!data.pipeline) return;
        
        step1Code.textContent = `Raw Text: "${data.raw_text}"`;
        step2Code.textContent = `Regex Cleaned: "${data.pipeline.step2_cleaned}"`;
        step3Code.textContent = `Stopwords Removed: "${data.pipeline.step3_no_stopwords}"`;
        step4Code.textContent = `TF-IDF Matrix -> RNN Output: Probability=${data.probability} (${data.sentiment})`;
    }

    // Batch Predictions
    async function runBatchPrediction(reviews) {
        batchAnalyzeBtn.disabled = true;
        batchAnalyzeBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Screening Batch...`;

        try {
            let results = [];
            const res = await fetch("/api/batch-predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ reviews })
            });

            if (res.ok) {
                results = await res.json();
            } else {
                results = reviews.map(r => fallbackPredict(r));
            }

            renderBatchTable(results);

        } catch (e) {
            console.warn("Batch API fallback:", e);
            const results = reviews.map(r => fallbackPredict(r));
            renderBatchTable(results);
        } finally {
            batchAnalyzeBtn.disabled = false;
            batchAnalyzeBtn.innerHTML = `<i class="fa-solid fa-clapperboard"></i> Screen Batch Reviews`;
        }
    }

    // Render Batch Table
    function renderBatchTable(results) {
        batchResultsWrapper.classList.remove("hidden");
        batchTableBody.innerHTML = "";

        let posCount = 0;
        let negCount = 0;

        results.forEach((item, index) => {
            if (item.sentiment === "Positive") posCount++;
            else negCount++;

            const tr = document.createElement("tr");
            const snippet = item.raw_text.length > 50 ? item.raw_text.substring(0, 50) + "..." : item.raw_text;
            const isPos = item.sentiment === "Positive";

            tr.innerHTML = `
                <td><strong>${index + 1}</strong></td>
                <td>"${snippet}"</td>
                <td><span class="sentiment-badge-table ${isPos ? 'pos' : 'neg'}">${isPos ? 'CRITICS PICK' : 'BOX OFFICE FLOP'}</span></td>
                <td>${item.confidence}%</td>
                <td><code>${item.matched_vocab_count || 0} words</code></td>
            `;
            batchTableBody.appendChild(tr);
        });

        batchTotalNum.textContent = results.length;
        batchPosNum.textContent = posCount;
        batchNegNum.textContent = negCount;
    }

    // Fallback Predictor Heuristic in JS if offline
    function fallbackPredict(text) {
        const lower = text.lower ? text.lower() : text.toLowerCase();
        const posKeywords = ["great", "good", "love", "awesome", "amazing", "excellent", "best", "wonderful", "fantastic", "masterpiece", "unbelievable", "superb"];
        const negKeywords = ["bad", "worst", "terrible", "awful", "horrible", "boring", "waste", "poor", "hate", "disappointing", "dull", "trash"];

        let posScore = 0;
        let negScore = 0;

        posKeywords.forEach(k => { if (lower.includes(k)) posScore += 2; });
        negKeywords.forEach(k => { if (lower.includes(k)) negScore += 2; });

        let prob = 0.85;
        if (posScore > negScore) prob = 0.88;
        else if (negScore > posScore) prob = 0.12;
        else prob = 0.50;

        const sentiment = prob >= 0.5 ? "Positive" : "Negative";
        const confidence = prob >= 0.5 ? prob * 100 : (1 - prob) * 100;
        
        const words = lower.replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(w => w.length > 0);
        const matched = words.filter(w => posKeywords.includes(w) || negKeywords.includes(w));

        return {
            raw_text: text,
            cleaned_text: lower,
            sentiment: sentiment,
            probability: prob,
            positive_score: Math.round(prob * 100),
            negative_score: Math.round((1 - prob) * 100),
            confidence: Math.round(confidence),
            matched_vocab_count: matched.length,
            matched_vocab_words: matched,
            pipeline: {
                step1_raw: text,
                step2_cleaned: lower,
                step3_no_stopwords: words.join(" "),
                token_list: words
            }
        }
    }
});
