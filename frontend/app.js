/* =========================================
   AI TEACHING ASSISTANT - APP JS
   Version: 3.0.0 (Agentic RAG)
========================================= */

// Automatically adapt API URL for both local development and cloud deployments
const API = (window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost")
    ? "http://127.0.0.1:8000"
    : (window.STUDYSYNC_API_URL || window.location.origin);

// Session ID for conversation memory (persists across questions in same tab)
const SESSION_ID = 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

// State
let currentMode = "default";
let currentLanguage = "english";
let lastContext = "";
let lastTopic = "general";
let attentionTimer = null;
let lastOcrText = "";

// Quiz session state
let quizSessionTotal = 0;      // total questions in current quiz
let quizSessionAnswered = 0;   // how many have been answered
let quizSessionCorrect = 0;    // how many correct

/* =========================================
   INITIALIZATION
========================================= */

document.addEventListener("DOMContentLoaded", () => {
    console.log("AI Teaching Assistant UI Initialized");
    loadFileList();
    loadDashboard();
    checkAttentionStatus();
});

/* =========================================
   UTILS
========================================= */

function parseMarkdownWithMath(content) {
    const placeholders = [];

    // Protect display math $$ ... $$
    let parsedContent = content.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
        placeholders.push(match);
        return `@@@MATH_PLACEHOLDER_${placeholders.length - 1}@@@`;
    });

    // Protect inline math $ ... $
    parsedContent = parsedContent.replace(/\$([^\$\n]+?)\$/g, (match) => {
        placeholders.push(match);
        return `@@@MATH_PLACEHOLDER_${placeholders.length - 1}@@@`;
    });

    // Run marked
    let html = typeof marked !== "undefined" ? marked.parse(parsedContent) : parsedContent;

    // Restore math blocks
    html = html.replace(/@@@MATH_PLACEHOLDER_(\d+)@@@/g, (match, index) => {
        return placeholders[parseInt(index, 10)];
    });

    return html;
}

function setResult(id, content, stateClass = "") {
    const el = document.getElementById(id);
    if (!el) return;

    if (id === "ocrResult") {
        el.className = `result ocr-result-box ${stateClass}`.trim();
    } else {
        el.className = `result ${stateClass}`.trim();
    }

    const isHtmlPlaceholder = typeof content === "string" && content.includes("ocr-placeholder-content");

    // Use marked for rich text / math
    if (stateClass === "state-success" || id === "chatResult") {
        if (isHtmlPlaceholder) {
            el.innerHTML = content;
        } else {
            el.innerHTML = parseMarkdownWithMath(content);
            if (window.MathJax) {
                MathJax.typesetPromise([el]);
            }
        }
    } else {
        if (isHtmlPlaceholder || (typeof content === "string" && content.startsWith("<div"))) {
            el.innerHTML = content;
        } else {
            el.textContent = content;
        }
    }

    el.classList.remove("state-loading", "state-success", "state-error");
    if (stateClass) el.classList.add(stateClass);
}

function setBusy(id, isBusy) {
    const el = document.getElementById(id);
    if (!el) return;

    if (isBusy) {
        el.disabled = true;
        el.classList.add("loading-pulse");
    } else {
        el.disabled = false;
        el.classList.remove("loading-pulse");
    }
}

function handleKeyPress(event) {
    if (event.key === "Enter") {
        sendChat();
    }
}

/* =========================================
   KNOWLEDGE BASE
========================================= */

async function uploadPDFs() {
    const fileInput = document.getElementById("pdfFiles");
    if (!fileInput.files.length) {
        setResult("pdfResult", "Please select files first.", "state-error");
        return;
    }

    setBusy("uploadBtn", true);
    setResult("pdfResult", "Uploading and indexing...", "state-loading");

    const formData = new FormData();
    for (const file of fileInput.files) {
        formData.append("files", file);
    }

    try {
        const response = await fetch(`${API}/upload_pdf/`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            setResult("pdfResult", data.message, "state-success");
            fileInput.value = "";
            loadFileList();
            loadDashboard(); // Update stats
        } else {
            setResult("pdfResult", data.detail || "Upload failed.", "state-error");
        }
    } catch (err) {
        setResult("pdfResult", "Could not reach backend.", "state-error");
    } finally {
        setBusy("uploadBtn", false);
    }
}

async function buildDB() {
    setBusy("buildBtn", true);
    setResult("pdfResult", "Building index from all files...", "state-loading");

    try {
        const response = await fetch(`${API}/build_db/`, { method: "POST" });
        const data = await response.json();

        if (response.ok) {
            setResult("pdfResult", data.message, "state-success");
            loadFileList();
            loadDashboard();
        } else {
            setResult("pdfResult", data.detail || "Rebuild failed.", "state-error");
        }
    } catch (err) {
        setResult("pdfResult", "Could not reach backend.", "state-error");
    } finally {
        setBusy("buildBtn", false);
    }
}

async function loadFileList() {
    const container = document.getElementById("fileListContainer");
    const section = document.getElementById("fileListSection");
    const headerStatus = document.getElementById("headerStatusText");

    try {
        const response = await fetch(`${API}/files/`);
        const data = await response.json();

        if (response.ok && data.files && data.files.length > 0) {
            section.style.display = "block";
            if (headerStatus) headerStatus.textContent = `${data.files.length} Document(s) Ready`;
            container.innerHTML = data.files.map(file => `
                <div class="file-item">
                    <span class="file-icon">📄</span>
                    <span class="file-name">${file.filename}</span>
                    <span class="file-meta">${file.pages} pages · ${file.chunks} chunks</span>
                </div>
            `).join("");
        } else {
            section.style.display = "none";
            if (headerStatus) headerStatus.textContent = "Upload Documents First";
        }
    } catch (err) {
        console.error("Failed to load file list:", err);
    }
}

/* =========================================
   OCR
========================================= */

async function extractText() {
    const imgInput = document.getElementById("imageFile");
    if (!imgInput.files.length) {
        setResult("ocrResult", "Select an image first.", "state-error");
        return;
    }

    setBusy("ocrBtn", true);
    setResult("ocrResult", "Running OCR...", "state-loading");

    // Hide the Index button on new search start
    const indexBtn = document.getElementById("addOcrToDbBtn");
    if (indexBtn) indexBtn.style.display = "none";

    const formData = new FormData();
    formData.append("image", imgInput.files[0]);

    try {
        const response = await fetch(`${API}/ocr/`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            setResult("ocrResult", data.detail || "OCR failed.", "state-error");
            return;
        }

        if (data.error) {
            setResult("ocrResult", `OCR failed: ${data.error}`, "state-error");
            return;
        }

        const output = data.text;
        if (!output || output.trim() === "") {
            setResult("ocrResult", "No text detected in the image.", "state-error");
            return;
        }

        lastOcrText = output;

        const metaParts = [];
        if (data.confidence) metaParts.push(`Confidence: ${data.confidence.toFixed(1)}%`);
        if (data.method) metaParts.push(`Method: ${data.method}`);
        if (data.processing_time_ms) metaParts.push(`${data.processing_time_ms.toFixed(0)}ms`);

        const metaLine = metaParts.length
            ? "\n\n---\n" + metaParts.join("  ·  ")
            : "";

        setResult("ocrResult", output + metaLine, "state-success");

        // Show the Index button on success
        const indexBtn = document.getElementById("addOcrToDbBtn");
        if (indexBtn) indexBtn.style.display = "inline-block";
    } catch (err) {
        setResult("ocrResult", "Could not reach backend.", "state-error");
    } finally {
        setBusy("ocrBtn", false);
    }
}

function copyToChat() {
    const chatInput = document.getElementById("chatInput");
    if (lastOcrText) {
        chatInput.value = lastOcrText;
        chatInput.focus();
    } else {
        const ocrResult = document.getElementById("ocrResult");
        const text = ocrResult ? ocrResult.textContent.trim() : "";
        if (text && !text.includes("OCR output and confidence score")) {
            chatInput.value = text;
            chatInput.focus();
        } else {
            alert("No OCR text available yet. Please select an image and click 'Extract Text & Math' first.");
        }
    }
}

async function addOcrToDatabase() {
    if (!lastOcrText || lastOcrText.trim() === "") {
        setResult("ocrResult", "No text to index. Run OCR first.", "state-error");
        return;
    }

    const imgInput = document.getElementById("imageFile");
    const filename = imgInput.files.length ? imgInput.files[0].name : "ocr_extracted_text.txt";

    setBusy("addOcrToDbBtn", true);
    setResult("ocrResult", "Adding text to Knowledge Base...", "state-loading");

    try {
        const response = await fetch(`${API}/index_text/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: lastOcrText,
                filename: filename
            })
        });

        const data = await response.json();

        if (response.ok) {
            setResult("ocrResult", `Successfully indexed!\n\n${data.message}\n\nYou can now ask StudySync AI questions about this slide!`, "state-success");

            const indexBtn = document.getElementById("addOcrToDbBtn");
            if (indexBtn) indexBtn.style.display = "none"; // Hide button once indexed

            loadFileList(); // Refresh the indexed files list
            loadDashboard(); // Refresh dashboard metrics
        } else {
            setResult("ocrResult", data.detail || "Indexing failed.", "state-error");
        }
    } catch (err) {
        setResult("ocrResult", "Could not reach backend.", "state-error");
    } finally {
        setBusy("addOcrToDbBtn", false);
    }
}

/* =========================================
   EXPLANATION MODES & LANGUAGE
========================================= */

function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll(".mode-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.mode === mode);
    });
}

function setLanguage(lang) {
    currentLanguage = lang;
    console.log("Language set to:", lang);
}

/* =========================================
   CHAT / RAG
========================================= */

async function sendChat() {
    const questionInput = document.getElementById("chatInput");
    const question = questionInput.value.trim();
    if (!question) return;

    setBusy("askBtn", true);
    setResult("chatResult", "⏳ Analyzing context and generating answer...", "state-loading");

    // Hide quiz area until new response comes in
    document.getElementById("quizActionArea").style.display = "none";
    document.getElementById("quizResult").style.display = "none";

    try {
        const response = await fetch(`${API}/chat/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                mode: currentMode,
                language: currentLanguage,
                session_id: SESSION_ID
            })
        });

        const data = await response.json();

        if (!response.ok) {
            setResult("chatResult", data.detail || "Failed to generate answer.", "state-error");
            return;
        }

        lastContext = data.context_text || "";
        lastTopic = data.source_details && data.source_details.length > 0
            ? data.source_details[0].subject
            : "general";

        const answerText = data.answer || "No answer returned.";

        // Build rich source citations
        let sourceMd = "";
        const sourceDetails = data.source_details || [];
        if (sourceDetails.length > 0) {
            const lines = sourceDetails.map(s => {
                const page = s.page ? ` (Page ${s.page})` : "";
                return `- 📄 **${s.filename}**${page}`;
            });
            const unique = [...new Set(lines)];
            sourceMd = "\n\n---\n### 📚 Sources\n" + unique.join("\n");
        }

        // Build Agentic RAG metadata footer
        let agentMd = "";
        const metaParts = [];

        // Mode and language
        const modeLabel = currentMode.charAt(0).toUpperCase() + currentMode.slice(1);
        const langLabel = currentLanguage.charAt(0).toUpperCase() + currentLanguage.slice(1);
        metaParts.push(`${modeLabel} mode`);
        metaParts.push(langLabel);

        // Timing
        if (data.processing_time_ms) {
            metaParts.push(`${(data.processing_time_ms / 1000).toFixed(1)}s`);
        }

        // Retrieval method
        if (data.retrieval_method) {
            metaParts.push(`🔍 ${data.retrieval_method === 'hybrid_bm25_vector' ? 'Hybrid Retrieval' : data.retrieval_method}`);
        }

        agentMd = `\n\n*${metaParts.join(' · ')}*`;

        // Query rewriting info
        if (data.rewritten_queries && data.rewritten_queries.length > 1) {
            agentMd += `\n*🔄 Query expanded into ${data.rewritten_queries.length} search queries*`;
        }

        // Retrieval retry info
        if (data.retrieval_attempts && data.retrieval_attempts > 1) {
            agentMd += `\n*🔁 Retrieval refined ${data.retrieval_attempts} time(s) for better results*`;
        }

        // Grounding verification badge
        if (data.grounding_verified === true) {
            const pct = data.grounding_score ? Math.round(data.grounding_score * 100) : 100;
            agentMd += `\n*🛡️ Answer verified against sources (${pct}% confidence)*`;
        } else if (data.grounding_verified === false) {
            agentMd += `\n*⚠️ Some claims could not be fully verified against sources*`;
        }

        setResult("chatResult", answerText + sourceMd + agentMd, "state-success");

        // Show Quiz button
        if (lastContext) {
            document.getElementById("quizActionArea").style.display = "block";
        }

        loadDashboard(); // Update stats
    } catch (err) {
        setResult("chatResult", "Could not reach backend.", "state-error");
    } finally {
        setBusy("askBtn", false);
    }
}

/* =========================================
   QUIZ LOGIC
========================================= */

async function generateQuiz() {
    const quizBtn = document.getElementById("genQuizBtn");
    const quizResult = document.getElementById("quizResult");

    setBusy("genQuizBtn", true);
    quizResult.style.display = "block";
    quizResult.innerHTML = "<p class='state-loading'>✨ Crafting custom quiz for you...</p>";

    try {
        const response = await fetch(`${API}/quiz/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                topic: lastTopic,
                context_text: lastContext,
                num_questions: 10,
                language: currentLanguage
            })
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            quizResult.innerHTML = `<p class='state-error'>${data.error || "Quiz generation failed."}</p>`;
            return;
        }

        // Reset quiz session counters
        quizSessionTotal = data.quiz.length;
        quizSessionAnswered = 0;
        quizSessionCorrect = 0;

        renderQuiz(data.quiz);
    } catch (err) {
        quizResult.innerHTML = "<p class='state-error'>Could not reach backend for quiz.</p>";
    } finally {
        setBusy("genQuizBtn", false);
    }
}

function renderQuiz(questions) {
    const container = document.getElementById("quizResult");

    let html = `<h3>🧠 Topic Check: ${lastTopic.replace(/_/g, ' ')}</h3>`;

    questions.forEach((q, idx) => {
        const qId = `q_${idx}`;
        // Render question text with markdown+math support
        const questionHtml = parseMarkdownWithMath(String(q.question || ""));
        html += `
            <div class="quiz-card" id="card_${qId}">
                <div class="quiz-question">${idx + 1}. ${questionHtml}</div>
                <div class="quiz-options">
                    ${q.options ? q.options.map((opt, oIdx) => {
            const isCorrect = opt.trim() === (q.answer || "").trim();
            const optHtml = parseMarkdownWithMath(String(opt || ""));
            return `<div class="quiz-option" data-correct="${isCorrect}" data-opt-idx="${oIdx}" onclick="selectOption('${qId}', ${oIdx}, '${(q.explanation || '').replace(/'/g, "\\'")}')">${optHtml}</div>`;
        }).join("") : `
                        <p class="muted">Short answer - think about it then check the answer.</p>
                        <button class="btn-outline" onclick="showShortAnswer('${qId}', '${(q.answer || '').replace(/'/g, "\\'")}', '${(q.explanation || '').replace(/'/g, "\\'")}')">Show Answer</button>
                    `}
                </div>
                <div id="feedback_${qId}" class="quiz-explanation" style="display: none;"></div>
            </div>
        `;
    });

    container.innerHTML = html;

    // Render LaTeX / math in the quiz
    if (window.MathJax) {
        MathJax.typesetPromise([container]);
    }
}

function selectOption(qId, oIdx, explanation) {
    const card = document.getElementById(`card_${qId}`);
    const options = card.querySelectorAll(".quiz-option");
    const feedback = document.getElementById(`feedback_${qId}`);

    // Prevent multiple selections
    if (card.dataset.answered === "true") return;
    card.dataset.answered = "true";

    let isCorrect = false;
    options.forEach((opt, idx) => {
        const correct = opt.dataset.correct === "true";
        if (idx === oIdx) {
            opt.classList.add("selected");
            if (correct) {
                opt.classList.add("correct");
                isCorrect = true;
            } else {
                opt.classList.add("incorrect");
            }
        }
        // Always highlight the correct option (even if not clicked)
        if (correct) {
            opt.classList.add("correct");
        }
    });

    feedback.style.display = "block";
    feedback.innerHTML = `<strong>${isCorrect ? "✅ Correct!" : "❌ Not quite."}</strong> ${explanation}`;

    // Track quiz session progress
    quizSessionAnswered++;
    if (isCorrect) quizSessionCorrect++;

    // When all questions answered, submit score for the whole quiz once
    if (quizSessionAnswered >= quizSessionTotal && quizSessionTotal > 0) {
        recordScore(quizSessionCorrect, quizSessionTotal);
    }
}

function showShortAnswer(qId, answer, explanation) {
    const feedback = document.getElementById(`feedback_${qId}`);
    feedback.style.display = "block";
    feedback.innerHTML = `<strong>Reference Answer:</strong> ${answer}<br><br><strong>Explanation:</strong> ${explanation}`;

    // Track quiz session progress (short answers count as correct review)
    quizSessionAnswered++;
    quizSessionCorrect++;

    if (quizSessionAnswered >= quizSessionTotal && quizSessionTotal > 0) {
        recordScore(quizSessionCorrect, quizSessionTotal);
    }
}

async function recordScore(score, total) {
    try {
        await fetch(`${API}/quiz/score/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                topic: lastTopic,
                score,
                total,
                quiz_type: "mcq"
            })
        });
        loadDashboard(); // Refresh stats
    } catch (err) {
        console.error("Score recording failed", err);
    }
}

/* =========================================
   DASHBOARD
========================================= */

async function loadDashboard() {
    try {
        const response = await fetch(`${API}/dashboard/`);
        const data = await response.json();

        if (!response.ok) return;

        // Stats
        document.getElementById("statQuestions").textContent = data.stats.total_questions;
        document.getElementById("statQuizzes").textContent = data.stats.total_quizzes;
        document.getElementById("statAvgScore").textContent = `${data.stats.average_quiz_score}%`;
        document.getElementById("statDocs").textContent = data.stats.documents_indexed;

        // Show last quiz score next to quizzes taken
        const lastScoreEl = document.getElementById("statLastQuizScore");
        if (lastScoreEl) {
            const ls = data.stats.last_quiz_score;
            lastScoreEl.textContent = ls ? `Last: ${ls}` : "";
        }

        // Topic Mastery
        const topicList = document.getElementById("topicList");
        if (data.topic_summary.length > 0) {
            topicList.innerHTML = data.topic_summary.map(t => `
                <div class="topic-item">
                    <div class="topic-info">
                        <strong>${t.topic.replace(/_/g, ' ')}</strong>
                        <span>${t.mastery_score}% mastery</span>
                    </div>
                    <div class="mastery-bar-bg">
                        <div class="mastery-bar-fill" style="width: ${t.mastery_score}%"></div>
                    </div>
                </div>
            `).join("");
        } else {
            topicList.innerHTML = "<p class='muted'>No data available yet.</p>";
        }

        // Activity
        const timeline = document.getElementById("activityTimeline");
        if (data.recent_activity.length > 0) {
            timeline.innerHTML = data.recent_activity.map(a => {
                const date = new Date(a.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                const icon = a.type === "question" ? "❓" : "📝";
                const label = a.type === "question" ? "Asked about" : "Took quiz on";
                return `
                    <div class="activity-item">
                        <div class="activity-time">${date}</div>
                        <div>${icon} ${label} <strong>${a.topic.replace(/_/g, ' ')}</strong></div>
                    </div>
                `;
            }).join("");
        } else {
            timeline.innerHTML = "<p class='muted'>No recent activity.</p>";
        }

    } catch (err) {
        console.error("Dashboard failed to load", err);
    }
}

/* =========================================
   ATTENTION TRACKING
========================================= */

async function startAttention() {
    try {
        const response = await fetch(`${API}/attention/start`, { method: "POST" });
        const data = await response.json();

        if (data.status === "started" || data.status === "already_active") {
            setResult("attentionResult", "Attention tracking started. Please look at the camera.", "state-success");

            // Poll for status
            if (attentionTimer) clearInterval(attentionTimer);
            attentionTimer = setInterval(checkAttentionStatus, 2000);

            // Try to start local video for feedback if on same machine
            startLocalVideo();
        }
    } catch (err) {
        setResult("attentionResult", "Failed to start attention tracker.", "state-error");
    }
}

async function stopAttention() {
    try {
        const response = await fetch(`${API}/attention/stop`, { method: "POST" });
        const data = await response.json();

        if (attentionTimer) clearInterval(attentionTimer);
        attentionTimer = null;

        const results = data.last_results || {};
        const score = results.attention_score || 0;
        setResult("attentionResult", `Session Stopped. Final Attention Score: ${score}%`, "state-success");

        stopLocalVideo();
    } catch (err) {
        setResult("attentionResult", "Failed to stop attention tracker.", "state-error");
    }
}

async function checkAttentionStatus() {
    try {
        const response = await fetch(`${API}/attention/status`);
        const data = await response.json();

        if (data.active) {
            const results = data.results || {};
            const score = results.attention_score || 0;
            const label = score > 70 ? "Focused" : "Distracted";
            document.getElementById("attentionResult").innerHTML = `
                <div style="font-size: 1.2rem; font-weight: 800; color: ${score > 70 ? 'green' : 'red'}">
                    ${label} (${score}%)
                </div>
                <div class="muted">Live monitoring active...</div>
            `;
        }
    } catch (err) {
        console.error("Failed to check attention status", err);
    }
}

function startLocalVideo() {
    const video = document.getElementById("attentionVideo");
    const placeholder = document.getElementById("videoPlaceholder");
    if (video) {
        // Stream the processed feed from FastAPI backend with cache buster
        video.src = `${API}/attention/video_feed?t=${Date.now()}`;
        video.style.display = "block";
    }
    if (placeholder) {
        placeholder.style.display = "none";
    }
}

function stopLocalVideo() {
    const video = document.getElementById("attentionVideo");
    const placeholder = document.getElementById("videoPlaceholder");
    if (video) {
        video.src = "";
        video.style.display = "none";
    }
    if (placeholder) {
        placeholder.style.display = "flex";
    }
}

/* =========================================
   STUDENT MICRO-INTERACTIONS & WORKSPACE
   ========================================= */

function fillPrompt(promptText) {
    const input = document.getElementById("chatInput");
    if (!input) return;

    input.value = promptText;
    input.focus();

    // Subtle micro-animation on prompt dock
    const dock = document.querySelector(".gemini-prompt-dock");
    if (dock) {
        dock.style.borderColor = "rgba(129, 140, 248, 0.8)";
        dock.style.boxShadow = "0 0 30px rgba(99, 102, 241, 0.4)";
        setTimeout(() => {
            dock.style.borderColor = "";
            dock.style.boxShadow = "";
        }, 600);
    }

    // Scroll smoothly to chat if needed
    const chatPanel = document.getElementById("chatPanel");
    if (chatPanel) {
        chatPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
}

function switchWorkspaceView(view) {
    document.querySelectorAll(".workspace-tab").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.view === view);
    });

    const panels = {
        chat: document.getElementById("chatPanel"),
        kb: document.getElementById("knowledgePanel"),
        ocr: document.getElementById("ocrPanel"),
        analytics: document.getElementById("dashboardPanel"),
        focus: document.getElementById("attentionPanel")
    };

    if (view === "all") {
        Object.values(panels).forEach(p => {
            if (p) p.classList.remove("workspace-view-hidden");
        });
    } else {
        Object.entries(panels).forEach(([key, p]) => {
            if (!p) return;
            if (key === view) {
                p.classList.remove("workspace-view-hidden");
                // If single view, give it wide layout
                p.style.gridColumn = "span 12";
            } else {
                p.classList.add("workspace-view-hidden");
                p.style.gridColumn = "";
            }
        });
    }
}

/* =========================================
   APPLICATION RESET LOGIC
   ========================================= */

async function resetApp() {
    const confirmed = confirm(
        "⚠️ RESET ALL DATA & INDEXED DOCUMENTS?\n\n" +
        "This will permanently:\n" +
        "• Delete all uploaded course documents and files\n" +
        "• Erase the ChromaDB vector index\n" +
        "• Reset all quiz scores, topic mastery, and study stats\n" +
        "• Clear conversation session memory\n\n" +
        "Are you sure you want to proceed?"
    );

    if (!confirmed) return;

    const resetBtns = [
        document.getElementById("resetAllBtn"),
        document.getElementById("headerResetBtn")
    ];

    resetBtns.forEach(btn => {
        if (btn) {
            btn.disabled = true;
            btn.dataset.originalText = btn.innerHTML;
            btn.innerHTML = "⏳ Resetting...";
        }
    });

    try {
        const response = await fetch(`${API}/reset/`, { method: "POST" });
        const data = await response.json();

        if (response.ok) {
            // 1. Clear text and file inputs
            const chatInput = document.getElementById("chatInput");
            if (chatInput) chatInput.value = "";

            const pdfFiles = document.getElementById("pdfFiles");
            if (pdfFiles) pdfFiles.value = "";

            const imgFile = document.getElementById("imageFile");
            if (imgFile) imgFile.value = "";

            // 2. Reset results panels
            setResult("pdfResult", "Knowledge base has been completely reset. Upload new files above to begin.", "state-success");
            setResult(
                "chatResult",
                "<div style='color: var(--text-muted); text-align: center; padding: 28px 12px;'>" +
                "<span style='font-size: 2.2rem; display: block; margin-bottom: 8px;'>📖</span>" +
                "System reset complete. Please upload course materials in Step 1 to begin asking questions." +
                "</div>"
            );
            lastOcrText = "";
            setResult("ocrResult", "<div class=\"ocr-placeholder-content\"><span class=\"ocr-placeholder-icon\">📷</span><span>OCR output and confidence score will appear here.</span></div>");
            const indexSlideBtn = document.getElementById("addOcrToDbBtn");
            if (indexSlideBtn) indexSlideBtn.style.display = "none";

            // 3. Hide quiz components
            const quizActionArea = document.getElementById("quizActionArea");
            if (quizActionArea) quizActionArea.style.display = "none";

            const quizResult = document.getElementById("quizResult");
            if (quizResult) {
                quizResult.style.display = "none";
                quizResult.innerHTML = "";
            }

            // 4. Reload file list and analytics dashboard
            await loadFileList();
            await loadDashboard();

            // 5. Update header status
            const headerStatus = document.getElementById("headerStatusText");
            if (headerStatus) headerStatus.textContent = "Upload Documents First";

            alert("✅ Reset Complete!\n\nAll indexed documents, vector stores, and analytics have been cleared.");
        } else {
            alert(`❌ Reset failed: ${data.detail || "Server error"}`);
        }
    } catch (err) {
        console.error("Reset request error:", err);
        alert("❌ Failed to reach backend server to perform reset.");
    } finally {
        resetBtns.forEach(btn => {
            if (btn && btn.dataset.originalText) {
                btn.disabled = false;
                btn.innerHTML = btn.dataset.originalText;
            }
        });
    }
}