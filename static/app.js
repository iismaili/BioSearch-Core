const state = {
  stage1: {
    imageData: null,
    fileName: null,
    result: null,
    referenceVector: null,
    referenceCropB64: null
  },
  stage2: {
    imageData: null,
    fileName: null,
    result: null,
    threshold: 0.363
  },
  samples: []
};

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initModals();
  initDropzones();
  initThresholdSlider();
  initButtons();
  loadSampleImages();
});

function initTabs() {
  const tabBtns = document.querySelectorAll(".nav-tab-btn");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-target");
      if (targetId === "tab-stage2" && !state.stage1.referenceVector) {
        showToast("Lütfen önce 1. aşamada bir kişinin görüntüsünü analiz edin.", "error");
        return;
      }
      switchTab(targetId);
    });
  });

  const btnProceed = document.getElementById("btn-proceed-to-stage2");
  if (btnProceed) {
    btnProceed.addEventListener("click", () => {
      switchTab("tab-stage2");
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  const btnBack = document.getElementById("btn-back-to-stage1");
  if (btnBack) {
    btnBack.addEventListener("click", () => {
      switchTab("tab-stage1");
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }
}

function switchTab(targetId) {
  document.querySelectorAll(".nav-tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-target") === targetId);
  });
  document.querySelectorAll(".tab-pane").forEach(pane => {
    pane.classList.toggle("active", pane.id === targetId);
  });
}

function initButtons() {
  const btnRun1 = document.getElementById("btn-run-stage1");
  if (btnRun1) {
    btnRun1.addEventListener("click", runStage1Analysis);
  }

  const btnRun2 = document.getElementById("btn-run-stage2");
  if (btnRun2) {
    btnRun2.addEventListener("click", runStage2Match);
  }

  const btnQuickDemo = document.getElementById("btn-quick-demo");
  if (btnQuickDemo) {
    btnQuickDemo.addEventListener("click", runQuickDemo);
  }
}

function initThresholdSlider() {
  const slider = document.getElementById("threshold-slider");
  const label = document.getElementById("threshold-val-label");
  if (slider && label) {
    slider.addEventListener("input", (e) => {
      state.stage2.threshold = parseFloat(e.target.value);
      label.textContent = state.stage2.threshold.toFixed(3);
    });
  }
}

function initDropzones() {
  setupDropzone(
    "dropzone-stage1",
    "file-input-stage1",
    (dataUrl, file) => {
      state.stage1.imageData = dataUrl;
      state.stage1.fileName = file.name;
      updateStage1Preview(dataUrl, file.name);
    }
  );

  setupDropzone(
    "dropzone-stage2",
    "file-input-stage2",
    (dataUrl, file) => {
      state.stage2.imageData = dataUrl;
      state.stage2.fileName = file.name;
      updateStage2Preview(dataUrl, file.name);
    }
  );
}

function setupDropzone(dropzoneId, inputId, onFileLoaded) {
  const dropzone = document.getElementById(dropzoneId);
  const input = document.getElementById(inputId);
  if (!dropzone || !input) return;

  dropzone.addEventListener("click", () => input.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0], onFileLoaded);
    }
  });

  input.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelect(e.target.files[0], onFileLoaded);
    }
  });
}

function handleFileSelect(file, callback) {
  if (!file.type.startsWith("image/")) {
    showToast("Lütfen geçerli bir resim dosyası seçin.", "error");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    callback(e.target.result, file);
  };
  reader.readAsDataURL(file);
}

function updateStage1Preview(dataUrl, name) {
  const previewRow = document.getElementById("preview-row-stage1");
  const thumb = document.getElementById("selected-thumb-stage1");
  const nameLabel = document.getElementById("selected-name-stage1");
  const btn = document.getElementById("btn-run-stage1");

  thumb.src = dataUrl;
  nameLabel.textContent = name || "Görüntü Hazır";
  previewRow.style.display = "flex";
  btn.disabled = false;
}

function updateStage2Preview(dataUrl, name) {
  const previewRow = document.getElementById("preview-row-stage2");
  const thumb = document.getElementById("selected-thumb-stage2");
  const nameLabel = document.getElementById("selected-name-stage2");
  const btn = document.getElementById("btn-run-stage2");

  thumb.src = dataUrl;
  nameLabel.textContent = name || "Görüntü Hazır";
  previewRow.style.display = "flex";
  btn.disabled = false;
}

async function loadSampleImages() {
  try {
    const res = await fetch("/api/sample-images");
    if (!res.ok) return;
    const samples = await res.json();
    state.samples = samples;

    const personSample = samples.find(s => s.id === "target_person");
    if (personSample) {
      const thumb = document.getElementById("thumb-sample-person");
      if (thumb) thumb.src = personSample.image_data;
      const chip = document.getElementById("chip-sample-person");
      if (chip) {
        chip.addEventListener("click", () => {
          state.stage1.imageData = personSample.image_data;
          state.stage1.fileName = personSample.filename;
          updateStage1Preview(personSample.image_data, personSample.title);
          showToast("Örnek referans portre yüklendi.", "success");
        });
      }
    }

    const groupMatchSample = samples.find(s => s.id === "group_match");
    if (groupMatchSample) {
      const thumb = document.getElementById("thumb-sample-group-match");
      if (thumb) thumb.src = groupMatchSample.image_data;
      const chip = document.getElementById("chip-sample-group-match");
      if (chip) {
        chip.addEventListener("click", () => {
          state.stage2.imageData = groupMatchSample.image_data;
          state.stage2.fileName = groupMatchSample.filename;
          updateStage2Preview(groupMatchSample.image_data, groupMatchSample.title);
          showToast("Örnek grup fotoğrafı (kişi mevcut) seçildi.", "success");
        });
      }
    }

    const groupNoMatchSample = samples.find(s => s.id === "group_nomatch");
    if (groupNoMatchSample) {
      const thumb = document.getElementById("thumb-sample-group-nomatch");
      if (thumb) thumb.src = groupNoMatchSample.image_data;
      const chip = document.getElementById("chip-sample-group-nomatch");
      if (chip) {
        chip.addEventListener("click", () => {
          state.stage2.imageData = groupNoMatchSample.image_data;
          state.stage2.fileName = groupNoMatchSample.filename;
          updateStage2Preview(groupNoMatchSample.image_data, groupNoMatchSample.title);
          showToast("Örnek grup fotoğrafı (kişi yok) seçildi.", "success");
        });
      }
    }
  } catch (err) {
    console.error("Failed to load sample images", err);
  }
}

async function runStage1Analysis() {
  if (!state.stage1.imageData) return;

  const btn = document.getElementById("btn-run-stage1");
  const originalHtml = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> <span>Görüntü İşleniyor...</span>`;

  const startTime = performance.now();

  try {
    const formData = new FormData();
    formData.append("image_base64", state.stage1.imageData);

    const res = await fetch("/api/analyze-stage1", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      throw new Error(`Sunucu hatası: ${res.statusText}`);
    }

    const data = await res.json();
    const elapsed = Math.round(performance.now() - startTime);
    state.stage1.result = data;

    if (data.feature_vector) {
      state.stage1.referenceVector = data.feature_vector;
      state.stage1.referenceCropB64 = data.cropped_face_b64;
    }

    renderStage1Telemetry(data, elapsed);
    renderStage1Cards(data.stages);
    updateReferenceLockBanner(data);

    showToast(`Görüntü işleme tamamlandı (${elapsed} ms)!`, "success");
  } catch (err) {
    console.error(err);
    showToast("Analiz sırasında hata oluştu: " + err.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalHtml;
  }
}

function renderStage1Telemetry(data, elapsedMs) {
  const teleBar = document.getElementById("telemetry-bar-stage1");
  teleBar.style.display = "flex";

  document.getElementById("tele-steps-count").textContent = `${data.stages.length} / 11`;
  document.getElementById("tele-elapsed-time").textContent = `${elapsedMs} ms`;
  
  const faceStatusEl = document.getElementById("tele-face-status");
  if (data.has_face) {
    faceStatusEl.textContent = "Doğrulandı (1 Yüz)";
    faceStatusEl.style.color = "var(--emerald)";
  } else {
    faceStatusEl.textContent = "Yüz Bulunamadı";
    faceStatusEl.style.color = "var(--rose)";
  }
}

function renderStage1Cards(stages) {
  const container = document.getElementById("stages-grid-container");
  container.innerHTML = "";

  stages.forEach(stage => {
    const card = document.createElement("article");
    card.className = "stage-card";

    let metricsHtml = "";
    if (stage.metrics) {
      for (const [k, v] of Object.entries(stage.metrics)) {
        if (Array.isArray(v)) {
          metricsHtml += `<span class="metric-pill">${k}: <strong>[${v.join(", ")}]</strong></span>`;
        } else {
          metricsHtml += `<span class="metric-pill">${k}: <strong>${v}</strong></span>`;
        }
      }
    }

    card.innerHTML = `
      <div class="stage-card-header">
        <span class="stage-badge">Adım ${String(stage.step_number).padStart(2, "0")}</span>
        <span class="stage-category">${stage.category}</span>
      </div>
      <div class="stage-img-wrap">
        <img src="${stage.image_data}" alt="${stage.title}" class="stage-img">
        <button class="zoom-overlay-btn" type="button">🔍 Büyüt</button>
      </div>
      <div class="stage-card-body">
        <h3 class="stage-card-title">${stage.title}</h3>
        
        <div class="explain-box">
          <div class="explain-label">Bu Aşama Nedir?</div>
          <p class="explain-text">${stage.what_is_it}</p>
        </div>

        <div class="explain-box" style="border-left-color: var(--violet)">
          <div class="explain-label" style="color: var(--violet)">Neden Kullanılır?</div>
          <p class="explain-text">${stage.why_it_matters}</p>
        </div>

        <div class="tech-box">
          <strong>Teknik Karşılık:</strong> ${stage.technical_details}
        </div>

        <div class="metrics-list">
          ${metricsHtml}
        </div>
      </div>
    `;

    const zoomBtn = card.querySelector(".zoom-overlay-btn");
    const img = card.querySelector(".stage-img");
    const triggerZoom = () => openImageModal(stage.title, stage.image_data, stage.technical_details);
    zoomBtn.addEventListener("click", triggerZoom);
    img.addEventListener("click", triggerZoom);

    container.appendChild(card);
  });
}

function updateReferenceLockBanner(data) {
  if (!data.has_face) return;

  const banner = document.getElementById("reference-lock-banner");
  const avatar = document.getElementById("ref-banner-avatar");
  avatar.src = data.cropped_face_b64;
  banner.style.display = "flex";

  const stage2Avatar = document.getElementById("stage2-ref-avatar");
  if (stage2Avatar) stage2Avatar.src = data.cropped_face_b64;
  
  const compareTargetImg = document.getElementById("compare-target-img");
  if (compareTargetImg) compareTargetImg.src = data.cropped_face_b64;
}

async function runStage2Match() {
  if (!state.stage2.imageData) {
    showToast("Lütfen aranacak 2. fotoğrafı seçin.", "error");
    return;
  }
  if (!state.stage1.referenceVector) {
    showToast("1. Aşamada henüz referans yüz çıkarılmadı.", "error");
    return;
  }

  const btn = document.getElementById("btn-run-stage2");
  const originalHtml = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> <span>Yüzler Taranıyor...</span>`;

  try {
    const payload = {
      image_base64: state.stage2.imageData,
      reference_vector: state.stage1.referenceVector,
      reference_crop_b64: state.stage1.referenceCropB64,
      threshold: state.stage2.threshold
    };

    const res = await fetch("/api/match-stage2", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error(`Sunucu hatası: ${res.statusText}`);
    }

    const data = await res.json();
    state.stage2.result = data;

    renderStage2Results(data);
    showToast(data.match_found ? "Hedef kişi tespit edildi!" : "Arama tamamlandı: Eşleşme bulunamadı.", data.match_found ? "success" : "info");
  } catch (err) {
    console.error(err);
    showToast("Eşleştirme hatası: " + err.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalHtml;
  }
}

function renderStage2Results(data) {
  const container = document.getElementById("stage2-results-container");
  container.style.display = "block";

  const verdictCard = document.getElementById("verdict-card");
  const verdictIcon = document.getElementById("verdict-icon");
  const verdictTitle = document.getElementById("verdict-title");
  const verdictDesc = document.getElementById("verdict-desc");
  const verdictScoreNum = document.getElementById("verdict-score-num");

  verdictCard.className = `verdict-card ${data.match_found ? "matched" : "nomatch"}`;
  verdictIcon.textContent = data.match_found ? "✓" : "✕";
  verdictTitle.textContent = data.verdict_title;
  verdictDesc.textContent = data.verdict_description;
  verdictScoreNum.textContent = data.best_match ? `%${data.best_match.similarity_pct.toFixed(1)}` : "%0";

  document.getElementById("badge-total-faces").textContent = `${data.total_faces_detected} Yüz Tespit Edildi`;
  document.getElementById("annotated-result-img").src = data.annotated_image_b64;

  const compareCandImg = document.getElementById("compare-cand-img");
  const compareCandLabel = document.getElementById("compare-cand-label");
  if (data.best_match) {
    compareCandImg.src = data.best_match.crop_b64;
    compareCandLabel.textContent = data.match_found 
      ? `Eşleşen Yüz (Aday #${data.best_match.index})` 
      : `En Yakın Aday #${data.best_match.index}`;
  }

  const heatmapBox = document.getElementById("heatmap-container");
  const heatmapImg = document.getElementById("compare-heatmap-img");
  if (data.diff_heatmap_b64) {
    heatmapImg.src = data.diff_heatmap_b64;
    heatmapBox.style.display = "block";
  } else {
    heatmapBox.style.display = "none";
  }

  const candList = document.getElementById("candidates-list-container");
  candList.innerHTML = "";

  data.candidates.forEach(cand => {
    const row = document.createElement("div");
    row.className = `candidate-row ${cand.is_match ? "winner" : ""}`;
    row.innerHTML = `
      <div class="cand-left">
        <img src="${cand.crop_b64}" alt="Aday ${cand.index}" class="cand-thumb">
        <div class="cand-info">
          <h5>${cand.is_match ? "✓ Hedef Kişi" : `Aday #${cand.index}`}</h5>
          <p>Kosinüs: ${cand.cosine_similarity.toFixed(4)} | L2: ${cand.l2_distance.toFixed(3)}</p>
        </div>
      </div>
      <div class="cand-score ${cand.is_match ? "winner" : "loser"}">
        %${cand.similarity_pct.toFixed(1)}
      </div>
    `;
    candList.appendChild(row);
  });

  container.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function runQuickDemo() {
  if (state.samples.length === 0) {
    await loadSampleImages();
  }

  const personSample = state.samples.find(s => s.id === "target_person");
  const groupSample = state.samples.find(s => s.id === "group_match");

  if (!personSample || !groupSample) {
    showToast("Örnek görseller hazır değil.", "error");
    return;
  }

  switchTab("tab-stage1");
  state.stage1.imageData = personSample.image_data;
  state.stage1.fileName = personSample.filename;
  updateStage1Preview(personSample.image_data, personSample.title);

  showToast("1. Aşama analizi başlatılıyor...", "info");
  await runStage1Analysis();

  setTimeout(async () => {
    switchTab("tab-stage2");
    state.stage2.imageData = groupSample.image_data;
    state.stage2.fileName = groupSample.filename;
    updateStage2Preview(groupSample.image_data, groupSample.title);

    showToast("2. Aşama: Fotoğrafta hedef kişi taranıyor...", "info");
    await runStage2Match();
  }, 1000);
}

function initModals() {
  const modal = document.getElementById("image-modal");
  const closeBtn = document.getElementById("modal-close-btn");

  if (closeBtn && modal) {
    closeBtn.addEventListener("click", () => modal.classList.remove("open"));
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.remove("open");
    });
  }
}

function openImageModal(title, imageSrc, desc) {
  const modal = document.getElementById("image-modal");
  document.getElementById("modal-title").textContent = title;
  document.getElementById("modal-img").src = imageSrc;
  document.getElementById("modal-desc").textContent = desc || "";
  modal.classList.add("open");
}

function showToast(msg, type = "info") {
  const toast = document.getElementById("alert-toast");
  const icon = document.getElementById("toast-icon");
  const text = document.getElementById("toast-msg");

  text.textContent = msg;
  toast.className = `alert-toast show ${type}`;
  
  if (type === "success") icon.textContent = "✓";
  else if (type === "error") icon.textContent = "⚠";
  else icon.textContent = "ℹ";

  setTimeout(() => {
    toast.classList.remove("show");
  }, 3500);
}
