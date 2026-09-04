# Cover Letter to the Editor-in-Chief

**To:**  
Editor-in-Chief,  
*IEEE Transactions on Medical Imaging (IEEE TMI)*  

**Date:** September 2025  
**Subject:** Submission of Original Research Manuscript  
**Manuscript Title:** *Causal-Interventional Grounding and Counterfactual Inpainting for Hallucination-Resistant and Calibrated Medical Visual Question Answering*  
**Authors:** Faezeh Miller and Co-Authors  
**Corresponding Author:** Faezeh Miller (Department of Biomedical Engineering and Computer Science, AI Medical Imaging Laboratory; Email: faezeh.miller@ieee.org)  

---

Dear Editor-in-Chief and Editorial Board Members,

We are pleased to submit our original research manuscript titled **"Causal-Interventional Grounding and Counterfactual Inpainting for Hallucination-Resistant and Calibrated Medical Visual Question Answering"** for consideration as a Regular Paper in *IEEE Transactions on Medical Imaging (IEEE TMI)*.

### Clinical Problem and Methodological Motivation
Medical Visual Question Answering (Med-VQA) systems hold tremendous potential to assist clinicians with interactive diagnostic interpretation, automated second opinions, and emergency department triage. However, current state-of-the-art multimodal vision-language models (Med-VLMs) exhibit critical clinical vulnerabilities that obstruct deployment in safety-critical clinical environments:
1. **Backdoor Confounding and Language Shortcuts:** Observational likelihood maximization causes models to exploit statistical dataset priors (e.g., predicting *"effusion"* whenever *"pleural"* appears in the prompt) without visually verifying pathology in the scan.
2. **Generative Anatomical Hallucinations:** Autoregressive decoders frequently fabricate non-existent pathological abnormalities or mischaracterize benign anatomical variants as malignant findings.
3. **Severe Miscalibration and Unwarranted Overconfidence:** Modern architectures output extreme confidence on hallucinated predictions, preventing dependable selective abstention or clinical referral.

### Methodological Innovations
To overcome these barriers, our manuscript reformulates Med-VQA from passive observational correlation fitting into an active physical causal intervention framework grounded in Pearl's Structural Causal Models (SCMs) and $do$-calculus. To our knowledge, this work is the first to employ photorealistic image-domain generative counterfactual inpainting ($do(I = I_{\text{cf}})$) to eliminate multimodal language bias and calibrate diagnostic uncertainty. Specifically, our framework (**CI-GCI**) introduces four synergistic components:
1. **Question-Conditioned ROI Locator (QCRL):** Computes cross-modal spatial attention between language embeddings and visual patch keys to dynamically extract continuous anatomical lesion masks $\mathbf{M}$.
2. **Generative Counterfactual Inpainter (CFI):** Executes physical image-domain interventions $do(I = I_{\text{cf}})$, seamlessly inpainting suspicious pathology into healthy background anatomy.
3. **Dynamic Causal Contrastive Decoder (CCD):** Evaluates the counterfactual logit contrast $(\mathbf{L}_{\text{orig}} - \mathbf{L}_{\text{cf}})$ scaled by a learned question-conditioned parameter $\gamma(Q)$, provably neutralizing language shortcuts and reducing Expected Calibration Error (ECE).
4. **Selective Abstention Triage Gate:** Establishes calibrated confidence thresholds (selected on validation splits) to automatically triage ambiguous cases to human specialists.

### Empirical Rigor and Methodological Positioning
We evaluate CI-GCI across four diverse multi-center benchmarks spanning radiology, histopathology, and gastrointestinal endoscopy: **VQA-RAD**, **SLAKE**, **PathVQA**, and **Kvasir-VQA-x1**. Our framework maintains competitive diagnostic accuracy while demonstrating transformative gains in clinical dependability:
- Expected Calibration Error (ECE) is reduced by up to **88.1% relative error** compared to standard uncalibrated baselines.
- Visual hallucinations are reduced by **over two-thirds** on rigorous programmatic POPE probes of documented absent pathology.
- Selective abstention triage achieves an error rate of **2.4% at 72.5% automated coverage** ($\\tau_2 = 0.85$), safely referring 27.5% of ambiguous cases.

**Positioning on Clinical Scope & Counterfactual Fidelity:**  
We explicitly frame this investigation as a methodological and algorithmic contribution. Rather than presenting subjective, non-standardized reader surveys or simulated radiologist ratings, we validate counterfactual inpainting realism through objective, automated empirical protocols:
- An independently trained pathology classifier demonstrates a substantial target finding probability drop ($\Delta P_{\text{target}} = 0.714 \pm 0.042$) on $I_{\text{cf}}$, confirming lesion neutralization.
- Two One-Sided Tests (TOST) confirm strict statistical equivalence of non-target anatomical findings ($p_{\text{TOST}} < 0.01$ with equivalence margin $\delta = 0.05$), ensuring no spurious pathological artifacts are introduced.
- Background anatomy outside the lesion mask is preserved near-identically by construction (PSNR = 38.64 dB, SSIM = 0.982).
- A linear patch discriminator achieves an ROC-AUC of $0.518 \pm 0.035$ (ideal $\approx 0.50$), confirming that inpainted tissue patches are distributionally indistinguishable from authentic healthy parenchyma.

### Reproducibility and Declarations
- **Originality:** This manuscript is an original work that has not been published elsewhere and is not under consideration by any other journal or conference.
- **Code & Open Science:** Complete source code, model checkpoints, per-sample evaluation records (`records.jsonl`), and automated verification pipelines are publicly available at:  
  [https://github.com/FaezehMillerAI/TMI-VQA.git](https://github.com/FaezehMillerAI/TMI-VQA.git)
- **Conflicts of Interest:** The authors declare that they have no competing financial or personal conflicts of interest.
- **Ethics:** All experiments were conducted on de-identified, publicly available benchmark datasets in compliance with ethical research standards.

Thank you very much for your time, consideration, and coordination of the peer-review process.

Sincerely,  
**Faezeh Miller**  
Department of Biomedical Engineering and Computer Science  
AI Medical Imaging Laboratory, London, UK  
Email: `faezeh.miller@ieee.org`
