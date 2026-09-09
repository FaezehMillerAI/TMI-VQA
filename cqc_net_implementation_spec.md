# CQC-Net: Counterfactual Question Curriculum Network for Hallucination-Aware Medical VQA

این سند یک specification اجرایی و پژوهشی برای پیاده‌سازی یک مدل ژورنالی در Medical VQA ارائه می‌کند که بر **grounded reasoning**، **hallucination mitigation**، **SLM-based curriculum generation**، و **multi-level verification** متمرکز است. Med-VQA در سال‌های اخیر به سمت reasoning-driven و multi-agent / verification-oriented systems حرکت کرده است، و benchmarkها و surveyهای جدید بر grounded evaluation، hallucination analysis، و multimodal reasoning تأکید کرده‌اند.[cite:30][cite:86][cite:91]

## هدف پژوهشی

هدف، طراحی یک مدل عمیق برای Medical VQA است که فقط به سؤال اصلی پاسخ ندهد، بلکه برای هر نمونه یک زنجیره‌ی ساخت‌یافته از سؤال‌های کمکی تولید کند، پاسخ‌های آن زنجیره را با evidence تصویری بسنجد، و از این مسیر hallucination را کشف و کاهش دهد. این framing با روندهای جدیدی که Med-VQA را از یک classification task به یک generation-and-reasoning task تبدیل کرده‌اند هم‌راستا است.[cite:48][cite:84]

## ایده‌ی اصلی

مدل پیشنهادی با نام **CQC-Net** از ایده‌ی Counterfactual Question Curriculum استفاده می‌کند. به‌جای اینکه تنها یک پاسخ برای سؤال ورودی تولید شود، مدل ابتدا چند سؤال کمکی در سه سطح می‌سازد: مشاهده و مکان‌یابی، ویژگی و رابطه، و استنتاج بالینی. سپس به این سؤال‌ها پاسخ می‌دهد، سازگاری پاسخ‌ها را با تصویر و با یکدیگر می‌سنجد، و در نهایت بر اساس این زنجیره تصمیم می‌گیرد که پاسخ نهایی قابل اعتماد است یا hallucinatory.[cite:30][cite:93]

## مسئله و نمادگذاری

برای هر نمونه، ورودی شامل تصویر پزشکی \\(I\\)، سؤال اصلی \\(Q_0\\)، و در صورت وجود متن کمکی مانند caption، metadata، یا گزارش است. خروجی شامل پاسخ اصلی \\(A_0\\)، مجموعه‌ای از سؤال‌های curriculum \\(\{Q_k^{(l)}\}\\)، پاسخ‌های متناظر \\(\{A_k^{(l)}\}\\)، نمرات grounding \\(\{s_k^{(l)}\}\\)، و یک hallucination score نهایی \\(h\\) است.

سطوح curriculum به این صورت تعریف می‌شوند:

- **Level 1 — Existence / Localization:** آیا یک ساختار یا یافته وجود دارد؟ در کدام ناحیه است؟
- **Level 2 — Attribute / Relation:** اندازه، شدت، شکل، رابطه با ساختارهای مجاور، یا تغییر نسبت به نمای دیگر چیست؟
- **Level 3 — Clinical Inference:** آیا الگوی موجود با یک تشخیص، مرحله، یا تفسیر بالینی مشخص سازگار است؟

این decomposition با benchmarkهای جدیدی که Med-VQA را به سمت reasoning chains، multi-image inference، و grounded medical understanding برده‌اند سازگار است.[cite:46][cite:51][cite:91]

## معماری کلی

CQC-Net از شش بلوک اصلی تشکیل می‌شود:

1. **Visual Encoder**
2. **Question Encoder / Text Encoder**
3. **Question Curriculum Generator (QCG)**
4. **Answer Generator**
5. **Grounding & Evidence Verifier**
6. **Curriculum Consistency and Hallucination Head**

### فلو کلی داده

1. تصویر وارد visual encoder می‌شود و feature map چندمقیاسی تولید می‌شود.
2. سؤال اصلی encode می‌شود.
3. QCG بر اساس تصویر و سؤال اصلی، زنجیره‌ای از سؤال‌های کمکی می‌سازد.
4. Answer Generator به سؤال اصلی و همه‌ی سؤال‌های کمکی پاسخ می‌دهد.
5. Verifier هر پاسخ را از نظر grounding و سازگاری با تصویر می‌سنجد.
6. Consistency Head ارتباط بین پاسخ‌های سطوح مختلف را تحلیل می‌کند و hallucination score می‌دهد.
7. Refiner یا final decision module پاسخ نهایی را تأیید، بازنویسی، یا reject می‌کند.

## بلوک 1: Visual Encoder

### گزینه‌های پیشنهادی

| گزینه | مزیت | ضعف | پیشنهاد استفاده |
|---|---|---|---|
| ResNet-50 / ResNet-101 | ساده، پایدار، کم‌هزینه | در global context ضعیف‌تر | baseline قوی [cite:40] |
| DenseNet-121 | رایج در پزشکی، مناسب برای CXR | محدودتر از ViT در long-range interactions | برای radiology baseline |
| ViT-B/16 | representation قوی‌تر | هزینه‌ی بیشتر | main encoder |
| Swin Transformer | چندمقیاسی، مناسب localization | پیچیدگی بیشتر | برای grounding-focused setup |
| CLIP-like vision tower | هم‌ترازی قوی‌تر vision-language | نیازمند adaptation پزشکی | در VLM setup |

### طراحی پیشنهادی

برای مقاله‌ی ژورنالی، یک **dual-scale visual encoder** پیشنهاد می‌شود:

- شاخه‌ی اول: global visual tokens برای semantic understanding.
- شاخه‌ی دوم: local patch tokens یا region features برای localization و grounding.

خروجی:

\\[
V_g = f_{global}(I), \qquad V_l = f_{local}(I)
\\]

و representation نهایی:

\\[
V = [V_g ; V_l]
\\]

## بلوک 2: Text Encoder

### گزینه‌های پیشنهادی

| گزینه | نوع | مزیت | پیشنهاد |
|---|---|---|---|
| BioClinicalBERT | encoder | خوب برای پرسش کوتاه و medical text | برای verifier |
| PubMedBERT | encoder | vocabulary پزشکی قوی | برای text-only branches |
| SciBERT | encoder | عمومی‌تر ولی علمی | baseline |
| Phi-small / Qwen-small / TinyLLaMA | SLM | سبک، قابل fine-tune | برای QCG و critic |

### طراحی پیشنهادی

دو encoder متنی مجزا استفاده شود:

- **Text-Enc-A:** برای encoding سؤال‌ها و پاسخ‌ها در verifier.
- **Text-Enc-B / SLM:** برای generation سؤال‌های curriculum و در صورت نیاز refinement پاسخ.

## بلوک 3: Question Curriculum Generator (QCG)

این بلوک novelty اصلی مدل است. QCG یک SLM است که از روی تصویر و سؤال اصلی، مجموعه‌ای از سؤال‌های کمکی سطح‌بندی‌شده می‌سازد.

### ورودی

- visual summary tokenها
- embedding سؤال اصلی
- optional: task tag مثل modality یا organ system

### خروجی

- Level 1 questions: \\(Q^{(1)}_1, ..., Q^{(1)}_{n_1}\\)
- Level 2 questions: \\(Q^{(2)}_1, ..., Q^{(2)}_{n_2}\\)
- Level 3 questions: \\(Q^{(3)}_1, ..., Q^{(3)}_{n_3}\\)

### تعداد سؤال‌ها

برای شروع:

- Level 1: دو سؤال
- Level 2: دو سؤال
- Level 3: یک سؤال

پس برای هر سؤال اصلی، پنج سؤال کمکی تولید می‌شود. این طراحی هزینه را کنترل می‌کند ولی هنوز برای consistency signal کافی است.

### روش پیاده‌سازی

#### روش A: Template-guided generation

برای هر modality، templateهای نیمه‌ساختاریافته تعریف شود. مثال در CXR:

- Level 1: آیا finding در لوب راست/چپ دیده می‌شود؟
- Level 2: آیا opacity diffuse است یا focal؟
- Level 3: آیا این الگو با pneumonia سازگار است؟

مزیت: کنترل بهتر و خطای کمتر.

#### روش B: SLM-based free generation

یک SLM کوچک با prompt کنترل‌شده سؤال‌ها را می‌سازد. این روش flexibleتر است، ولی نیاز به filtering بیشتری دارد.

#### روش C: Hybrid

بهترین گزینه برای مقاله: ابتدا template پیشنهاد شود، سپس SLM متن را naturalize و تخصصی‌تر کند.

### supervision برای QCG

سه نوع supervision قابل استفاده است:

1. **Weak supervision** از caption/report، با استخراج entities و relations.
2. **Synthetic supervision** با LLM-generated question chains؛ pipelineهای جدید synthetic Med-VQA نشان داده‌اند که QA تولیدشده می‌تواند برای training مفید باشد.[cite:79][cite:87][cite:89]
3. **Human refinement** روی subset کوچک برای تضمین کیفیت.

### loss برای QCG

\\[
\mathcal{L}_{QCG} = \mathcal{L}_{gen} + \lambda_{lvl}\mathcal{L}_{level} + \lambda_{div}\mathcal{L}_{diversity}
\\]

- \\(\mathcal{L}_{gen}\\): loss تولید توکن‌ها
- \\(\mathcal{L}_{level}\\): تشخیص درست level هر سؤال
- \\(\mathcal{L}_{diversity}\\): جلوگیری از تولید سؤال‌های تکراری

## بلوک 4: Answer Generator

این بلوک به سؤال اصلی و سؤال‌های curriculum پاسخ می‌دهد.

### گزینه‌های معماری

| گزینه | مزیت | ضعف | توصیه |
|---|---|---|---|
| Encoder-decoder VLM | پایدار برای generation | پیچیده‌تر | برای پاسخ‌های آزاد |
| Decoder-only SLM + visual prefix | سبک‌تر و modular | ممکن است grounding ضعیف‌تر شود | اگر efficiency مهم است |
| BLIP2-style Q-Former + LLM | bridge خوب بین vision و language | پیاده‌سازی سنگین‌تر | برای نسخه‌ی full |
| Flamingo-style cross-attention | چندوجهی قوی | نیازمند compute | در صورت دسترسی به GPU قوی |

### طراحی پیشنهادی

- اگر هدف thesis-quality و TMI-level است: **Vision encoder + Q-Former + SLM decoder**.
- اگر هدف prototype سریع است: **ViT + projection layer + decoder-only SLM**.

### خروجی

برای هر سؤال \\(Q_k\\):

\\[
P(A_k | I, Q_k)
\\]

برای پاسخ‌های آزاد، beam search یا constrained decoding استفاده شود. برای سوالات yes/no و closed-form، classification head جداگانه می‌تواند کمک کند.

## بلوک 5: Grounding & Evidence Verifier

این بلوک تعیین می‌کند که پاسخ هر سؤال واقعاً از تصویر پشتیبانی می‌شود یا نه.

### ورودی‌ها

- پاسخ تولیدشده
- embedding سؤال
- visual tokens
- optional: report/caption embedding

### خروجی‌ها

- grounding score برای هر پاسخ
- attention map یا region importance
- contradiction score بین answer و evidence

### طراحی پیشنهادی

Verifier شامل سه sub-module باشد:

1. **Answer-Image Compatibility Head**
2. **Answer-Question Entailment Head**
3. **Region Attribution Head**

score کلی:

\\[
s_k = \alpha s_k^{img} + \beta s_k^{qa} + \gamma s_k^{reg}
\\]

### supervision

- اگر region annotation نیست، از soft grounding بر پایه‌ی attention consistency استفاده شود.
- اگر گزارش یا caption موجود است، از text entailment بین answer و report استفاده شود.
- برای داده‌های دارای explanation یا report، weak evidence labels ساخته می‌شود.

## بلوک 6: Curriculum Consistency and Hallucination Head

این بلوک مهم‌ترین بخش تصمیم‌گیری است. ایده این است که پاسخ اصلی باید با پاسخ‌های سطوح پایین‌تر سازگار باشد.

### شهود

اگر مدل ادعا کند «pleural effusion وجود دارد»، ولی در Level 1 نتواند وجود fluid را تشخیص دهد یا در Level 2 نتواند موقعیت/شدت را توضیح دهد، پاسخ اصلی احتمالاً hallucination است.

### طراحی ریاضی

برای هر level:

\\[
\bar{s}^{(l)} = \frac{1}{n_l}\sum_i s_i^{(l)}
\\]

سپس بردار consistency ساخته می‌شود:

\\[
c = [\bar{s}^{(1)}, \bar{s}^{(2)}, \bar{s}^{(3)}, s_0]
\\]

که در آن \\(s_0\\) grounding score سؤال اصلی است. سپس:

\\[
h = \sigma(MLP(c))
\\]

که \\(h\\) احتمال hallucination است.

### گزینه‌ی خلاقانه‌ی تکمیلی

یک **directional inconsistency detector** اضافه شود که نه فقط میانگین levelها، بلکه الگوی ناسازگاری را تحلیل کند. مثلاً افت شدید از Level 1 به Level 3 با pattern خاصی رخ دهد. این ماژول می‌تواند با transformer کوچک روی sequence scoreها پیاده شود.

## Refiner / Decision Module

در inference سه حالت تعریف شود:

- **Accept:** اگر confidence بالا و hallucination score پایین باشد.
- **Revise:** اگر پاسخ اصلی قابل قبول است ولی inconsistency متوسط دارد؛ در این حالت SLM refiner پاسخ را بازنویسی می‌کند.
- **Abstain / Flag:** اگر hallucination score بالا است؛ مدل پاسخ را با هشدار یا حالت no-answer برمی‌گرداند.

این نوع abstention برای ارزیابی safety و calibration مفید است و با مطالعات جدید درباره‌ی medical hallucination و reliability هم‌راستا است.[cite:16][cite:86]

## توابع هدف نهایی

### 1) QA Loss

\\[
\mathcal{L}_{QA} = -\sum_k \log P(A_k^{gt} | I, Q_k)
\\]

### 2) Grounding Loss

\\[
\mathcal{L}_{ground} = \sum_k \ell(s_k, y_k^{ground})
\\]

### 3) Consistency Loss

\\[
\mathcal{L}_{cons} = \sum_{l=1}^{2} \max(0, \bar{s}^{(l)} - \bar{s}^{(l+1)} - \delta)
\\]

### 4) Hallucination Classification Loss

\\[
\mathcal{L}_{hallu} = BCE(h, y^{hallu})
\\]

### 5) Optional Calibration Loss

\\[
\mathcal{L}_{cal} = ECE\text{-surrogate or Brier-style loss}
\\]

### 6) Loss کل

\\[
\mathcal{L}_{total} = \lambda_1\mathcal{L}_{QA} + \lambda_2\mathcal{L}_{ground} + \lambda_3\mathcal{L}_{cons} + \lambda_4\mathcal{L}_{hallu} + \lambda_5\mathcal{L}_{cal}
\\]

## برنامه‌ی پیاده‌سازی مرحله‌به‌مرحله

## فاز 1: Baseline آماده‌سازی

### هدف

یک baseline Medical VQA قوی بسازید که بعداً CQC روی آن سوار شود.

### انتخاب baseline

- Visual encoder: ViT-B یا DenseNet-121
- Text/decoder: Qwen-small / Phi-small / TinyLLaMA
- Datasets: VQA-RAD, SLAKE, PathVQA، و در صورت امکان PMC-VQA.[cite:39][cite:40][cite:48][cite:50][cite:84]

### خروجی این فاز

- baseline accuracy و NLG metrics
- pipeline تمیز برای training و evaluation

## فاز 2: ساخت curriculum data

### روش ساخت داده

1. برای هر نمونه، entityها و relationها از answer/caption/report استخراج شود.
2. templateها بر اساس modality و organ طراحی شوند.
3. SLM یا LLM از روی template + context، پنج سؤال کمکی تولید کند.
4. heuristic filters:
   - عدم تکرار
   - سازگاری با modality
   - محدودیت طول
   - قابل پاسخ بودن از روی تصویر
5. subset کوچکی توسط human expert بررسی شود.

### annotationهای مورد نیاز

- level label برای هر سؤال
- optional hallucination label برای بعضی پاسخ‌ها
- optional grounding validity label

## فاز 3: آموزش QCG

QCG را ابتدا جداگانه train کنید تا سؤال‌های curriculum معنادار تولید کند. متریک‌های مناسب:

- diversity
- level accuracy
- relevance to main question
- answerability

## فاز 4: آموزش joint

در این فاز QCG، answer generator، verifier، و consistency head به‌صورت مشترک fine-tune می‌شوند. ابتدا QCG را freeze کنید و بقیه‌ی مدل را train کنید؛ سپس همه‌ی ماژول‌ها را jointly fine-tune کنید.

## فاز 5: Refiner و Abstention

در انتها policy تصمیم‌گیری اضافه شود. این بخش novelty کاربردی و clinical-safety value مقاله را بالا می‌برد.

## مدل‌های مناسب برای هر بلوک

## گزینه‌های SLM برای QCG و Critic

| مدل | اندازه تقریبی | مزیت | کاربرد پیشنهادی |
|---|---:|---|---|
| Phi family (small) | سبک | instruction-following خوب | QCG, refiner |
| Qwen small | سبک تا متوسط | multilingual و قوی در reasoning | QCG, answer refinement |
| TinyLLaMA | بسیار سبک | ارزان برای آزمایش | prototype |
| SmolVLM-like compact models | سبک | vision-language compact | end-to-end lightweight setup [cite:79] |

## گزینه‌های VLM برای Answer Generator

| مدل | مزیت | ضعف | استفاده |
|---|---|---|---|
| BLIP-2 style | modular و محبوب | adaptation پزشکی لازم دارد | main baseline |
| LLaVA-style medical adaptation | پیاده‌سازی آسان‌تر | hallucination بالا ممکن است | baseline مقایسه |
| Florence-style compact adaptation | مناسب lightweight | domain adaptation لازم | efficient setup [cite:83] |
| Custom ViT + Q-Former + SLM | کنترل کامل روی architecture | زمان توسعه بیشتر | paper model |

## گزینه‌های Verifier

| مدل | مزیت | کاربرد |
|---|---|---|
| PubMedBERT cross-encoder | entailment خوب روی text | answer-question-report consistency |
| BiomedCLIP-like scorer | cross-modal similarity | answer-image compatibility |
| MLP on fused embeddings | ساده و سریع | baseline verifier |
| transformer over score sequence | captures structured inconsistency | advanced hallucination head |

## دیتاست‌های مناسب

| دیتاست | نقش در پروژه | دلیل |
|---|---|---|
| VQA-RAD | benchmark کلاسیک | استاندارد Med-VQA [cite:40][cite:50] |
| SLAKE | generalization | تنوع خوب و multilingual flavor [cite:50] |
| PathVQA | modality variation | پاتولوژی و الگوهای متفاوت [cite:47][cite:50] |
| PMC-VQA | scale-up | 227k QA از 149k تصویر [cite:48][cite:84] |
| Kvasir-VQA-x1 | reasoning-heavy | QA پیچیده‌ی GI و مناسب curriculum [cite:41][cite:80] |
| MedFrameQA | multi-image reasoning | reasoning چندتصویری [cite:46][cite:51] |
| 3D-RAD | extension thesis-level | 3D و temporality [cite:44] |

## پیشنهاد setup عملی

برای اینکه پروژه قابل اجرا و در عین حال ژورنالی بماند:

### نسخه‌ی 1: Main paper setup

- Datasetها: VQA-RAD + SLAKE + PathVQA + Kvasir-VQA-x1
- Visual encoder: Swin / ViT
- QCG: Qwen-small یا Phi-small
- Answerer: custom ViT + Q-Former + small decoder
- Verifier: PubMedBERT + grounding head

### نسخه‌ی 2: Scalable setup

- اضافه کردن PMC-VQA برای pretraining
- fine-tune روی datasetهای کوچک‌تر
- evaluate cross-dataset generalization

### نسخه‌ی 3: Thesis extension

- اضافه کردن MedFrameQA و 3D-RAD
- توسعه به multi-image و 3D reasoning

## الگوریتم پیشنهادی آموزش

```text
Input: training set D = {(I_i, Q_i, A_i)}
Initialize visual encoder, QCG, answer generator, verifier, consistency head
Stage 1: Train baseline answer generator on original QA pairs
Stage 2: Build synthetic curriculum questions for each sample
Stage 3: Train QCG on generated curriculum data
Stage 4: Freeze QCG, jointly train answer generator + verifier + consistency head
Stage 5: Unfreeze all modules and fine-tune end-to-end
Stage 6: Calibrate confidence and abstention threshold on validation set
Output: hallucination-aware Med-VQA model
```

## شبه‌کد inference

```text
Given image I and main question Q0:
1. Extract visual features V
2. Generate curriculum questions {Qk(l)} using QCG
3. Answer main and curriculum questions using Answer Generator
4. Compute grounding scores sk(l) and s0 using Verifier
5. Compute hallucination score h using Consistency Head
6. If h < tau_low: return answer
7. Else if tau_low <= h < tau_high: refine answer with SLM Refiner
8. Else: abstain or return flagged answer
```

## آبلیشن‌هایی که باید حتماً انجام شوند

برای قوی شدن مقاله، آبلیشن‌ها بسیار مهم‌اند:

1. بدون QCG
2. بدون Verifier
3. بدون Consistency Loss
4. بدون Hallucination Head
5. template-only curriculum vs SLM-generated curriculum
6. small vs large number of auxiliary questions
7. single-level vs three-level curriculum
8. با و بدون abstention policy
9. با و بدون synthetic pretraining

## متریک‌های ارزیابی

### VQA / NLG

- Accuracy
- Exact Match
- BLEU-1/2/4
- ROUGE-L
- METEOR
- CIDEr
- F1

### hallucination / safety

- Hallucination rate
- Hallucination precision / recall / F1
- AUROC / AUPRC برای hallucination detection
- Faithfulness score
- Consistency score
- Abstention accuracy
- Selective risk / coverage

مطالعات جدید در medical hallucination evaluation و benchmark design روی ارزیابی دقیق hallucination، safety، و benchmarkهای structured تأکید کرده‌اند.[cite:24][cite:28][cite:32][cite:86]

### calibration

- ECE
- Brier score
- Negative log-likelihood

### grounding

- region overlap اگر annotation موجود باشد
- pointing game
- token-region alignment consistency

## نوآوری‌های قابل ادعا در مقاله

این model plan به شما اجازه می‌دهد contributionهای زیر را claim کنید:

1. یک **curriculum-based hallucination-aware architecture** برای Medical VQA.
2. یک **SLM-based question curriculum generator** به‌جای external knowledge graph.
3. یک **multi-level consistency verifier** که hallucination را از روی ناسازگاری سلسله‌مراتبی کشف می‌کند.
4. یک **clinical abstention/refinement policy** برای reliability بیشتر.
5. یک **data construction pipeline** برای question-chain supervision در Med-VQA.[cite:79][cite:84][cite:87]

## ریسک‌ها و راه‌حل‌ها

### ریسک 1: سؤال‌های کمکی بی‌کیفیت

راه‌حل: hybrid generation + template constraints + human validation روی subset.

### ریسک 2: verifier ضعیف

راه‌حل: text entailment branch و cross-modal branch را باهم استفاده کنید، نه یک scorer ساده.

### ریسک 3: هزینه‌ی محاسباتی بالا

راه‌حل: ابتدا curriculum را offline تولید کنید؛ سپس فقط در fine-tuning بخشی از generation را online کنید.

### ریسک 4: novelty ناکافی

راه‌حل: تمرکز مقاله را فقط روی performance نگذارید؛ بر **structured hallucination mitigation**, **consistency reasoning**, و **reliability-aware inference** تأکید کنید، چون ادبیات جدید نیز همین شکاف‌ها را برجسته کرده است.[cite:16][cite:30][cite:86]

## ساختار پیشنهادی مخزن کد

```text
project/
  configs/
  data/
    raw/
    processed/
    curriculum/
  models/
    visual_encoder.py
    text_encoder.py
    qcg.py
    answer_generator.py
    verifier.py
    consistency_head.py
    refiner.py
  training/
    train_baseline.py
    train_qcg.py
    train_joint.py
    calibrate.py
  evaluation/
    eval_vqa.py
    eval_hallucination.py
    eval_grounding.py
  scripts/
    build_curriculum_data.py
    preprocess_datasets.py
  notebooks/
  outputs/
```

## پیشنهاد نهایی برای تحویل به مدل پیاده‌ساز

اگر این specification را به یک مدل دیگر برای implementation می‌دهید، دستور نهایی مناسب این است:

- ابتدا baseline Med-VQA را روی VQA-RAD و SLAKE پیاده‌سازی کند.
- سپس pipeline ساخت curriculum data را بسازد.
- بعد QCG را به‌عنوان ماژول جدا آموزش دهد.
- سپس verifier و consistency head را اضافه کند.
- در پایان training joint، ablation، و evaluation کامل را انجام دهد.

## جمع‌بندی عملی

بهترین نسخه‌ی اجرایی برای شروع این است که یک baseline با **ViT + Q-Former + small decoder** ساخته شود، curriculum به‌صورت hybrid تولید شود، و verifier از ترکیب **cross-modal compatibility** و **text entailment** استفاده کند. این طراحی هم از نظر novelty برای مقاله‌ی ژورنالی مناسب است و هم از نظر engineering قابل پیاده‌سازی مرحله‌ای است.[cite:30][cite:48][cite:84][cite:86]

## پروتکل ارزیابی بسیار کامل

برای اینکه مقاله از نظر داوری ژورنالی قوی باشد، ارزیابی باید چندلایه و عمیق باشد و فقط به accuracy اکتفا نکند. مرورها و benchmarkهای جدید در Med-VQA و medical vision-language generation نشان داده‌اند که مجموعه‌ای از متریک‌های lexical، semantic، clinical، hallucination-specific، calibration، و grounding باید کنار هم گزارش شوند، زیرا BLEU و ROUGE به‌تنهایی برای سنجش clinical faithfulness کافی نیستند.[cite:168][cite:172][cite:176][cite:178]

### اصل طراحی evaluation

چهار محور اصلی باید هم‌زمان پوشش داده شوند:

1. **Task performance** برای کیفیت پاسخ‌دهی و generation.
2. **Semantic and clinical correctness** برای سنجش معنا و factuality.
3. **Hallucination and reliability** برای شناسایی خطاهای خطرناک.
4. **Grounding and calibration** برای سنجش اتکای پاسخ به تصویر و confidence model.

## گروه 1: متریک‌های پایه VQA

### Accuracy

برای سوالات close-ended و classification-like، accuracy باید گزارش شود. Accuracy در reviewهای Med-VQA همچنان متریک پایه برای correctness کلی مدل است.[cite:168]

**کتابخانه پیشنهادی:**
- `scikit-learn` → `accuracy_score`
- یا `torchmetrics.classification.MulticlassAccuracy`

### Exact Match (EM)

برای پاسخ‌های کوتاه و canonicalized، Exact Match مناسب است. در برخی ارزیابی‌های جدید Med-VQA، Exact Match کنار ROUGE، BLEU، METEOR و BERTScore استفاده شده است.[cite:166]

**کتابخانه پیشنهادی:**
- پیاده‌سازی سفارشی در Python بعد از normalization
- یا utility داخلی پروژه

### Precision / Recall / F1

برای closed-form answers، multi-label outputs، یا ارزیابی clinical labels، precision، recall و F1 لازم‌اند. در medical VLM evaluation و clinical efficacy tables، این متریک‌ها برای سنجش balance بین false positives و false negatives استفاده می‌شوند.[cite:167][cite:172]

**کتابخانه پیشنهادی:**
- `scikit-learn.metrics`
- `torchmetrics`

## گروه 2: متریک‌های NLG و lexical overlap

### BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4

BLEU همچنان در VQA و report generation برای سنجش شباهت n-gram استفاده می‌شود، هرچند به‌تنهایی کافی نیست.[cite:167][cite:168][cite:172]

**کتابخانه پیشنهادی:**
- `pycocoevalcap`
- `nltk.translate.bleu_score`
- `evaluate` در Hugging Face برای setupهای ساده‌تر

### ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum

ROUGE برای recall-oriented overlap و به‌خصوص در پاسخ‌های طولانی‌تر مفید است و در Med-VQA و GI-VQA جدید هم استفاده شده است.[cite:166][cite:177]

**کتابخانه پیشنهادی:**
- `rouge-score`
- `evaluate`
- `pycocoevalcap` برای ROUGE-L

### METEOR

METEOR معمولاً بهتر از BLEU می‌تواند synonymy و matching سطح واژه را لحاظ کند و در گزارش‌های Med-VQA و report generation پرتکرار است.[cite:166][cite:167][cite:177]

**کتابخانه پیشنهادی:**
- `pycocoevalcap`
- `nltk.translate.meteor_score`

### CIDEr

CIDEr در image-to-text و Med-VQA generation برای سنجش consensus با referenceها به‌کار می‌رود و در گزارش‌های پزشکی نیز گزارش شده است.[cite:167][cite:169][cite:173]

**کتابخانه پیشنهادی:**
- `pycocoevalcap`
- `caption_eval` forkهای سازگار با Python 3 [cite:175]

### CHRF++

در برخی benchmarkهای جدید GI-VQA، CHRF++ هم کنار BLEU و ROUGE گزارش شده است و برای character-level matching مفید است، به‌ویژه وقتی پاسخ‌ها کوتاه ولی حساس به spelling باشند.[cite:177]

**کتابخانه پیشنهادی:**
- `sacrebleu`

## گروه 3: متریک‌های semantic similarity

### BERTScore

BERTScore برای سنجش similarity معنایی بین prediction و reference مناسب است و در Med-VQA جدید نیز گزارش شده است.[cite:166][cite:168][cite:177]

**کتابخانه پیشنهادی:**
- `bert-score`

### Sentence Embedding Similarity

برای ارزیابی semantic closeness، cosine similarity بین embeddingهای prediction و reference نیز می‌تواند گزارش شود. این متریک مکمل خوبی برای lexical metrics است، به‌خصوص وقتی چند عبارت بالینی هم‌معنا وجود دارد.[cite:168]

**کتابخانه پیشنهادی:**
- `sentence-transformers`
- `scikit-learn` برای cosine similarity

## گروه 4: متریک‌های clinical correctness

### Clinical Precision / Recall / F1

اگر پاسخ‌ها به disease tags، abnormality tags، anatomy labels، یا finding ontology map شوند، clinical precision/recall/F1 باید جداگانه محاسبه شود. بسیاری از ارزیابی‌های پزشکی تأکید می‌کنند که متریک‌های lexical می‌توانند template similarity را بالا نشان دهند، در حالی که correctness بالینی پایین باشد.[cite:164][cite:172]

**کتابخانه پیشنهادی:**
- `scikit-learn`
- rule-based concept extraction با `spaCy` یا `scispaCy`

### Label-overlap Score

برای هر پاسخ، یافته‌های بالینی استخراج شود و overlap با reference label set محاسبه شود. این متریک برای Med-VQA datasetهایی که answerهای کوتاه ولی clinically dense دارند بسیار مفید است.[cite:168]

**کتابخانه پیشنهادی:**
- `scispaCy`
- `medspacy`
- `pandas` + custom code

### Optional report-style metrics

اگر پاسخ‌های مدل طولانی‌تر و شبیه mini-report شوند، metricهای report generation مانند **RadGraph F1** یا image-aware clinical metrics هم ارزشمندند، چون مرورهای جدید همین ضعف ارزیابی صرفاً lexical را برجسته کرده‌اند.[cite:163][cite:168]

**کتابخانه پیشنهادی:**
- `radgraph` implementations در GitHub یا custom wrapper
- metric-specific repositories

## گروه 5: متریک‌های hallucination

### Hallucination Rate

نرخ نمونه‌هایی که پاسخ مدل حاوی information unsupported by image/context است باید گزارش شود. benchmarkها و مرورهای جدید medical hallucination این موضوع را هسته‌ی ارزیابی reliability می‌دانند.[cite:176][cite:178]

**کتابخانه پیشنهادی:**
- custom implementation after hallucination labeling
- `pandas`, `numpy`

### Hallucination Precision / Recall / F1

اگر مدل علاوه بر answer، hallucination flag هم بدهد، precision/recall/F1 برای detection لازم است. Med-HallMark و benchmarkهای hallucination-oriented دقیقاً روی detection quality تأکید دارند.[cite:176][cite:178]

**کتابخانه پیشنهادی:**
- `scikit-learn.metrics`
- `torchmetrics`

### AUROC

AUC-ROC برای سنجش رتبه‌بندی hallucinated vs non-hallucinated outputs در benchmarkهای جدید hallucination detection استفاده می‌شود.[cite:170][cite:179]

**کتابخانه پیشنهادی:**
- `scikit-learn.metrics.roc_auc_score`
- `torchmetrics.AUROC`

### AUPRC

AUPRC به‌خصوص برای کلاس مثبت hallucination وقتی class imbalance وجود دارد informative است.[cite:170]

**کتابخانه پیشنهادی:**
- `scikit-learn.metrics.average_precision_score`
- `torchmetrics.AveragePrecision`

### FPR@95TPR و TPR@Fixed-FPR

برای سنجش رفتار detector در thresholdهای حساس، متریک‌های thresholded نیز پیشنهاد می‌شوند. این متریک‌ها در کارهای safety-oriented معمول‌اند و برای مقاله‌ی قابل قبول ژورنالی ارزش افزوده دارند.[cite:170]

**کتابخانه پیشنهادی:**
- custom implementation با `numpy`
- یا استخراج از curveهای `scikit-learn`

### Severity-weighted Hallucination Score

کارهای جدید مثل MediHall Score نشان داده‌اند که hallucinationها باید از نظر severity هم سنجیده شوند، نه فقط occurrence.[cite:176]

**کتابخانه پیشنهادی:**
- custom implementation
- `pandas`, `numpy`

### Cause-specific Hallucination Scores

اگر annotation یا heuristic labeling داشته باشید، hallucinationها را به visual misinterpretation، knowledge deficiency، و context misalignment تفکیک کنید؛ MedHEval دقیقاً چنین framing علّی را پیشنهاد می‌کند.[cite:178]

**کتابخانه پیشنهادی:**
- custom taxonomy-based scoring
- `pandas`

## گروه 6: متریک‌های calibration و uncertainty

### ECE (Expected Calibration Error)

ECE برای سنجش انطباق confidence و correctness ضروری است و کارهای calibration-aware VLM بر کاهش ECE تأکید کرده‌اند.[cite:179]

**کتابخانه پیشنهادی:**
- `netcal`
- `torchmetrics.CalibrationError`
- custom binning

### MCE (Maximum Calibration Error)

MCE مکمل ECE است و worst-case miscalibration را نشان می‌دهد.

**کتابخانه پیشنهادی:**
- `netcal`
- custom code

### Brier Score

Brier score برای quality احتمالات خروجی مفید است و در ارزیابی reliability و safety توصیه می‌شود.[cite:170]

**کتابخانه پیشنهادی:**
- `scikit-learn.metrics.brier_score_loss`

### Negative Log-Likelihood (NLL)

NLL برای توزیع احتمالات مدل و calibration مفید است، به‌ویژه اگر output probabilistic نگه داشته شود.

**کتابخانه پیشنهادی:**
- `torch.nn.functional.cross_entropy`
- custom logging

### Selective Risk / Coverage

اگر مدل abstain می‌کند، selective risk و coverage باید گزارش شوند تا trade-off reliability و utility مشخص شود.

**کتابخانه پیشنهادی:**
- custom implementation
- `numpy`, `pandas`

## گروه 7: متریک‌های grounding و explainability

### Pointing Game Accuracy

اگر heatmap یا region attribution دارید، Pointing Game یک متریک استاندارد برای سنجش قرار گرفتن نقطه‌ی بیشینه روی ناحیه‌ی درست است.

**کتابخانه پیشنهادی:**
- custom implementation
- `numpy`, `opencv-python`

### IoU / Dice برای localization

اگر dataset یا subset دارای bounding box / segmentation annotation باشد، IoU یا Dice برای ناحیه‌ی evidence گزارش شود.

**کتابخانه پیشنهادی:**
- `torchmetrics.segmentation`
- `monai.metrics`

### Deletion / Insertion Curves

برای faithfulness explanation، می‌توان deletion/insertion curves را با mask کردن patchهای مهم محاسبه کرد. این متریک‌ها برای نشان دادن causality explanation مفیدند.

**کتابخانه پیشنهادی:**
- custom implementation
- `captum`

### Attention / Attribution Consistency

اگر چند view یا چند سطح سؤال دارید، consistency attribution بین آن‌ها را گزارش کنید. این متریک آماده‌ی کتابخانه‌ای استاندارد ندارد ولی برای novelty مقاله مفید است.

**کتابخانه پیشنهادی:**
- custom implementation
- `numpy`, `scipy`

## گروه 8: متریک‌های quality برای Question Curriculum Generator

چون QCG خودش یک جزء پژوهشی مهم است، باید جداگانه ارزیابی شود.

### Relevance to Main Question

سؤال‌های کمکی باید به سؤال اصلی و تصویر مرتبط باشند.

**کتابخانه پیشنهادی:**
- `sentence-transformers`
- cosine similarity
- human evaluation for spot checks

### Answerability Rate

نسبت سؤال‌های کمکی که واقعاً از روی تصویر قابل پاسخ‌اند باید گزارش شود.

**کتابخانه پیشنهادی:**
- human evaluation subset
- custom rule-based validation

### Level Accuracy

درستی تخصیص level برای سؤال‌های Level 1/2/3 باید سنجیده شود.

**کتابخانه پیشنهادی:**
- `scikit-learn`

### Diversity

برای جلوگیری از تکرار، self-BLEU یا pairwise semantic diversity محاسبه شود.

**کتابخانه پیشنهادی:**
- `nltk` برای self-BLEU
- `sentence-transformers` برای semantic diversity

## گروه 9: متریک‌های انسانی

مرورهای جدید ارزیابی medical generation تأکید می‌کنند که automated metrics به‌تنهایی کافی نیستند و human review هنوز لازم است، چون lexical similarity ممکن است quality بالینی را اشتباه تخمین بزند.[cite:162][cite:163][cite:164][cite:172]

### Clinical Correctness (Likert 1–5)

ارزیابی توسط متخصص برای صحت بالینی پاسخ.

### Image Grounding (Likert 1–5)

آیا پاسخ واقعاً از تصویر قابل استنتاج است؟

### Helpfulness / Completeness

آیا پاسخ برای decision support مفید است؟

### Hallucination Severity

شدت خطای بالینی در صورت hallucination.

**ابزار پیشنهادی:**
- `pandas` برای فرم annotation
- Google Forms / REDCap / spreadsheets برای جمع‌آوری داده
- `statsmodels` یا `pingouin` برای inter-rater agreement

### Inter-rater agreement

- Cohen’s kappa
- Fleiss’ kappa
- Krippendorff’s alpha

**کتابخانه پیشنهادی:**
- `scikit-learn`
- `statsmodels`
- `krippendorff`
- `pingouin`

## جدول نهایی متریک‌ها و کتابخانه‌ها

| گروه | متریک | باید گزارش شود؟ | کتابخانه پیشنهادی |
|---|---|---|---|
| VQA | Accuracy | بله | `scikit-learn`, `torchmetrics` |
| VQA | Exact Match | بله | custom |
| VQA | Precision / Recall / F1 | بله | `scikit-learn`, `torchmetrics` |
| NLG | BLEU-1/2/3/4 | بله | `pycocoevalcap`, `nltk`, `evaluate` |
| NLG | ROUGE-1/2/L/Lsum | بله | `rouge-score`, `evaluate` |
| NLG | METEOR | بله | `pycocoevalcap`, `nltk` |
| NLG | CIDEr | بله | `pycocoevalcap`, `caption_eval` |
| NLG | CHRF++ | ترجیحاً | `sacrebleu` |
| Semantic | BERTScore P/R/F1 | بله | `bert-score` |
| Semantic | Embedding cosine similarity | ترجیحاً | `sentence-transformers` |
| Clinical | Clinical P/R/F1 | بله | `scikit-learn`, `scispaCy` |
| Clinical | Label-overlap score | بله | `scispaCy`, custom |
| Clinical | RadGraph F1 / clinical metric | اگر پاسخ‌ها طولانی باشند | metric-specific repos |
| Hallucination | Hallucination rate | بله | custom |
| Hallucination | Hallucination P/R/F1 | بله | `scikit-learn`, `torchmetrics` |
| Hallucination | AUROC | بله | `scikit-learn`, `torchmetrics` |
| Hallucination | AUPRC | بله | `scikit-learn`, `torchmetrics` |
| Hallucination | FPR@95TPR | ترجیحاً | custom |
| Hallucination | Severity-weighted score | ترجیحاً | custom |
| Calibration | ECE | بله | `netcal`, `torchmetrics` |
| Calibration | MCE | ترجیحاً | `netcal` |
| Calibration | Brier score | بله | `scikit-learn` |
| Calibration | NLL | ترجیحاً | PyTorch |
| Selective prediction | Risk / Coverage | بله اگر abstain دارید | custom |
| Grounding | Pointing Game | اگر explanation map دارید | custom |
| Grounding | IoU / Dice | اگر region GT دارید | `torchmetrics`, `MONAI` |
| Explainability | Deletion / Insertion | ترجیحاً | `captum`, custom |
| QCG | Relevance | بله | `sentence-transformers` |
| QCG | Answerability | بله | custom + human review |
| QCG | Level accuracy | بله | `scikit-learn` |
| QCG | Diversity | ترجیحاً | `nltk`, `sentence-transformers` |
| Human | Clinical correctness | بله روی subset | annotation workflow |
| Human | Grounding / Severity | بله روی subset | annotation workflow |
| Human | Inter-rater agreement | بله | `scikit-learn`, `krippendorff` |

## کتابخانه‌های پیشنهادی برای requirements.txt

```text
numpy
pandas
scipy
scikit-learn
torch
torchvision
torchmetrics
evaluate
nltk
rouge-score
bert-score
sentence-transformers
sacrebleu
pycocoevalcap
caption-eval
netcal
captum
opencv-python
monai
spacy
scispacy
statsmodels
pingouin
krippendorff
```

## پیشنهاد اجرایی برای pipeline ارزیابی

بهترین کار این است که evaluation را به پنج اسکریپت مستقل تقسیم کنید:

1. `eval_vqa_core.py` برای Accuracy, EM, Precision, Recall, F1
2. `eval_nlg.py` برای BLEU, ROUGE, METEOR, CIDEr, CHRF++, BERTScore
3. `eval_hallucination.py` برای hallucination rate, P/R/F1, AUROC, AUPRC, severity scores
4. `eval_calibration_grounding.py` برای ECE, Brier, NLL, risk-coverage, IoU, pointing game
5. `eval_human_study.py` برای human ratings و inter-rater agreement

این تفکیک باعث می‌شود هر بخش reproducible بماند و داور هم حس کند ارزیابی مطالعه عمیق و ساخت‌یافته است. از آنجا که benchmarkها و reviewهای جدید repeatedly نشان داده‌اند که ارزیابی تک‌متریکی در medical generation گمراه‌کننده است، داشتن این پروتکل چندلایه یک مزیت واقعی برای پذیرش مقاله خواهد بود.[cite:163][cite:164][cite:168][cite:176][cite:178]

## حداقل بسته‌ی متریک‌های لازم برای submission قوی

اگر compute و زمان محدود باشد ولی همچنان بخواهید submission قوی بماند، حداقل این بسته باید گزارش شود:

- Accuracy, EM, F1
- BLEU-1/4, ROUGE-L, METEOR, CIDEr, BERTScore
- Hallucination rate, hallucination F1, AUROC, AUPRC
- ECE, Brier score
- Clinical correctness human study روی subset
- QCG relevance و answerability

## بسته‌ی کامل و ایده‌آل برای مقاله‌ی ژورنالی عمیق

برای نسخه‌ی کاملاً thesis-level و journal-level، همه‌ی متریک‌های جدول بالا باید گزارش شوند و نتایج به تفکیک dataset، question type، modality، و complexity level هم ارائه شوند. در benchmarkهای جدید GI-VQA و medical hallucination evaluation نیز تأکید بر breakdownهای چندسطحی و علت‌محور دیده می‌شود، نه فقط یک میانگین کلی.[cite:177][cite:178]

## قالب‌های جدول برای بخش نتایج مقاله

برای داوری قوی در ژورنال، ارائه‌ی نتایج باید ساخت‌یافته، مقایسه‌پذیر، و چندلایه باشد. مرورها و benchmarkهای جدید نشان داده‌اند که گزارش یک یا دو metric نهایی کافی نیست و breakdown بر اساس نوع سؤال، modality، و reliability ضروری است.[cite:168][cite:172][cite:178]

### Table Template 1: Main comparison on standard Med-VQA datasets

این جدول برای مقایسه‌ی مدل پیشنهادی با baselineها و SOTA روی دیتاست‌های استاندارد استفاده می‌شود.

```markdown
| Model | VQA-RAD Acc | VQA-RAD F1 | SLAKE Acc | SLAKE F1 | PathVQA Acc | PathVQA F1 | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore-F1 | Halluc. Rate ↓ | Halluc. F1 | AUROC | ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline-1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Baseline-2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Medical VLM |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Proposed CQC-Net |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
```

### Table Template 2: Kvasir-VQA-x1 reasoning breakdown

این دیتاست برای reasoning پیچیده مناسب است، بنابراین باید breakdown بر اساس complexity level گزارش شود.[cite:41][cite:80]

```markdown
| Model | L1 Acc | L2 Acc | L3 Acc | Overall Acc | BLEU-4 | BERTScore-F1 | Halluc. Rate ↓ | Cause-Visual ↓ | Cause-Knowledge ↓ | Cause-Context ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline |  |  |  |  |  |  |  |  |  |  |
| Proposed CQC-Net |  |  |  |  |  |  |  |  |  |  |
```

### Table Template 3: Hallucination detection performance

```markdown
| Model | Halluc. Precision | Halluc. Recall | Halluc. F1 | AUROC | AUPRC | FPR@95TPR ↓ | Severity Score ↓ | ECE ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Detector-1 |  |  |  |  |  |  |  |  |  |
| Detector-2 |  |  |  |  |  |  |  |  |  |
| Proposed Consistency Head |  |  |  |  |  |  |  |  |  |
```

### Table Template 4: Grounding and explanation quality

```markdown
| Model | Pointing Game ↑ | IoU ↑ | Dice ↑ | Deletion AUC ↓ | Insertion AUC ↑ | Attribution Consistency ↑ | Human Grounding Score ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline |  |  |  |  |  |  |  |
| Proposed CQC-Net |  |  |  |  |  |  |  |
```

### Table Template 5: Human evaluation

چون متریک‌های خودکار همیشه با judgement بالینی همبسته نیستند، human evaluation باید جدا گزارش شود.[cite:162][cite:163][cite:172]

```markdown
| Model | Clinical Correctness ↑ | Image Grounding ↑ | Helpfulness ↑ | Hallucination Severity ↓ | Cohen's Kappa | Fleiss' Kappa |
|---|---:|---:|---:|---:|---:|---:|
| Baseline |  |  |  |  |  |  |
| Proposed CQC-Net |  |  |  |  |  |  |
```

### Table Template 6: Calibration and abstention

```markdown
| Model | ECE ↓ | MCE ↓ | Brier ↓ | NLL ↓ | Coverage @ tau1 | Risk @ tau1 ↓ | Coverage @ tau2 | Risk @ tau2 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline |  |  |  |  |  |  |  |  |
| Proposed CQC-Net |  |  |  |  |  |  |  |  |
```

## قالب‌های جدول برای آبلیشن

برای اینکه contributionهای مقاله قانع‌کننده شوند، آبلیشن‌ها باید دقیقاً نشان دهند هر جزء چه اثری داشته است. داوران TMI معمولاً به‌شدت به این بخش حساس‌اند، به‌خصوص وقتی روش چندماژوله و novelty-driven باشد.[cite:6][cite:168]

### Ablation Template 1: ماژول‌های اصلی

```markdown
| Setting | QCG | Verifier | Consistency Head | Refiner | Abstention | Acc | F1 | BLEU-4 | CIDEr | Halluc. Rate ↓ | Halluc. F1 | AUROC | ECE ↓ |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full model | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |
| w/o QCG | ✗ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |
| w/o Verifier | ✓ | ✗ | ✓ | ✓ | ✓ |  |  |  |  |  |  |  |  |
| w/o Consistency | ✓ | ✓ | ✗ | ✓ | ✓ |  |  |  |  |  |  |  |  |
| w/o Refiner | ✓ | ✓ | ✓ | ✗ | ✓ |  |  |  |  |  |  |  |  |
| w/o Abstention | ✓ | ✓ | ✓ | ✓ | ✗ |  |  |  |  |  |  |  |  |
```

### Ablation Template 2: ساخت curriculum

```markdown
| Curriculum Strategy | Template-only | SLM-only | Hybrid | #Aux Questions | Acc | BERTScore-F1 | Halluc. Rate ↓ | QCG Relevance ↑ | QCG Answerability ↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Strategy-1 | ✓ | ✗ | ✗ | 5 |  |  |  |  |  |
| Strategy-2 | ✗ | ✓ | ✗ | 5 |  |  |  |  |  |
| Strategy-3 | ✗ | ✗ | ✓ | 5 |  |  |  |  |  |
| Strategy-4 | ✗ | ✗ | ✓ | 3 |  |  |  |  |  |
| Strategy-5 | ✗ | ✗ | ✓ | 7 |  |  |  |  |  |
```

### Ablation Template 3: سطح‌های curriculum

```markdown
| Setting | Level-1 | Level-2 | Level-3 | Acc | F1 | Halluc. Rate ↓ | AUROC | ECE ↓ |
|---|---|---|---|---:|---:|---:|---:|---:|
| Full (1+2+3) | ✓ | ✓ | ✓ |  |  |  |  |  |
| Only L1 | ✓ | ✗ | ✗ |  |  |  |  |  |
| L1+L2 | ✓ | ✓ | ✗ |  |  |  |  |  |
| L2+L3 | ✗ | ✓ | ✓ |  |  |  |  |  |
```

### Ablation Template 4: loss functions

```markdown
| Setting | L_QA | L_ground | L_cons | L_hallu | L_cal | Acc | BLEU-4 | CIDEr | Halluc. F1 | AUROC | ECE ↓ |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Full loss | ✓ | ✓ | ✓ | ✓ | ✓ |  |  |  |  |  |  |
| w/o L_ground | ✓ | ✗ | ✓ | ✓ | ✓ |  |  |  |  |  |  |
| w/o L_cons | ✓ | ✓ | ✗ | ✓ | ✓ |  |  |  |  |  |  |
| w/o L_hallu | ✓ | ✓ | ✓ | ✗ | ✓ |  |  |  |  |  |  |
| w/o L_cal | ✓ | ✓ | ✓ | ✓ | ✗ |  |  |  |  |  |  |
```

### Ablation Template 5: backbone comparison

```markdown
| Vision Encoder | Language Model | Params | Trainable Params | Acc | F1 | BLEU-4 | Halluc. Rate ↓ | ECE ↓ | Train Time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DenseNet-121 | Phi-small |  |  |  |  |  |  |  |  |
| ViT-B/16 | Phi-small |  |  |  |  |  |  |  |  |
| Swin-B | Qwen-small |  |  |  |  |  |  |  |  |
| Proposed Final | Qwen-small |  |  |  |  |  |  |  |  |
```

## پرامپت آماده برای AI coder

بخش زیر به‌صورت مستقیم برای دادن به یک مدل کدنویس طراحی شده است. این prompt باید به‌عنوان instruction اصلی استفاده شود تا implementation مرحله‌ای، تمیز، و قابل بازتولید تولید شود.

### Prompt اصلی

```text
You are an expert ML engineer and medical vision-language researcher.
Implement a full PyTorch project for the paper idea below.

Project title:
CQC-Net: Counterfactual Question Curriculum Network for Hallucination-Aware Medical VQA.

Main goal:
Build a medical VQA system that, for each image and main question, generates a structured curriculum of auxiliary questions at three levels:
(1) existence/localization,
(2) attribute/relation,
(3) clinical inference.
The model must answer the main and auxiliary questions, verify answer grounding against image evidence, estimate hallucination risk from hierarchical inconsistency, and optionally refine or abstain.

Core modules to implement:
1. Visual encoder
2. Text encoder
3. Question Curriculum Generator (QCG)
4. Answer generator
5. Grounding & Evidence Verifier
6. Curriculum Consistency Head
7. Optional Refiner
8. Evaluation pipeline

Implementation constraints:
- Use Python and PyTorch.
- Organize code as a clean research repo.
- Provide configs for datasets, training, and evaluation.
- Make the code modular so encoders and LMs can be swapped.
- Support at least VQA-RAD, SLAKE, and PathVQA first.
- Add placeholders/hooks for PMC-VQA and Kvasir-VQA-x1.
- Implement deterministic seeds and reproducible evaluation.
- Save metrics as CSV and JSON.
- Save predictions, hallucination flags, confidence, and grounding outputs.

Model design requirements:
- QCG should generate 3-level auxiliary questions.
- The answer generator should answer both main and auxiliary questions.
- The verifier should compute grounding/compatibility scores.
- The consistency head should aggregate scores and produce hallucination probability.
- Add abstention thresholds and a refinement path.

Training stages:
Stage 1: baseline Med-VQA training
Stage 2: curriculum data construction
Stage 3: QCG training
Stage 4: joint training with verifier and consistency head
Stage 5: optional end-to-end fine-tuning
Stage 6: calibration and threshold tuning

Losses to implement:
- QA loss
- Grounding loss
- Consistency loss
- Hallucination detection loss
- Calibration loss

Evaluation metrics to implement:
- Accuracy, Exact Match, Precision, Recall, F1
- BLEU-1/2/3/4, ROUGE-1/2/L/Lsum, METEOR, CIDEr, CHRF++
- BERTScore
- Clinical label overlap metrics
- Hallucination rate, hallucination precision/recall/F1
- AUROC, AUPRC, FPR@95TPR
- ECE, MCE, Brier score, NLL
- Selective risk / coverage
- Pointing game, IoU, Dice, attribution consistency
- QCG relevance, QCG answerability, QCG level accuracy, QCG diversity

Libraries to use where appropriate:
- torch, torchvision, torchmetrics
- transformers
- scikit-learn
- evaluate
- rouge-score
- nltk
- pycocoevalcap
- bert-score
- sentence-transformers
- sacrebleu
- netcal
- captum
- monai
- spacy / scispacy
- pandas / numpy / scipy

Repo structure to create:
configs/
data/
models/
training/
evaluation/
scripts/
utils/
outputs/

Required deliverables:
1. Complete codebase
2. requirements.txt
3. README with setup and training instructions
4. dataset preparation scripts
5. training scripts for each stage
6. evaluation scripts for each metric group
7. ablation script or config variants
8. inference script
9. sample experiment config files

Coding style requirements:
- Production-quality but research-friendly
- Clear docstrings
- No unnecessary abstraction
- Type hints where useful
- Logging with tqdm + python logging
- YAML config support
- Simple trainer first; avoid heavy frameworks unless necessary

Output expectations:
- Start by generating the full repo tree.
- Then implement files one by one.
- For each file, provide complete code.
- Do not leave placeholders like TODO unless explicitly marked as extension points.
- Prefer working baseline code over overengineered designs.
```

### Prompt برای فازبندی اجرایی

```text
Implement the project in this order:
1. Repo skeleton
2. Dataset loaders and preprocessing
3. Baseline Med-VQA model
4. Core training loop
5. Core evaluation metrics
6. QCG module
7. Verifier module
8. Consistency head
9. Hallucination metrics
10. Calibration and abstention pipeline
11. Full ablation support
12. README and reproducibility utilities

After each phase, show the exact files created and how to run a minimal experiment.
```

### Prompt برای تولید جدول‌های نتایج به‌صورت خودکار

```text
Add scripts that automatically aggregate experiment outputs into publication-ready CSV tables for:
- main results
- hallucination results
- grounding results
- human evaluation summaries
- ablations
- calibration/abstention results

Each script should read per-run JSON/CSV outputs and create a merged summary table.
```

### Prompt برای dataset preparation

```text
Implement dataset preparation pipelines for VQA-RAD, SLAKE, and PathVQA.
Normalize answer text, create train/val/test splits if needed, generate canonical IDs, and export unified JSON format:
{
  "image_path": ...,
  "question": ...,
  "answer": ...,
  "question_type": ...,
  "dataset": ...,
  "meta": {...}
}
Also add hooks to attach curriculum questions and hallucination labels later.
```

### Prompt برای QCG data builder

```text
Implement a curriculum-data builder that creates auxiliary question chains with three levels:
- Level 1: existence/localization
- Level 2: attribute/relation
- Level 3: clinical inference

Support:
- template-only mode
- SLM-generated mode
- hybrid mode

Export outputs in JSONL and include fields for:
main_question, auxiliary_questions, level_labels, answerability_flag, source_method.
```

## توصیه برای استفاده از این promptها

بهتر است AI coder را وادار کنید پروژه را در یک مرحله‌ی غول‌آسا تولید نکند، بلکه فازبندی‌شده جلو برود. این کار هم خطا را کم می‌کند و هم کنترل شما را بر کیفیت engineering بالا می‌برد. همچنین برای مقاله‌ی پذیرش‌پذیر، reproducibility، config-driven experimentation، و auto-generated result tables بسیار مهم‌اند؛ بسیاری از benchmarkها و مطالعات جدید روی standardized evaluation و structured comparison تأکید کرده‌اند.[cite:174][cite:178]
