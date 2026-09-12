const { GoogleGenerativeAI } = require('@google/generative-ai');

const GEMINI_API_KEY = process.env.EXPO_PUBLIC_GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
  console.warn(
    '[AI] Missing EXPO_PUBLIC_GEMINI_API_KEY; AI routes will error.'
  );
}

const genAI = GEMINI_API_KEY ? new GoogleGenerativeAI(GEMINI_API_KEY) : null;

// NIM / Gemma fallbacks disabled by default
const NIM_API_KEY = null;
const NIM_BASE_URL =
  process.env.NVIDIA_BASE_URL || 'https://integrate.api.nvidia.com/v1';
const NIM_QUESTION_MODELS = [];
const NIM_DIAGNOSIS_MODELS = [];
const NIM_CONVERSATION_MODELS = [];

const OPENROUTER_BASE_URL =
  process.env.OPENROUTER_BASE_URL || 'https://openrouter.ai/api/v1';

const OPENROUTER_API_KEY =
  process.env.OPENROUTER_API_KEY || process.env.EXPO_PUBLIC_OPENROUTER_API_KEY;

const OPENROUTER_VISION_MODEL =
  process.env.OPENROUTER_VISION_MODEL || 'nvidia/nemotron-nano-12b-v2-vl:free';

const NIM_OCR_MODEL =
  process.env.EXPO_PUBLIC_NVIDIA_OCR_MODEL ||
  process.env.NVIDIA_OCR_MODEL ||
  'nvidia/nemotron-ocr-v1';

const INFERMEDICA_BASE_URL =
  process.env.INFERMEDICA_BASE_URL || 'https://api.infermedica.com/v3';

const INFERMEDICA_APP_ID = process.env.INFERMEDICA_APP_ID || null;
const INFERMEDICA_APP_KEY = process.env.INFERMEDICA_APP_KEY || null;

const URGENCY_PATTERNS = [
  /chest\s+pain/i,
  /shortness\s+of\s+breath/i,
  /cannot\s+breathe/i,
  /stroke/i,
  /suicid/i,
  /fainted|fainting|unconscious/i,
  /severe\s+bleeding/i,
  /बेहोश|सांस\s*नहीं\s*आ\s*रही|सीने\s*में\s*दर्द|आत्महत्या/i
];

const URGENCY_TEXT =
  'This may be urgent. Call 108 now or go to the nearest emergency facility immediately.';

function parseModelList(raw, fallback) {
  if (!raw || typeof raw !== 'string') return fallback;
  const parsed = raw
    .split(',')
    .map((m) => m.trim().replace(/^models\//i, ''))
    .filter(Boolean);
  return parsed.length ? parsed : fallback;
}

const GEMINI_MODELS = parseModelList(
  process.env.EXPO_PUBLIC_GEMINI_MODEL_FALLBACKS ||
    process.env.GEMINI_MODEL_FALLBACKS ||
    process.env.EXPO_PUBLIC_GEMINI_MODEL ||
    process.env.GEMINI_MODEL,
  [
    'gemini-flash-latest',
    'gemini-flash-lite-latest',
    'gemini-2.5-flash-lite',
    'gemini-3.1-flash-lite'
  ]
);

function stripJson(text = '') {
  return text
    .replace(/```json/gi, '')
    .replace(/```/g, '')
    .trim();
}

function safeParseJson(text) {
  try {
    return JSON.parse(stripJson(text));
  } catch (e) {
    return null;
  }
}

function extractJsonObject(text = '') {
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start < 0 || end <= start) return null;
  return text.slice(start, end + 1);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

async function callGemini(prompt) {
  if (!genAI) throw new Error('Gemini client not configured');

  let lastErr = null;
  for (const modelName of GEMINI_MODELS) {
    try {
      const model = genAI.getGenerativeModel({ model: modelName });
      const result = await model.generateContent(prompt);
      const response = await result.response;
      const text = await response.text();
      if (!text?.trim()) {
        throw new Error(`Gemini model ${modelName} returned empty response`);
      }
      return { text, modelName };
    } catch (err) {
      console.warn(
        `[AI] Gemini model ${modelName} failed: ${err?.message || err}`
      );
      lastErr = err;
    }
  }

  throw new Error(
    `Gemini failed for all configured models (${GEMINI_MODELS.join(', ')}). Last error: ${String(lastErr)}`
  );
}

async function callNimChat(model, messages, opts = {}) {
  if (!NIM_API_KEY) {
    throw new Error('NVIDIA API key missing for NIM fallback');
  }

  const res = await fetch(`${NIM_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${NIM_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model,
      messages,
      temperature: opts.temperature ?? 0.2,
      max_tokens: opts.maxTokens ?? 2048
    })
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(
      `NIM ${model} failed (${res.status}): ${body.slice(0, 220)}`
    );
  }

  const payload = await res.json();
  const content = payload?.choices?.[0]?.message?.content;
  if (typeof content !== 'string' || !content.trim()) {
    throw new Error(`NIM ${model} returned empty content`);
  }
  return content;
}

async function callOpenRouterChat(messages, opts = {}) {
  if (!OPENROUTER_API_KEY) {
    throw new Error('OPENROUTER_API_KEY missing for vision tasks');
  }

  const response = await fetch(`${OPENROUTER_BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${OPENROUTER_API_KEY}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': process.env.APP_URL || 'https://nirogya.app',
      'X-Title': 'Nirogya'
    },
    body: JSON.stringify({
      model: opts.model || OPENROUTER_VISION_MODEL,
      messages,
      temperature: opts.temperature ?? 0.2,
      max_tokens: opts.maxTokens ?? 900
    })
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(
      `OpenRouter failed (${response.status}): ${body.slice(0, 220)}`
    );
  }

  const payload = await response.json();
  const content = payload?.choices?.[0]?.message?.content;
  if (typeof content !== 'string' || !content.trim()) {
    throw new Error('OpenRouter returned empty content');
  }

  return content;
}

function safeParseJsonObject(text = '') {
  const direct = safeParseJson(text);
  if (direct && typeof direct === 'object') return direct;

  const extracted = extractJsonObject(text);
  if (!extracted) return null;

  try {
    return JSON.parse(extracted);
  } catch {
    return null;
  }
}

function mapWhisperLanguageCode(language = 'en') {
  const key = String(language || 'en')
    .trim()
    .toLowerCase();
  const map = {
    en: 'en',
    english: 'en',
    hi: 'hi',
    hindi: 'hi',
    ta: 'ta',
    tamil: 'ta',
    te: 'te',
    telugu: 'te',
    kn: 'kn',
    kannada: 'kn',
    ml: 'ml',
    malayalam: 'ml',
    bn: 'bn',
    bengali: 'bn',
    mr: 'mr',
    marathi: 'mr',
    gu: 'gu',
    gujarati: 'gu',
    pa: 'pa',
    punjabi: 'pa',
    or: 'or',
    odia: 'or',
    as: 'as',
    assamese: 'as',
    ur: 'ur',
    urdu: 'ur'
  };

  return map[key] || 'en';
}

async function generateTextWithFallback(task, prompt, nimModels) {
  try {
    const generated = await callGemini(prompt);
    return { text: generated.text, source: generated.modelName };
  } catch (geminiErr) {
    // No NIM fallback: propagate so caller can use rule-based defaults
    throw geminiErr;
  }
}

function fallbackQuestions(symptom) {
  return {
    questions: [
      {
        id: 'q1',
        text: `How long have you had ${symptom}?`,
        options: ['<1 week', '1-4 weeks', '1-3 months', '3+ months']
      },
      {
        id: 'q2',
        text: 'How severe is it?',
        options: ['Mild', 'Moderate', 'Severe', 'Unbearable']
      },
      {
        id: 'q3',
        text: 'Any related symptoms?',
        options: ['Fatigue', 'Fever', 'Headache', 'None']
      },
      {
        id: 'q4',
        text: 'Does it affect daily activities?',
        options: ['No', 'Slightly', 'Yes', 'Cannot function']
      },
      {
        id: 'q5',
        text: 'Have you seen a doctor?',
        options: [
          'No',
          'Yes recently',
          'Yes >6 months ago',
          'Prefer not to say'
        ]
      }
    ]
  };
}

function normalizeQuestionSet(candidate, symptom) {
  if (!candidate || !Array.isArray(candidate.questions)) {
    return fallbackQuestions(symptom);
  }

  const questions = candidate.questions
    .slice(0, 5)
    .map((item, idx) => {
      const text = typeof item?.text === 'string' ? item.text.trim() : '';
      const options = Array.isArray(item?.options)
        ? item.options
            .map((o) => (typeof o === 'string' ? o.trim() : ''))
            .filter(Boolean)
            .slice(0, 4)
        : [];

      if (!text || options.length < 2) return null;

      return {
        id:
          typeof item.id === 'string' && item.id.trim()
            ? item.id
            : `q${idx + 1}`,
        text,
        options
      };
    })
    .filter(Boolean);

  if (questions.length < 3) {
    return fallbackQuestions(symptom);
  }

  return { questions };
}

function ruleDiagnosis(symptom, answers = {}) {
  const text =
    Object.values(answers).join(' ').toLowerCase() +
    ' ' +
    symptom.toLowerCase();
  let riskScore = 40;
  let riskLevel = 'low';
  const nextSteps = [];

  if (
    text.includes('heavy') ||
    text.includes('dizzy') ||
    text.includes('fatigue')
  ) {
    riskScore = 70;
    riskLevel = 'medium';
    nextSteps.push('Check hemoglobin (Hb)');
    nextSteps.push('Increase iron-rich foods (spinach, jaggery, lentils)');
  }
  if (
    text.includes('insomnia') ||
    text.includes('stress') ||
    text.includes('tired')
  ) {
    riskScore = Math.max(riskScore, 60);
    riskLevel = riskScore >= 70 ? 'medium' : 'low';
    nextSteps.push(
      'Follow sleep hygiene: fixed bedtime, no screens 60 mins prior'
    );
  }

  return {
    diagnosis: `Possible ${riskLevel === 'medium' ? 'nutrient or stress-related issue' : 'mild concern'}`,
    description:
      'Based on your answers, this is a preliminary risk indicator. Please monitor and consider basic tests.',
    riskScore,
    riskLevel,
    nextSteps: nextSteps.length ? nextSteps : ['Monitor symptoms for 3-5 days'],
    seeDoctor: riskScore >= 70,
    urgency: riskScore >= 70 ? 'Within 2 weeks' : 'Monitor'
  };
}

function normalizeDiagnosis(candidate, symptom) {
  const fallback = ruleDiagnosis(symptom, {}, {});
  if (!candidate || typeof candidate !== 'object') return fallback;

  const diagnosis =
    typeof candidate.diagnosis === 'string' && candidate.diagnosis.trim()
      ? candidate.diagnosis.trim()
      : fallback.diagnosis;

  const description =
    typeof candidate.description === 'string' && candidate.description.trim()
      ? candidate.description.trim()
      : fallback.description;

  const riskScore = clamp(
    Number(candidate.riskScore) || fallback.riskScore,
    0,
    100
  );
  const riskLevelRaw = String(candidate.riskLevel || '').toLowerCase();
  const riskLevel = ['low', 'medium', 'high'].includes(riskLevelRaw)
    ? riskLevelRaw
    : riskScore >= 75
      ? 'high'
      : riskScore >= 40
        ? 'medium'
        : 'low';

  const nextSteps = Array.isArray(candidate.nextSteps)
    ? candidate.nextSteps
        .map((s) => (typeof s === 'string' ? s.trim() : ''))
        .filter(Boolean)
        .slice(0, 6)
    : [];

  return {
    diagnosis,
    description,
    riskScore,
    riskLevel,
    nextSteps: nextSteps.length ? nextSteps : fallback.nextSteps,
    seeDoctor:
      typeof candidate.seeDoctor === 'boolean'
        ? candidate.seeDoctor
        : riskLevel === 'high' || riskScore >= 70,
    urgency:
      typeof candidate.urgency === 'string' && candidate.urgency.trim()
        ? candidate.urgency.trim()
        : fallback.urgency
  };
}

function needsTranslation(targetLanguage) {
  if (!targetLanguage || typeof targetLanguage !== 'string') return false;
  const normalized = targetLanguage.trim().toLowerCase();
  return normalized && normalized !== 'en' && normalized !== 'english';
}

function normalizeTargetLanguage(targetLanguage) {
  const value = targetLanguage.trim().toLowerCase();
  const map = {
    hi: 'Hindi',
    ta: 'Tamil',
    te: 'Telugu',
    bn: 'Bengali',
    kn: 'Kannada',
    ml: 'Malayalam',
    mr: 'Marathi',
    gu: 'Gujarati',
    pa: 'Punjabi',
    or: 'Odia',
    ur: 'Urdu'
  };
  return map[value] || targetLanguage;
}

async function translateMany(texts, targetLanguage) {
  if (
    !needsTranslation(targetLanguage) ||
    !Array.isArray(texts) ||
    texts.length === 0
  ) {
    return texts;
  }
  // Translation disabled (no NIM); return original
  return texts;
}

async function translateQuestionSet(questionSet, targetLanguage) {
  if (!needsTranslation(targetLanguage)) return questionSet;
  const questions = questionSet.questions || [];
  if (!questions.length) return questionSet;

  const questionTexts = questions.map((q) => q.text);
  const translatedQuestionTexts = await translateMany(
    questionTexts,
    targetLanguage
  );

  const flatOptions = questions.flatMap((q) => q.options || []);
  const translatedOptions = await translateMany(flatOptions, targetLanguage);

  let optionCursor = 0;
  const translatedQuestions = questions.map((q, idx) => {
    const options = (q.options || []).map((originalOpt) => {
      const next = translatedOptions[optionCursor];
      optionCursor += 1;
      return next || originalOpt;
    });

    return {
      ...q,
      text: translatedQuestionTexts[idx] || q.text,
      options
    };
  });

  return { questions: translatedQuestions };
}

async function translateDiagnosis(diagnosisResult, targetLanguage) {
  if (!needsTranslation(targetLanguage)) return diagnosisResult;
  const bundle = [
    diagnosisResult.diagnosis,
    diagnosisResult.description,
    ...(diagnosisResult.nextSteps || []),
    diagnosisResult.urgency
  ];

  const translated = await translateMany(bundle, targetLanguage);
  const stepsCount = diagnosisResult.nextSteps.length;

  return {
    ...diagnosisResult,
    diagnosis: translated[0] || diagnosisResult.diagnosis,
    description: translated[1] || diagnosisResult.description,
    nextSteps: diagnosisResult.nextSteps.map(
      (step, idx) => translated[2 + idx] || step
    ),
    urgency: translated[2 + stepsCount] || diagnosisResult.urgency
  };
}

async function translateTextIfNeeded(text, targetLanguage) {
  if (!needsTranslation(targetLanguage) || !text?.trim()) return text;
  const translated = await translateMany([text], targetLanguage);
  return translated?.[0] || text;
}

function computeFairnessBreakdown(profile = {}, diagnosisResult = {}) {
  const monthlyIncome = Number(profile.monthlyIncome ?? profile.income ?? 0);
  const distanceKm = Number(
    profile.hospitalDistanceKm ?? profile.distanceToFacility ?? 0
  );
  const hasPmjay = Boolean(
    profile.hasPmjay ||
    String(profile.insuranceType || '').toUpperCase() === 'PMJAY'
  );

  const estimatedCost =
    typeof diagnosisResult.estimatedCost === 'number'
      ? diagnosisResult.estimatedCost
      : diagnosisResult.riskLevel === 'high'
        ? 3500
        : diagnosisResult.riskLevel === 'medium'
          ? 1500
          : 500;

  const effectiveCost = hasPmjay ? 0 : estimatedCost;
  const costRatio =
    monthlyIncome > 0 ? effectiveCost / Math.max(monthlyIncome, 1) : 0.4;

  const affordability =
    costRatio <= 0.1
      ? 1
      : costRatio <= 0.2
        ? 0.8
        : costRatio <= 0.3
          ? 0.6
          : costRatio <= 0.5
            ? 0.3
            : 0;

  const travelTimeMins = (distanceKm / 30) * 60;
  const accessibility =
    travelTimeMins <= 30
      ? 1
      : travelTimeMins <= 60
        ? 0.8
        : travelTimeMins <= 90
          ? 0.5
          : 0.2;

  const relevance = needsTranslation(profile.language || 'en') ? 0.9 : 1;

  const equityScore = Number(
    clamp(
      affordability * 0.4 + accessibility * 0.4 + relevance * 0.2,
      0,
      1
    ).toFixed(2)
  );

  const explanation =
    `Affordability ${affordability.toFixed(2)}, accessibility ${accessibility.toFixed(2)}, ` +
    `relevance ${relevance.toFixed(2)}. Final fairness score ${equityScore.toFixed(2)}.`;

  return {
    equityScore,
    affordability,
    accessibility,
    relevance,
    estimatedCost,
    explanation
  };
}

function computeFairnessScore(profile = {}, diagnosisResult = {}) {
  return computeFairnessBreakdown(profile, diagnosisResult).equityScore;
}

function buildContextMap(profile = {}, language = 'en') {
  const monthlyIncome = Number(profile.monthlyIncome ?? profile.income ?? 0);
  return {
    language: language || profile.language || 'en',
    city: profile.city || null,
    district: profile.district || null,
    state: profile.state || null,
    districtHdiTier: profile.districtHdiTier || profile.hdiTier || null,
    monthlyIncome,
    affordabilityThreshold: monthlyIncome > 0 ? monthlyIncome * 0.3 : 0,
    hospitalDistanceKm: Number(
      profile.hospitalDistanceKm ?? profile.distanceToFacility ?? 0
    ),
    hasPmjay: Boolean(
      profile.hasPmjay ||
      String(profile.insuranceType || '').toUpperCase() === 'PMJAY'
    ),
    conditions: Array.isArray(profile.conditions) ? profile.conditions : [],
    medications: Array.isArray(profile.medications)
      ? profile.medications
      : profile.medications
        ? [profile.medications]
        : []
  };
}

function deriveRiskFlags(symptom = '', diagnosis = {}) {
  const text =
    `${symptom} ${diagnosis?.diagnosis || ''} ${diagnosis?.description || ''}`.toLowerCase();
  const flags = [];

  if (diagnosis?.riskLevel === 'high' || diagnosis?.riskScore >= 75) {
    flags.push({
      condition: 'HIGH_RISK_PATTERN',
      severity: 'HIGH',
      confidence: 0.8,
      triggerRules: ['riskLevelHighOrScore>=75']
    });
  }

  if (detectUrgency(text)) {
    flags.push({
      condition: 'URGENT_RED_FLAG',
      severity: 'URGENT',
      confidence: 0.95,
      triggerRules: ['urgencyKeywords']
    });
  }

  if (text.includes('fatigue') && text.includes('heavy')) {
    flags.push({
      condition: 'ANEMIA_RISK',
      severity: diagnosis?.riskLevel === 'high' ? 'HIGH' : 'MEDIUM',
      confidence: 0.7,
      triggerRules: ['fatigue+heavyFlowPattern']
    });
  }

  return flags;
}

function buildRecommendationFromAssessment(diagnosis = {}, contextMap = {}) {
  const score = Number(diagnosis?.riskScore || 0);
  const hasPmjay = Boolean(contextMap.hasPmjay);
  const distance = Number(contextMap.hospitalDistanceKm || 0);

  const facilityType =
    score >= 80
      ? 'Emergency'
      : distance > 25
        ? 'Teleconsultation / PHC'
        : 'Nearest Hospital';

  const estimatedCost = score >= 80 ? 4000 : score >= 55 ? 1500 : 500;

  const carePathway =
    score >= 80
      ? 'Seek urgent in-person evaluation immediately. Call 108 or go to the nearest emergency facility.'
      : score >= 55
        ? 'Book a doctor consultation within 1-2 weeks and start the suggested lifestyle adjustments today.'
        : 'Monitor symptoms for a few days, follow self-care guidance, and consult a doctor if symptoms persist.';

  return {
    carePathway,
    facilityType,
    estimatedCostLow: Math.max(0, estimatedCost - 300),
    estimatedCostHigh: estimatedCost + 500,
    pmjayApplicable: hasPmjay
  };
}

function detectUrgency(text = '') {
  const normalized = String(text || '').trim();
  if (!normalized) return false;
  return URGENCY_PATTERNS.some((pattern) => pattern.test(normalized));
}

async function getUrgencyReply(language) {
  return translateTextIfNeeded(URGENCY_TEXT, language || 'en');
}

async function generateQuestions(symptom, profile, language) {
  const safeSymptom = String(symptom || '').trim();
  if (!safeSymptom) {
    return { ...fallbackQuestions('your symptom'), source: 'rules' };
  }

  const prompt = `
You are a medical AI assistant for Indian women's health.
Symptom: "${safeSymptom}"
User profile (JSON): ${JSON.stringify(profile || {})}
Language: ${language || 'en'}
Generate exactly 5 follow-up questions with 4 options each in JSON format {"questions":[{"id":"q1","text":"...","options":["A","B","C","D"]}]}.
If you cannot comply, return the JSON fallback with simple severity/duration/options.
`;

  let questionSet = fallbackQuestions(safeSymptom);
  let source = 'rules';

  try {
    const generated = await generateTextWithFallback(
      'questions',
      prompt,
      NIM_QUESTION_MODELS
    );
    source = generated.source;
    const parsed = safeParseJson(generated.text);
    questionSet = normalizeQuestionSet(parsed, safeSymptom);
  } catch (err) {
    console.warn(
      '[AI] Question generation fallback used:',
      err?.message || err
    );
  }

  const translated = await translateQuestionSet(questionSet, language || 'en');
  return { ...translated, source };
}

async function generateDiagnosis(symptom, answers, profile, language) {
  const safeSymptom = String(symptom || '').trim();
  if (!safeSymptom) {
    return {
      ...ruleDiagnosis('your symptom', answers || {}, profile || {}),
      source: 'rules'
    };
  }

  const prompt = `
You are a medical AI assistant for Indian women.
Symptom: "${safeSymptom}"
Answers: ${JSON.stringify(answers || {})}
Profile: ${JSON.stringify(profile || {})}
Language: ${language || 'en'}
Respond ONLY with JSON:
{
  "diagnosis": "...",
  "description": "...",
  "riskScore": 0-100,
  "riskLevel": "low" | "medium" | "high",
  "nextSteps": ["..."],
  "seeDoctor": true|false,
  "urgency": "..." 
}
If you cannot comply, still return a JSON in that shape.
`;

  let diagnosis = ruleDiagnosis(safeSymptom, answers || {}, profile || {});
  let source = 'rules';

  try {
    const generated = await generateTextWithFallback(
      'diagnosis',
      prompt,
      NIM_DIAGNOSIS_MODELS
    );
    source = generated.source;
    const parsed = safeParseJson(generated.text);
    diagnosis = normalizeDiagnosis(parsed, safeSymptom);
  } catch (err) {
    console.warn(
      '[AI] Diagnosis generation fallback used:',
      err?.message || err
    );
  }

  const translated = await translateDiagnosis(diagnosis, language || 'en');
  return { ...translated, source };
}

async function generateConversationReply({
  message,
  profile,
  language,
  history = []
}) {
  if (detectUrgency(message)) {
    return {
      reply: await getUrgencyReply(language || 'en'),
      isUrgent: true,
      source: 'urgency-guard'
    };
  }

  const historyText = history
    .slice(-8)
    .map((item) => {
      const role = item?.role === 'assistant' ? 'Assistant' : 'User';
      return `${role}: ${String(item?.content || '').slice(0, 700)}`;
    })
    .join('\n');

  const prompt = `
You are Nira, an empathetic AI health assistant for Indian women.
Use plain language, be concise, and avoid alarmist wording.
You are not a doctor; include safety advice when needed.

User profile JSON: ${JSON.stringify(profile || {})}
Preferred language: ${language || 'en'}

Recent conversation history:
${historyText || 'No prior history'}

User message: ${String(message || '')}

Reply with practical next steps in 2-6 short lines.
`;

  try {
    const generated = await generateTextWithFallback(
      'conversation',
      prompt,
      NIM_CONVERSATION_MODELS
    );

    const reply =
      (await translateTextIfNeeded(
        stripJson(generated.text),
        language || 'en'
      )) || 'I am here to help. Please tell me more about your symptoms.';

    return {
      reply,
      isUrgent: false,
      source: generated.source
    };
  } catch (err) {
    console.warn('[AI] Conversation fallback used:', err?.message || err);
    return {
      reply: await translateTextIfNeeded(
        'I could not process that right now. Please try again in a moment.',
        language || 'en'
      ),
      isUrgent: false,
      source: 'conversation-fallback'
    };
  }
}

async function transcribeAudio({ audioBase64, language = 'en', mimeType }) {
  throw new Error('Transcription disabled: no NIM/Whisper backend configured.');
}

async function analyzeClinicalImage({
  imageBase64,
  profile = {},
  language = 'en'
}) {
  if (!imageBase64 || typeof imageBase64 !== 'string') {
    throw new Error('imageBase64 is required');
  }

  const dataUrl = imageBase64.startsWith('data:')
    ? imageBase64
    : `data:image/jpeg;base64,${imageBase64}`;

  const prompt = `
User context: ${JSON.stringify(profile || {})}
Preferred language: ${language || 'en'}

Return ONLY valid JSON with shape:
{
  "analysis": "short non-alarming observation",
  "followUpQuestion": "one targeted follow-up question",
  "warning": "always include: This is not a medical diagnosis."
}
`;

  try {
    const content = await callOpenRouterChat([
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: dataUrl } },
          { type: 'text', text: prompt }
        ]
      }
    ]);

    const parsed = safeParseJsonObject(content) || {};
    const analysis = await translateTextIfNeeded(
      parsed.analysis ||
        'I can see this needs closer evaluation. Please describe when this started and if it is worsening.',
      language
    );
    const followUpQuestion = await translateTextIfNeeded(
      parsed.followUpQuestion ||
        'When did this begin and have you noticed pain, fever, or discharge?',
      language
    );

    return {
      analysis,
      followUpQuestion,
      warning:
        parsed.warning ||
        'This is not a medical diagnosis. Please consult a doctor.',
      source: OPENROUTER_VISION_MODEL
    };
  } catch (err) {
    return {
      analysis: await translateTextIfNeeded(
        'I could not analyze the image right now. Please share a clearer image and symptom details.',
        language
      ),
      followUpQuestion: await translateTextIfNeeded(
        'Can you describe your main symptom and when it started?',
        language
      ),
      warning: 'This is not a medical diagnosis. Please consult a doctor.',
      source: 'vision-fallback'
    };
  }
}

async function extractDocumentData({
  imageBase64,
  documentType = 'OTHER',
  language = 'en'
}) {
  if (!imageBase64 || typeof imageBase64 !== 'string') {
    throw new Error('imageBase64 is required');
  }

  const dataUrl = imageBase64.startsWith('data:')
    ? imageBase64
    : `data:image/jpeg;base64,${imageBase64}`;

  const prompt = `
You are a medical document extraction assistant.
Document type: ${documentType}
Respond ONLY as JSON with shape:
{
  "rawText": "...",
  "extracted": {
    "doctorName": null,
    "documentDate": null,
    "diagnosis": null,
    "medications": [{"drugName":"","dosage":"","frequency":"","duration":""}],
    "labValues": [{"testName":"","value":"","unit":"","refRange":""}]
  },
  "confidence": 0.0,
  "clarificationNeeded": [{"field":"...","question":"..."}]
}
`;

  try {
    const content = await callOpenRouterChat([
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: dataUrl } },
          { type: 'text', text: prompt }
        ]
      }
    ]);

    const parsed = safeParseJsonObject(content) || {};
    return {
      rawText: parsed.rawText || '',
      extracted: parsed.extracted || {},
      confidence:
        typeof parsed.confidence === 'number'
          ? clamp(parsed.confidence, 0, 1)
          : 0.55,
      clarificationNeeded: Array.isArray(parsed.clarificationNeeded)
        ? parsed.clarificationNeeded
        : [],
      source: OPENROUTER_VISION_MODEL,
      language
    };
  } catch (err) {
    return {
      rawText: '',
      extracted: {},
      confidence: 0,
      clarificationNeeded: [
        {
          field: 'document',
          question:
            'Image parsing failed. Please upload a clearer image or enter details manually.'
        }
      ],
      source: 'ocr-fallback',
      language
    };
  }
}

async function triageSymptoms({
  symptomText = '',
  sex = 'female',
  age,
  language = 'en'
}) {
  const normalized = String(symptomText || '').trim();
  if (!normalized) {
    return {
      triage: 'SELF_CARE',
      shouldStop: true,
      conditions: [],
      question: null,
      source: 'rules'
    };
  }

  if (INFERMEDICA_APP_ID && INFERMEDICA_APP_KEY) {
    try {
      const headers = {
        'App-Id': INFERMEDICA_APP_ID,
        'App-Key': INFERMEDICA_APP_KEY,
        'Content-Type': 'application/json'
      };

      const parseRes = await fetch(`${INFERMEDICA_BASE_URL}/parse`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ text: normalized })
      });
      const parsePayload = await parseRes.json();

      const evidence = Array.isArray(parsePayload?.mentions)
        ? parsePayload.mentions.map((m) => ({
            id: m.id,
            choice_id: m.choice_id || 'present'
          }))
        : [];

      const diagnosisRes = await fetch(`${INFERMEDICA_BASE_URL}/diagnosis`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          sex:
            String(sex || 'female').toLowerCase() === 'male'
              ? 'male'
              : 'female',
          age: { value: Number(age) || 30 },
          evidence,
          extras: { enable_explanations: true }
        })
      });

      const diagnosisPayload = await diagnosisRes.json();

      return {
        triage: diagnosisPayload?.triage || 'SELF_CARE',
        shouldStop: Boolean(diagnosisPayload?.should_stop),
        conditions: diagnosisPayload?.conditions || [],
        question: diagnosisPayload?.question || null,
        source: 'infermedica'
      };
    } catch (err) {
      console.warn('[AI] Infermedica triage fallback:', err?.message || err);
    }
  }

  const urgent = detectUrgency(normalized);
  const triage = urgent
    ? 'EMERGENCY'
    : /fever|pain|bleeding|vomit|dizzy/i.test(normalized)
      ? 'CONSULT'
      : 'SELF_CARE';

  return {
    triage,
    shouldStop: true,
    conditions: [],
    question: null,
    source: 'rules'
  };
}

async function runFullPipeline({
  symptom,
  answers = {},
  profile = {},
  language = 'en',
  history = []
}) {
  const contextMap = buildContextMap(profile, language);
  const triage = await triageSymptoms({
    symptomText: symptom,
    sex: profile.gender,
    age: profile.age,
    language
  });

  const questions = await generateQuestions(symptom, profile, language);
  const assessment = await generateDiagnosis(
    symptom,
    answers,
    profile,
    language
  );

  const fairness = computeFairnessBreakdown(profile, assessment);
  const riskFlags = deriveRiskFlags(symptom, assessment);
  const recommendation = buildRecommendationFromAssessment(
    assessment,
    contextMap
  );

  const followupReply = await generateConversationReply({
    message: symptom,
    profile,
    language,
    history
  });

  return {
    contextMap,
    triage,
    questions: questions.questions,
    assessment: {
      ...assessment,
      fairnessScore: fairness.equityScore,
      fairness
    },
    riskFlags,
    recommendation,
    assistantReply: followupReply.reply,
    sources: {
      questions: questions.source,
      assessment: assessment.source,
      conversation: followupReply.source
    }
  };
}

module.exports = {
  generateQuestions,
  generateDiagnosis,
  generateConversationReply,
  transcribeAudio,
  analyzeClinicalImage,
  extractDocumentData,
  triageSymptoms,
  runFullPipeline,
  computeFairnessScore,
  computeFairnessBreakdown,
  buildContextMap,
  deriveRiskFlags,
  buildRecommendationFromAssessment,
  detectUrgency,
  translateTextIfNeeded
};
