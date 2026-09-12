require('dotenv').config({ path: '.env' });
const crypto = require('crypto');
const express = require('express');
const multer = require('multer');
const { PrismaClient } = require('@prisma/client');
const { createClient } = require('@supabase/supabase-js');
const {
  generateQuestions,
  generateDiagnosis,
  generateConversationReply,
  computeFairnessScore,
  computeFairnessBreakdown,
  buildContextMap,
  deriveRiskFlags,
  buildRecommendationFromAssessment,
  detectUrgency,
  transcribeAudio,
  analyzeClinicalImage,
  extractDocumentData,
  triageSymptoms,
  runFullPipeline
} = require('./ai');
const { getNearbyFacilities } = require('./facilities');
const { parseVCF, extractRsIds } = require('./genomics/vcfParser');
const { fetchAllVariants } = require('./genomics/clinvarClient');
const { annotateAll } = require('./genomics/ensemblClient');
const { calculateAllPRS } = require('./genomics/prsCalculator');
const { generateGeneticFlags } = require('./genomics/riskFlagGenerator');

const prisma = new PrismaClient();
const app = express();
app.use(express.json({ limit: '15mb' }));

// ── Request logging: one line per request, incl. auth failures ───────────
app.use((req, res, next) => {
  const startedAt = Date.now();
  res.on('finish', () => {
    const ms = Date.now() - startedAt;
    const line = `${req.method} ${req.originalUrl} → ${res.statusCode} (${ms}ms)`;
    if (res.statusCode >= 500) console.error(`[http] ${line}`);
    else if (res.statusCode >= 400) console.warn(`[http] ${line}`);
    else console.log(`[http] ${line}`);
  });
  next();
});

// ── CORS: the Expo web build calls this API cross-origin (8081 → 4000) ───
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,PATCH,DELETE,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') {
    res.sendStatus(204);
    return;
  }
  next();
});

const genomicUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }
});

const conversations = new Map();
const SHARE_LINK_SECRET =
  process.env.SHARE_LINK_SECRET || process.env.JWT_SECRET || 'nirogya-dev-share-secret';
const DEFAULT_SHARE_LINK_TTL_HOURS = 72;
const MAX_SHARE_LINK_TTL_HOURS = 720;

function createConversationFlow() {
  return {
    stage: 'idle',
    symptom: null,
    questions: [],
    questionIndex: 0,
    answers: {}
  };
}

function formatQuestionForChat(question, index, total) {
  void index;
  void total;
  const options = Array.isArray(question?.options)
    ? question.options
        .map((option, i) => `${i + 1}. ${option}`)
        .join('\n')
    : '';

  return `${question?.text || 'Please provide more details.'}${
    options ? `\n\n${options}` : ''
  }`;
}

function buildAssessmentAlerts({
  symptom,
  assessment,
  riskFlags = [],
  recommendation,
  nearbyFacilities = []
}) {
  const alerts = [];
  const mergedText = `${symptom || ''} ${assessment?.diagnosis || ''} ${assessment?.description || ''}`;

  const isCritical =
    assessment?.riskLevel === 'high' ||
    Number(assessment?.riskScore || 0) >= 80 ||
    detectUrgency(mergedText) ||
    riskFlags.some(
      (flag) => String(flag?.severity || '').toUpperCase() === 'URGENT'
    );

  if (isCritical) {
    alerts.push({
      id: `critical-${Date.now()}`,
      severity: 'critical',
      title: 'Emergency alert',
      message:
        'Possible urgent symptoms detected. Seek in-person care now and call 108 if symptoms are severe.',
      action:
        nearbyFacilities[0]?.mapUrl ||
        'https://www.google.com/maps/search/?api=1&query=nearest+emergency+hospital'
    });
  } else if (assessment?.riskLevel === 'medium') {
    alerts.push({
      id: `warning-${Date.now()}`,
      severity: 'warning',
      title: 'Doctor follow-up recommended',
      message:
        'Book a consultation within 1-2 weeks and monitor symptom progression.',
      action: nearbyFacilities[0]?.mapUrl || null
    });
  } else {
    alerts.push({
      id: `info-${Date.now()}`,
      severity: 'info',
      title: 'Self-care monitoring',
      message:
        'Current risk appears low. Continue self-care and seek medical advice if symptoms worsen.',
      action: null
    });
  }

  if (
    Array.isArray(riskFlags) &&
    riskFlags.some((flag) => flag?.condition === 'ANEMIA_RISK')
  ) {
    alerts.push({
      id: `anemia-${Date.now()}`,
      severity: 'warning',
      title: 'Possible anemia pattern',
      message:
        'Please consider CBC and ferritin testing soon to confirm iron status.',
      action: nearbyFacilities[1]?.mapUrl || nearbyFacilities[0]?.mapUrl || null
    });
  }

  return alerts;
}

function formatAssessmentReply({ assessment, recommendation, nearbyFacilities }) {
  const nextSteps = Array.isArray(assessment?.nextSteps)
    ? assessment.nextSteps.slice(0, 4)
    : [];

  const stepsText =
    nextSteps.length > 0
      ? nextSteps.map((step, idx) => `${idx + 1}. ${step}`).join('\n')
      : '1. Track your symptoms and consult a doctor if they persist.';

  const nearbyText = Array.isArray(nearbyFacilities)
    ? nearbyFacilities
        .slice(0, 3)
        .map((facility, idx) => {
          const distance =
            typeof facility?.distanceKm === 'number'
              ? ` (${facility.distanceKm} km)`
              : '';
          return `${idx + 1}. ${facility?.name || 'Nearby facility'}${distance} - ${facility?.facilityType || 'Hospital'}`;
        })
        .join('\n')
    : '';

  const affordabilityHint = recommendation?.pmjayApplicable
    ? '\nPMJAY may reduce out-of-pocket costs for this care path.'
    : '';

  return (
    `Likely condition: ${assessment?.diagnosis || 'Needs further evaluation'}\n` +
    `Risk level: ${(assessment?.riskLevel || 'medium').toUpperCase()} (${Math.round(
      Number(assessment?.riskScore || 0)
    )}%)\n` +
    `Urgency: ${assessment?.urgency || 'Follow up soon'}\n\n` +
    `${assessment?.description || 'Please continue with a focused health assessment.'}\n\n` +
    `What to do next:\n${stepsText}` +
    `${nearbyText ? `\n\nNearby care options:\n${nearbyText}` : ''}` +
    `${affordabilityHint}`
  );
}

function toBase64Url(value) {
  return Buffer.from(value)
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/g, '');
}

function hashValue(value) {
  return crypto.createHash('sha256').update(String(value)).digest('hex');
}

function signShareNonce(nonce) {
  return toBase64Url(
    crypto.createHmac('sha256', SHARE_LINK_SECRET).update(String(nonce)).digest()
  );
}

function createSignedShareToken() {
  const nonce = toBase64Url(crypto.randomBytes(24));
  const signature = signShareNonce(nonce);
  const token = `${nonce}.${signature}`;
  return {
    token,
    tokenHash: hashValue(token)
  };
}

function isShareTokenValid(token) {
  if (!token || typeof token !== 'string') return false;

  const [nonce, signature] = token.split('.');
  if (!nonce || !signature) return false;

  const expected = signShareNonce(nonce);
  if (expected.length !== signature.length) return false;

  try {
    return crypto.timingSafeEqual(
      Buffer.from(expected),
      Buffer.from(signature)
    );
  } catch {
    return false;
  }
}

function normalizeBaseUrl(raw) {
  if (!raw || typeof raw !== 'string') return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  return trimmed.endsWith('/') ? trimmed.slice(0, -1) : trimmed;
}

function getPublicShareBaseUrl(req) {
  const explicit = normalizeBaseUrl(
    process.env.SHARE_BASE_URL ||
      process.env.BACKEND_PUBLIC_URL ||
      process.env.EXPO_PUBLIC_BACKEND_URL
  );

  if (explicit) return explicit;

  const forwardedProto = String(req.headers['x-forwarded-proto'] || '')
    .split(',')[0]
    .trim();
  const protocol = forwardedProto || req.protocol || 'http';
  return `${protocol}://${req.get('host')}`;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function ensureStringArray(value, limit = 6) {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => (typeof item === 'string' ? item.trim() : ''))
    .filter(Boolean)
    .slice(0, limit);
}

function parseAssessmentSnapshot(assessment) {
  const parsedRaw = (() => {
    try {
      return assessment?.rawAiText ? JSON.parse(assessment.rawAiText) : null;
    } catch {
      return null;
    }
  })();

  const diagnosisCandidate =
    parsedRaw &&
    typeof parsedRaw === 'object' &&
    parsedRaw.diagnosis &&
    typeof parsedRaw.diagnosis === 'object'
      ? parsedRaw.diagnosis
      : parsedRaw && typeof parsedRaw === 'object'
        ? parsedRaw
        : null;

  const pipelineCandidate =
    parsedRaw &&
    typeof parsedRaw === 'object' &&
    parsedRaw.pipeline &&
    typeof parsedRaw.pipeline === 'object'
      ? parsedRaw.pipeline
      : null;

  const assessmentCandidate =
    pipelineCandidate &&
    pipelineCandidate.assessment &&
    typeof pipelineCandidate.assessment === 'object'
      ? pipelineCandidate.assessment
      : null;

  const recommendation =
    pipelineCandidate &&
    pipelineCandidate.recommendation &&
    typeof pipelineCandidate.recommendation === 'object'
      ? pipelineCandidate.recommendation
      : parsedRaw &&
          typeof parsedRaw === 'object' &&
          parsedRaw.recommendation &&
          typeof parsedRaw.recommendation === 'object'
        ? parsedRaw.recommendation
        : null;

  const nearbyFacilities = Array.isArray(pipelineCandidate?.nearbyFacilities)
    ? pipelineCandidate.nearbyFacilities
    : Array.isArray(parsedRaw?.nearbyFacilities)
      ? parsedRaw.nearbyFacilities
      : [];

  const alerts = Array.isArray(pipelineCandidate?.alerts)
    ? pipelineCandidate.alerts
    : Array.isArray(parsedRaw?.alerts)
      ? parsedRaw.alerts
      : [];

  const diagnosis =
    (typeof diagnosisCandidate?.diagnosis === 'string' &&
      diagnosisCandidate.diagnosis.trim()) ||
    (typeof assessment?.diagnosis === 'string' && assessment.diagnosis.trim()) ||
    'Assessment summary';

  const description =
    (typeof diagnosisCandidate?.description === 'string' &&
      diagnosisCandidate.description.trim()) ||
    (typeof assessmentCandidate?.description === 'string' &&
      assessmentCandidate.description.trim()) ||
    'No additional summary was provided.';

  const riskScore = Number(
    diagnosisCandidate?.riskScore ??
      assessmentCandidate?.riskScore ??
      assessment?.riskScore ??
      0
  );

  const riskLevelRaw = String(
    diagnosisCandidate?.riskLevel ??
      assessmentCandidate?.riskLevel ??
      assessment?.riskLevel ??
      'medium'
  ).toLowerCase();
  const riskLevel = ['low', 'medium', 'high'].includes(riskLevelRaw)
    ? riskLevelRaw
    : 'medium';

  const urgency =
    (typeof diagnosisCandidate?.urgency === 'string' &&
      diagnosisCandidate.urgency.trim()) ||
    (typeof assessmentCandidate?.urgency === 'string' &&
      assessmentCandidate.urgency.trim()) ||
    (riskLevel === 'high' ? 'Immediate' : 'Within a few days');

  const fairnessScore =
    typeof diagnosisCandidate?.fairnessScore === 'number'
      ? diagnosisCandidate.fairnessScore
      : typeof assessmentCandidate?.fairnessScore === 'number'
        ? assessmentCandidate.fairnessScore
        : null;

  const nextSteps =
    ensureStringArray(diagnosisCandidate?.nextSteps, 8).length > 0
      ? ensureStringArray(diagnosisCandidate?.nextSteps, 8)
      : ensureStringArray(assessmentCandidate?.nextSteps, 8).length > 0
        ? ensureStringArray(assessmentCandidate?.nextSteps, 8)
        : ensureStringArray(assessment?.nextSteps, 8);

  return {
    diagnosis,
    description,
    riskScore: Number.isFinite(riskScore) ? Math.round(riskScore) : 0,
    riskLevel,
    urgency,
    fairnessScore:
      typeof fairnessScore === 'number'
        ? Math.max(0, Math.min(1, fairnessScore))
        : null,
    nextSteps,
    recommendation,
    nearbyFacilities,
    alerts
  };
}

function uniqText(values, limit = 10) {
  const out = [];
  const seen = new Set();

  for (const entry of values || []) {
    const normalized = typeof entry === 'string' ? entry.trim() : '';
    if (!normalized) continue;
    const key = normalized.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(normalized);
    if (out.length >= limit) break;
  }

  return out;
}

function rankByFrequency(values, limit = 8) {
  const tally = new Map();

  for (const value of values || []) {
    const normalized = typeof value === 'string' ? value.trim() : '';
    if (!normalized) continue;
    tally.set(normalized, (tally.get(normalized) || 0) + 1);
  }

  return [...tally.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([label, count]) => ({ label, count }));
}

function normalizeDetectedTitles(value, limit = 10) {
  if (!Array.isArray(value)) return [];
  return uniqText(
    value
      .map((entry) => (typeof entry === 'string' ? entry : ''))
      .map((entry) => entry.trim())
      .filter(
        (entry) =>
          entry &&
          entry.toLowerCase() !== 'nira conversation' &&
          entry.toLowerCase() !== 'ai chat finding'
      ),
    limit
  );
}

function buildOverallSummaryReport({ profile, assessments, chatDetectedTitles }) {
  const sortedAssessments = [...(assessments || [])].sort(
    (a, b) =>
      new Date(b?.createdAt || 0).getTime() - new Date(a?.createdAt || 0).getTime()
  );

  const snapshots = sortedAssessments.map((assessment) => ({
    assessment,
    snapshot: parseAssessmentSnapshot(assessment)
  }));

  const riskCounts = snapshots.reduce(
    (acc, item) => {
      const level = String(item?.snapshot?.riskLevel || '').toLowerCase();
      if (level === 'high' || level === 'medium' || level === 'low') {
        acc[level] += 1;
      }
      return acc;
    },
    { high: 0, medium: 0, low: 0 }
  );

  const topDiagnoses = rankByFrequency(
    snapshots.map((item) => item?.snapshot?.diagnosis || ''),
    8
  );
  const topSymptoms = rankByFrequency(
    sortedAssessments.map((item) => item?.symptom || ''),
    8
  );

  const topAlerts = uniqText(
    snapshots.flatMap((item) => {
      const alerts = Array.isArray(item?.snapshot?.alerts)
        ? item.snapshot.alerts
        : [];

      return alerts.map((alert) => {
        const severity = String(alert?.severity || '').trim().toUpperCase();
        const title = String(alert?.title || '').trim();
        const message = String(alert?.message || '').trim();
        const body = title || message;
        if (!body) return '';
        return severity ? `${severity}: ${body}` : body;
      });
    }),
    8
  );

  const nextSteps = uniqText(
    snapshots.flatMap((item) => ensureStringArray(item?.snapshot?.nextSteps, 8)),
    10
  );

  const cleanedChatTitles = normalizeDetectedTitles(chatDetectedTitles, 10);

  const fromDate = sortedAssessments.at(-1)?.createdAt || null;
  const toDate = sortedAssessments[0]?.createdAt || null;

  return {
    title: 'Nirogya Overall Health Report',
    generatedAt: new Date().toISOString(),
    patientName:
      (typeof profile?.name === 'string' && profile.name.trim()) ||
      (typeof profile?.email === 'string' && profile.email.trim()) ||
      'Patient',
    coverage: {
      assessmentCount: sortedAssessments.length,
      chatDetectionCount: cleanedChatTitles.length,
      fromDate,
      toDate
    },
    riskCounts,
    topDiagnoses,
    topSymptoms,
    topAlerts,
    nextSteps,
    chatDetections: cleanedChatTitles,
    clinicalNote:
      'This is an AI-assisted summary and must be interpreted by a licensed doctor with clinical judgment.'
  };
}

function renderOverallDoctorShareHtml(payload) {
  const report = payload?.report && typeof payload.report === 'object'
    ? payload.report
    : {};

  const patientName =
    (typeof report.patientName === 'string' && report.patientName) ||
    payload?.profile?.name ||
    'Not specified';
  const generatedAt = report.generatedAt
    ? new Date(report.generatedAt).toLocaleString('en-IN')
    : 'Unknown';
  const expiresAt = payload?.shareLink?.expiresAt
    ? new Date(payload.shareLink.expiresAt).toLocaleString('en-IN')
    : 'Unknown';
  const fromDate = report?.coverage?.fromDate
    ? new Date(report.coverage.fromDate).toLocaleDateString('en-IN')
    : 'N/A';
  const toDate = report?.coverage?.toDate
    ? new Date(report.coverage.toDate).toLocaleDateString('en-IN')
    : 'N/A';

  const riskCounts = report?.riskCounts || { high: 0, medium: 0, low: 0 };
  const topDiagnoses = Array.isArray(report?.topDiagnoses)
    ? report.topDiagnoses
    : [];
  const topSymptoms = Array.isArray(report?.topSymptoms)
    ? report.topSymptoms
    : [];
  const topAlerts = Array.isArray(report?.topAlerts) ? report.topAlerts : [];
  const nextSteps = Array.isArray(report?.nextSteps) ? report.nextSteps : [];
  const chatDetections = Array.isArray(report?.chatDetections)
    ? report.chatDetections
    : [];

  const renderRanked = (items) => {
    if (!items.length) return '<p>No strong recurring pattern captured yet.</p>';
    return `<ul>${items
      .map(
        (item) =>
          `<li>${escapeHtml(String(item?.label || 'Unknown'))} (${escapeHtml(
            String(item?.count || 0)
          )})</li>`
      )
      .join('')}</ul>`;
  };

  const renderList = (items, emptyText) => {
    if (!items.length) return `<p>${escapeHtml(emptyText)}</p>`;
    return `<ul>${items
      .map((entry) => `<li>${escapeHtml(String(entry || ''))}</li>`)
      .join('')}</ul>`;
  };

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Nirogya Overall Doctor Share</title>
    <style>
      :root {
        --bg: #f6f9ff;
        --card: #ffffff;
        --text: #1b2640;
        --muted: #5a6b91;
        --line: rgba(36, 66, 120, 0.15);
        --accent: #d5457a;
      }
      body { margin: 0; font-family: "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--text); }
      .wrap { max-width: 860px; margin: 0 auto; padding: 24px 16px 40px; }
      .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px; margin-bottom: 12px; }
      h1 { margin: 0 0 10px; font-size: 24px; }
      h2 { margin: 0 0 8px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
      p { margin: 0 0 8px; line-height: 1.55; }
      ul { margin: 0; padding-left: 20px; }
      li { margin: 6px 0; }
      .meta { display: grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); gap: 10px; }
      .meta div { background: #f8fbff; border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
      .k { color: var(--muted); font-size: 12px; }
      .v { font-weight: 600; margin-top: 2px; }
      .foot { color: var(--muted); font-size: 12px; margin-top: 10px; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <h1>${escapeHtml(report?.title || 'Nirogya Overall Health Report')}</h1>
        <p>Shared for clinical review. This consolidated summary combines all completed assessment findings and detected chat outcomes.</p>
      </div>

      <div class="card">
        <h2>Patient Snapshot</h2>
        <div class="meta">
          <div><div class="k">Patient</div><div class="v">${escapeHtml(patientName)}</div></div>
          <div><div class="k">Generated</div><div class="v">${escapeHtml(generatedAt)}</div></div>
          <div><div class="k">Report Timeline</div><div class="v">${escapeHtml(fromDate)} to ${escapeHtml(toDate)}</div></div>
          <div><div class="k">Link Expires</div><div class="v">${escapeHtml(expiresAt)}</div></div>
        </div>
      </div>

      <div class="card">
        <h2>Coverage</h2>
        <p>Total assessments reviewed: <strong>${escapeHtml(
          String(report?.coverage?.assessmentCount || 0)
        )}</strong></p>
        <p>Chat detections reviewed: <strong>${escapeHtml(
          String(report?.coverage?.chatDetectionCount || 0)
        )}</strong></p>
      </div>

      <div class="card">
        <h2>Risk Overview</h2>
        <p>High risk episodes: <strong>${escapeHtml(String(riskCounts.high || 0))}</strong></p>
        <p>Medium risk episodes: <strong>${escapeHtml(String(riskCounts.medium || 0))}</strong></p>
        <p>Low risk episodes: <strong>${escapeHtml(String(riskCounts.low || 0))}</strong></p>
      </div>

      <div class="card">
        <h2>Top AI Detected Conditions</h2>
        ${renderRanked(topDiagnoses)}
      </div>

      <div class="card">
        <h2>Top Reported Symptoms</h2>
        ${renderRanked(topSymptoms)}
      </div>

      <div class="card">
        <h2>Key Alerts Detected</h2>
        ${renderList(topAlerts, 'No alert events captured in this reporting window.')}
      </div>

      <div class="card">
        <h2>Consolidated Next Steps</h2>
        ${renderList(nextSteps, 'No consolidated next-step guidance captured yet.')}
      </div>

      <div class="card">
        <h2>Detected In Chat Conversations</h2>
        ${renderList(chatDetections, 'No finalized chat detections captured yet.')}
      </div>

      <div class="card">
        <h2>Clinical Note</h2>
        <p>${escapeHtml(
          report?.clinicalNote ||
            'This is an AI-assisted summary and must be interpreted by a licensed doctor with clinical judgment.'
        )}</p>
      </div>

      <p class="foot">Share ID: ${escapeHtml(payload?.shareLink?.id || '')} · Views: ${escapeHtml(payload?.shareLink?.viewCount || 0)}</p>
    </div>
  </body>
</html>`;
}

function renderDoctorShareHtml(payload) {
  const diagnosis = payload?.snapshot?.diagnosis || 'Assessment summary';
  const description = payload?.snapshot?.description || '';
  const nextSteps = ensureStringArray(payload?.snapshot?.nextSteps, 8);
  const alerts = Array.isArray(payload?.snapshot?.alerts)
    ? payload.snapshot.alerts.slice(0, 3)
    : [];
  const nearbyFacilities = Array.isArray(payload?.snapshot?.nearbyFacilities)
    ? payload.snapshot.nearbyFacilities.slice(0, 3)
    : [];

  const riskLevel = String(payload?.snapshot?.riskLevel || 'medium').toUpperCase();
  const riskScore = Number(payload?.snapshot?.riskScore || 0);
  const urgency = payload?.snapshot?.urgency || 'Not specified';
  const fairness =
    typeof payload?.snapshot?.fairnessScore === 'number'
      ? `${Math.round(payload.snapshot.fairnessScore * 100)}%`
      : 'N/A';

  const createdAt = payload?.assessment?.createdAt
    ? new Date(payload.assessment.createdAt).toLocaleString('en-IN')
    : 'Unknown';

  const expiresAt = payload?.shareLink?.expiresAt
    ? new Date(payload.shareLink.expiresAt).toLocaleString('en-IN')
    : 'Unknown';

  const recommendationText =
    payload?.snapshot?.recommendation?.carePathway ||
    'Follow the care guidance provided in the assessment.';

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Nirogya Doctor Share</title>
    <style>
      :root {
        --bg: #f6f9ff;
        --card: #ffffff;
        --text: #1b2640;
        --muted: #5a6b91;
        --line: rgba(36, 66, 120, 0.15);
        --accent: #d5457a;
      }
      body { margin: 0; font-family: "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--text); }
      .wrap { max-width: 860px; margin: 0 auto; padding: 24px 16px 40px; }
      .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px; margin-bottom: 12px; }
      h1 { margin: 0 0 10px; font-size: 24px; }
      h2 { margin: 0 0 8px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
      p { margin: 0 0 8px; line-height: 1.55; }
      ul { margin: 0; padding-left: 20px; }
      li { margin: 6px 0; }
      .meta { display: grid; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); gap: 10px; }
      .meta div { background: #f8fbff; border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
      .k { color: var(--muted); font-size: 12px; }
      .v { font-weight: 600; margin-top: 2px; }
      a { color: var(--accent); text-decoration: none; }
      .foot { color: var(--muted); font-size: 12px; margin-top: 10px; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <h1>Nirogya Shared Assessment</h1>
        <p>Shared for clinical review. This summary was generated by an AI-assisted health workflow and should be interpreted with medical judgment.</p>
      </div>

      <div class="card">
        <h2>Patient Snapshot</h2>
        <div class="meta">
          <div><div class="k">Patient</div><div class="v">${escapeHtml(payload?.profile?.name || 'Not specified')}</div></div>
          <div><div class="k">Symptom</div><div class="v">${escapeHtml(payload?.assessment?.symptom || 'Not specified')}</div></div>
          <div><div class="k">Created</div><div class="v">${escapeHtml(createdAt)}</div></div>
          <div><div class="k">Link Expires</div><div class="v">${escapeHtml(expiresAt)}</div></div>
        </div>
      </div>

      <div class="card">
        <h2>Assessment Summary</h2>
        <p><strong>${escapeHtml(diagnosis)}</strong></p>
        <p>${escapeHtml(description)}</p>
        <div class="meta" style="margin-top: 8px;">
          <div><div class="k">Risk Level</div><div class="v">${escapeHtml(riskLevel)}</div></div>
          <div><div class="k">Risk Score</div><div class="v">${escapeHtml(`${riskScore}%`)}</div></div>
          <div><div class="k">Urgency</div><div class="v">${escapeHtml(urgency)}</div></div>
          <div><div class="k">Fairness Score</div><div class="v">${escapeHtml(fairness)}</div></div>
        </div>
      </div>

      <div class="card">
        <h2>Recommended Next Steps</h2>
        ${
          nextSteps.length > 0
            ? `<ul>${nextSteps.map((step) => `<li>${escapeHtml(step)}</li>`).join('')}</ul>`
            : '<p>No specific next steps captured.</p>'
        }
      </div>

      <div class="card">
        <h2>Care Pathway Context</h2>
        <p>${escapeHtml(recommendationText)}</p>
      </div>

      <div class="card">
        <h2>Alerts</h2>
        ${
          alerts.length > 0
            ? `<ul>${alerts
                .map(
                  (alert) =>
                    `<li><strong>${escapeHtml(alert?.title || 'Alert')}:</strong> ${escapeHtml(alert?.message || '')}</li>`
                )
                .join('')}</ul>`
            : '<p>No alert events captured for this assessment.</p>'
        }
      </div>

      <div class="card">
        <h2>Nearby Facilities</h2>
        ${
          nearbyFacilities.length > 0
            ? `<ul>${nearbyFacilities
                .map((facility) => {
                  const name = escapeHtml(facility?.name || 'Facility');
                  const type = escapeHtml(facility?.facilityType || 'Hospital');
                  const distance =
                    typeof facility?.distanceKm === 'number'
                      ? ` (${escapeHtml(facility.distanceKm)} km)`
                      : '';
                  const mapUrl =
                    typeof facility?.mapUrl === 'string' ? facility.mapUrl : null;
                  return `<li>${name} - ${type}${distance}${
                    mapUrl
                      ? ` - <a href="${escapeHtml(mapUrl)}" target="_blank" rel="noreferrer">Open Map</a>`
                      : ''
                  }</li>`;
                })
                .join('')}</ul>`
            : '<p>No nearby facility details were captured for this shared assessment.</p>'
        }
      </div>

      <p class="foot">Share ID: ${escapeHtml(payload?.shareLink?.id || '')} · Views: ${escapeHtml(payload?.shareLink?.viewCount || 0)}</p>
    </div>
  </body>
</html>`;
}

app.get('/health', (_req, res) => {
  res.json({ ok: true, time: new Date().toISOString() });
});

app.get(['/share/:token', '/api/share/:token'], async (req, res) => {
  try {
    const rawToken = String(req.params.token || '').trim();
    const isApiShareRequest = req.path.startsWith('/api/share/');

    const sendShareError = (statusCode, message) => {
      if (isApiShareRequest) {
        return res.status(statusCode).json({ error: message });
      }
      return res.status(statusCode).send(message);
    };

    if (!isShareTokenValid(rawToken)) {
      return sendShareError(404, 'This shared link is invalid.');
    }

    const tokenHash = hashValue(rawToken);
    const shareLink = await prisma.shareLink.findUnique({
      where: { tokenHash },
      include: {
        assessment: true,
        profile: {
          select: {
            name: true,
            city: true,
            language: true
          }
        }
      }
    });

    if (!shareLink || !shareLink.assessment || shareLink.assessment.deletedAt) {
      return sendShareError(404, 'This shared link is unavailable.');
    }

    const now = new Date();
    if (shareLink.revokedAt) {
      return sendShareError(410, 'This shared link has been revoked.');
    }

    if (shareLink.expiresAt <= now) {
      return sendShareError(410, 'This shared link has expired.');
    }

    const updated = await prisma.shareLink.update({
      where: { id: shareLink.id },
      data: {
        lastViewedAt: now,
        viewCount: { increment: 1 }
      }
    });

    const payload = {
      shareLink: {
        id: updated.id,
        createdAt: updated.createdAt,
        expiresAt: updated.expiresAt,
        viewCount: updated.viewCount
      },
      profile: shareLink.profile,
      assessment: {
        id: shareLink.assessment.id,
        symptom: shareLink.assessment.symptom,
        createdAt: shareLink.assessment.createdAt
      },
      snapshot: parseAssessmentSnapshot(shareLink.assessment)
    };

    if (isApiShareRequest) {
      return res.json(payload);
    }

    res.status(200).type('html').send(renderDoctorShareHtml(payload));
  } catch (err) {
    console.error(err);
    if (req.path.startsWith('/api/share/')) {
      return res
        .status(500)
        .json({ error: 'Could not open shared assessment right now.' });
    }
    res.status(500).send('Could not open shared assessment right now.');
  }
});

app.get(['/overall-share/:token', '/api/overall-share/:token'], async (req, res) => {
  try {
    const rawToken = String(req.params.token || '').trim();
    const isApiShareRequest = req.path.startsWith('/api/overall-share/');

    const sendShareError = (statusCode, message) => {
      if (isApiShareRequest) {
        return res.status(statusCode).json({ error: message });
      }
      return res.status(statusCode).send(message);
    };

    if (!isShareTokenValid(rawToken)) {
      return sendShareError(404, 'This shared link is invalid.');
    }

    const tokenHash = hashValue(rawToken);
    const shareLink = await prisma.overallShareLink.findUnique({
      where: { tokenHash },
      include: {
        profile: {
          select: {
            name: true,
            city: true,
            language: true
          }
        }
      }
    });

    if (!shareLink) {
      return sendShareError(404, 'This shared link is unavailable.');
    }

    const now = new Date();
    if (shareLink.revokedAt) {
      return sendShareError(410, 'This shared link has been revoked.');
    }

    if (shareLink.expiresAt <= now) {
      return sendShareError(410, 'This shared link has expired.');
    }

    const updated = await prisma.overallShareLink.update({
      where: { id: shareLink.id },
      data: {
        lastViewedAt: now,
        viewCount: { increment: 1 }
      }
    });

    const payload = {
      shareLink: {
        id: updated.id,
        createdAt: updated.createdAt,
        expiresAt: updated.expiresAt,
        viewCount: updated.viewCount
      },
      profile: shareLink.profile,
      report:
        shareLink.report && typeof shareLink.report === 'object'
          ? shareLink.report
          : null
    };

    if (isApiShareRequest) {
      return res.json(payload);
    }

    res.status(200).type('html').send(renderOverallDoctorShareHtml(payload));
  } catch (err) {
    console.error(err);
    if (req.path.startsWith('/api/overall-share/')) {
      return res
        .status(500)
        .json({ error: 'Could not open shared overall report right now.' });
    }
    res.status(500).send('Could not open shared overall report right now.');
  }
});

const supabase = createClient(
  process.env.EXPO_PUBLIC_SUPABASE_URL,
  process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY
);

// Auth middleware using Supabase JWT
app.use(async (req, res, next) => {
  const authHeader = req.header('authorization');
  const token = authHeader?.startsWith('Bearer ')
    ? authHeader.slice('Bearer '.length)
    : null;
  if (!token) {
    return res
      .status(401)
      .json({ error: 'Missing Authorization Bearer token' });
  }

  const { data, error } = await supabase.auth.getUser(token);
  if (error || !data?.user) {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }

  req.uid = data.user.id;
  req.user = data.user;

  // Ensure profile exists
  await prisma.profile.upsert({
    where: { uid: req.uid },
    update: { lastActiveAt: new Date() },
    create: {
      uid: req.uid,
      email: data.user.email ?? null,
      name: data.user.user_metadata?.full_name ?? null,
      lastActiveAt: new Date()
    }
  });

  next();
});

app.get(['/profiles/me', '/api/profiles/me'], async (req, res) => {
  try {
    const profile = await prisma.profile.findUnique({
      where: { uid: req.uid }
    });
    if (!profile) return res.status(404).json({ error: 'Profile not found' });
    res.json(profile);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.post(['/profiles', '/api/profiles'], async (req, res) => {
  try {
    const data = req.body || {};
    const profile = await prisma.profile.upsert({
      where: { uid: req.uid },
      update: { ...data, updatedAt: new Date(), lastActiveAt: new Date() },
      create: { ...data, uid: req.uid, lastActiveAt: new Date() }
    });
    res.json(profile);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.get(['/assessments', '/api/assessments'], async (req, res) => {
  try {
    const list = await prisma.assessment.findMany({
      where: { uid: req.uid, deletedAt: null },
      orderBy: { createdAt: 'desc' },
      take: 50
    });
    res.json(list);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.post(['/assessments', '/api/assessments'], async (req, res) => {
  try {
    const data = req.body || {};
    const created = await prisma.assessment.create({
      data: {
        uid: req.uid,
        symptom: data.symptom ?? null,
        answers: data.answers ?? null,
        diagnosis: data.diagnosis ?? null,
        riskScore: data.riskScore ?? null,
        riskLevel: data.riskLevel ?? null,
        nextSteps: data.nextSteps ?? [],
        rawAiText: data.rawAiText ?? null
      }
    });
    res.json(created);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Server error' });
  }
});

app.post(
  ['/genomics/upload', '/api/genomics/upload'],
  genomicUpload.single('file'),
  async (req, res) => {
    try {
      if (!req.file?.buffer) {
        return res.status(400).json({ error: 'VCF file is required.' });
      }

      const originalName = String(req.file.originalname || '').toLowerCase();
      if (!originalName.endsWith('.vcf')) {
        return res
          .status(400)
          .json({ error: 'Please upload a valid .vcf file.' });
      }

      const vcfText = req.file.buffer.toString('utf8');
      const parsedVariants = parseVCF(vcfText);
      const rsIds = extractRsIds(parsedVariants);

      if (!rsIds.length) {
        return res.status(400).json({
          error:
            'No valid rsID variants found. Ensure your VCF has PASS entries with rs identifiers.'
        });
      }

      const [clinvarVariants, ensemblVariants] = await Promise.all([
        fetchAllVariants(rsIds),
        annotateAll(rsIds)
      ]);

      const ensemblByRsId = new Map(
        ensemblVariants.map((variant) => [variant.rsId, variant])
      );

      const annotatedVariants = clinvarVariants.map((variant) => ({
        ...variant,
        ensembl: ensemblByRsId.get(variant.rsId) || null
      }));

      const prsScores = calculateAllPRS(rsIds);
      const rawFlags = await generateGeneticFlags(annotatedVariants, prsScores);

      const savedProfile = await prisma.geneticProfile.upsert({
        where: { uid: req.uid },
        update: {
          rsIds,
          prsT2d: prsScores.t2d.score,
          prsCad: prsScores.cad.score,
          prsHtn: prsScores.htn.score
        },
        create: {
          uid: req.uid,
          rsIds,
          prsT2d: prsScores.t2d.score,
          prsCad: prsScores.cad.score,
          prsHtn: prsScores.htn.score
        }
      });

      await prisma.geneticFlag.deleteMany({ where: { uid: req.uid } });

      const normalizedFlags = rawFlags.map((flag) => ({
        uid: req.uid,
        geneticProfileId: savedProfile.id,
        type: String(flag.type || 'genetic_insight'),
        gene: typeof flag.gene === 'string' ? flag.gene : null,
        severity: String(flag.severity || 'LOW').toUpperCase(),
        conditions: Array.isArray(flag.conditions)
          ? flag.conditions.map((item) => String(item))
          : typeof flag.condition === 'string'
            ? [flag.condition]
            : [],
        plainLanguage: String(
          flag.plainLanguage || 'No plain-language interpretation available.'
        ),
        actionRequired: String(
          flag.actionRequired || 'Discuss this finding with your doctor.'
        ),
        drugWarning:
          typeof flag.drugWarning === 'string' && flag.drugWarning.trim()
            ? flag.drugWarning.trim()
            : null,
        source: String(flag.source || 'Nirogya genomics pipeline')
      }));

      if (normalizedFlags.length > 0) {
        await prisma.geneticFlag.createMany({ data: normalizedFlags });
      }

      res.status(201).json({
        geneticProfile: {
          id: savedProfile.id,
          uid: savedProfile.uid,
          rsIds: savedProfile.rsIds,
          prsT2d: Number(savedProfile.prsT2d),
          prsCad: Number(savedProfile.prsCad),
          prsHtn: Number(savedProfile.prsHtn),
          createdAt: savedProfile.createdAt,
          updatedAt: savedProfile.updatedAt
        },
        flags: normalizedFlags,
        summary: {
          rsIdCount: rsIds.length,
          clinvarAnnotatedCount: clinvarVariants.length,
          generatedFlagCount: normalizedFlags.length
        }
      });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: err.message || 'Genomic upload failed' });
    }
  }
);

app.get(['/genomics/profile', '/api/genomics/profile'], async (req, res) => {
  try {
    const profile = await prisma.geneticProfile.findUnique({
      where: { uid: req.uid },
      include: { flags: true }
    });

    if (!profile) {
      return res.json({ geneticProfile: null, flags: [] });
    }

    const severityRank = { HIGH: 3, MEDIUM: 2, LOW: 1 };
    const sortedFlags = [...profile.flags].sort((a, b) => {
      const scoreA = severityRank[String(a.severity || '').toUpperCase()] || 0;
      const scoreB = severityRank[String(b.severity || '').toUpperCase()] || 0;
      if (scoreA !== scoreB) return scoreB - scoreA;
      return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    });

    res.json({
      geneticProfile: {
        id: profile.id,
        uid: profile.uid,
        rsIds: profile.rsIds,
        prsT2d: Number(profile.prsT2d),
        prsCad: Number(profile.prsCad),
        prsHtn: Number(profile.prsHtn),
        createdAt: profile.createdAt,
        updatedAt: profile.updatedAt
      },
      flags: sortedFlags
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Could not fetch genomic profile' });
  }
});

app.post(['/share-links', '/api/share-links'], async (req, res) => {
  try {
    const data = req.body || {};
    const assessmentId = String(data.assessmentId || '').trim();

    if (!assessmentId) {
      return res.status(400).json({ error: 'assessmentId is required' });
    }

    const assessment = await prisma.assessment.findFirst({
      where: {
        id: assessmentId,
        uid: req.uid,
        deletedAt: null
      }
    });

    if (!assessment) {
      return res.status(404).json({ error: 'Assessment not found' });
    }

    const ttlHoursRaw = Number(data.expiresInHours);
    const ttlHours = Number.isFinite(ttlHoursRaw)
      ? Math.max(1, Math.min(Math.round(ttlHoursRaw), MAX_SHARE_LINK_TTL_HOURS))
      : DEFAULT_SHARE_LINK_TTL_HOURS;
    const expiresAt = new Date(Date.now() + ttlHours * 60 * 60 * 1000);

    const issued = createSignedShareToken();
    const created = await prisma.shareLink.create({
      data: {
        uid: req.uid,
        assessmentId: assessment.id,
        tokenHash: issued.tokenHash,
        expiresAt
      }
    });

    const shareUrl = `${getPublicShareBaseUrl(req)}/share/${encodeURIComponent(
      issued.token
    )}`;

    res.status(201).json({
      id: created.id,
      assessmentId: created.assessmentId,
      createdAt: created.createdAt,
      expiresAt: created.expiresAt,
      revokedAt: created.revokedAt,
      shareUrl
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Share link create error' });
  }
});

app.get(['/share-links', '/api/share-links'], async (req, res) => {
  try {
    const assessmentId = String(req.query.assessmentId || '').trim();

    const where = {
      uid: req.uid,
      ...(assessmentId ? { assessmentId } : {})
    };

    const list = await prisma.shareLink.findMany({
      where,
      orderBy: { createdAt: 'desc' },
      take: 50
    });

    const now = Date.now();
    res.json({
      shareLinks: list.map((item) => ({
        id: item.id,
        assessmentId: item.assessmentId,
        createdAt: item.createdAt,
        expiresAt: item.expiresAt,
        revokedAt: item.revokedAt,
        lastViewedAt: item.lastViewedAt,
        viewCount: item.viewCount,
        status: item.revokedAt
          ? 'revoked'
          : item.expiresAt.getTime() <= now
            ? 'expired'
            : 'active'
      }))
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Share link list error' });
  }
});

app.delete(['/share-links/:id', '/api/share-links/:id'], async (req, res) => {
  try {
    const shareLinkId = String(req.params.id || '').trim();
    if (!shareLinkId) {
      return res.status(400).json({ error: 'share link id is required' });
    }

    const updated = await prisma.shareLink.updateMany({
      where: {
        id: shareLinkId,
        uid: req.uid,
        revokedAt: null
      },
      data: {
        revokedAt: new Date()
      }
    });

    if (updated.count === 0) {
      return res.status(404).json({ error: 'Share link not found or already revoked' });
    }

    res.json({ ok: true, id: shareLinkId });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Share link revoke error' });
  }
});

app.post(['/overall-share-links', '/api/overall-share-links'], async (req, res) => {
  try {
    const data = req.body || {};
    const ttlHoursRaw = Number(data.expiresInHours);
    const ttlHours = Number.isFinite(ttlHoursRaw)
      ? Math.max(1, Math.min(Math.round(ttlHoursRaw), MAX_SHARE_LINK_TTL_HOURS))
      : DEFAULT_SHARE_LINK_TTL_HOURS;
    const expiresAt = new Date(Date.now() + ttlHours * 60 * 60 * 1000);

    const profile = await prisma.profile.findUnique({
      where: { uid: req.uid }
    });

    const assessments = await prisma.assessment.findMany({
      where: { uid: req.uid, deletedAt: null },
      orderBy: { createdAt: 'desc' },
      take: 250
    });

    const chatDetectedTitles = normalizeDetectedTitles(data.chatDetectedTitles, 12);

    if (assessments.length === 0 && chatDetectedTitles.length === 0) {
      return res.status(400).json({
        error: 'No overall report data yet. Complete at least one diagnosis or finalized chat detection.'
      });
    }

    const report = buildOverallSummaryReport({
      profile,
      assessments,
      chatDetectedTitles
    });

    const issued = createSignedShareToken();
    const created = await prisma.overallShareLink.create({
      data: {
        uid: req.uid,
        tokenHash: issued.tokenHash,
        report,
        expiresAt
      }
    });

    const shareUrl = `${getPublicShareBaseUrl(req)}/overall-share/${encodeURIComponent(
      issued.token
    )}`;

    res.status(201).json({
      id: created.id,
      createdAt: created.createdAt,
      expiresAt: created.expiresAt,
      revokedAt: created.revokedAt,
      status: 'active',
      shareUrl
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Overall share link create error' });
  }
});

app.get(['/overall-share-links', '/api/overall-share-links'], async (req, res) => {
  try {
    const list = await prisma.overallShareLink.findMany({
      where: { uid: req.uid },
      orderBy: { createdAt: 'desc' },
      take: 50
    });

    const now = Date.now();
    res.json({
      shareLinks: list.map((item) => ({
        id: item.id,
        createdAt: item.createdAt,
        expiresAt: item.expiresAt,
        revokedAt: item.revokedAt,
        lastViewedAt: item.lastViewedAt,
        viewCount: item.viewCount,
        status: item.revokedAt
          ? 'revoked'
          : item.expiresAt.getTime() <= now
            ? 'expired'
            : 'active'
      }))
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Overall share link list error' });
  }
});

app.delete(['/overall-share-links/:id', '/api/overall-share-links/:id'], async (req, res) => {
  try {
    const shareLinkId = String(req.params.id || '').trim();
    if (!shareLinkId) {
      return res.status(400).json({ error: 'overall share link id is required' });
    }

    const updated = await prisma.overallShareLink.updateMany({
      where: {
        id: shareLinkId,
        uid: req.uid,
        revokedAt: null
      },
      data: {
        revokedAt: new Date()
      }
    });

    if (updated.count === 0) {
      return res
        .status(404)
        .json({ error: 'Overall share link not found or already revoked' });
    }

    res.json({ ok: true, id: shareLinkId });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Overall share link revoke error' });
  }
});

app.get(['/facilities/nearby', '/api/facilities/nearby'], async (req, res) => {
  try {
    const dbProfile = await prisma.profile.findUnique({
      where: { uid: req.uid }
    });

    const city =
      String(req.query.city || '').trim() ||
      String(dbProfile?.city || '').trim() ||
      null;
    const district =
      String(req.query.district || '').trim() ||
      String(dbProfile?.district || '').trim() ||
      null;
    const state =
      String(req.query.state || '').trim() ||
      String(dbProfile?.state || '').trim() ||
      null;
    const riskLevelRaw = String(req.query.riskLevel || '').trim().toLowerCase();
    const riskLevel = ['low', 'medium', 'high'].includes(riskLevelRaw)
      ? riskLevelRaw
      : 'medium';
    const limit = Number(req.query.limit);

    const nearbyFacilities = await getNearbyFacilities({
      city,
      district,
      state,
      riskLevel,
      limit: Number.isFinite(limit) ? limit : 3
    });

    res.json({
      location: [city, district, state].filter(Boolean).join(', ') || null,
      riskLevel,
      nearbyFacilities
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Facility lookup error' });
  }
});

// AI: generate follow-up questions
app.post(['/ai/questions', '/api/ai/questions'], async (req, res) => {
  try {
    const { symptom, profile, language } = req.body || {};
    if (!symptom) return res.status(400).json({ error: 'symptom is required' });

    const dbProfile = await prisma.profile.findUnique({
      where: { uid: req.uid }
    });
    const mergedProfile = {
      ...(dbProfile || {}),
      ...(profile || {})
    };

    const result = await generateQuestions(symptom, mergedProfile, language);
    res.json({
      questions: result.questions,
      source: result.source,
      language: language || mergedProfile.language || 'en'
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'AI error' });
  }
});

// AI: generate diagnosis + fairness score
app.post(['/ai/diagnosis', '/api/ai/diagnosis'], async (req, res) => {
  try {
    const { symptom, answers, profile, language } = req.body || {};
    if (!symptom) return res.status(400).json({ error: 'symptom is required' });

    const dbProfile = await prisma.profile.findUnique({
      where: { uid: req.uid }
    });
    const mergedProfile = {
      ...(dbProfile || {}),
      ...(profile || {})
    };

    const assessment = await generateDiagnosis(
      symptom,
      answers || {},
      mergedProfile,
      language
    );

    const contextMap = buildContextMap(
      mergedProfile,
      language || mergedProfile.language || 'en'
    );
    const fairness = computeFairnessBreakdown(mergedProfile, assessment);
    const fairnessScore = computeFairnessScore(mergedProfile, assessment);
    const riskFlags = deriveRiskFlags(symptom, assessment);
    const recommendation = buildRecommendationFromAssessment(
      assessment,
      contextMap
    );
    const nearbyFacilities = await getNearbyFacilities({
      city: mergedProfile.city,
      district: mergedProfile.district,
      state: mergedProfile.state,
      riskLevel: assessment.riskLevel,
      limit: 3
    });
    const alerts = buildAssessmentAlerts({
      symptom,
      assessment,
      riskFlags,
      recommendation,
      nearbyFacilities
    });

    res.json({
      assessment: {
        diagnosis: assessment.diagnosis,
        description: assessment.description,
        riskScore: assessment.riskScore,
        riskLevel: assessment.riskLevel,
        nextSteps: assessment.nextSteps,
        seeDoctor: assessment.seeDoctor,
        urgency: assessment.urgency,
        fairnessScore,
        fairness
      },
      context: contextMap,
      riskFlags,
      recommendation,
      alerts,
      nearbyFacilities,
      source: assessment.source,
      language: language || mergedProfile.language || 'en'
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'AI error' });
  }
});

// AI: speech-to-text (Whisper/NIM)
app.post(
  ['/ai/voice/transcribe', '/api/ai/voice/transcribe'],
  async (req, res) => {
    try {
      const { audioBase64, language, mimeType } = req.body || {};
      if (!audioBase64) {
        return res.status(400).json({ error: 'audioBase64 is required' });
      }

      const result = await transcribeAudio({
        audioBase64,
        language,
        mimeType
      });

      res.json(result);
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: err.message || 'Transcription error' });
    }
  }
);

// AI: clinical image analysis
app.post(['/ai/image/analyze', '/api/ai/image/analyze'], async (req, res) => {
  try {
    const { imageBase64, language, profile } = req.body || {};
    if (!imageBase64) {
      return res.status(400).json({ error: 'imageBase64 is required' });
    }

    const dbProfile = await prisma.profile.findUnique({
      where: { uid: req.uid }
    });
    const mergedProfile = {
      ...(dbProfile || {}),
      ...(profile || {})
    };

    const result = await analyzeClinicalImage({
      imageBase64,
      profile: mergedProfile,
      language: language || mergedProfile.language || 'en'
    });

    res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Image analysis error' });
  }
});

// AI: OCR + extraction for medical documents
app.post(
  ['/ai/documents/extract', '/api/ai/documents/extract'],
  async (req, res) => {
    try {
      const { imageBase64, documentType, language } = req.body || {};
      if (!imageBase64) {
        return res.status(400).json({ error: 'imageBase64 is required' });
      }

      const result = await extractDocumentData({
        imageBase64,
        documentType,
        language
      });

      res.json(result);
    } catch (err) {
      console.error(err);
      res
        .status(500)
        .json({ error: err.message || 'Document extraction error' });
    }
  }
);

// AI: symptom triage (Infermedica + fallback rules)
app.post(['/ai/triage', '/api/ai/triage'], async (req, res) => {
  try {
    const { symptomText, sex, age, language } = req.body || {};
    if (!symptomText) {
      return res.status(400).json({ error: 'symptomText is required' });
    }

    const result = await triageSymptoms({
      symptomText,
      sex,
      age,
      language
    });

    res.json(result);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Triage error' });
  }
});

// AI: end-to-end pipeline orchestration
app.post(['/ai/pipeline/run', '/api/ai/pipeline/run'], async (req, res) => {
  try {
    const { symptom, answers, profile, language, history } = req.body || {};
    if (!symptom) {
      return res.status(400).json({ error: 'symptom is required' });
    }

    const dbProfile = await prisma.profile.findUnique({
      where: { uid: req.uid }
    });
    const mergedProfile = {
      ...(dbProfile || {}),
      ...(profile || {})
    };

    const result = await runFullPipeline({
      symptom,
      answers: answers || {},
      profile: mergedProfile,
      language: language || mergedProfile.language || 'en',
      history: Array.isArray(history) ? history : []
    });

    const nearbyFacilities = await getNearbyFacilities({
      city: mergedProfile.city,
      district: mergedProfile.district,
      state: mergedProfile.state,
      riskLevel: result?.assessment?.riskLevel,
      limit: 3
    });
    const alerts = buildAssessmentAlerts({
      symptom,
      assessment: result.assessment,
      riskFlags: result.riskFlags,
      recommendation: result.recommendation,
      nearbyFacilities
    });

    res.json({
      ...result,
      alerts,
      nearbyFacilities
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Pipeline execution error' });
  }
});

// AI: start conversation session
app.post(
  [
    '/ai/conversations/start',
    '/api/ai/conversations/start',
    '/api/conversations/start'
  ],
  async (req, res) => {
    try {
      const { language, profile } = req.body || {};
      const dbProfile = await prisma.profile.findUnique({
        where: { uid: req.uid }
      });
      const mergedProfile = {
        ...(dbProfile || {}),
        ...(profile || {})
      };

      const conversationId =
        typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
          ? crypto.randomUUID()
          : `${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;

      const conversationKey = `${req.uid}:${conversationId}`;
      const greeting =
        'Hi! I am Nira, your AI health assistant. Tell me what you are feeling today.';

      conversations.set(conversationKey, {
        uid: req.uid,
        profile: mergedProfile,
        language: language || mergedProfile.language || 'en',
        messages: [{ role: 'assistant', content: greeting }],
        flow: createConversationFlow(),
        createdAt: new Date().toISOString()
      });

      res.json({
        conversationId,
        greeting,
        language: language || mergedProfile.language || 'en'
      });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: err.message || 'AI error' });
    }
  }
);

// AI: send a message in an existing conversation
app.post(
  [
    '/ai/conversations/:id/message',
    '/api/ai/conversations/:id/message',
    '/api/conversations/:id/message'
  ],
  async (req, res) => {
    try {
      const conversationId = req.params.id;
      const { message, language, profile } = req.body || {};

      if (!message || typeof message !== 'string') {
        return res.status(400).json({ error: 'message is required' });
      }

      const key = `${req.uid}:${conversationId}`;
      const current = conversations.get(key);
      if (!current) {
        return res.status(404).json({ error: 'Conversation not found' });
      }

      const dbProfile = await prisma.profile.findUnique({
        where: { uid: req.uid }
      });
      const mergedProfile = {
        ...(dbProfile || {}),
        ...(current.profile || {}),
        ...(profile || {})
      };
      const activeLanguage =
        language || current.language || mergedProfile.language || 'en';

      const flow =
        current.flow && typeof current.flow === 'object'
          ? {
              stage: current.flow.stage || 'idle',
              symptom: current.flow.symptom || null,
              questions: Array.isArray(current.flow.questions)
                ? current.flow.questions
                : [],
              questionIndex: Number(current.flow.questionIndex) || 0,
              answers:
                current.flow.answers && typeof current.flow.answers === 'object'
                  ? current.flow.answers
                  : {}
            }
          : createConversationFlow();

      const incomingMessage = String(message || '').trim();
      const priorMessages = Array.isArray(current.messages) ? current.messages : [];
      const historyWithUser = [
        ...priorMessages,
        { role: 'user', content: incomingMessage }
      ];

      if (
        flow.stage === 'questionnaire' &&
        flow.symptom &&
        Array.isArray(flow.questions) &&
        flow.questions.length > 0 &&
        flow.questionIndex < flow.questions.length
      ) {
        const activeQuestion = flow.questions[flow.questionIndex];
        const updatedAnswers = {
          ...(flow.answers || {}),
          [activeQuestion?.id || `q${flow.questionIndex + 1}`]: incomingMessage
        };

        const nextQuestionIndex = flow.questionIndex + 1;
        if (nextQuestionIndex < flow.questions.length) {
          const nextQuestion = flow.questions[nextQuestionIndex];
          const reply = formatQuestionForChat(
            nextQuestion,
            nextQuestionIndex + 1,
            flow.questions.length
          );

          conversations.set(key, {
            ...current,
            profile: mergedProfile,
            language: activeLanguage,
            flow: {
              ...flow,
              questionIndex: nextQuestionIndex,
              answers: updatedAnswers
            },
            messages: [...historyWithUser, { role: 'assistant', content: reply }].slice(
              -20
            ),
            updatedAt: new Date().toISOString()
          });

          return res.json({
            reply,
            isUrgent: false,
            source: 'question-flow',
            conversationId,
            followUpQuestion: {
              id: nextQuestion.id,
              text: nextQuestion.text,
              options: nextQuestion.options || [],
              index: nextQuestionIndex + 1,
              total: flow.questions.length
            },
            alerts: [
              {
                id: `info-${Date.now()}`,
                severity: 'info',
                title: 'Clinical follow-up',
                message: 'Please answer the next follow-up question.',
                action: null
              }
            ],
            nearbyFacilities: []
          });
        }

        const assessment = await generateDiagnosis(
          flow.symptom,
          updatedAnswers,
          mergedProfile,
          activeLanguage
        );
        const contextMap = buildContextMap(mergedProfile, activeLanguage);
        const fairness = computeFairnessBreakdown(mergedProfile, assessment);
        const fairnessScore = computeFairnessScore(mergedProfile, assessment);
        const riskFlags = deriveRiskFlags(flow.symptom, assessment);
        const recommendation = buildRecommendationFromAssessment(
          assessment,
          contextMap
        );
        const nearbyFacilities = await getNearbyFacilities({
          city: mergedProfile.city,
          district: mergedProfile.district,
          state: mergedProfile.state,
          riskLevel: assessment.riskLevel,
          limit: 3
        });
        const alerts = buildAssessmentAlerts({
          symptom: flow.symptom,
          assessment,
          riskFlags,
          recommendation,
          nearbyFacilities
        });
        const assessmentPayload = {
          diagnosis: assessment.diagnosis,
          description: assessment.description,
          riskScore: assessment.riskScore,
          riskLevel: assessment.riskLevel,
          nextSteps: assessment.nextSteps,
          seeDoctor: assessment.seeDoctor,
          urgency: assessment.urgency,
          fairnessScore,
          fairness
        };

        const reply = formatAssessmentReply({
          assessment: assessmentPayload,
          recommendation,
          nearbyFacilities
        });

        conversations.set(key, {
          ...current,
          profile: mergedProfile,
          language: activeLanguage,
          flow: createConversationFlow(),
          messages: [...historyWithUser, { role: 'assistant', content: reply }].slice(
            -20
          ),
          updatedAt: new Date().toISOString()
        });

        return res.json({
          reply,
          isUrgent:
            assessment.riskLevel === 'high' ||
            alerts.some((alert) => alert.severity === 'critical'),
          source: assessment.source || 'conversation-assessment',
          conversationId,
          followUpQuestion: null,
          assessment: assessmentPayload,
          recommendation,
          riskFlags,
          alerts,
          nearbyFacilities
        });
      }

      const combinedUrgencyText = `${incomingMessage} ${flow?.symptom || ''}`;
      if (detectUrgency(combinedUrgencyText)) {
        const emergencyAssessment = {
          diagnosis: 'Potential emergency symptom pattern',
          description:
            'Your message suggests urgent warning signs. Immediate in-person evaluation is recommended.',
          riskScore: 92,
          riskLevel: 'high',
          nextSteps: [
            'Call emergency services (108) now if symptoms are severe.',
            'Go to the nearest emergency facility immediately.',
            'Carry current medications and prior reports if available.'
          ],
          seeDoctor: true,
          urgency: 'Immediate'
        };
        const contextMap = buildContextMap(mergedProfile, activeLanguage);
        const recommendation = buildRecommendationFromAssessment(
          emergencyAssessment,
          contextMap
        );
        const nearbyFacilities = await getNearbyFacilities({
          city: mergedProfile.city,
          district: mergedProfile.district,
          state: mergedProfile.state,
          riskLevel: 'high',
          limit: 3
        });
        const riskFlags = deriveRiskFlags(incomingMessage, emergencyAssessment);
        const alerts = buildAssessmentAlerts({
          symptom: incomingMessage,
          assessment: emergencyAssessment,
          riskFlags,
          recommendation,
          nearbyFacilities
        });
        const reply = formatAssessmentReply({
          assessment: emergencyAssessment,
          recommendation,
          nearbyFacilities
        });

        conversations.set(key, {
          ...current,
          profile: mergedProfile,
          language: activeLanguage,
          flow: createConversationFlow(),
          messages: [...historyWithUser, { role: 'assistant', content: reply }].slice(
            -20
          ),
          updatedAt: new Date().toISOString()
        });

        return res.json({
          reply,
          isUrgent: true,
          source: 'urgency-guard',
          conversationId,
          followUpQuestion: null,
          assessment: emergencyAssessment,
          recommendation,
          riskFlags,
          alerts,
          nearbyFacilities
        });
      }

      const generatedQuestions = await generateQuestions(
        incomingMessage,
        mergedProfile,
        activeLanguage
      );
      const selectedQuestions = Array.isArray(generatedQuestions?.questions)
        ? generatedQuestions.questions.slice(0, 5)
        : [];

      if (selectedQuestions.length > 0) {
        const firstQuestion = selectedQuestions[0];
        const intro =
          selectedQuestions.length <= 2
            ? 'I need a couple of quick follow-up checks before recommendations.'
            : 'I will ask a few focused follow-up questions before I recommend next steps.';
        const reply = `${intro}\n\n${formatQuestionForChat(
          firstQuestion,
          1,
          selectedQuestions.length
        )}`;

        conversations.set(key, {
          ...current,
          profile: mergedProfile,
          language: activeLanguage,
          flow: {
            stage: 'questionnaire',
            symptom: incomingMessage,
            questions: selectedQuestions,
            questionIndex: 0,
            answers: {}
          },
          messages: [...historyWithUser, { role: 'assistant', content: reply }].slice(
            -20
          ),
          updatedAt: new Date().toISOString()
        });

        return res.json({
          reply,
          isUrgent: false,
          source: generatedQuestions.source || 'questions',
          conversationId,
          followUpQuestion: {
            id: firstQuestion.id,
            text: firstQuestion.text,
            options: firstQuestion.options || [],
            index: 1,
            total: selectedQuestions.length
          },
          alerts: [
            {
              id: `info-${Date.now()}`,
              severity: 'info',
              title: 'Follow-up started',
              message: 'I need a few quick follow-up answers before recommendations.',
              action: null
            }
          ],
          nearbyFacilities: []
        });
      }

      const fallbackResult = await generateConversationReply({
        message: incomingMessage,
        profile: mergedProfile,
        language: activeLanguage,
        history: historyWithUser
      });

      conversations.set(key, {
        ...current,
        profile: mergedProfile,
        language: activeLanguage,
        flow: createConversationFlow(),
        messages: [...historyWithUser, { role: 'assistant', content: fallbackResult.reply }].slice(
          -20
        ),
        updatedAt: new Date().toISOString()
      });

      return res.json({
        reply: fallbackResult.reply,
        isUrgent: fallbackResult.isUrgent,
        source: fallbackResult.source,
        conversationId,
        followUpQuestion: null,
        alerts: fallbackResult.isUrgent
          ? [
              {
                id: `critical-${Date.now()}`,
                severity: 'critical',
                title: 'Urgent guidance',
                message: fallbackResult.reply,
                action: null
              }
            ]
          : []
      });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: err.message || 'AI error' });
    }
  }
);

// ── Global error handler: logs every unhandled route error with context ─
app.use((err, req, res, _next) => {
  console.error(`[error] ${req.method} ${req.originalUrl}:`, err);
  if (res.headersSent) return;
  res.status(err?.status || 500).json({ error: err?.message || 'Internal server error' });
});

// ── Never let async failures vanish silently ────────────────────────────
process.on('unhandledRejection', (reason) => {
  console.error('[process] Unhandled promise rejection:', reason);
});
process.on('uncaughtException', (err) => {
  console.error('[process] Uncaught exception:', err);
});

const port = Number(process.env.PORT || 4000);
const host = process.env.HOST || '0.0.0.0';

const server = app.listen(port, host, () => {
  console.log(`Backend running on http://${host}:${port}`);
});

server.on('error', (err) => {
  if (err?.code === 'EADDRINUSE') {
    console.error(
      `Port ${port} is already in use. Stop the existing backend process before starting a new one.`
    );
    process.exit(1);
    return;
  }

  console.error('Backend startup failed:', err?.message || err);
  process.exit(1);
});
