import React, { useEffect, useState } from "react";

// ThreatSense — single-file React component (Tailwind CSS assumed)
// To run: create a React app (Vite or CRA), add Tailwind if you want the styling to match,
// and replace App.jsx with this file. This component trains a tiny Multinomial Naive Bayes
// on a handful of example messages in-browser (so it's "ML-powered" but lightweight).

export default function ThreatSenseApp() {
  // UI state
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);
  const [model, setModel] = useState(null);
  const [score, setScore] = useState({ human: 0, ai: 0 });
  const [lastUserGuess, setLastUserGuess] = useState(null);

  // tiny training dataset — short, synthetic examples for classroom demo
  // label: 'safe' | 'threat' | 'blackmail' | 'harass'
  const TRAINING = [
    ["Hey can I borrow your notes?", "safe"],
    ["Are you free to meet tomorrow?", "safe"],
    ["Please send the pics again, I lost them.", "safe"],

    ["Send me those photos or I'll share them", "blackmail"],
    ["If you don't, I'll leak everything", "blackmail"],
    ["You better pay me or I'll expose you", "blackmail"],

    ["I'm coming to your place tonight", "threat"],
    ["If you talk I'll make you regret it", "threat"],
    ["Come alone or else", "threat"],

    ["You're useless, everyone will know", "harass"],
    ["Stop talking like that, idiot", "harass"],
    ["Nobody likes you, just disappear", "harass"],

    // borderline / disguised threats
    ["Maybe it's better if you just do what I say", "threat"],
    ["Don't test me. You know I can make trouble", "threat"],

    // polite but manipulative (coercion)
    ["If you loved me you'd send them", "harass"],
    ["This will ruin you if you don't cooperate", "blackmail"],

    // more safe phrases
    ["Thanks for the help!", "safe"],
    ["Can you share the link?", "safe"],

    // training extra
    ["I will ruin your reputation", "blackmail"],
    ["Meet me now or I'll call everyone", "threat"],
    ["You should be ashamed", "harass"],
  ];

  // Keyword triggers that raise severity or give reasons
  const KEYWORDS = {
    leak: ["leak", "expose", "share", "publish"],
    coercion: ["or else", "you better", "if you don't", "do what I say"],
    meeting: ["come to", "meet me", "alone"],
    threatWords: ["kill", "hurt", "regret", "come to your"],
  };

  // Utility: simple tokenizer
  function tokenize(text) {
    return text
      .toLowerCase()
      .replace(/["'`(),.?;:!\/\\]/g, " ")
      .split(/\s+/)
      .filter(Boolean);
  }

  // Train a tiny multinomial naive bayes in-browser
  function trainNaiveBayes(dataset) {
    const classes = {};
    let vocab = new Set();

    for (const [text, label] of dataset) {
      if (!classes[label]) classes[label] = { count: 0, wordCounts: {} };
      classes[label].count += 1;
      const words = tokenize(text);
      for (const w of words) {
        vocab.add(w);
        classes[label].wordCounts[w] = (classes[label].wordCounts[w] || 0) + 1;
      }
    }

    const V = vocab.size;
    const totalDocs = dataset.length;

    // compute priors and likelihoods with Laplace smoothing
    const model = { classes: {}, V, totalDocs, vocab: Array.from(vocab) };

    for (const label of Object.keys(classes)) {
      const cls = classes[label];
      const totalWordsInClass = Object.values(cls.wordCounts).reduce((a, b) => a + b, 0);
      const prior = Math.log(cls.count / totalDocs);
      const likelihood = {}; // log-likelihoods
      for (const w of model.vocab) {
        const countW = cls.wordCounts[w] || 0;
        // Laplace smoothing
        likelihood[w] = Math.log((countW + 1) / (totalWordsInClass + V));
      }
      model.classes[label] = { prior, likelihood, totalWordsInClass };
    }

    return model;
  }

  function predictNB(model, text) {
    const words = tokenize(text);
    const scores = {};
    for (const label of Object.keys(model.classes)) {
      let s = model.classes[label].prior;
      for (const w of words) {
        if (model.vocab.includes(w)) s += model.classes[label].likelihood[w] || 0;
        else {
          // unknown word: approx uniform small probability
          s += Math.log(1 / (model.classes[label].totalWordsInClass + model.V));
        }
      }
      scores[label] = s;
    }

    // convert log-scores to probabilities (softmax)
    const maxS = Math.max(...Object.values(scores));
    const exps = Object.fromEntries(
      Object.entries(scores).map(([k, v]) => [k, Math.exp(v - maxS)])
    );
    const sumExps = Object.values(exps).reduce((a, b) => a + b, 0);
    const probs = Object.fromEntries(
      Object.entries(exps).map(([k, v]) => [k, v / sumExps])
    );

    return { scores, probs };
  }

  // severity mapping function
  function severityFromProbs(probs) {
    // threat-ish classes: threat, blackmail, harass
    const threatProb = (probs.threat || 0) + (probs.blackmail || 0) + (probs.harass || 0);

    // Keyword escalation
    let keywordBoost = 0;
    const lower = input.toLowerCase();
    for (const k of KEYWORDS.threatWords) if (lower.includes(k)) keywordBoost += 0.15;
    for (const k of KEYWORDS.leak) if (lower.includes(k)) keywordBoost += 0.2;
    for (const k of KEYWORDS.coercion) if (lower.includes(k)) keywordBoost += 0.12;

    const final = Math.min(1, threatProb + keywordBoost);

    let label = "Safe";
    if (final >= 0.91) label = "Critical";
    else if (final >= 0.76) label = "High";
    else if (final >= 0.56) label = "Medium";
    else if (final >= 0.36) label = "Low";
    else label = "Safe";

    return { final, label, threatProb };
  }

  // reason extraction
  function getReasons(text) {
    const reasons = [];
    const l = text.toLowerCase();
    for (const w of KEYWORDS.leak) if (l.includes(w)) reasons.push("Blackmail / threat to expose content");
    for (const w of KEYWORDS.coercion) if (l.includes(w)) reasons.push("Coercion / conditional demand");
    for (const w of KEYWORDS.meeting) if (l.includes(w)) reasons.push("Request to meet alone / potential physical risk");
    for (const w of KEYWORDS.threatWords) if (l.includes(w)) reasons.push("Direct threatening language");
    return Array.from(new Set(reasons));
  }

  // recommended actions
  function recommendedActions(sevLabel) {
    const base = [
      "Stop contact immediately (do not reply)",
      "Take clean screenshots: include handle, timestamp, full conversation",
      "Save evidence to device and secure cloud (do not delete)"
    ];
    if (sevLabel === "Safe") return ["No immediate drastic action needed. Monitor and avoid sharing personal data.", ...base];

    if (sevLabel === "Low")
      return [
        "Block or mute the sender",
        "Tell a trusted friend / peer",
        ...base
      ];

    if (sevLabel === "Medium")
      return [
        "Block the sender immediately",
        "Report to platform (social media / messaging app)",
        "Inform campus support or a trusted adult",
        ...base
      ];

    if (sevLabel === "High" || sevLabel === "Critical")
      return [
        "Cut contact immediately and preserve all evidence",
        "Inform local authorities or campus security",
        "Get help from peers or a trusted adult — do NOT confront the sender alone",
        ...base
      ];

    return base;
  }

  // Generate sample prompts for quick demo
  const SAMPLES = [
    "Send me those pictures or I'll show everyone.",
    "Hey, can you drop the assignment here?",
    "If you don't come I'm going to expose things about you.",
    "Let's meet up at your place tonight.",
    "Thanks a lot! You're a lifesaver.",
  ];

  useEffect(() => {
    const m = trainNaiveBayes(TRAINING);
    setModel(m);
  }, []);

  function analyze(text, userGuess = null) {
    if (!model) return;
    setInput(text);
    setLastUserGuess(userGuess);
    const { scores, probs } = predictNB(model, text);
    const sev = severityFromProbs(probs);
    const reasons = getReasons(text);
    const actions = recommendedActions(sev.label);

    // determine top predicted class
    const topClass = Object.entries(probs).sort((a, b) => b[1] - a[1])[0];

    // update ai/human score if user guessed
    if (userGuess) {
      const aiCorrect = topClass[0] !== "safe" ? true : false; // crude correctness: non-safe predicted = threat
      const userCorrect = (userGuess === (topClass[0] !== "safe" ? "threat" : "safe"));
      setScore((s) => ({ human: s.human + (userCorrect ? 1 : 0), ai: s.ai + (aiCorrect ? 1 : 0) }));
    }

    setResult({ probs, scores, sev, reasons, actions, topClass });
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-6">
      <div className="max-w-4xl mx-auto bg-white shadow-lg rounded-2xl p-6">
        <h1 className="text-2xl font-bold mb-2">ThreatSense — AI Safety Classifier</h1>
        <p className="text-sm text-gray-600 mb-4">Type or paste a message and press Analyze. The tiny in-browser ML will score it and suggest actions.</p>

        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type message here... (e.g. 'If you don't send those photos I'll leak them')"
          className="w-full border rounded-md p-3 mb-3 h-28"
        />

        <div className="flex gap-2 mb-4">
          <button
            onClick={() => analyze(input)}
            className="px-4 py-2 bg-indigo-600 text-white rounded-md shadow-sm"
          >
            Analyze
          </button>

          <button
            onClick={() => {
              const sample = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
              analyze(sample);
            }}
            className="px-4 py-2 bg-emerald-600 text-white rounded-md shadow-sm"
          >
            Try sample
          </button>

          <button
            onClick={() => {
              setInput("");
              setResult(null);
            }}
            className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md"
          >
            Clear
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-2">
            <div className="bg-slate-50 p-4 rounded-md mb-4">
              <h3 className="font-semibold">Input Preview</h3>
              <p className="text-sm text-gray-700 mt-2">{input || "(no input)"}</p>
            </div>

            {result ? (
              <div className="bg-white p-4 rounded-md border">
                <h3 className="font-semibold">AI Classification</h3>
                <p className="mt-2">Top class: <strong className="capitalize">{result.topClass[0]}</strong> (confidence {(result.probs[result.topClass[0]]*100).toFixed(0)}%)</p>

                <div className="mt-3">
                  <h4 className="font-medium">Severity</h4>
                  <div className="mt-1 flex items-center gap-3">
                    <div className="text-lg font-bold">{result.sev.label}</div>
                    <div className="text-sm text-gray-500">Score: {(result.sev.final*100).toFixed(0)}%</div>
                  </div>
                </div>

                <div className="mt-3">
                  <h4 className="font-medium">Why the model flagged this</h4>
                  {result.reasons.length ? (
                    <ul className="list-disc list-inside text-sm mt-2">
                      {result.reasons.map((r, i) => <li key={i}>{r}</li>)}
                    </ul>
                  ) : (
                    <p className="text-sm text-gray-500 mt-2">No high-confidence keyword triggers detected — model used general patterns.</p>
                  )}
                </div>

                <div className="mt-3">
                  <h4 className="font-medium">Recommended actions</h4>
                  <ol className="list-decimal list-inside mt-2 text-sm">
                    {result.actions.map((a, i) => <li key={i}>{a}</li>)}
                  </ol>
                </div>

                <div className="mt-4">
                  <h4 className="font-medium">Model probabilities (by class)</h4>
                  <div className="mt-2 space-y-1 text-sm">
                    {Object.entries(result.probs).map(([k, v]) => (
                      <div key={k} className="flex justify-between">
                        <div className="capitalize">{k}</div>
                        <div>{(v*100).toFixed(1)}%</div>
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            ) : (
              <div className="bg-white p-4 rounded-md border text-sm text-gray-600">No analysis yet. Type a message and press <strong>Analyze</strong>.</div>
            )}

          </div>

          <aside className="bg-slate-50 p-4 rounded-md">
            <h4 className="font-semibold mb-2">Quick actions</h4>
            <div className="space-y-2 text-sm">
              <div>
                <strong>Sample messages</strong>
                <ul className="list-disc list-inside mt-1">
                  {SAMPLES.map((s, i) => (
                    <li key={i} className="cursor-pointer text-indigo-600" onClick={() => analyze(s)}>{s}</li>
                  ))}
                </ul>
              </div>

              <div>
                <strong>Explain mode</strong>
                <p className="text-xs text-gray-600 mt-1">The model is a tiny Naive Bayes trained in-browser on a small synthetic dataset. It's for demo/education — not a production classifier.</p>
              </div>

              <div>
                <strong>Scoreboard</strong>
                <p className="text-sm mt-1">Human: {score.human} &nbsp; | &nbsp; AI: {score.ai}</p>
              </div>

              <div className="mt-3">
                <strong>Presentation tips</strong>
                <ul className="list-disc list-inside text-xs mt-1">
                  <li>Demo 3–5 real-like messages to show model strengths and limits.</li>
                  <li>Explain shortcomings: small dataset, biases, false positives.</li>
                  <li>Show clean-screenshot checklist after a critical result.</li>
                </ul>
              </div>
            </div>
          </aside>
        </div>

        <div className="mt-6 bg-gray-50 p-4 rounded-md text-sm">
          <strong>Note:</strong> This demo uses a tiny dataset and a simple in-browser Naive Bayes. It is intended for classroom demonstration only — do not rely on it for real-life safety decisions. Always follow campus-approved reporting and authorities.
        </div>
      </div>
    </div>
  );
}

