"use client";

import { useState } from "react";

type Language = "en" | "hi";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000";

const questions = [
  {
    key: "state",
    type: "text",
    en: "Which state do you live in?",
    hi: "आप किस राज्य में रहते हैं?",
    placeholderEn: "e.g. Bihar",
    placeholderHi: "जैसे बिहार",
  },
  {
    key: "cultivable_land",
    type: "choice",
    en: "Do you own cultivable agricultural land?",
    hi: "क्या आपके पास खेती योग्य कृषि भूमि है?",
  },
  {
    key: "institutional_landholder",
    type: "choice",
    en: "Are you an institutional landholder?",
    hi: "क्या आप संस्थागत भूमि धारक हैं?",
  },
  {
    key: "government_employee",
    type: "choice",
    en: "Are you a government employee?",
    hi: "क्या आप सरकारी कर्मचारी हैं?",
  },
  {
    key: "retired_pensioner",
    type: "choice",
    en: "Are you a retired or superannuated pensioner?",
    hi: "क्या आप सेवानिवृत्त पेंशनभोगी हैं?",
  },
  {
    key: "paid_income_tax",
    type: "choice",
    en: "Did you pay income tax in the last assessment year?",
    hi: "क्या आपने पिछले आकलन वर्ष में आयकर दिया था?",
  },
  {
    key: "registered_professional",
    type: "choice",
    en: "Are you a registered professional?",
    hi: "क्या आप पंजीकृत पेशेवर हैं?",
  },
  {
    key: "nri",
    type: "choice",
    en: "Are you an NRI?",
    hi: "क्या आप NRI हैं?",
  },
];

export default function Home() {
  const [language, setLanguage] =
    useState<Language>("en");

  const [step, setStep] = useState(0);

  const [form, setForm] = useState({
    state: "",
    cultivable_land: "",
    institutional_landholder: "",
    government_employee: "",
    retired_pensioner: "",
    monthly_pension: "",
    paid_income_tax: "",
    registered_professional: "",
    nri: "",
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState("");
  const [eligibility, setEligibility] = useState("");
  const [error, setError] = useState("");

  const currentQuestion = questions[step];

  const currentValue =
    form[currentQuestion.key as keyof typeof form];

  const handleChange = (value: string) => {
    setForm({
      ...form,
      [currentQuestion.key]: value,
    });
  };

  const nextStep = () => {
    if (!currentValue) return;

    if (step < questions.length - 1) {
      setStep(step + 1);
    }
  };

  const previousStep = () => {
    if (step > 0) {
      setStep(step - 1);
    }
  };

  const checkEligibility = async () => {
    if (!currentValue) return;

    setLoading(true);
    setResult("");
    setEligibility("");
    setError("");

    try {
      const response = await fetch(
        `${API_URL}/eligibility`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            scheme: "PM-KISAN",
            language,
            user_profile: form,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Request failed: ${response.status}`
        );
      }

      const data = await response.json();

      setResult(data.answer || "");
      setEligibility(data.eligibility || "");

    } catch (error) {
      console.error(error);

      setError(
        language === "hi"
          ? "Eligibility service से कनेक्ट नहीं हो पाया।"
          : "Unable to connect to the eligibility service."
      );
    } finally {
      setLoading(false);
    }
  };

  const restart = () => {
    setStep(0);

    setForm({
      state: "",
      cultivable_land: "",
      institutional_landholder: "",
      government_employee: "",
      retired_pensioner: "",
      monthly_pension: "",
      paid_income_tax: "",
      registered_professional: "",
      nri: "",
    });

    setResult("");
    setEligibility("");
    setError("");
  };

  const progress =
    ((step + 1) / questions.length) * 100;

  return (
    <main className="min-h-screen bg-[#f5f7f4] text-gray-900">

      {/* NAVBAR */}

      <nav className="border-b bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-4">

          <div className="flex items-center gap-3">

            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-green-700 text-lg text-white">
              🇮🇳
            </div>

            <div>
              <p className="font-bold">
                Scheme Navigator
              </p>

              <p className="text-xs text-gray-500">
                Government Benefits
              </p>
            </div>

          </div>

          {/* LANGUAGE */}

          <div className="flex rounded-full border bg-gray-50 p-1">

            <button
              onClick={() => setLanguage("en")}
              className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                language === "en"
                  ? "bg-green-700 text-white"
                  : "text-gray-600"
              }`}
            >
              English
            </button>

            <button
              onClick={() => setLanguage("hi")}
              className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                language === "hi"
                  ? "bg-green-700 text-white"
                  : "text-gray-600"
              }`}
            >
              हिंदी
            </button>

          </div>

        </div>
      </nav>

      {/* HERO */}

      {!result && !loading && (
        <section className="mx-auto max-w-6xl px-5 pb-6 pt-12">

          <div className="max-w-3xl">

            <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-green-100 px-4 py-2 text-sm font-medium text-green-800">
              <span>✓</span>

              {language === "hi"
                ? "सरकारी योजनाओं की पात्रता आसानी से जानें"
                : "Check government scheme eligibility easily"}
            </div>

            <h1 className="text-4xl font-bold leading-tight tracking-tight md:text-5xl">
              {language === "hi"
                ? "आप किन सरकारी योजनाओं के लिए पात्र हैं?"
                : "Which government schemes are you eligible for?"}
            </h1>

            <p className="mt-5 max-w-2xl text-lg leading-8 text-gray-600">
              {language === "hi"
                ? "कुछ आसान सवालों के जवाब दें। हम सरकारी दस्तावेज़ों के आधार पर आपकी संभावित पात्रता समझाने में मदद करेंगे।"
                : "Answer a few simple questions. We use official government documents to explain which schemes you may qualify for."}
            </p>

          </div>

        </section>
      )}

      {/* MAIN CONTENT */}

      <section className="mx-auto max-w-3xl px-5 pb-16">

        {/* RESULT */}

        {result ? (
          <div className="mt-8 space-y-5">

            {/* Result Header */}

            <div className="rounded-2xl border bg-white p-6 shadow-sm">

              <div className="flex items-start justify-between gap-4">

                <div>

                  <p className="text-sm font-medium text-gray-500">
                    {language === "hi"
                      ? "पात्रता परिणाम"
                      : "Eligibility result"}
                  </p>

                  <h2 className="mt-2 text-3xl font-bold">
                    {eligibility === "Likely eligible"
                      ? language === "hi"
                        ? "संभावित रूप से पात्र"
                        : "Likely eligible"
                      : eligibility === "Likely not eligible"
                        ? language === "hi"
                          ? "संभावित रूप से पात्र नहीं"
                          : "Likely not eligible"
                        : language === "hi"
                          ? "निर्धारित नहीं किया जा सकता"
                          : "Cannot determine"}
                  </h2>

                </div>

                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-100 text-xl">
                  ✓
                </div>

              </div>

              <div className="mt-5 rounded-xl bg-gray-50 p-4">

                <p className="text-sm font-medium text-gray-500">
                  {language === "hi"
                    ? "योजना"
                    : "Scheme"}
                </p>

                <p className="mt-1 font-semibold">
                  PM-KISAN
                </p>

              </div>

            </div>

            {/* AI RESULT */}

            <div className="rounded-2xl border bg-white p-6 shadow-sm">

              <h3 className="text-xl font-bold">
                {language === "hi"
                  ? "विस्तृत जानकारी"
                  : "Detailed assessment"}
              </h3>

              <div className="mt-5 whitespace-pre-wrap leading-8 text-gray-700">
                {result}
              </div>

            </div>

            {/* DISCLAIMER */}

            <div className="rounded-2xl border border-yellow-200 bg-yellow-50 p-5">

              <p className="font-semibold text-yellow-900">
                {language === "hi"
                  ? "महत्वपूर्ण"
                  : "Important"}
              </p>

              <p className="mt-2 text-sm leading-6 text-yellow-800">
                {language === "hi"
                  ? "यह परिणाम उपलब्ध सरकारी दस्तावेज़ों और आपके द्वारा दी गई जानकारी के आधार पर एक संभावित आकलन है। अंतिम पात्रता संबंधित सरकारी विभाग द्वारा निर्धारित की जाती है।"
                  : "This is an indicative assessment based on the government documents available to the system and the information you provided. Final eligibility is determined by the relevant government authority."}
              </p>

            </div>

            <button
              onClick={restart}
              className="w-full rounded-xl bg-green-700 px-5 py-4 font-semibold text-white transition hover:bg-green-800"
            >
              {language === "hi"
                ? "फिर से जाँच करें"
                : "Check another profile"}
            </button>

          </div>

        ) : (

          /* QUESTION CARD */

          <div className="mt-6 rounded-3xl border bg-white p-6 shadow-sm md:p-8">

            {/* CARD HEADER */}

            <div className="flex items-center justify-between">

              <div>

                <p className="text-sm font-medium text-gray-500">
                  {language === "hi"
                    ? "PM-KISAN"
                    : "PM-KISAN"}
                </p>

                <p className="mt-1 text-sm text-gray-500">
                  {language === "hi"
                    ? `सवाल ${step + 1} / ${questions.length}`
                    : `Question ${step + 1} of ${questions.length}`}
                </p>

              </div>

              <span className="text-sm font-semibold text-green-700">
                {Math.round(progress)}%
              </span>

            </div>

            {/* PROGRESS */}

            <div className="mt-4 h-2 overflow-hidden rounded-full bg-gray-100">

              <div
                className="h-full rounded-full bg-green-700 transition-all duration-300"
                style={{
                  width: `${progress}%`,
                }}
              />

            </div>

            {/* QUESTION */}

            <div className="mt-10">

              <h2 className="text-2xl font-bold leading-tight md:text-3xl">
                {language === "hi"
                  ? currentQuestion.hi
                  : currentQuestion.en}
              </h2>

              {/* TEXT INPUT */}

              {currentQuestion.type === "text" && (
                <input
                  type="text"
                  value={currentValue}
                  onChange={(e) =>
                    handleChange(e.target.value)
                  }
                  placeholder={
                    language === "hi"
                      ? currentQuestion.placeholderHi
                      : currentQuestion.placeholderEn
                  }
                  className="mt-8 w-full rounded-xl border-2 border-gray-200 px-5 py-4 text-lg outline-none transition focus:border-green-600"
                />
              )}

              {/* YES / NO */}

              {currentQuestion.type === "choice" && (
                <div className="mt-8 grid gap-4 sm:grid-cols-2">

                  <button
                    type="button"
                    onClick={() =>
                      handleChange("yes")
                    }
                    className={`rounded-2xl border-2 p-6 text-left transition ${
                      currentValue === "yes"
                        ? "border-green-600 bg-green-50"
                        : "border-gray-200 hover:border-green-400 hover:bg-gray-50"
                    }`}
                  >
                    <div className="text-2xl">
                      ✓
                    </div>

                    <div className="mt-3 text-lg font-semibold">
                      {language === "hi"
                        ? "हाँ"
                        : "Yes"}
                    </div>
                  </button>

                  <button
                    type="button"
                    onClick={() =>
                      handleChange("no")
                    }
                    className={`rounded-2xl border-2 p-6 text-left transition ${
                      currentValue === "no"
                        ? "border-green-600 bg-green-50"
                        : "border-gray-200 hover:border-green-400 hover:bg-gray-50"
                    }`}
                  >
                    <div className="text-2xl">
                      ✕
                    </div>

                    <div className="mt-3 text-lg font-semibold">
                      {language === "hi"
                        ? "नहीं"
                        : "No"}
                    </div>
                  </button>

                </div>
              )}

              {/* PENSION INPUT */}

              {currentQuestion.key ===
                "retired_pensioner" &&
                form.retired_pensioner ===
                  "yes" && (
                  <div className="mt-6">

                    <label className="text-sm font-medium text-gray-700">
                      {language === "hi"
                        ? "आपकी मासिक पेंशन कितनी है?"
                        : "What is your monthly pension?"}
                    </label>

                    <input
                      type="number"
                      value={form.monthly_pension}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          monthly_pension:
                            e.target.value,
                        })
                      }
                      placeholder="₹ 10,000"
                      className="mt-2 w-full rounded-xl border-2 border-gray-200 px-5 py-4 text-lg outline-none focus:border-green-600"
                    />

                  </div>
                )}

            </div>

            {/* NAVIGATION */}

            <div className="mt-10 flex gap-3">

              {step > 0 && (
                <button
                  type="button"
                  onClick={previousStep}
                  className="rounded-xl border-2 border-gray-200 px-6 py-4 font-semibold transition hover:bg-gray-50"
                >
                  ←{" "}
                  {language === "hi"
                    ? "पीछे"
                    : "Back"}
                </button>
              )}

              {step < questions.length - 1 ? (
                <button
                  type="button"
                  onClick={nextStep}
                  disabled={!currentValue}
                  className="flex-1 rounded-xl bg-green-700 px-6 py-4 font-semibold text-white transition hover:bg-green-800 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {language === "hi"
                    ? "आगे बढ़ें"
                    : "Continue"}
                  {" →"}
                </button>
              ) : (
                <button
                  type="button"
                  onClick={checkEligibility}
                  disabled={!currentValue || loading}
                  className="flex-1 rounded-xl bg-green-700 px-6 py-4 font-semibold text-white transition hover:bg-green-800 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {loading
                    ? language === "hi"
                      ? "जाँच हो रही है..."
                      : "Checking..."
                    : language === "hi"
                      ? "पात्रता जाँचें"
                      : "Check eligibility"}
                </button>
              )}

            </div>

          </div>

        )}

        {/* ERROR */}

        {error && (
          <div className="mt-5 rounded-xl border border-red-200 bg-red-50 p-5 text-red-700">
            {error}
          </div>
        )}

      </section>

      {/* FOOTER */}

      <footer className="border-t bg-white">

        <div className="mx-auto max-w-6xl px-5 py-8">

          <p className="text-sm text-gray-500">
            {language === "hi"
              ? "यह सेवा सरकारी योजनाओं की जानकारी को आसान भाषा में समझने में सहायता करती है।"
              : "This service helps users understand government scheme information in simple language."}
          </p>

          <p className="mt-2 text-xs text-gray-400">
            Not an official government website.
          </p>

        </div>

      </footer>

    </main>
  );
}