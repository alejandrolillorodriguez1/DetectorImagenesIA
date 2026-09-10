import { useEffect, useRef, useState } from "react";
import "./App.css";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Cpu,
  Image as ImageIcon,
  LockKeyhole,
  RefreshCw,
  ScanSearch,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  X,
} from "lucide-react";

const API_URL = "http://127.0.0.1:8000";
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_FILE_SIZE = 10 * 1024 * 1024;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Home() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const selectFile = (nextFile) => {
    if (!nextFile) return;

    if (!ACCEPTED_TYPES.includes(nextFile.type)) {
      setError("Formato no compatible. Utiliza una imagen JPG, PNG o WEBP.");
      return;
    }

    if (nextFile.size > MAX_FILE_SIZE) {
      setError("La imagen supera el límite de 10 MB.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
    setResult(null);
    setError(null);
  };

  const handleInput = (event) => {
    selectFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  };

  const clearImage = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const analyzeImage = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail ?? "No se ha podido analizar la imagen.");
      }

      setResult({
        ...data,
        prediction: String(data.prediction).toUpperCase(),
        confidence: Number(data.confidence),
      });
    } catch (requestError) {
      setError(
        requestError instanceof TypeError
          ? "No se puede conectar con el detector. Comprueba que FastAPI está ejecutándose."
          : requestError instanceof Error
            ? requestError.message
            : "Ha ocurrido un error inesperado.",
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const confidence = result ? Math.min(100, Math.max(0, result.confidence)) : 0;
  const fakeProbability = result
    ? (result.probabilities?.FAKE ??
      (result.prediction === "FAKE" ? confidence : 100 - confidence))
    : 0;
  const realProbability = result
    ? (result.probabilities?.REAL ??
      (result.prediction === "REAL" ? confidence : 100 - confidence))
    : 0;
  const resultIsFake = result?.prediction === "FAKE";

  return (
    <main className="site-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <header className="topbar">
        <a className="brand" href="#top" aria-label="Authentica AI, inicio">
          <span className="brand-mark"><ScanSearch size={21} /></span>
          <span>Authentica<span className="brand-accent">AI</span></span>
        </a>

        <div className="system-status">
          <span className="status-dot" />
          Modelo operativo
        </div>
      </header>

      <section className="hero" id="top">
        <div className="eyebrow"><Sparkles size={15} /> Análisis visual mediante IA</div>
        <h1>Descubre qué hay detrás<br />de cada <span>imagen.</span></h1>
        <p>
          Sube una imagen y nuestro modelo convolucional analizará sus patrones
          para estimar si es real o ha sido generada artificialmente.
        </p>

        <div className="hero-facts" aria-label="Características del detector">
          <div><ShieldCheck size={17} /><span><strong>91,55 %</strong> precisión en test</span></div>
          <div><Cpu size={17} /><span><strong>CNN</strong> entrenada desde cero</span></div>
          <div><LockKeyhole size={17} /><span><strong>Privado</strong> y sin almacenar</span></div>
        </div>
      </section>

      <section className="analyzer-card" aria-label="Analizador de imágenes">
        <div className="workspace-head">
          <div>
            <span className="step-label">Detector visual</span>
            <h2>Analiza una imagen</h2>
          </div>
          <span className="file-hint">JPG · PNG · WEBP · Máx. 10 MB</span>
        </div>

        <div className="analyzer-grid">
          <div className="upload-column">
            <input
              ref={inputRef}
              className="sr-only"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleInput}
            />

            {!preview ? (
              <div
                className={`dropzone ${isDragging ? "is-dragging" : ""}`}
                onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
              >
                <div className="upload-icon"><UploadCloud size={31} /></div>
                <h3>Arrastra tu imagen aquí</h3>
                <p>o selecciónala directamente desde tu equipo</p>
                <button className="secondary-button" type="button" onClick={() => inputRef.current?.click()}>
                  Seleccionar imagen <ArrowRight size={17} />
                </button>
              </div>
            ) : (
              <div className="preview-panel">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={preview} alt="Vista previa de la imagen seleccionada" />
                {isAnalyzing && (
                  <div className="scan-overlay" aria-label="Analizando imagen">
                    <span className="scan-line" />
                    <div><RefreshCw size={22} className="spin" /> Analizando patrones…</div>
                  </div>
                )}
                <button className="clear-button" type="button" onClick={clearImage} aria-label="Eliminar imagen">
                  <X size={18} />
                </button>
                <div className="file-chip">
                  <ImageIcon size={16} />
                  <span>{file?.name}</span>
                  <small>{file ? formatBytes(file.size) : ""}</small>
                </div>
              </div>
            )}

            <button
              className="analyze-button"
              type="button"
              disabled={!file || isAnalyzing}
              onClick={analyzeImage}
            >
              {isAnalyzing ? (
                <><RefreshCw size={19} className="spin" /> Analizando imagen</>
              ) : (
                <><ScanSearch size={19} /> Iniciar análisis</>
              )}
            </button>
          </div>

          <div className="result-column">
            {!result && !error && (
              <div className="empty-result">
                <div className="radar-icon"><ScanSearch size={35} /></div>
                <span>Resultado del análisis</span>
                <h3>Todo listo para empezar</h3>
                <p>Selecciona una imagen y nuestro modelo mostrará aquí su clasificación y nivel de confianza.</p>
                <div className="result-skeleton"><i /><i /><i /></div>
              </div>
            )}

            {error && (
              <div className="error-result" role="alert">
                <span><AlertTriangle size={24} /></span>
                <div>
                  <h3>No se pudo completar el análisis</h3>
                  <p>{error}</p>
                  <button type="button" onClick={analyzeImage} disabled={!file}>Intentar de nuevo</button>
                </div>
              </div>
            )}

            {result && (
              <div className={`result-card ${resultIsFake ? "fake-result" : "real-result"}`}>
                <div className="result-topline">
                  <span>Análisis completado</span>
                  <CheckCircle2 size={19} />
                </div>

                <div className="verdict">
                  <div className="verdict-icon">
                    {resultIsFake ? <Sparkles size={28} /> : <ShieldCheck size={28} />}
                  </div>
                  <div>
                    <span>Clasificación</span>
                    <h3>{resultIsFake ? "Generada por IA" : "Imagen real"}</h3>
                  </div>
                </div>

                <div className="confidence-block">
                  <div className="confidence-copy">
                    <span>Confianza del modelo</span>
                    <strong>{confidence.toFixed(2)}%</strong>
                  </div>
                  <div className="confidence-track"><span style={{ width: `${confidence}%` }} /></div>
                </div>

                <div className="probability-list">
                  <div>
                    <span><i className="fake-dot" /> IA generada</span>
                    <strong>{fakeProbability.toFixed(2)}%</strong>
                    <div><i className="fake-bar" style={{ width: `${fakeProbability}%` }} /></div>
                  </div>
                  <div>
                    <span><i className="real-dot" /> Imagen real</span>
                    <strong>{realProbability.toFixed(2)}%</strong>
                    <div><i className="real-bar" style={{ width: `${realProbability}%` }} /></div>
                  </div>
                </div>

                <p className="result-note">
                  Este resultado es una estimación del modelo y no constituye una verificación absoluta.
                </p>
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="how-it-works">
        <div className="section-heading">
          <span>Proceso transparente</span>
          <h2>De la imagen al resultado</h2>
        </div>
        <div className="process-grid">
          <article><b>01</b><UploadCloud size={23} /><h3>Carga segura</h3><p>La imagen se envía al detector únicamente para realizar el análisis.</p></article>
          <article><b>02</b><Cpu size={23} /><h3>Análisis neuronal</h3><p>La CNN examina patrones visuales aprendidos durante el entrenamiento.</p></article>
          <article><b>03</b><ScanSearch size={23} /><h3>Resultado claro</h3><p>Recibes la clasificación junto con la confianza estimada del modelo.</p></article>
        </div>
      </section>

      <footer>
        <a className="brand footer-brand" href="#top"><span className="brand-mark"><ScanSearch size={18} /></span>Authentica<span className="brand-accent">AI</span></a>
        <p>Proyecto de detección de imágenes generadas mediante inteligencia artificial.</p>
        <span>Modelo CNN · PyTorch · FastAPI</span>
      </footer>
    </main>
  );
}
