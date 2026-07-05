import { FormEvent, useMemo, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

type ApiResult = {
  title: string;
  body: string;
  downloadUrl?: string;
  raw?: unknown;
  kind: 'success' | 'error' | 'info';
};

function formatJson(value: unknown) {
  return JSON.stringify(value, null, 2);
}

export default function App() {
  const [encodeImage, setEncodeImage] = useState<File | null>(null);
  const [encodeMessage, setEncodeMessage] = useState('Hello Jyoti');
  const [decodeImage, setDecodeImage] = useState<File | null>(null);
  const [totalBits, setTotalBits] = useState('384');
  const [loading, setLoading] = useState<'encode' | 'decode' | null>(null);
  const [result, setResult] = useState<ApiResult>({
    title: 'Ready',
    body: 'Upload an image and test encode or decode from the same screen.',
    kind: 'info',
  });

  const encodePreview = useMemo(() => encodeImage?.name ?? 'No file selected', [encodeImage]);
  const decodePreview = useMemo(() => decodeImage?.name ?? 'No file selected', [decodeImage]);

  async function handleEncode(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!encodeImage) {
      setResult({ title: 'Encode failed', body: 'Select an image first.', kind: 'error' });
      return;
    }

    const formData = new FormData();
    formData.append('image', encodeImage);
    formData.append('secret_message', encodeMessage.trim());

    setLoading('encode');
    setResult({ title: 'Encoding...', body: 'Sending file to backend.', kind: 'info' });

    try {
      const response = await fetch(`${API_BASE}/encode`, {
        method: 'POST',
        body: formData,
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload?.error ?? 'Encoding failed');
      }

      setResult({
        title: 'Encode complete',
        body: 'Watermarked image generated successfully.',
        downloadUrl: payload.download ? `${API_BASE}${payload.download}` : undefined,
        raw: payload,
        kind: 'success',
      });
    } catch (error) {
      setResult({
        title: 'Encode error',
        body: error instanceof Error ? error.message : 'Unexpected error',
        kind: 'error',
      });
    } finally {
      setLoading(null);
    }
  }

  async function handleDecode(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!decodeImage) {
      setResult({ title: 'Decode failed', body: 'Select an encoded image first.', kind: 'error' });
      return;
    }

    const bitsValue = Number(totalBits);
    if (!Number.isInteger(bitsValue) || bitsValue <= 0) {
      setResult({ title: 'Decode failed', body: 'Enter a valid total bits value.', kind: 'error' });
      return;
    }

    const formData = new FormData();
    formData.append('image', decodeImage);
    formData.append('total_bits', String(bitsValue));

    setLoading('decode');
    setResult({ title: 'Decoding...', body: 'Reading hidden data from the image.', kind: 'info' });

    try {
      const response = await fetch(`${API_BASE}/decode`, {
        method: 'POST',
        body: formData,
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload?.error ?? 'Decoding failed');
      }

      setResult({
        title: 'Decode complete',
        body: payload.secret_message ? `Recovered message: ${payload.secret_message}` : 'Decoded successfully.',
        raw: payload,
        kind: 'success',
      });
    } catch (error) {
      setResult({
        title: 'Decode error',
        body: error instanceof Error ? error.message : 'Unexpected error',
        kind: 'error',
      });
    } finally {
      setLoading(null);
    }
  }

  return (
    <div className="shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <main className="app">
        <section className="hero card">
          <div className="eyebrow">Robust Digital Watermarking</div>
          <h1>Encode and decode hidden messages with a clean, focused studio UI.</h1>
          <p>
            A lightweight frontend for your FastAPI backend. Upload an image, hide a secret,
            then decode it back from the same interface.
          </p>
          <div className="meta-row">
            <span>API: {API_BASE}</span>
            <span>Format: PNG / JPG / JPEG</span>
            <span>Modes: Encode · Decode</span>
          </div>
        </section>

        <section className="grid">
          <form className="card form-card" onSubmit={handleEncode}>
            <div className="section-head">
              <h2>Encode</h2>
              <span>Build a watermarked image</span>
            </div>

            <label className="field">
              <span>Image</span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={(event) => setEncodeImage(event.target.files?.[0] ?? null)}
              />
              <small>{encodePreview}</small>
            </label>

            <label className="field">
              <span>Secret message</span>
              <textarea
                rows={5}
                value={encodeMessage}
                onChange={(event) => setEncodeMessage(event.target.value)}
                placeholder="Write the hidden text here..."
              />
            </label>

            <button type="submit" disabled={loading !== null}>
              {loading === 'encode' ? 'Encoding...' : 'Encode image'}
            </button>
          </form>

          <form className="card form-card" onSubmit={handleDecode}>
            <div className="section-head">
              <h2>Decode</h2>
              <span>Recover the hidden message</span>
            </div>

            <label className="field">
              <span>Encoded image</span>
              <input
                type="file"
                accept="image/png,image/jpeg,image/jpg"
                onChange={(event) => setDecodeImage(event.target.files?.[0] ?? null)}
              />
              <small>{decodePreview}</small>
            </label>

            <label className="field">
              <span>Total bits</span>
              <input
                type="number"
                min={8}
                step={8}
                value={totalBits}
                onChange={(event) => setTotalBits(event.target.value)}
              />
            </label>

            <button type="submit" disabled={loading !== null}>
              {loading === 'decode' ? 'Decoding...' : 'Decode image'}
            </button>
          </form>
        </section>

        <section className={`card result-card ${result.kind}`}>
          <div className="section-head">
            <h2>{result.title}</h2>
            <span>{result.kind === 'success' ? 'Operation finished' : 'Live output'}</span>
          </div>

          <p>{result.body}</p>

          {result.downloadUrl ? (
            <a className="download-link" href={result.downloadUrl} target="_blank" rel="noreferrer">
              Download encoded image
            </a>
          ) : null}

          {result.raw ? <pre>{formatJson(result.raw)}</pre> : null}
        </section>
      </main>
    </div>
  );
}
