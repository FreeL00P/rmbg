/* eslint-disable @next/next/no-img-element */
"use client";
import React, { useRef, useState, useEffect, useCallback } from "react";
import * as ort from "onnxruntime-web";

const MODEL_URL = "/u2net.onnx";

export default function RMBGPage() {
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [status, setStatus] = useState<{ text: string; type: "info" | "success" | "error" | "loading" }>({ text: "", type: "info" });
  const [file, setFile] = useState<File | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string>("");
  const [processedUrl, setProcessedUrl] = useState<string>("");
  const [showResult, setShowResult] = useState(false);
  const [sliderRatio, setSliderRatio] = useState(0.4);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const originalImgRef = useRef<HTMLImageElement>(null);
  const processedImgRef = useRef<HTMLImageElement>(null);

  function updateStatus(text: string, type: "info" | "success" | "error" | "loading" = "info") {
    setStatus({ text, type });
  }

  const loadModel = useCallback(async () => {
    try {
      updateStatus("AI模型加载中...", "loading");
      const s = await ort.InferenceSession.create(MODEL_URL);
      setSession(s);
      updateStatus("模型已就绪", "success");
    } catch (e) {
      updateStatus("模型加载失败，请检查模型文件。", "error");
      throw e;
    }
  }, []);

  function preprocess(image: HTMLImageElement, size = 320) {
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(image, 0, 0, size, size);
    const imageData = ctx.getImageData(0, 0, size, size);
    const data = new Float32Array(3 * size * size);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const i = y * size + x;
        data[i] = imageData.data[i * 4] / 255;
        data[i + size * size] = imageData.data[i * 4 + 1] / 255;
        data[i + 2 * size * size] = imageData.data[i * 4 + 2] / 255;
      }
    }
    return new ort.Tensor("float32", data, [1, 3, size, size]);
  }

  async function runInference(image: HTMLImageElement) {
    if (!session) throw new Error("模型未加载");
    const inputTensor = preprocess(image);
    const feeds: Record<string, ort.Tensor> = {};
    feeds[session.inputNames[0]] = inputTensor;
    const results = await session.run(feeds);
    return results[session.outputNames[0]].data as Float32Array;
  }

  function postprocess(maskData: Float32Array, width: number, height: number) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d")!;
    const imageData = ctx.createImageData(width, height);
    for (let i = 0; i < width * height; i++) {
      const x = Math.floor(((i % width) * 320) / width);
      const y = Math.floor((Math.floor(i / width) * 320) / height);
      const alpha = maskData[y * 320 + x] * 255;
      imageData.data[i * 4] = 0;
      imageData.data[i * 4 + 1] = 0;
      imageData.data[i * 4 + 2] = 0;
      imageData.data[i * 4 + 3] = alpha;
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas;
  }

  async function handleRemoveBg() {
    if (!file) {
      updateStatus("请上传图片", "error");
      return;
    }
    setIsLoading(true);
    updateStatus("图片分析中...", "loading");
    const img = new window.Image();
    img.onload = async () => {
      try {
        updateStatus("AI智能抠图中...", "loading");
        const maskData = await runInference(img);
        updateStatus("生成透明图...", "loading");
        const outCanvas = document.createElement("canvas");
        outCanvas.width = img.width;
        outCanvas.height = img.height;
        const ctx = outCanvas.getContext("2d")!;
        ctx.drawImage(img, 0, 0);
        const imgData = ctx.getImageData(0, 0, outCanvas.width, outCanvas.height);
        const maskCanvas = postprocess(maskData, outCanvas.width, outCanvas.height);
        const maskCtx = maskCanvas.getContext("2d")!;
        const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        for (let i = 0; i < imgData.data.length; i += 4) {
          imgData.data[i + 3] = maskImageData.data[i + 3];
        }
        ctx.putImageData(imgData, 0, 0);
        setOriginalUrl(img.src);
        setProcessedUrl(outCanvas.toDataURL("image/png"));
        setShowResult(true);
        setSliderRatio(0.4);
        updateStatus("完成！可滑动对比，点击下载透明图", "success");
      } catch (e) {
        updateStatus("AI处理失败，请重试或更换图片。}"+e, "error");
      } finally {
        setIsLoading(false);
      }
    };
    img.onerror = () => {
      updateStatus("图片加载失败，请重试。", "error");
      setIsLoading(false);
    };
    img.src = URL.createObjectURL(file);
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length) {
      setFile(e.dataTransfer.files[0]);
      updateStatus("图片已选择，点击去背景", "info");
    }
  }

  function handleSliderDown(e: React.MouseEvent | React.TouchEvent) {
    setIsDragging(true);
    e.preventDefault();
  }

  const handleSliderMove = useCallback((e: MouseEvent | TouchEvent) => {
    if (!isDragging || !wrapperRef.current) return;
    const rect = wrapperRef.current.getBoundingClientRect();
    let clientX = (e as MouseEvent).clientX;
    if ((e as TouchEvent).touches) {
      clientX = (e as TouchEvent).touches[0].clientX;
    }
    let x = clientX - rect.left;
    if (x < 0) x = 0;
    if (x > rect.width) x = rect.width;
    setSliderRatio(x / rect.width);
  }, [isDragging]);

  function handleSliderUp() {
    setIsDragging(false);
  }

  useEffect(() => {
    if (isDragging) {
      window.addEventListener("mousemove", handleSliderMove);
      window.addEventListener("touchmove", handleSliderMove, { passive: false });
      window.addEventListener("mouseup", handleSliderUp);
      window.addEventListener("touchend", handleSliderUp);
      window.addEventListener("touchcancel", handleSliderUp);
    } else {
      window.removeEventListener("mousemove", handleSliderMove);
      window.removeEventListener("touchmove", handleSliderMove);
      window.removeEventListener("mouseup", handleSliderUp);
      window.removeEventListener("touchend", handleSliderUp);
      window.removeEventListener("touchcancel", handleSliderUp);
    }
    return () => {
      window.removeEventListener("mousemove", handleSliderMove);
      window.removeEventListener("touchmove", handleSliderMove);
      window.removeEventListener("mouseup", handleSliderUp);
      window.removeEventListener("touchend", handleSliderUp);
      window.removeEventListener("touchcancel", handleSliderUp);
    };
  }, [isDragging, handleSliderMove]);

  useEffect(() => {
    function adjustHeight() {
      if (!wrapperRef.current || !originalImgRef.current) return;
      const width = wrapperRef.current.offsetWidth;
      const ratio = originalImgRef.current.naturalHeight / originalImgRef.current.naturalWidth;
      wrapperRef.current.style.height = width * ratio + "px";
    }
    window.addEventListener("resize", adjustHeight);
    setTimeout(adjustHeight, 100);
    return () => window.removeEventListener("resize", adjustHeight);
  }, [originalUrl, showResult]);

  useEffect(() => {
    if (!session) loadModel();
  }, [session, loadModel]);

  function reset() {
    setFile(null);
    setOriginalUrl("");
    setProcessedUrl("");
    setShowResult(false);
    setSliderRatio(0.4);
    updateStatus("", "info");
    if (inputRef.current) inputRef.current.value = "";
  }

  function handleDownload() {
    if (!processedUrl) return;
    const a = document.createElement("a");
    a.href = processedUrl;
    a.download = "removed-bg.png";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  const iconMap = {
    info: "🟦",
    success: "✅",
    error: "❌",
    loading: "⏳"
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#f8fafc] to-[#e0e7ef] flex items-center justify-center px-2">
      <div className="w-full max-w-4xl flex flex-col gap-8 items-center justify-center py-10">
        {/* 顶部操作区 */}
        <div className="w-full max-w-md bg-white/90 rounded-3xl shadow-2xl p-8 flex flex-col gap-8 border border-slate-100 mb-4">
          <div className="flex flex-col items-center gap-2">
            <h2 className="text-2xl font-bold text-slate-800 tracking-wide mb-1">AI去背景工具</h2>
            <p className="text-slate-500 text-sm text-center">上传图片，AI自动抠图，滑动对比，下载透明PNG</p>
          </div>
          <div
            className="flex flex-col gap-4 items-center"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            <label
              htmlFor="inputImage"
              className="w-full flex flex-col items-center justify-center gap-2 border-2 border-dashed border-sky-300 rounded-2xl py-6 px-4 bg-sky-50 hover:bg-sky-100 cursor-pointer transition text-sky-700 font-semibold text-base text-center"
            >
              <svg width="36" height="36" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 22v-8m0 0l-4 4m4-4l4 4"/><rect x="8" y="24" width="20" height="6" rx="3"/></svg>
              <span>{file ? file.name : "点击或拖拽图片到此"}</span>
            </label>
            <input
              type="file"
              id="inputImage"
              accept="image/*"
              className="hidden"
              ref={inputRef}
              onChange={e => {
                if (e.target.files && e.target.files.length) {
                  setFile(e.target.files[0]);
                  updateStatus("图片已选择，点击去背景", "info");
                } else {
                  setFile(null);
                }
              }}
            />
            <button
              type="button"
              className="w-full py-3 rounded-xl font-bold text-lg bg-gradient-to-r from-indigo-600 to-sky-400 text-white shadow-lg transition hover:scale-105 active:scale-95 disabled:bg-slate-300 disabled:text-slate-400 disabled:cursor-not-allowed"
              disabled={!file || isLoading}
              onClick={async () => {
                if (!session) await loadModel();
                await handleRemoveBg();
              }}
            >
              {isLoading ? (
                <span className="flex items-center justify-center gap-2"><span className="animate-spin inline-block w-5 h-5 border-2 border-white border-t-transparent rounded-full"></span>处理中...</span>
              ) : (
                "去背景"
              )}
            </button>
            {(file || showResult) && (
              <button
                type="button"
                className="w-full py-3 rounded-xl font-bold text-base bg-gradient-to-r from-slate-200 to-slate-100 text-slate-500 shadow transition hover:bg-slate-200 hover:text-slate-700"
                onClick={reset}
                title="重置"
              >
                重置
              </button>
            )}
          </div>
          <div className="min-h-[1.5em] text-center flex items-center justify-center gap-2 text-base font-semibold" style={{ color: status.type === "error" ? "#dc2626" : status.type === "success" ? "#2563eb" : undefined }}>
            <span className="text-lg">{iconMap[status.type]}</span> {status.text}
          </div>
        </div>
        {/* 对比区 */}
        <section className="w-full max-w-2xl flex flex-col items-center justify-center relative overflow-visible py-8 px-2 bg-transparent">
          {showResult ? (
            <div className="relative w-full flex flex-col items-center justify-center">
              <div
                ref={wrapperRef}
                className="relative flex items-center justify-center mx-auto bg-white/70 rounded-2xl overflow-hidden"
                style={{ width: '100%', height: 'min(60vw, 480px)', maxWidth: 800, maxHeight: 480 }}
              >
                {/* 对比模式：原图左侧，去背景右侧 */}
                <img
                  ref={originalImgRef}
                  src={originalUrl}
                  alt="原图"
                  className="absolute top-0 left-0 w-full h-full object-contain select-none rounded-2xl"
                  style={{
                    zIndex: 1,
                    clipPath: showResult
                      ? `inset(0 ${(1 - sliderRatio) * 100}% 0 0)`
                      : undefined
                  }}
                  draggable={false}
                />
                <img
                  ref={processedImgRef}
                  src={processedUrl}
                  alt="去背景"
                  className="absolute top-0 left-0 w-full h-full object-contain select-none rounded-2xl pointer-events-none"
                  style={{
                    zIndex: 2,
                    clipPath: showResult
                      ? `inset(0 0 0 ${sliderRatio * 100}%)`
                      : undefined
                  }}
                  draggable={false}
                />
                {/* 滑块线和按钮 */}
                <div
                  className="absolute top-0 bottom-0 flex flex-col items-center z-30 group"
                  style={{ left: `calc(${sliderRatio * 100}% - 16px)`, width: 32 }}
                  onMouseDown={handleSliderDown}
                  onTouchStart={handleSliderDown}
                >
                  {/* 滑块线 */}
                  <div className="w-0.5 h-full bg-gradient-to-b from-slate-300 to-slate-400 group-hover:from-sky-400 group-hover:to-indigo-400 transition" />
                  {/* 滑块按钮 */}
                  <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white/90 backdrop-blur rounded-full border-2 border-slate-300 group-hover:border-sky-400 shadow-md w-7 h-7 flex items-center justify-center text-lg font-extrabold text-slate-500 group-hover:text-sky-500 select-none pointer-events-none transition group-hover:scale-110 group-hover:shadow-sky-200">
                    <svg width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="8" stroke="currentColor" strokeWidth="2"/></svg>
                  </div>
                </div>
                {/* 下载按钮美化 */}
                <button
                  type="button"
                  className="fixed md:absolute bottom-8 right-8 md:bottom-6 md:right-6 w-12 h-12 rounded-full bg-gradient-to-br from-sky-400 to-indigo-500 text-white flex items-center justify-center shadow-md hover:scale-110 hover:shadow-sky-200 transition z-40 border-none outline-none"
                  onClick={handleDownload}
                  title="下载透明PNG"
                >
                  <svg width="22" height="22" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="9" stroke="white" strokeWidth="2"/><path d="M11 6v7m0 0l-3-3m3 3l3-3" stroke="white" strokeWidth="2"/></svg>
                </button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center w-full h-64 text-slate-400 text-lg font-semibold gap-2">
              <svg width="60" height="60" fill="none" stroke="currentColor" strokeWidth="2" className="mb-2"><circle cx="30" cy="30" r="28" strokeDasharray="4 4"/><path d="M30 18v16m0 0l-6-6m6 6l6-6"/></svg>
              <span>请上传图片，AI将自动抠图</span>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
