/* eslint-disable @next/next/no-img-element */
"use client";
import React, { useRef, useState, useEffect, useCallback } from "react";
import * as ort from "onnxruntime-web";
import { removeBackground as imglyRemoveBackground } from "@imgly/background-removal";

type NormalizeType = "simple" | "imagenet" | "center";

interface ModelConfig {
  id: string;
  name: string;
  description: string;
  badge: string;
  badgeClass: string;
  quality: number;
  speed: number;
  size: string;
  type: "onnx" | "imgly";
  modelPath?: string;
  inputSize?: number;
  normalize?: NormalizeType;
  available: boolean;
  downloadHint?: string;
}

const MODELS: ModelConfig[] = [
  {
    id: "u2net",
    name: "U2-Net",
    description: "轻量快速，通用场景",
    badge: "内置",
    badgeClass: "bg-emerald-100 text-emerald-700",
    quality: 3,
    speed: 5,
    size: "4.5 MB",
    type: "onnx",
    modelPath: "/u2net.onnx",
    inputSize: 320,
    normalize: "simple",
    available: true,
  },
  {
    id: "imgly",
    name: "IMG.LY",
    description: "质量优秀，自动下载模型",
    badge: "推荐",
    badgeClass: "bg-indigo-100 text-indigo-700",
    quality: 4,
    speed: 3,
    size: "~40 MB",
    type: "imgly",
    available: true,
  },
  {
    id: "bria",
    name: "BRIA RMBG-2.0",
    description: "业界领先，精细抠图",
    badge: "高质量",
    badgeClass: "bg-amber-100 text-amber-700",
    quality: 5,
    speed: 2,
    size: "~976 MB",
    type: "onnx",
    modelPath: "/bria_rmbg_2.0.onnx",
    inputSize: 1024,
    normalize: "imagenet",
    available: false,
    downloadHint: "将 bria-rmbg-2.0.onnx 放入 public/ 目录（模型较大）",
  },
  {
    id: "isnet",
    name: "ISNet",
    description: "迭代收缩网络，质量均衡",
    badge: "平衡",
    badgeClass: "bg-sky-100 text-sky-700",
    quality: 4,
    speed: 2,
    size: "170 MB",
    type: "onnx",
    modelPath: "/isnet.onnx",
    inputSize: 1024,
    normalize: "center",
    available: true,
  },
  {
    id: "modnet",
    name: "MODNet",
    description: "人像专用，轻量高效",
    badge: "人像",
    badgeClass: "bg-rose-100 text-rose-700",
    quality: 3,
    speed: 4,
    size: "~25 MB",
    type: "onnx",
    modelPath: "/modnet.onnx",
    inputSize: 512,
    normalize: "simple",
    available: false,
    downloadHint: "将 modnet.onnx 放入 public/ 目录",
  },
];

function Dots({ count, max = 5 }: { count: number; max?: number }) {
  return (
    <span className="inline-flex gap-0.5">
      {Array.from({ length: max }).map((_, i) => (
        <span
          key={i}
          className={`w-1.5 h-1.5 rounded-full ${i < count ? "bg-indigo-400" : "bg-slate-200"}`}
        />
      ))}
    </span>
  );
}

export default function RMBGPage() {
  const [selectedModelId, setSelectedModelId] = useState("u2net");
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [sessions, setSessions] = useState<Record<string, ort.InferenceSession>>({});
  const [modelAvailability, setModelAvailability] = useState<Record<string, boolean>>({});
  const [status, setStatus] = useState<{ text: string; type: "info" | "success" | "error" | "loading" }>({
    text: "",
    type: "info",
  });
  const [file, setFile] = useState<File | null>(null);
  const [originalUrl, setOriginalUrl] = useState("");
  const [processedUrl, setProcessedUrl] = useState("");
  const [showResult, setShowResult] = useState(false);
  const [sliderRatio, setSliderRatio] = useState(0.5);
  const [isDragging, setIsDragging] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const originalImgRef = useRef<HTMLImageElement>(null);
  const processedImgRef = useRef<HTMLImageElement>(null);
  const pickerRef = useRef<HTMLDivElement>(null);

  const selectedModel = MODELS.find((m) => m.id === selectedModelId)!;

  function updateStatus(text: string, type: "info" | "success" | "error" | "loading" = "info") {
    setStatus({ text, type });
  }

  useEffect(() => {
    async function checkAvailability() {
      const results: Record<string, boolean> = {};
      for (const m of MODELS) {
        if (m.type === "imgly") {
          results[m.id] = true;
          continue;
        }
        if (m.available) {
          results[m.id] = true;
          continue;
        }
        try {
          const res = await fetch(m.modelPath!, { method: "HEAD" });
          results[m.id] = res.ok;
        } catch {
          results[m.id] = false;
        }
      }
      setModelAvailability(results);
    }
    checkAvailability();
  }, []);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
        setShowModelPicker(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const loadOnnxSession = useCallback(
    async (model: ModelConfig) => {
      if (sessions[model.id]) return sessions[model.id];
      try {
        updateStatus(`正在加载 ${model.name} 模型...`, "loading");
        const s = await ort.InferenceSession.create(model.modelPath!);
        setSessions((prev) => ({ ...prev, [model.id]: s }));
        updateStatus(`${model.name} 模型已就绪`, "success");
        return s;
      } catch {
        updateStatus(`${model.name} 模型加载失败，请检查模型文件`, "error");
        return null;
      }
    },
    [sessions]
  );

  useEffect(() => {
    if (selectedModel.type === "onnx" && selectedModel.available) {
      loadOnnxSession(selectedModel);
    } else if (selectedModel.type === "imgly") {
      updateStatus("IMG.LY 模型首次使用时会自动下载", "info");
    } else {
      updateStatus(`${selectedModel.name} 已就绪`, "info");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModelId]);

  function preprocessOnnx(image: HTMLImageElement, model: ModelConfig) {
    const size = model.inputSize || 320;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(image, 0, 0, size, size);
    const imageData = ctx.getImageData(0, 0, size, size);
    const data = new Float32Array(3 * size * size);

    if (model.normalize === "imagenet") {
      const mean = [0.485, 0.456, 0.406];
      const std = [0.229, 0.224, 0.225];
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const i = y * size + x;
          data[i] = (imageData.data[i * 4] / 255 - mean[0]) / std[0];
          data[i + size * size] = (imageData.data[i * 4 + 1] / 255 - mean[1]) / std[1];
          data[i + 2 * size * size] = (imageData.data[i * 4 + 2] / 255 - mean[2]) / std[2];
        }
      }
    } else if (model.normalize === "center") {
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const i = y * size + x;
          data[i] = imageData.data[i * 4] / 255 - 0.5;
          data[i + size * size] = imageData.data[i * 4 + 1] / 255 - 0.5;
          data[i + 2 * size * size] = imageData.data[i * 4 + 2] / 255 - 0.5;
        }
      }
    } else {
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const i = y * size + x;
          data[i] = imageData.data[i * 4] / 255;
          data[i + size * size] = imageData.data[i * 4 + 1] / 255;
          data[i + 2 * size * size] = imageData.data[i * 4 + 2] / 255;
        }
      }
    }
    return new ort.Tensor("float32", data, [1, 3, size, size]);
  }

  async function runOnnxInference(image: HTMLImageElement, model: ModelConfig) {
    const session = sessions[model.id] || (await loadOnnxSession(model));
    if (!session) throw new Error("模型未加载");
    const inputTensor = preprocessOnnx(image, model);
    const feeds: Record<string, ort.Tensor> = {};
    feeds[session.inputNames[0]] = inputTensor;
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];
    return {
      maskData: output.data as Float32Array,
      size: model.inputSize || 320,
    };
  }

  function postprocessMask(maskData: Float32Array, maskSize: number, width: number, height: number) {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < maskData.length; i++) {
      if (maskData[i] < min) min = maskData[i];
      if (maskData[i] > max) max = maskData[i];
    }
    const range = max - min || 1;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d")!;
    const imageData = ctx.createImageData(width, height);
    for (let i = 0; i < width * height; i++) {
      const x = Math.floor(((i % width) * maskSize) / width);
      const y = Math.floor((Math.floor(i / width) * maskSize) / height);
      const alpha = Math.max(0, Math.min(255, ((maskData[y * maskSize + x] - min) / range) * 255));
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

    if (selectedModel.type === "imgly") {
      try {
        updateStatus("IMG.LY 处理中（首次使用需下载模型）...", "loading");
        const blob = await imglyRemoveBackground(file, {
          output: { format: "image/png" },
        });
        const resultUrl = URL.createObjectURL(blob);
        const img = new window.Image();
        img.onload = async () => {
          setOriginalUrl(URL.createObjectURL(file!));
          setProcessedUrl(resultUrl);
          setShowResult(true);
          setSliderRatio(0.5);
          updateStatus("完成！滑动对比，点击下载透明图", "success");
          setIsLoading(false);
        };
        img.onerror = () => {
          updateStatus("结果图片加载失败", "error");
          setIsLoading(false);
        };
        img.src = resultUrl;
      } catch {
        updateStatus("IMG.LY 处理失败，请重试", "error");
        setIsLoading(false);
      }
      return;
    }

    // ONNX path
    const img = new window.Image();
    img.onload = async () => {
      try {
        updateStatus("AI智能抠图中...", "loading");
        const { maskData, size: maskSize } = await runOnnxInference(img, selectedModel);
        updateStatus("生成透明图...", "loading");
        const outCanvas = document.createElement("canvas");
        outCanvas.width = img.width;
        outCanvas.height = img.height;
        const ctx = outCanvas.getContext("2d")!;
        ctx.drawImage(img, 0, 0);
        const imgData = ctx.getImageData(0, 0, outCanvas.width, outCanvas.height);
        const maskCanvas = postprocessMask(maskData, maskSize, outCanvas.width, outCanvas.height);
        const maskCtx = maskCanvas.getContext("2d")!;
        const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        for (let i = 0; i < imgData.data.length; i += 4) {
          imgData.data[i + 3] = maskImageData.data[i + 3];
        }
        ctx.putImageData(imgData, 0, 0);
        setOriginalUrl(img.src);
        setProcessedUrl(outCanvas.toDataURL("image/png"));
        setShowResult(true);
        setSliderRatio(0.5);
        updateStatus("完成！滑动对比，点击下载透明图", "success");
      } catch (e) {
        updateStatus("AI处理失败，请重试或更换图片。" + e, "error");
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
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length) {
      setFile(e.dataTransfer.files[0]);
      updateStatus("图片已选择，点击去背景", "info");
    }
  }

  function handleSliderDown(e: React.MouseEvent | React.TouchEvent) {
    setIsDragging(true);
    e.preventDefault();
  }

  const handleSliderMove = useCallback(
    (e: MouseEvent | TouchEvent) => {
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
    },
    [isDragging]
  );

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

  function reset() {
    setFile(null);
    setOriginalUrl("");
    setProcessedUrl("");
    setShowResult(false);
    setSliderRatio(0.5);
    updateStatus("", "info");
    if (inputRef.current) inputRef.current.value = "";
  }

  function handleDownload() {
    if (!processedUrl) return;
    const a = document.createElement("a");
    a.href = processedUrl;
    a.download = `removed-bg-${selectedModel.id}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  const statusDotMap = {
    info: "bg-slate-400",
    success: "bg-emerald-500",
    error: "bg-rose-500",
    loading: "bg-indigo-500 animate-pulse",
  };

  const isAvailable = (m: ModelConfig) => modelAvailability[m.id] ?? m.available;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50/40 to-sky-50/60 flex flex-col">
      {/* Header */}
      <header className="w-full py-6 px-4">
        <div className="max-w-5xl mx-auto flex items-center justify-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-sky-400 flex items-center justify-center shadow-lg shadow-indigo-200/50">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <path d="M21 15l-5-5L5 21"/>
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-600 to-sky-500 bg-clip-text text-transparent">
              AI 去背景
            </h1>
            <p className="text-xs text-slate-400 font-medium -mt-0.5">智能抠图工具</p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-start justify-center px-4 pb-12">
        <div className="w-full max-w-5xl flex flex-col lg:flex-row gap-8 items-start justify-center">
          {/* Left Panel */}
          <div className="w-full lg:w-96 flex-shrink-0 animate-fade-in-up">
            <div className="bg-white/80 backdrop-blur-xl rounded-2xl shadow-xl shadow-slate-200/50 border border-white/60 p-7 flex flex-col gap-6">
              {/* Model Selector */}
              <div className="relative" ref={pickerRef}>
                <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 block">
                  选择模型
                </label>
                <button
                  type="button"
                  className="w-full flex items-center justify-between gap-3 px-4 py-3 rounded-xl bg-slate-50 border border-slate-200 hover:border-indigo-300 transition-colors text-left"
                  onClick={() => setShowModelPicker(!showModelPicker)}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${selectedModel.badgeClass}`}>
                      {selectedModel.badge}
                    </span>
                    <div className="min-w-0">
                      <p className="text-sm font-semibold text-slate-700 truncate">{selectedModel.name}</p>
                      <p className="text-xs text-slate-400 truncate">{selectedModel.description}</p>
                    </div>
                  </div>
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 16 16"
                    fill="none"
                    stroke="#94a3b8"
                    strokeWidth="2"
                    className={`flex-shrink-0 transition-transform ${showModelPicker ? "rotate-180" : ""}`}
                  >
                    <path d="M4 6l4 4 4-4"/>
                  </svg>
                </button>

                {/* Dropdown */}
                {showModelPicker && (
                  <div className="absolute z-50 top-full left-0 right-0 mt-2 bg-white rounded-xl shadow-2xl shadow-slate-300/40 border border-slate-100 overflow-hidden animate-fade-in-up">
                    {MODELS.map((m) => {
                      const avail = isAvailable(m);
                      const isSelected = m.id === selectedModelId;
                      return (
                        <button
                          key={m.id}
                          type="button"
                          disabled={!avail}
                          className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-colors ${
                            isSelected
                              ? "bg-indigo-50"
                              : avail
                              ? "hover:bg-slate-50"
                              : "opacity-50 cursor-not-allowed"
                          }`}
                          onClick={() => {
                            if (avail) {
                              setSelectedModelId(m.id);
                              setShowModelPicker(false);
                              if (showResult) reset();
                            }
                          }}
                        >
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-0.5">
                              <span className="text-sm font-semibold text-slate-700">{m.name}</span>
                              <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${m.badgeClass}`}>
                                {m.badge}
                              </span>
                              {!avail && (
                                <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-slate-100 text-slate-400">
                                  未安装
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-3 text-xs text-slate-400">
                              <span>质量 <Dots count={m.quality}/></span>
                              <span>速度 <Dots count={m.speed}/></span>
                              <span>{m.size}</span>
                            </div>
                            {!avail && m.downloadHint && (
                              <p className="text-xs text-amber-500 mt-1">{m.downloadHint}</p>
                            )}
                          </div>
                          {isSelected && (
                            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                              <circle cx="8" cy="8" r="8" fill="#6366f1"/>
                              <path d="M5 8l2 2 4-4" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                            </svg>
                          )}
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Upload Area */}
              <div
                className={`flex flex-col items-center justify-center gap-3 border-2 border-dashed rounded-xl py-8 px-4 cursor-pointer transition-all duration-300 ${
                  isDragOver
                    ? "drag-active border-indigo-400 bg-indigo-50"
                    : "border-slate-200 bg-slate-50/50 hover:border-indigo-300 hover:bg-indigo-50/50"
                }`}
                onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={handleDrop}
                onClick={() => inputRef.current?.click()}
              >
                <div className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-colors duration-300 ${isDragOver ? "bg-indigo-100 text-indigo-500" : "bg-slate-100 text-slate-400"}`}>
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                </div>
                <div className="text-center">
                  <p className="text-sm font-semibold text-slate-600">
                    {file ? file.name : "点击上传或拖拽图片"}
                  </p>
                  <p className="text-xs text-slate-400 mt-1">支持 JPG, PNG, WebP</p>
                </div>
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  ref={inputRef}
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length) {
                      setFile(e.target.files[0]);
                      updateStatus("图片已选择，点击去背景", "info");
                    } else {
                      setFile(null);
                    }
                  }}
                />
              </div>

              {/* File Preview */}
              {file && !showResult && (
                <div className="flex items-center gap-3 bg-slate-50 rounded-xl p-3 animate-fade-in-up">
                  <div className="w-12 h-12 rounded-lg bg-slate-200 overflow-hidden flex-shrink-0">
                    <img src={URL.createObjectURL(file)} alt="preview" className="w-full h-full object-cover rounded-lg"/>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-slate-700 truncate">{file.name}</p>
                    <p className="text-xs text-slate-400">{(file.size / 1024).toFixed(1)} KB</p>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex flex-col gap-3">
                <button
                  type="button"
                  className={`w-full py-3.5 rounded-xl font-semibold text-base text-white shadow-lg transition-all duration-200 hover:shadow-xl disabled:shadow-none ${
                    file && !isLoading
                      ? "bg-gradient-to-r from-indigo-600 to-sky-500 hover:from-indigo-500 hover:to-sky-400 hover:-translate-y-0.5 active:translate-y-0 animate-pulse-glow"
                      : "bg-slate-300 text-slate-500 cursor-not-allowed"
                  }`}
                  disabled={!file || isLoading}
                  onClick={handleRemoveBg}
                >
                  {isLoading ? (
                    <span className="flex items-center justify-center gap-2">
                      <span className="animate-spin inline-block w-5 h-5 border-2 border-white border-t-transparent rounded-full"/>
                      处理中...
                    </span>
                  ) : (
                    <span className="flex items-center justify-center gap-2">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M18 6L6 18M6 6l12 12"/>
                      </svg>
                      一键去背景
                    </span>
                  )}
                </button>
                {(file || showResult) && (
                  <button
                    type="button"
                    className="w-full py-3 rounded-xl font-semibold text-sm bg-slate-100 text-slate-500 hover:bg-slate-200 hover:text-slate-700 transition-all duration-200"
                    onClick={reset}
                  >
                    重新上传
                  </button>
                )}
              </div>

              {/* Status */}
              {status.text && (
                <div className={`flex items-center gap-2.5 px-4 py-3 rounded-xl text-sm font-medium animate-fade-in-up ${
                  status.type === "error" ? "bg-rose-50 text-rose-600" :
                  status.type === "success" ? "bg-emerald-50 text-emerald-600" :
                  status.type === "loading" ? "bg-indigo-50 text-indigo-600" :
                  "bg-slate-50 text-slate-500"
                }`}>
                  <span className={`w-2 h-2 rounded-full flex-shrink-0 ${statusDotMap[status.type]}`}/>
                  {status.text}
                </div>
              )}
            </div>

            {/* Tips */}
            <div className="mt-4 px-2">
              <div className="flex items-start gap-2 text-xs text-slate-400">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="flex-shrink-0 mt-0.5">
                  <circle cx="12" cy="12" r="10"/>
                  <path d="M12 16v-4M12 8h.01"/>
                </svg>
                <span>所有处理均在浏览器本地完成，图片不会上传到服务器</span>
              </div>
            </div>
          </div>

          {/* Right Panel - Result */}
          <div className="flex-1 w-full min-h-0">
            {showResult ? (
              <div className="animate-fade-in-up">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-4">
                    <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">对比模式</span>
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded-sm bg-slate-300"/>
                      <span className="text-xs text-slate-500">原图</span>
                      <span className="w-3 h-3 rounded-sm checkerboard border border-slate-200"/>
                      <span className="text-xs text-slate-500">去背景</span>
                    </div>
                  </div>
                  <button
                    type="button"
                    className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-indigo-600 to-sky-500 text-white text-sm font-semibold shadow-lg shadow-indigo-200/50 hover:shadow-xl hover:-translate-y-0.5 transition-all duration-200"
                    onClick={handleDownload}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                      <polyline points="7 10 12 15 17 10"/>
                      <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                    下载 PNG
                  </button>
                </div>

                <div
                  ref={wrapperRef}
                  className="relative w-full rounded-2xl overflow-hidden shadow-2xl shadow-slate-300/30 border border-slate-200/50 bg-white"
                  style={{ maxWidth: 800 }}
                >
                  <div className="checkerboard absolute inset-0 z-0"/>
                  <img
                    ref={originalImgRef}
                    src={originalUrl}
                    alt="原图"
                    className="absolute top-0 left-0 w-full h-full object-contain select-none"
                    style={{ zIndex: 3, clipPath: `inset(0 ${(1 - sliderRatio) * 100}% 0 0)` }}
                    draggable={false}
                  />
                  <img
                    ref={processedImgRef}
                    src={processedUrl}
                    alt="去背景"
                    className="absolute top-0 left-0 w-full h-full object-contain select-none"
                    style={{ zIndex: 2, clipPath: `inset(0 0 0 ${sliderRatio * 100}%)` }}
                    draggable={false}
                  />
                  <div
                    className="absolute top-0 bottom-0 z-20 pointer-events-none"
                    style={{ left: `${sliderRatio * 100}%`, transform: "translateX(-50%)" }}
                  >
                    <div className="w-0.5 h-full bg-white/80 shadow-lg shadow-black/10"/>
                  </div>
                  <div
                    className="slider-handle absolute top-1/2 z-30 -translate-y-1/2 -translate-x-1/2"
                    style={{ left: `${sliderRatio * 100}%` }}
                    onMouseDown={handleSliderDown}
                    onTouchStart={handleSliderDown}
                  >
                    <div className="w-10 h-10 rounded-full bg-white shadow-xl shadow-black/15 border-2 border-white flex items-center justify-center hover:scale-110 transition-transform duration-150">
                      <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                        <path d="M6 10H2M14 10H18" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round"/>
                        <circle cx="10" cy="10" r="3" fill="#6366f1"/>
                      </svg>
                    </div>
                  </div>
                  <div
                    className="absolute top-4 left-4 z-10 px-3 py-1.5 rounded-lg bg-white/80 backdrop-blur-sm text-xs font-semibold text-slate-600 shadow-sm"
                    style={{ clipPath: `inset(0 ${(1 - sliderRatio) * 100 + 5}% 0 0)` }}
                  >
                    原图
                  </div>
                  <div
                    className="absolute top-4 right-4 z-10 px-3 py-1.5 rounded-lg bg-white/80 backdrop-blur-sm text-xs font-semibold text-slate-600 shadow-sm"
                    style={{ clipPath: `inset(0 0 0 ${sliderRatio * 100 + 5}%)` }}
                  >
                    去背景
                  </div>
                </div>

                {/* Model info badge */}
                <div className="flex items-center gap-2 mt-3">
                  <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${selectedModel.badgeClass}`}>
                    {selectedModel.badge}
                  </span>
                  <span className="text-xs text-slate-400">
                    使用 {selectedModel.name} 处理
                  </span>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-80 lg:h-[480px] bg-white/40 backdrop-blur-sm rounded-2xl border border-slate-200/50 border-dashed">
                <div className="animate-float">
                  <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-indigo-100 to-sky-100 flex items-center justify-center mb-4 mx-auto">
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="3" y="3" width="18" height="18" rx="2"/>
                      <circle cx="8.5" cy="8.5" r="1.5"/>
                      <path d="M21 15l-5-5L5 21"/>
                      <path d="M3 3l18 18" strokeDasharray="2 3"/>
                    </svg>
                  </div>
                </div>
                <p className="text-slate-400 font-medium text-sm">上传图片后，AI 将自动去除背景</p>
                <p className="text-slate-300 text-xs mt-1">当前模型：{selectedModel.name} | 结果支持滑动对比和一键下载</p>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="py-5 px-4 text-center">
        <p className="text-xs text-slate-400">
          AI 去背景工具 &middot; 支持 U2-Net / IMG.LY / BRIA / ISNet / MODNet &middot; 浏览器本地处理
        </p>
      </footer>
    </div>
  );
}
