'use client';

import { useState } from 'react';
import { AnalyzeResponse } from '@/types/tdr';
import FileUpload from '@/components/FileUpload';
import ParameterInput from '@/components/ParameterInput';
import ResultsDisplay from '@/components/ResultsDisplay';
import WaveformChart from '@/components/WaveformChart';

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [velocityFactor, setVelocityFactor] = useState<string>('0.67');
  const [characteristicImpedance, setCharacteristicImpedance] = useState<string>('50');
  const [results, setResults] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please upload a CSV file');
      return;
    }

    setError('');
    setLoading(true);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('velocityFactor', velocityFactor);
      formData.append('characteristicImpedance', characteristicImpedance);

      const response = await fetch('http://localhost:8000/api/analyze-tdr', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to analyze TDR data');
      }

      setResults(data);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8">
      <div className="container mx-auto max-w-7xl">
        {/* Header */}
        <div className="text-center mb-10">
          <h1 className="text-5xl font-bold text-white mb-3 tracking-tight">
            TDR Cable Fault Detection
          </h1>
          <p className="text-xl text-white/70">
            AI-Powered Time Domain Reflectometry Analysis
          </p>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <FileUpload onFileSelect={setFile} />
          <ParameterInput
            velocityFactor={velocityFactor}
            characteristicImpedance={characteristicImpedance}
            onVelocityFactorChange={setVelocityFactor}
            onCharacteristicImpedanceChange={setCharacteristicImpedance}
          />
        </div>

        {/* Analyze Button */}
        <div className="flex justify-center mb-6">
          <button
            onClick={handleAnalyze}
            disabled={loading || !file}
            className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:from-gray-500 disabled:to-gray-600 disabled:cursor-not-allowed text-white font-semibold rounded-lg shadow-lg transform transition hover:scale-105 disabled:scale-100"
          >
            {loading ? 'Analyzing...' : 'Analyze Cable'}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="glass-card p-4 mb-6 bg-red-500/20 border-red-500/50">
            <p className="text-red-200 text-center">{error}</p>
          </div>
        )}

        {/* Results */}
        {results && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ResultsDisplay results={results} />
            <WaveformChart waveform={results.waveform} />
          </div>
        )}
      </div>
    </div>
  );
}
