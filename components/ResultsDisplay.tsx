'use client';

import { AnalyzeResponse } from '@/types/tdr';

interface ResultsDisplayProps {
  results: AnalyzeResponse;
}

export default function ResultsDisplay({ results }: ResultsDisplayProps) {
  const getFaultColor = (type: string) => {
    switch (type) {
      case 'No fault':
        return 'text-green-400';
      case 'Short Circuit':
        return 'text-red-400';
      case 'Resistive Fault':
        return 'text-orange-400';
      default:
        return 'text-gray-400';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'text-green-400';
    if (confidence > 0.6) return 'text-yellow-400';
    return 'text-orange-400';
  };

  const isNoFault = results.faultType === 'No fault';
  
  // Check if modelOutput exists
  if (!results.modelOutput) {
    return (
      <div className="glass-card p-6 animate-fade-in">
        <h2 className="text-xl font-semibold text-white mb-4">Analysis Results</h2>
        <div className="grid grid-cols-1 gap-4">
          <div className="bg-white/5 rounded-lg p-4">
            <p className="text-white/60 text-sm">Fault Type</p>
            <p className={`text-2xl font-bold ${getFaultColor(results.faultType)}`}>
              {results.faultType}
            </p>
          </div>
        </div>
      </div>
    );
  }

  const modelOut = results.modelOutput;
  const predictedConfidence = Object.values(modelOut.confidence_scores)[modelOut.predicted_class];

  return (
    <div className="glass-card p-6 animate-fade-in">
      <h2 className="text-xl font-semibold text-white mb-4">🏁 Final Report</h2>
      
      {/* Main Result */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-white/5 rounded-lg p-4 border-2 border-white/20">
          <p className="text-white/60 text-sm mb-1">Predicted Fault Type</p>
          <p className={`text-3xl font-bold ${getFaultColor(results.faultType)}`}>
            {results.faultType}
          </p>
          <p className={`text-lg font-semibold mt-2 ${getConfidenceColor(predictedConfidence)}`}>
            Confidence: {(predictedConfidence * 100).toFixed(2)}%
          </p>
        </div>

        {/* Distance Information (only if fault detected) */}
        {!isNoFault && (
          <div className="bg-white/5 rounded-lg p-4 border-2 border-white/20">
            <p className="text-white/60 text-sm mb-1">Fault Distance</p>
            {results.distance !== null ? (
              <>
                <p className="text-3xl font-bold text-blue-400">
                  {results.distance.toFixed(2)} m
                </p>
                {results.deltaT !== null && (
                  <p className="text-sm text-white/70 mt-2">
                    Round-trip time (ΔT): {(results.deltaT * 1e9).toFixed(2)} ns
                  </p>
                )}
              </>
            ) : (
              <p className="text-lg text-yellow-400">
                Distance model not available
              </p>
            )}
          </div>
        )}
      </div>

      {/* Model Output Details */}
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-white mb-3">📊 Model Output (Raw)</h3>
        
        {/* Confidence Scores */}
        <div className="bg-white/5 rounded-lg p-4 mb-3">
          <p className="text-white/80 font-semibold mb-2">Confidence Scores:</p>
          <div className="space-y-2">
            {Object.entries(modelOut.confidence_scores).map(([label, score]) => (
              <div key={label} className="flex justify-between items-center">
                <span className="text-white/70">{label}:</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 bg-white/10 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        label.includes(results.faultType) ? 'bg-green-400' : 'bg-white/30'
                      }`}
                      style={{ width: `${score * 100}%` }}
                    />
                  </div>
                  <span className="text-white font-mono text-sm w-16 text-right">
                    {(score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Technical Details */}
        <div className="bg-white/5 rounded-lg p-4">
          <p className="text-white/80 font-semibold mb-2">Technical Details:</p>
          <div className="space-y-1 font-mono text-xs text-white/70">
            <div><span className="text-white/50">Predicted Class:</span> {modelOut.predicted_class}</div>
            <div><span className="text-white/50">Device:</span> {modelOut.model_info.device}</div>
            <div><span className="text-white/50">Input Shape:</span> [{modelOut.model_info.input_shape.join(', ')}]</div>
            <div><span className="text-white/50">Num Samples:</span> {modelOut.model_info.num_samples}</div>
            <div><span className="text-white/50">Time Range:</span> {modelOut.model_info.time_range}</div>
          </div>
        </div>

        {/* Raw Logits (collapsible) */}
        <details className="bg-white/5 rounded-lg p-4 mt-3">
          <summary className="text-white/80 font-semibold cursor-pointer">Raw Logits (Click to expand)</summary>
          <pre className="mt-2 text-xs text-white/60 font-mono overflow-x-auto">
            {JSON.stringify(modelOut.raw_logits, null, 2)}
          </pre>
        </details>
      </div>
    </div>
  );
}
