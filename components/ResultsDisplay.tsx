'use client';

import { AnalyzeResponse } from '@/types/tdr';

interface ResultsDisplayProps {
  results: AnalyzeResponse;
}

export default function ResultsDisplay({ results }: ResultsDisplayProps) {
  const getFaultColor = (type: string) => {
    switch (type) {
      case 'Open':
        return 'text-red-400';
      case 'Short':
        return 'text-yellow-400';
      case 'Resistive':
        return 'text-orange-400';
      default:
        return 'text-gray-400';
    }
  };

  const isOpen = results.faultType === 'Open';

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
        
        {!isOpen && results.faultDistance !== undefined && (
          <>
            <div className="bg-white/5 rounded-lg p-4">
              <p className="text-white/60 text-sm">Fault Distance</p>
              <p className="text-2xl font-bold text-white">
                {results.faultDistance.toFixed(2)} m
              </p>
            </div>
            <div className="bg-white/5 rounded-lg p-4">
              <p className="text-white/60 text-sm">Time Delay (Δt)</p>
              <p className="text-xl font-bold text-white">
                {results.deltaT ? results.deltaT.toExponential(2) : 'N/A'} s
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
