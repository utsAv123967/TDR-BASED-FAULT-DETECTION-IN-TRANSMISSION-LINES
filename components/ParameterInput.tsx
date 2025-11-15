'use client';

interface ParameterInputProps {
  velocityFactor: string;
  characteristicImpedance: string;
  onVelocityFactorChange: (value: string) => void;
  onCharacteristicImpedanceChange: (value: string) => void;
}

export default function ParameterInput({
  velocityFactor,
  characteristicImpedance,
  onVelocityFactorChange,
  onCharacteristicImpedanceChange,
}: ParameterInputProps) {
  return (
    <div className="glass-card p-6">
      <h2 className="text-xl font-semibold text-white mb-4">Cable Parameters</h2>
      <div className="space-y-4">
        <div>
          <label className="block text-white/80 text-sm font-medium mb-2">
            Velocity Factor (VF)
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={velocityFactor}
            onChange={(e) => onVelocityFactorChange(e.target.value)}
            className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-400"
            placeholder="e.g., 0.67"
          />
        </div>
        <div>
          <label className="block text-white/80 text-sm font-medium mb-2">
            Characteristic Impedance (Z₀) Ω
          </label>
          <input
            type="number"
            step="0.1"
            min="0"
            value={characteristicImpedance}
            onChange={(e) => onCharacteristicImpedanceChange(e.target.value)}
            className="w-full px-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/40 focus:outline-none focus:ring-2 focus:ring-blue-400"
            placeholder="e.g., 50"
          />
        </div>
      </div>
    </div>
  );
}
