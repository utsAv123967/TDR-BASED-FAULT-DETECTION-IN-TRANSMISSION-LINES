'use client';

import { WaveformData } from '@/types/tdr';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface WaveformChartProps {
  waveform: WaveformData;
}

export default function WaveformChart({ waveform }: WaveformChartProps) {
  const data = waveform.time.map((t, i) => ({
    time: t,
    voltage: waveform.voltage[i],
  }));

  return (
    <div className="glass-card p-6">
      <h2 className="text-xl font-semibold text-white mb-4">Waveform Visualization</h2>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
          <XAxis
            dataKey="time"
            stroke="#ffffff80"
            label={{ value: 'Time (ns)', position: 'insideBottom', offset: -5, fill: '#fff' }}
          />
          <YAxis
            stroke="#ffffff80"
            label={{ value: 'Voltage (V)', angle: -90, position: 'insideLeft', fill: '#fff' }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
            labelStyle={{ color: '#fff' }}
          />
          <Legend />
          <Line type="monotone" dataKey="voltage" stroke="#3b82f6" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
