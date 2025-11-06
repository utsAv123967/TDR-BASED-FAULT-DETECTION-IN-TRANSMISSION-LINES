export type FaultType = 'Open' | 'Short' | 'Resistive';

export interface WaveformData {
  time: number[];
  voltage: number[];
}

export interface AnalyzeResponse {
  faultType: FaultType;
  faultDistance?: number;  // Only present if not Open
  deltaT?: number;         // Only present if not Open
  waveform: WaveformData;
  waveformImage?: string;
}
