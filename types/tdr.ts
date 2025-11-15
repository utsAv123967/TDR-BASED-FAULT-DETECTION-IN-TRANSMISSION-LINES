export type FaultType = 'No fault' | 'Short Circuit' | 'Resistive Fault';

export interface ModelOutput {
  predicted_class: number;
  predicted_fault_type: string;
  confidence_scores: {
    'No fault (Open)': number;
    'Short Circuit': number;
    'Resistive Fault': number;
  };
  raw_logits: number[];
  model_info: {
    device: string;
    input_shape: number[];
    num_samples: number;
    time_range: string;
  };
}

export interface AnalyzeResponse {
  faultType: FaultType;
  distance: number | null;  // Distance in meters (null if no fault or not available)
  deltaT: number | null;    // Round-trip time in seconds (null if no distance)
  plotData: string | null;
  modelOutput?: ModelOutput;  // Optional for backward compatibility
}
