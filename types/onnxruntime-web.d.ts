declare module "onnxruntime-web" {
  export interface InferenceSession {
    inputNames: string[];
    outputNames: string[];
    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  }

  export interface Tensor {
    data: Float32Array | Uint8Array;
    dims: number[];
    type: string;
  }

  export namespace InferenceSession {
    function create(path: string, options?: any): Promise<InferenceSession>;
  }

  export class Tensor {
    constructor(type: string, data: Float32Array | Uint8Array, dims: number[]);
  }
}
