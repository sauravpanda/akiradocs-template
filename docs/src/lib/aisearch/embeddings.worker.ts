import { pipeline, ProgressCallback, type PipelineType } from "@huggingface/transformers";

class EmbeddingPipelineSingleton {
    static task: PipelineType = 'feature-extraction';
    static model = 'sauravpanda/gte-small-onnx';
    static instance: Promise<any> | null = null;

    static async getInstance(progress_callback: ProgressCallback | null = null) {
        if (!this.instance) {
            console.log('[Embeddings Worker] Initializing embedding pipeline...');
            const startTime = performance.now();
            
            this.instance = pipeline(this.task, this.model, { 
                progress_callback: (progress) => {
                    progress_callback?.(progress);
                }
            }).then((pipeline) => {
                const loadTime = (performance.now() - startTime) / 1000;
                console.log(`[Embeddings Worker] Pipeline ready in ${loadTime.toFixed(2)}s`);
                return pipeline;
            });
        }
        
        return this.instance;
    }
}

self.addEventListener('message', async (event) => {
    const embedder = await EmbeddingPipelineSingleton.getInstance(x => {
        self.postMessage({ status: 'progress', progress: x });
    });

    const startTime = performance.now();
    const output = await embedder(event.data.text, {
        pooling: 'mean',
        normalize: true
    });
    const inferenceTime = (performance.now() - startTime) / 1000;
    console.log(`[Embeddings Worker] Embedding generated in ${inferenceTime.toFixed(2)}s`);

    self.postMessage({
        status: 'complete',
        output: Array.from(output.data),
    });
}); 