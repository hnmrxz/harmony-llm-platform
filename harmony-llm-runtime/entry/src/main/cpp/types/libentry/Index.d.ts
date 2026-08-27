/**
 * HarmonyOS LLM Runtime native entry module.
 *
 * The ArkTS layer imports this module to import, load and run the CANN LLM
 * Engine against models installed from a `.hllm` package:
 *   init            -> set the model store root (optional; defaults to app files)
 *   importmodel     -> import a .hllm into the durable model store
 *   loadmodel       -> load a model (by id) into the LLM Engine
 *   modelinfer      -> start async generation for a prompt
 *   answerget       -> register the streaming token callback
 *   deinitmodel     -> release the engine
 */
export const init: (installRoot?: string) => void;
export const importmodel: (hllmPath: string) => string;
export const loadmodel: (modelId: string) => string;
export const modelinfer: (question: string) => void;
export const answerget: (callback: (token: string) => void) => void;
export const deinitmodel: () => void;
