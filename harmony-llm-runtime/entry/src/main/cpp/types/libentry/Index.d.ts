/**
 * HarmonyOS LLM Runtime native entry module.
 *
 * The ArkTS layer imports this module to import, load and run the CANN LLM
 * Engine against models installed from a `.hllm` package OR a folder of
 * pre-converted CANN engine files.
 */
export const init: (installRoot?: string) => void;
export const importmodel: (hllmPath: string) => string;
export const importfolder: (folderPath: string) => string;
export const loadmodel: (modelId: string) => string;
export const modelinfer: (question: string) => void;
export const answerget: (callback: (token: string) => void) => void;
export const deinitmodel: () => void;
