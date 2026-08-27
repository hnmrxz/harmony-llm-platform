/**
 * HarmonyOS LLM Runtime native entry module.
 *
 * The ArkTS layer imports this module and drives the CANN LLM Engine through
 * the same four calls the official DemoNext sample exposes:
 *   loadmodel -> load the .omc from context.json + executor.json
 *   modelinfer -> start async generation for a prompt
 *   answerget  -> register the streaming token callback
 *   deinitmodel-> release the engine
 */
export const loadmodel: (contextJsonPath: string, executorJsonPath: string) => string;
export const modelinfer: (question: string) => void;
export const answerget: (callback: (token: string) => void) => void;
export const deinitmodel: () => void;
