import { promptAction } from '@kit.ArkUI';
import llmEngine from 'libentry.so';
import { hilog } from '@kit.PerformanceAnalysisKit';

const DOMAIN = 0x0000;

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
}

/*
 * Where the model lives on device, under the app's sandbox files directory.
 * The installed .hllm package is exported to this directory and contains:
 *   config/context.json, config/executor.json
 *   model/<model>.omc, model/SubGraph_0.weight, model/*.embedding_weights, ...
 *   model/tokenizer.json
 * `contextJsonPath` / `executorJsonPath` point at the two engine config files.
 */
const MODEL_DIR = '/data/storage/el2/base/files/installed/qwen3-8b';
const CONTEXT_JSON = `${MODEL_DIR}/config/context.json`;
const EXECUTOR_JSON = `${MODEL_DIR}/config/executor.json`;

@Entry
@Component
struct Index {
  @State messages: ChatMessage[] = [];
  @State input: string = '';
  @State loading: string = '';
  @State streamText: string = '';
  private scroller: Scroller = new Scroller();

  aboutToAppear(): void {
    // Register the streaming token callback once.
    llmEngine.answerget((token: string) => {
      Promise.resolve().then(() => {
        this.streamText += token;
        this.messages[this.messages.length - 1] = { role: 'assistant', text: this.streamText };
        this.scroller.scrollEdge(Edge.Bottom);
      });
    });
  }

  loadModel(): string {
    // context.json / executor.json must already be exported from the .hllm.
    const result = llmEngine.loadmodel(CONTEXT_JSON, EXECUTOR_JSON);
    this.messages.push({ role: 'assistant', text: result });
    return result;
  }

  /*
   * Apply the chat template (string-level) before handing the text to the
   * engine. The CANN LLM Engine tokenizes the resulting text internally, so no
   * tokenizer work is needed here. Replace with the model's own template.
   */
  private formatPrompt(messages: ChatMessage[]): string {
    let text = '';
    for (const msg of messages) {
      text += `<|im_start|>${msg.role}\n${msg.text}<|im_end|>\n`;
    }
    return `${text}<|im_start|>assistant\n`;
  }

  sendMessage(): void {
    const trimmed = this.input.trim();
    if (trimmed === '') {
      promptAction.showToast({ message: '请输入内容', duration: 1500 });
      return;
    }
    this.streamText = '';
    this.messages.push({ role: 'user', text: trimmed });
    this.messages.push({ role: 'assistant', text: 'Waiting for response...' });
    this.scroller.scrollEdge(Edge.Bottom);
    this.input = '';

    const prompt = this.formatPrompt(this.messages.filter((m) => m.role === 'user'));
    llmEngine.modelinfer(prompt);
    hilog.info(DOMAIN, 'HLLM', 'modelinfer sent: %{public}s', prompt);
  }

  build() {
    Column() {
      Text('HarmonyOS LLM Runtime')
        .fontSize(22)
        .fontWeight(FontWeight.Bold)
        .width('100%')
        .height(56)
        .padding({ left: 16 })
        .textAlign(TextAlign.Start)

      Row({ space: 8 }) {
        Button('加载模型')
          .onClick(() => this.loadModel())
        Button('清空')
          .onClick(() => {
            this.messages = [];
            this.streamText = '';
          })
      }
      .width('100%')
      .padding({ left: 16, right: 16, bottom: 8 })

      List({ space: 12, scroller: this.scroller }) {
        ForEach(this.messages, (msg: ChatMessage, index: number) => {
          ListItem() {
            Row() {
              Text(msg.text)
                .backgroundColor(msg.role === 'user' ? '#33c4fd' : '#e0e0e0')
                .fontSize(15)
                .fontColor('#1C1B21')
                .padding(10)
                .borderRadius(10)
                .textAlign(TextAlign.Start)
            }
            .width('100%')
            .justifyContent(msg.role === 'user' ? FlexAlign.End : FlexAlign.Start)
          }
        }, (msg: ChatMessage) => JSON.stringify(msg))
      }
      .layoutWeight(1)
      .width('100%')
      .padding({ left: 12, right: 12 })

      Row() {
        TextArea({ placeholder: '请输入内容', text: $$this.input })
          .layoutWeight(1)
          .height(48)
          .fontSize(18)
          .enterKeyType(EnterKeyType.Send)
          .borderWidth(1)
          .borderColor(Color.Black)
          .onSubmit(() => this.sendMessage())
        Button('发送')
          .width('22%')
          .height(48)
          .onClick(() => this.sendMessage())
      }
      .width('100%')
      .height(64)
      .padding({ left: 12, right: 12 })
    }
    .width('100%')
    .height('100%')
  }
}
