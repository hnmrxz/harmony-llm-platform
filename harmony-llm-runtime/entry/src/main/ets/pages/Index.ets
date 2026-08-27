import { promptAction } from '@kit.ArkUI';
import { common } from '@kit.AbilityKit';
import { picker } from '@kit.CoreFileKit';
import { fileIo as fs } from '@kit.CoreFileKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import llmEngine from 'libentry.so';

const DOMAIN = 0x0000;

interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
}

@Entry
@Component
struct Index {
  @State messages: ChatMessage[] = [];
  @State input: string = '';
  @State streamText: string = '';
  @State currentModel: string = '';
  private scroller: Scroller = new Scroller();

  private appContext(): common.UIAbilityContext {
    return getContext(this) as common.UIAbilityContext;
  }

  private modelStore(): string {
    return `${this.appContext().filesDir}/models`;
  }

  aboutToAppear(): void {
    llmEngine.init(this.modelStore());
    // Register the streaming token callback once.
    llmEngine.answerget((token: string) => {
      Promise.resolve().then(() => {
        this.streamText += token;
        this.messages[this.messages.length - 1] = { role: 'assistant', text: this.streamText };
        this.scroller.scrollEdge(Edge.Bottom);
      });
    });
  }

  /*
   * Pick a local `.hllm` with the system file picker, copy it into the app
   * sandbox (native fopen needs a real path, not an SAF URI), then import and
   * load it through the native model store.
   */
  async pickModelFile(): Promise<void> {
    try {
      const documentPicker = new picker.DocumentViewPicker(this.appContext());
      const result = await documentPicker.select({
        fileSuffixFilters: ['.hllm'],
        maxSelectNumber: 1,
      });
      const uri: string = result[0] ?? '';
      if (uri === '') {
        promptAction.showToast({ message: '未选择文件', duration: 1500 });
        return;
      }

      const destDir: string = `${this.appContext().filesDir}/imports`;
      fs.mkdirSync(destDir, true);
      const dest: string = `${destDir}/model_${Date.now()}.hllm`;
      fs.copyFileSync(uri, dest);
      hilog.info(DOMAIN, 'HLLM', 'picked model: %{public}s', dest);

      this.loadFromStore(dest);
    } catch (err) {
      hilog.error(DOMAIN, 'HLLM', 'pickModelFile failed: %{public}s', JSON.stringify(err));
      promptAction.showToast({ message: '模型文件选择失败', duration: 1500 });
    }
  }

  private loadFromStore(path: string): void {
    const id: string = llmEngine.importmodel(path);
    const msg: string = id === 'IMPORT_FAILED' ? '导入失败' : llmEngine.loadmodel(id);
    this.currentModel = id === 'IMPORT_FAILED' ? '' : id;
    this.messages.push({ role: 'assistant', text: msg });
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
        Button('选择模型文件')
          .onClick(() => this.pickModelFile())
        Button(this.currentModel !== '' ? `已加载: ${this.currentModel}` : '清空')
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
