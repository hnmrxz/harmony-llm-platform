import { AbilityConstant, ConfigurationConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window } from '@kit.ArkUI';
import llmEngine from 'libentry.so';

const DOMAIN = 0x0000;

export default class EntryAbility extends UIAbility {
  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    this.context.getApplicationContext()
      .setColorMode(ConfigurationConstant.ColorMode.COLOR_MODE_NOT_SET);
    hilog.info(DOMAIN, 'HLLM', '%{public}s', 'Ability onCreate');
  }

  onDestroy(): void {
    // Release the CANN LLM Engine when the ability goes away.
    llmEngine.deinitmodel();
    hilog.info(DOMAIN, 'HLLM', '%{public}s', 'Ability onDestroy');
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    hilog.info(DOMAIN, 'HLLM', '%{public}s', 'Ability onWindowStageCreate');
    windowStage.loadContent('pages/Index', (err) => {
      if (err.code) {
        hilog.error(DOMAIN, 'HLLM', 'Failed to load the content. Cause: %{public}s',
          JSON.stringify(err));
        return;
      }
      hilog.info(DOMAIN, 'HLLM', 'Succeeded in loading the content.');
    });
  }

  onWindowStageDestroy(): void {
    hilog.info(DOMAIN, 'HLLM', '%{public}s', 'Ability onWindowStageDestroy');
  }

  onForeground(): void {
    hilog.info(DOMAIN, 'HLLM', '%{public}s', 'Ability onForeground');
  }

  onBackground(): void {
    hilog.info(DOMAIN, 'HLLM', '%{public}s', 'Ability onBackground');
  }
}
