import type { ControllerType } from '@/lib/deltrel/ai/controllers';
import type { AiRuntime } from '@/lib/store';

const ENGINE_LABELS: Record<AiRuntime, string> = {
  server: 'Mac engine — current champion',
  local: 'Browser AI — lightweight',
};

export function deltrelAiDevtoolsEnabled(): boolean {
  return process.env.NEXT_PUBLIC_DELTREL_AI_DEVTOOLS === '1';
}

export function engineControllerLabel(controller: ControllerType): string {
  return controller === 'human' ? 'Human' : ENGINE_LABELS[controller];
}
