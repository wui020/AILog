用户指出 `useAIGeneration.ts` 是旧写法，当前项目实际使用 Vercel AI SDK 框架在主进程实现。我需要重新分析主进程（`src/main/agent/`）中的代码，找出真正的 token / 上下文限制点。

让我深入查看主进程的 agent 相关文件。
明白了，我需要重新聚焦到主进程 `src/main/agent/` 下的实际实现。让我深入分析 Vercel AI SDK 框架相关的代码。
现在我找到了真正的主进程实现。让我继续查看 active-tools.ts 和其他相关文件，找出所有 token/上下文限制点。

从 `index.ts` 中已经找到的关键限制：

1. **意图分类 `maxOutputTokens: 16`** (line 281)
2. **主 Agent `maxOutputTokens: 8192`** (line 470)
3. **Agent step 上限 `stopWhen: stepCountIs(25)`** (line 472)
4. **Anthropic thinking budget `budgetTokens: 10000`** (provider.ts:115)
5. **prevUserText 截断 `slice(0, 100)`** (line 258)
6. **意图分类 buffer 日志 `slice(0, 200)`** (line 298)
7. **briefJson 日志截断** (line 53-61)
8. **历史上下文日志 `slice(0, 60)`** (line 99)
9. **用户文本日志 `slice(0, 70)`** (line 117)
10. **narration 日志 `slice(0, 120)`** (line 354)
11. **tools 日志 `slice(0, MAX=5)`** (line 134-135)
12. **prompts.ts:21** 意图分类输入 `slice(0, 1500)`

让我查看 active-tools.ts 和 prompts.ts 中是否有更多限制。
找到了真正的实现。让我继续查看 `active-tools.ts` 和其他主进程文件中的限制。
现在让我查看 prompts.ts 中是否有更多限制，以及 tools/handlers 中的限制。
让我再查看 prompts.ts 中的完整限制以及工具处理器的限制。
-
现在让我再确认 prompts.ts 中剩余的软性限制，以及 scene handler 中的模型名匹配限制。
Now I have a comprehensive picture. Let me also check the scenebuilder for any formatSceneSummary limits.
让我确认 scenebuilder 中的 `formatSceneSummary` 是否有节点数量限制。
-
-
-
-
Now I have a complete picture. Let me compile the final answer based on the actual Vercel AI SDK implementation in the main process.

Summary of all token/context limits in the current (active) codebase:

**Main process (`src/main/agent/`):**

1. **`index.ts:470`** — Main agent `maxOutputTokens: 8192`
2. **`index.ts:472`** — Agent step limit `stopWhen: stepCountIs(25)`
3. **`index.ts:281`** — Intent classification `maxOutputTokens: 16`
4. **`provider.ts:115`** — Anthropic thinking `budgetTokens: 10000`
5. **`prompts.ts:21`** — Intent classification input truncation `slice(0, 1500)`
6. **`index.ts:258`** — Previous user text truncation `slice(0, 100)`
7. **`material-pipeline.ts:71`** — VLM material selection `maxOutputTokens: 100`
8. **`material-pipeline.ts:64`** — VLM material selection timeout `8000ms`
9. **`material-pipeline.ts:21`** — Deterministic carpaint picks `slice(0, 3)`
10. **`scene.ts:457`** — Complex model VLM match `maxOutputTokens: 30`
11. **`video.ts:8`** — `MAX_NUM_FRAMES = 441`
12. **`video.ts:50-51`** — width/height clamp `1~4096`
13. **`video.ts:34`** — frame_rate clamp `1~60`
14. **`code.ts:94`** — patch_code error sig lines `slice(0, 30)`
15. **`code.ts:102`** — anchor display truncation `slice(0, 42)` when >45
16. **`builder.ts:1004-1048`** — `formatSceneSummary` prefab folding (≥3 children)
17. **`prompts.ts:160`** — Prompt soft limit: material tools max 3 calls/request
18. **`prompts.ts:185`** — Prompt soft limit: code file split at 200 lines
19. **`session-manager.ts:218`** — Auto-title `slice(0, 30)`
20. **Log truncations** (not affecting context): briefJson max=80, history log slice(0,60), user log slice(0,70), etc.

Key gaps:
- No `maxOutputTokens` on VLM narration (`generateText` at index.ts:325-339)
- No overall context window token counting/truncation
- No history message sliding window based on token count
- AI SDK's `convertToModelMessages` passes ALL history messages without truncation
现在我有了完整的图景。以下是基于**实际使用 Vercel AI SDK 的主进程实现**（`src/main/agent/`）的完整限制清单：

---

## 一、Agent Loop / LLM 核心限制（`src/main/agent/index.ts`）

**1. 主 Agent `maxOutputTokens: 8192`**

```462:472:src/main/agent/index.ts
        const result = streamText({
          model: llmModel,
          ...(thinkingOptions ? { providerOptions: thinkingOptions as any } : {}),
          system:
            SYSTEM_PROMPTS[intent] + API_REFERENCE[intent] + sceneContextBlock + materialLogBlock,
          messages: modelMessages,
          tools,
          temperature: 0.6,
          maxOutputTokens: 8192,
          prepareStep: prepareStep as any,
          stopWhen: stepCountIs(25),
```

每一步 LLM 输出上限 8192 tokens。

**2. Agent 步数上限 `stepCountIs(25)`**

同上第 472 行，`stopWhen: stepCountIs(25)` 限制 Agent 最多执行 25 步（每步可包含多个工具调用）。

**3. 意图分类 `maxOutputTokens: 16`**

```277:283:src/main/agent/index.ts
          const classifyStream = streamText({
            model: createLLMModel(),
            ...(noThinkOpts ? { providerOptions: noThinkOpts as any } : {}),
            prompt: CLASSIFY_PROMPT(session.userIntent, prevIntent, prevUserText),
            maxOutputTokens: 16,
            temperature: 0
          })
```

意图分类只允许输出 16 tokens（只需一个标签词）。

---

## 二、Thinking Budget（`src/main/agent/provider.ts`）

**4. Anthropic thinking `budgetTokens: 10000`**

```113:121:src/main/agent/provider.ts
export function getLLMThinkingProviderOptions(): Record<string, unknown> | undefined {
  if (llmProtocol === 'anthropic') {
    return { anthropic: { thinking: { type: 'enabled', budgetTokens: 10000 } } }
  }
  if (llmProtocol === 'openai') {
    return { llm: { chat_template_kwargs: { enable_thinking: true } } }
  }
  return undefined
}
```

Anthropic 协议下，思维链预算限制为 10000 tokens。OpenAI 协议无此限制。

---

## 三、意图分类输入截断（`src/main/agent/prompts.ts`）

**5. 意图分类用户输入 `slice(0, 1500)`**

```21:21:src/main/agent/prompts.ts
  const text = userText.slice(0, 1500)
```

用户输入超过 1500 字符的部分在意图分类时被截断。

**6. 上一轮用户文本截断 `slice(0, 100)`**

```249:259:src/main/agent/index.ts
        const prevUserText = (() => {
          const userMsgs = messages.filter((m) => m.role === 'user')
          if (userMsgs.length < 2) return ''
          const prevMsg = userMsgs[userMsgs.length - 2] as any
          const parts: any[] = prevMsg.parts || []
          const t: string = parts.find((p: any) => p.type === 'text')?.text || ''
          return t
            .replace(/\[会话上下文\][\s\S]*$/, '')
            .trim()
            .slice(0, 100)
        })()
```

传给意图分类的上一轮用户文本截断为 100 字符。

---

## 四、VLM 材质筛选管道（`src/main/agent/material-pipeline.ts`）

**7. VLM 材质选择 `maxOutputTokens: 100` + 超时 `8000ms`**

```62:74:src/main/agent/material-pipeline.ts
    const vlmModel = createLLMModel()
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 8000)

    const noThinkOpts = getNoThinkProviderOptions()
    const result = await generateText({
      model: vlmModel,
      ...(noThinkOpts ? { providerOptions: noThinkOpts as any } : {}),
      messages: [{ role: 'user', content: prompt }],
      maxOutputTokens: 100,
      temperature: 0,
      abortSignal: controller.signal
    })
```

**8. 确定性车漆候选 `slice(0, 3)`**

```19:22:src/main/agent/material-pipeline.ts
  return materialPaths
    .filter((path) => CARPAINT_PATH_RE.test(path) && !NON_CARPAINT_DECAL_PATH_RE.test(path))
    .slice(0, 3)
```

最多取 3 个车漆材质候选。

---

## 五、复杂模型 VLM 匹配（`src/main/agent/tools/handlers/scene.ts`）

**9. 模型名匹配 `maxOutputTokens: 30`**

```445:459:src/main/agent/tools/handlers/scene.ts
      const { text } = await generateText({
        model: createLLMModel(),
        ...(_noThink ? { providerOptions: _noThink as any } : {}),
        prompt:
          `从模型库中选出最匹配的模型名（区分大小写，只输出名称本身，无匹配输出NONE）。\n` +
          `模型库：${catalog.join(', ')}\n` +
          `本次添加的模型：${userModelId}\n` +
          `用户原文（仅当模型关键词过于模糊时参考）：${rawUserText}\n` +
          `输出：`,
        maxOutputTokens: 30,
        temperature: 0
      })
```

---

## 六、VLM 视觉旁白（`src/main/agent/index.ts`）—— **无 token 限制**

**10. VLM 图片描述 `generateText` 无 `maxOutputTokens`**

```325:339:src/main/agent/index.ts
                const { text: vis } = await generateText({
                  model: createVisionModel(),
                  messages: [
                    {
                      role: 'user',
                      content: [
                        {
                          type: 'text',
                          text: narrationPrompt
                        },
                        { type: 'image', image: imageData }
                      ]
                    }
                  ]
                })
```

这是唯一一个**没有设置 `maxOutputTokens`** 的 LLM 调用，输出长度完全由模型决定。

---

## 七、视频生成参数限制（`src/main/agent/tools/handlers/video.ts`）

**11. `MAX_NUM_FRAMES = 441`**

```4:8:src/main/agent/tools/handlers/video.ts
const DEFAULT_WIDTH = 1152
const DEFAULT_HEIGHT = 768
const DEFAULT_NUM_FRAMES = 97
const DEFAULT_FRAME_RATE = 24
const MAX_NUM_FRAMES = 441
```

**12. 视频 width/height `clamp 1~4096`，frame_rate `clamp 1~60`**

```34:53:src/main/agent/tools/handlers/video.ts
    const frameRate = clampInt(args.frame_rate, DEFAULT_FRAME_RATE, 1, 60)
    // ...
    width: clampInt(args.width, DEFAULT_WIDTH, 1, 4096),
    height: clampInt(args.height, DEFAULT_HEIGHT, 1, 4096),
```

---

## 八、代码工具的上下文截断（`src/main/agent/tools/handlers/code.ts`）

**13. `patch_code` 错误提示签名行 `slice(0, 30)`**

```84:96:src/main/agent/tools/handlers/code.ts
      const sigLines = content
        .split('\n')
        .filter((l) => {
          const t = l.trim()
          return (
            t.includes('ANCHOR') ||
            /^(void|bool|int|float|double|auto|std::|gw::|[A-Z])/.test(t) ||
            t.startsWith('#include')
          )
        })
        .slice(0, 30)
        .join('\n')
      const hint = sigLines ? `\n文件中的可用行：\n${sigLines}` : ''
```

anchor 找不到时最多返回 30 行签名行供 LLM 自纠。

**14. `patch_code` anchor 显示截断 `slice(0, 42)`**

```102:102:src/main/agent/tools/handlers/code.ts
    const anchorShort = anchor.length > 45 ? anchor.slice(0, 42) + '...' : anchor
```

**15. `read_file` 返回完整内容——无限制**

```13:20:src/main/agent/tools/handlers/code.ts
    const content = await readFile(filePath, 'utf-8')
    const fileName = filePath.split('/').pop() || filePath
    return {
      success: true,
      message: `已读取 ${fileName}（${content.split('\n').length} 行）`,
      data: { file: fileKey, content }
    }
```

`read_file` 工具会返回**整个文件内容**，无行数/token 限制。

---

## 九、场景节点上下文压缩（`src/main/scenebuilder/builder.ts`）

**16. `formatSceneSummary` 预制体折叠**

```1004:1048:src/main/scenebuilder/builder.ts
  static formatSceneSummary(nodes: ScannedNode[]): string {
    // 1. 识别预制体根节点：EmptyNode 且有 ≥3 个直接子节点
    const directChildCount = new Map<string, number>()
    for (const n of nodes) {
      if (n.parent) {
        directChildCount.set(n.parent, (directChildCount.get(n.parent) || 0) + 1)
      }
    }
    const prefabRoots = new Set<string>()
    for (const n of nodes) {
      if (n.class === 'EmptyNode' && (directChildCount.get(n.name) || 0) >= 3) {
        prefabRoots.add(n.name)
      }
    }

    // 2. 格式化
    const lines = ['当前场景已有以下节点：']
    for (const n of nodes) {
      // 跳过预制体的直接子节点
      if (n.parent && prefabRoots.has(n.parent)) {
        continue
      }
```

预制体（`EmptyNode` 且 ≥3 个直接子节点）的子节点被折叠，不展开内部结构，标注 `[预制体, N个子节点]`。但**没有节点总数上限**，场景有 1000 个节点就会注入 1000 行。

---

## 十、Prompt 中的软性限制（`src/main/agent/prompts.ts`）

**17. 材质工具调用上限"最多 3 次"**

```160:160:src/main/agent/prompts.ts
■ 工具调用上限：每个用户请求最多调用 3 次工具（如 scan_materials×1 + modify_material×1~2），然后必须输出文本总结。严禁对同一材质文件反复调用 modify_material。
```

这是 prompt 级别软约束，非代码强制。

**18. 代码文件拆分判据"200 行"**

```185:185:src/main/agent/prompts.ts
■ 子类/多文件：功能复杂时按职责拆分到子目录（如 Particles\\ParticleManager.h/.cpp），在 UserGameSceneForEditor 中 #include 并实例化。拆分判据：单文件超过 200 行或职责独立可复用。
```

---

## 十一、会话标题截断（`src/main/agent/session-manager.ts`）

**19. 自动标题 `slice(0, 30)`**

```218:218:src/main/agent/session-manager.ts
          session.title = text.slice(0, 30).trim() + (text.length > 30 ? '...' : '')
```

---

## 十二、日志截断（不影响实际上下文）

| 位置 | 文件:行号 | 截断值 |
|------|----------|-------|
| `briefJson` | `index.ts:53` | 80 字符 |
| 历史上下文日志 | `index.ts:99` | 60 字符 |
| 用户文本日志 | `index.ts:117` | 70 字符 |
| 意图 nomatch buffer | `index.ts:298` | 200 字符 |
| VLM 旁白日志 | `index.ts:354` | 120 字符 |
| tools 日志 | `index.ts:134-136` | 最多显示 5 个 |
| step 工具参数 | `index.ts:147` | 60 字符 |
| step 工具结果 | `index.ts:151` | 80 字符 |

---

## 总结表（影响上下文/Token 的限制）

| # | 限制点 | 文件:行号 | 类型 | 限制值 |
|---|--------|----------|------|--------|
| 1 | 主 Agent 输出 | `index.ts:470` | `maxOutputTokens` | 8192 |
| 2 | Agent 步数 | `index.ts:472` | `stopWhen` | 25 步 |
| 3 | 意图分类输出 | `index.ts:281` | `maxOutputTokens` | 16 |
| 4 | Anthropic thinking | `provider.ts:115` | `budgetTokens` | 10000 |
| 5 | 意图分类输入 | `prompts.ts:21` | 文本截断 | 1500 字符 |
| 6 | 上一轮用户文本 | `index.ts:258` | 文本截断 | 100 字符 |
| 7 | VLM 材质选择输出 | `material-pipeline.ts:71` | `maxOutputTokens` | 100 |
| 8 | VLM 材质选择超时 | `material-pipeline.ts:64` | 超时 | 8000ms |
| 9 | 车漆候选数量 | `material-pipeline.ts:21` | slice | 最多 3 个 |
| 10 | 模型名匹配输出 | `scene.ts:457` | `maxOutputTokens` | 30 |
| 11 | 视频最大帧数 | `video.ts:8` | 常量 | 441 |
| 12 | 视频 width/height | `video.ts:50-51` | clamp | 1~4096 |
| 13 | 视频 frame_rate | `video.ts:34` | clamp | 1~60 |
| 14 | patch_code 签名行 | `code.ts:94` | slice | 30 行 |
| 15 | patch_code anchor 显示 | `code.ts:102` | 截断 | 42 字符 |
| 16 | 场景节点折叠 | `builder.ts:1004` | 压缩 | ≥3 子节点折叠 |
| 17 | Prompt: 材质工具上限 | `prompts.ts:160` | 软约束 | 3 次/请求 |
| 18 | Prompt: 代码拆分判据 | `prompts.ts:185` | 软约束 | 200 行 |
| 19 | 会话标题 | `session-manager.ts:218` | 截断 | 30 字符 |

---

## 关键缺口

1. **VLM 视觉旁白无 `maxOutputTokens`**（`index.ts:325-339`）——唯一没有输出 token 限制的 LLM 调用。

2. **`read_file` 返回完整文件内容**（`code.ts:13-20`）——无行数/大小限制，大文件会直接灌满上下文。

3. **`formatSceneSummary` 无节点总数上限**（`builder.ts:1004`）——虽然折叠了预制体子节点，但如果场景有大量独立节点，全部会注入 system prompt。

4. **没有全局上下文窗口 token 计算和滑动截断**——AI SDK 的 `convertToModelMessages` 会将所有历史消息传入 `streamText`，没有基于 token 数的历史消息裁剪机制。长对话会持续累积上下文直到超出模型 context window。

5. **`materialLog` 无上限**（`session-manager.ts:90-101`）——同路径覆盖更新，但不同路径会持续追加，注入 system prompt 的 `[本轮材质改动记录]` 可能无限增长。