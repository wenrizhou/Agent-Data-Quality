## LLM-Judge Prompt（CN 初版）

注意：

1. 输入：Prompt 中不包含留空的模板内容（如 “user query: {query}”），需要根据 prompt 模板灵活修改
2. 输出：如果不用 pydantic 等自动限定输出格式，则需要在 prompt 中添加指定

## 建议的统一输入/输出形式

### Canonical 样本输入

LLM-Judge 的原始输入建议保持与 `agent/data/10_canonical` 一致，即每条样本是一条完整的 general agent 多轮轨迹：

```json
{
  "id": "sample id",
  "query": "optional first user query",
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "tool name",
        "description": "tool description",
        "parameters": {}
      }
    }
  ],
  "metadata": {}
}
```

### Judge Task 输入封装

为便于 prompt 修改和指标迭代，当前建议默认采用“整条样本打分”：每个指标对一条 canonical 样本生成一个 `judge task`，LLM-Judge 直接看到完整多轮对话、可用工具和样本元信息，并输出该指标在样本级别的分数。

后续如果某个指标需要定位到具体轮次，可以在输出中补充 `affected_turns`，但不要求 runner 预先把样本拆成多个 turn-level task。

```json
{
  "sample_id": "sample id",
  "metric": "specificity",
  "task_id": "sample",
  "task_granularity": "sample",
  "query": "optional first user query",
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tools": [],
  "metadata": {}
}
```

字段含义：

- `query`：样本提供的原始 query。如果为空，则以 `conversations` 中第一个 user message 作为初始 query。
- `conversations`：完整多轮轨迹，是当前所有指标的默认核心输入。
- `tools`：当前样本可用工具/API 文档。
- `metadata`：数据来源、难度、语言等辅助信息。
- `task_granularity`：默认为 `sample`，表示该 task 直接输出样本级分数。

### 通用输出格式

所有指标建议统一返回 JSON object。不同指标可以扩展字段，但必须包含以下公共字段：

```json
{
  "score": 0,
  "issue_types": ["..."],
  "affected_turns": [],
  "explanation": "简短说明打分依据",
  "evidence": [
    {
      "source": "query | history | tool_schema | tool_call | tool_response | final_answer",
      "quote": "可选，引用关键片段，不要过长"
    }
  ],
  "confidence": 0.0
}
```

公共字段约定：

- `score`：主分数，只能取 `0`、`1`、`2` 三个枚举值。
  - `0`：存在明显问题，不采用。
  - `1`：稍欠合理性，考虑保留。
  - `2`：合理，保留。
- `issue_types`：命中的错误类型，必须与对应 prompt 中“请判断是否存在以下问题”列出的“问题”一一对应；无问题时返回空列表，不要新增 prompt 未定义的问题类型。
- `affected_turns`：可选，用于定位问题所在轮次。例如 `[0, 3]` 表示第 0 和第 3 个 `conversations` 元素相关。若不确定或无问题，返回空列表。
- `explanation`：一句到三句话，说明主要依据。
- `evidence`：可选但推荐，用于人工复查。只放最关键证据。
- `confidence`：judge 对自己判断的置信度，范围 `0.0-1.0`。

如果 LLM 输出无法解析，框架层建议写成：

```json
{
  "score": null,
  "issue_types": ["parse_error"],
  "affected_turns": [],
  "explanation": "LLM output is not valid JSON.",
  "evidence": [],
  "confidence": 0.0
}
```

### 各指标建议输入/输出

#### 具体性（Specificity）

建议粒度：`sample`。直接评估整条样本中所有用户请求的具体性，综合判断用户是否在任务动作和任务对象上给出了足够信息。多轮对话中允许后续 user message 依赖前文，只要上下文能唯一补全，就不应扣分。

建议输入字段：

```json
{
  "query": "...",
  "conversations": [],
  "tools": [],
  "metadata": {}
}
```

建议输出字段：

```json
{
  "score": 0,
  "action_specificity": 0,
  "object_specificity": 0,
  "intent_missing": false,
  "object_missing": false,
  "context_resolved": false,
  "affected_turns": [],
  "issue_types": [
    "意图不清",
    "意图缺失",
    "对象不清",
    "对象缺失"
  ],
  "explanation": "...",
  "evidence": [],
  "confidence": 0.0
}
```

分数建议：`score = min(action_specificity, object_specificity)`，因为动作或对象任一严重缺失都会影响具体性。若某些工具不需要参数，则相关请求的对象具体性不应被扣分。若整条样本包含多个用户请求，可按整体最严重问题给分，并在 `affected_turns` 中标出问题轮次。

#### 一致性（Coherence）

建议粒度：`sample`。评估整条样本中用户请求、实体、约束、工具/API 需求和多步任务之间是否语义连贯。多轮追问、补充条件、主题延续属于正常一致；无共同目标的突然跳转或矛盾约束应扣分。

建议输入字段：

```json
{
  "query": "...",
  "conversations": [],
  "tools": []
}
```

建议输出字段：

```json
{
  "score": 0,
  "affected_turns": [],
  "issue_types": [
    "语义割裂",
    "步骤不合理"
  ],
  "conflicting_spans": [
    {"span": "...", "reason": "..."}
  ],
  "explanation": "...",
  "evidence": [],
  "confidence": 0.0
}
```

#### 可解性（Solvability）

建议粒度：`sample`。判断给定 tools 的能力是否覆盖整条样本中用户提出的核心需求。不评价 assistant 实际调用是否正确，只看工具能力描述与用户需求是否匹配。

建议输入字段：

```json
{
  "query": "...",
  "conversations": [],
  "tools": []
}
```

建议输出字段：

```json
{
  "score": 0,
  "solvable": false,
  "matched_tools": ["tool_name"],
  "missing_capabilities": ["..."],
  "affected_turns": [],
  "issue_types": [
    "指向性不可解",
    "整体不可解"
  ],
  "explanation": "...",
  "evidence": [],
  "confidence": 0.0
}
```

分数建议：整条样本的核心需求均可由工具覆盖时返回 `2`；部分核心需求工具能力覆盖不充分、需要明显外部假设或工具只能满足主要需求的一部分时返回 `1`；任一核心需求明显无法由给定工具或工具组合覆盖时返回 `0`，并在 `missing_capabilities` 和 `affected_turns` 中说明。只是缺参数但工具能力匹配，仍应返回 `2`。

#### 参数对齐（Parameter alignment）

建议粒度：`sample`。评估整条样本中所有 assistant tool calls 的参数是否与用户需求、工具文档和前序 tool response 对齐。输出中可以列出每个被检查的工具调用，便于定位问题。

建议输入字段：

```json
{
  "query": "...",
  "conversations": [],
  "tools": []
}
```

建议输出字段：

```json
{
  "score": 0,
  "affected_turns": [],
  "checked_tool_calls": [
    {
      "tool_call_id": "optional id",
      "tool_name": "tool name",
      "score": 0,
      "invalid_parameters": [
        {
          "name": "parameter name",
          "value": "parameter value",
          "issue_type": "参数值不合法 | Query 参数提取错误 | 轨迹参数提取错误",
          "reason": "..."
        }
      ]
    }
  ],
  "issue_types": [
    "参数值不合法",
    "Query 参数提取错误",
    "轨迹参数提取错误"
  ],
  "explanation": "...",
  "evidence": [],
  "confidence": 0.0
}
```

分数建议：整条样本中所有核心 tool call 参数正确为 `2`；只有非核心参数或轻微格式问题为 `1`；任一核心参数错误、编造参数、ID/日期/实体继承错误为 `0`。

#### 充分性（Sufficiency）

建议粒度：`sample`。评估整条样本的 assistant/tool 轨迹和最终回答是否足以满足样本中用户提出的所有核心需求。需要同时看工具调用、tool response、assistant final answer 和用户格式要求。

建议输入字段：

```json
{
  "query": "...",
  "conversations": [],
  "tools": []
}
```

建议输出字段：

```json
{
  "score": 0,
  "covered_requirements": ["..."],
  "missing_requirements": ["..."],
  "final_answer_uses_tool_results": false,
  "format_followed": true,
  "blocked_by_permission": false,
  "affected_turns": [],
  "issue_types": [
    "解答错误",
    "轨迹不合理",
    "格式错误",
    "权限不足"
  ],
  "explanation": "...",
  "evidence": [],
  "confidence": 0.0
}
```

#### 最小性（Minimality）

建议粒度：`sample`。评估整条样本中的工具调用轨迹是否足够简洁必要，是否存在冗余、重复、可合并、无关或最终答案完全没有使用的工具结果。

建议输入字段：

```json
{
  "query": "...",
  "conversations": [],
  "tools": []
}
```

建议输出字段：

```json
{
  "score": 0,
  "necessary_tool_calls": ["tool_call_id_or_name"],
  "redundant_tool_calls": [
    {
      "tool_call_id": "optional id",
      "tool_name": "tool name",
      "reason": "..."
    }
  ],
  "unused_tool_responses": ["tool_call_id"],
  "final_answer_uses_any_tool_response": false,
  "affected_turns": [],
  "issue_types": [
    "冗余重复",
    "无结果引用"
  ],
  "explanation": "...",
  "evidence": [],
  "confidence": 0.0
}
```

分数建议：所有必要调用且无明显冗余为 `2`；少量不影响任务的冗余为 `1`；大量无关/重复调用或 final answer 完全不用工具结果为 `0`。

- **具体性（Specificity）**

```Plain
你是一名数据质量评估专家。你的任务是评估用户 query 的“具体性”。
“具体性”指的是：用户 query 是否清楚表达了用户想让 assistant 执行的任务动作，以及该任务作用的核心对象、实体、参数或目标是否明确。

一个 query 可能包含一个或多个请求。请判断 query 中是否存在以下问题：

1. 意图不清  
指 query 没有说明具体操作，或者任务动作表达模糊，导致 assistant 难以判断用户到底想做什么。
注意：如果 query 本身没有明确操作，但可以通过上下文、工具功能或向用户追问补全，则通常判为 1 分或 2 分，而不是直接判 0 分。
2. 意图缺失  
指整个 query 都没有说明任何具体任务，assistant 无法判断用户希望执行什么操作，也无法合理选择工具。
3. 对象不清  
指 query 没有说明任务作用的核心对象、实体、目标或必要参数，但可以通过上下文、工具功能或追问补全。
注意：对象不清的严重程度需要结合对象复杂度判断。如果对象本身简单，缺失时需要通过追问补全，可考虑 1 分；如果对象较复杂，本身需要更细致的确认，通常更倾向于 2 分。
4. 对象缺失  
指所有 query 都没有说明具体对象，且无法从上下文或工具中确定对象。
注意：如果调用工具本身不需要参数，或任务天然不需要具体对象，则不要因为对象缺失而扣分。例如：
- “讲个笑话”
- “生成一个随机数”
- “显示当前时间”
- “列出所有可用语言代码”

评分标准：

- 2 分：具体  
  query 明确说明了用户想执行的任务动作，并且给出了完成任务所需的核心对象、实体、参数或目标；或者这些信息可以从上下文中唯一确定。assistant 可以直接判断要做什么、对什么做，以及大致如何执行。
- 1 分：部分具体 / 可澄清  
  query 中存在意图不清或对象不清，但可以通过上下文、工具说明或向用户追问补全。assistant 仍能大致理解用户的方向，但无法直接完整执行任务。
- 0 分：不具体  
  query 缺少具体任务意图，或缺少所有关键对象信息，导致 assistant 无法判断用户想做什么或对什么做，也无法合理选择工具。

注意事项：
1. 只评估具体性，不评价 query 是否安全、真实等无关标准。
2. 不要因为 query 简短就扣分；只要动作和对象明确，就可以判为 2 分。
7. 如果工具不需要参数，或任务不依赖具体对象，不应因为对象缺失而判低分。
8. 如果 query 包含多个子任务，需要分别检查每个子任务的动作和对象是否明确。
9. 如果用户使用“这个”“它”“上面那个”等指代表达，但上下文中可以唯一确定所指对象，则不算对象缺失。
```

- **一致性（Coherence）**

```Plain
你是一名数据质量评估专家。你的任务是评估用户 query 的“一致性”。
“一致性”指的是：用户 query 中的多个请求、实体、约束条件以及需要调用的工具/API 之间是否语义连贯；如果 query 涉及多个步骤，还需要判断这些步骤是否符合合理的工具链调用顺序或现实逻辑。

一个 query 可能包含一个或多个请求。请判断 query 内部是否存在以下问题：

1. 语义割裂
指 query 中前后请求、实体、约束或工具/API 之间缺乏合理关联，出现明显的主题跳转或语义断裂。
2. 步骤不合理
指 query 中提出的动作顺序不符合工具链调用顺序逻辑，或不符合现实任务执行逻辑。

评分标准：
- 2 分：一致。query 内部语义连贯，多个请求之间有明确关联；如果涉及多个步骤，动作顺序合理，符合工具链调用逻辑或现实逻辑。
- 1 分：轻微不一致 / 轻微语义割裂。query 中存在轻微的主题跳转、弱关联请求或工具/API 语义不完全匹配，但整体仍可以被理解为服务于一个较宽泛的目标，assistant 可以在有限澄清或合理解释下继续执行。
- 0 分：不一致。query 中存在明显无关的请求、矛盾约束，或动作顺序明显违反工具链调用逻辑/现实逻辑，assistant 难以直接执行，需要重排任务、拒绝部分请求或向用户澄清。

注意事项：
1. 如果两个请求属于不同领域，但它们之间有清晰的用户目标连接，可以判为一致。
2. 如果 query 要求先执行后置动作，再执行前置准备动作，应判为步骤不合理。
3. 如果 query 中出现多个无共同目标的工具/API 领域，应判为语义割裂。
4. 只评估一致性，不要评价 query 是否真实、安全等无关标准。
```

- **可解性（Solvability）**

```Plain
你是一名数据质量评估专家。你的任务是评估用户 query 在给定工具/API 条件下的“可解性”。
“可解性”指的是：根据用户 query 中表达的需求，以及给定工具/API 的功能说明，判断这些工具是否有可能满足用户需求。

请判断 query 与 tools 之间是否存在以下问题：

1. 指向性不可解  
指用户 query 明确指定了某个工具/API，但该工具/API 的功能无法满足用户需求。
2. 整体不可解  
指给出的所有工具/API 都无法满足用户 query 的需求。

评分标准：

- 2 分：可解  
  给定工具/API 中至少有一个工具，或多个工具组合后，能够满足用户 query 的核心需求。即使缺少部分参数，也可以通过追问或上下文补全，不影响可解性判断。
- 0 分：不可解  
  用户指定的工具无法满足需求，或给定的所有工具/API 都无法满足用户需求。工具能力与用户核心需求明显不匹配。

注意事项：
1. 只评估“工具/API 的功能是否能够覆盖用户需求”的可解性，不评价 query 是否真实、安全等无关标准。
2. 如果 query 明确指定了某个工具/API，应优先判断该指定工具是否能满足需求；如果 query 没有指定具体工具/API，则判断所有给定工具中是否至少存在一个工具或工具组合可以满足需求。
3. 如果 query 缺少必要参数或对象不明确，但工具能力方向正确，不要判为不可解。
4. 如果工具只能满足 query 的非核心需求，而核心需求无法满足，应判为 0 分。
5. 不要因为需要多步工具调用就判为不可解，只要工具组合能够完成需求即可。
6. 不要因为工具返回结果可能为空、失败或不完整就判为不可解；只要工具能力描述上支持该需求即可。
7. 只评估 query 与工具/API 功能之间的匹配关系，不评估后续调用参数是否正确，也不评估最终答案是否正确。
```

- **参数对齐（Parameter alignment）**

```Plain
你是一名数据质量评估专家。你的任务是评估工具调用轨迹中的“参数对齐”质量。
“参数对齐”指的是：assistant 在调用工具/API 时，所填写的参数名称、参数值和参数来源是否与用户 query、工具文档以及前序 tool response 保持一致。

请判断工具调用中的参数是否存在以下问题：

1. 参数值不合法  
指 assistant 调用工具时填写的参数值超出了工具文档规定的取值范围、格式要求、上下限约束，或违反了现实/业务规则。
2. Query 参数提取错误  
指 assistant 从用户 query 中错误提取、错误推断、遗漏或篡改了参数，导致工具调用中的参数值与用户真实意图不一致。
3. 轨迹参数提取错误  
指 assistant 在多步工具调用中，错误提取或使用了前序 tool response 中的参数、ID、结果或中间变量，导致后续工具调用参数错误。

评分标准：

- 2 分：参数对齐正确。  
  工具调用中的参数名称、参数值和参数来源均与用户 query、工具文档和前序 tool response 一致；没有明显错误提取、非法参数或幻觉参数。
- 1 分：轻微参数问题。  
  存在轻微参数不完整、格式轻微不规范、可通过上下文修正的小错误，或非核心参数存在不影响主要任务执行的偏差。  
- 0 分：参数对齐错误。  
  存在明显参数值不合法、query 参数提取错误、前序 tool response 参数提取错误，导致工具调用无法正确执行或明显偏离用户需求。

注意事项：
1. 只评估工具调用中的参数是否对齐，不评价 query 是否真实、安全等无关标准。
2. 如果用户 query 本身缺少必要参数，而 assistant 没有编造参数，而是留空、使用 #missing 或提出澄清，不应判为参数对齐错误。
3. 如果 assistant 在用户没有提供参数的情况下自行编造具体值，应判为 Query 参数提取错误。
4. 如果后续工具调用依赖前序 tool response，应重点检查 ID、名称、日期、数量、选项编号等是否被正确继承。
5. 如果 query 中存在多个实体或多个约束，需要检查参数是否对应正确，尤其是起止时间等容易反转的参数。
6. 不要因为工具最终可能执行失败就直接判 0；只有当失败原因来自参数提取或参数填写错误时，才属于参数对齐问题。
7. 如果工具调用参数虽然与用户表达不完全一致，但属于合理标准化，例如 “tomorrow” 被转换为正确日期，“New York City” 被规范为 “NYC”，可以判为正确。
8. 如果存在多个工具调用，应综合评估所有调用；只要核心工具调用参数存在严重错误，整体应判为 0。
9. 如果错误只出现在无关紧要的可选参数上，且不影响核心任务，可以考虑判为 1。
```

- **充分性（Sufficiency）**

```Plain
你是一名数据质量评估专家。你的任务是评估 assistant 对用户 query 的回答或工具调用轨迹是否具有“充分性”。
“充分性”指的是：assistant 的工具调用轨迹、推理过程和 final answer 是否足以完整满足用户 query 中提出的所有需求。

请判断 assistant 的回答或轨迹是否存在以下问题：

1. 解答错误  
指 final answer 的内容与工具返回结果、用户需求或事实依据不一致，导致最终回答错误或无法满足用户需求。
2. 轨迹不合理  
指工具调用轨迹不符合工具链调用顺序逻辑或现实逻辑，导致任务无法被正确完成。
3. 格式错误  
指用户 query 明确提出输出格式要求，但 assistant 的 final answer 没有遵守该格式。
4. 权限不足  
指任务中的部分必要步骤由于权限、认证、用户批准、外部系统限制等原因无法继续执行，导致轨迹中断或 final answer 无法完全满足用户需求。

评分标准：

- 2 分：充分  
  assistant 的工具调用轨迹合理，final answer 与工具结果或用户需求一致，能够完整满足用户 query 中的所有核心需求；如果用户有格式要求，也正确遵守。
- 1 分：部分充分 / 权限或轻微缺失  
  assistant 完成了用户的主要需求，但存在轻微遗漏、非核心信息缺失、格式轻微不规范，或因为权限不足/需要用户批准导致部分步骤无法执行。整体仍有一定帮助，但没有完全满足用户所有需求。
- 0 分：不充分  
  assistant 的 final answer 明显错误、无法覆盖用户所有核心需求，或工具调用轨迹严重不合理，导致任务无法正确完成；如果用户有明确格式要求但 response 完全不匹配，也判为 0。

注意事项：
1. 只评估充分性，不评价 query 是否真实、是否安全等无关标准。
2. 如果 assistant 已调用工具，应检查 final answer 是否正确使用了 tool response，而不是编造或篡改结果。
3. 如果用户没有明确格式要求，不要因为回答格式扣分；如果用户明确要求某种输出格式，response 未能匹配，应判为格式错误。
4. 如果 assistant 没有完成任务是因为缺少权限、需要用户确认或外部系统限制，可以判为 1 分；但如果 assistant 本应先说明限制或请求确认却直接编造完成结果，应判为 0 分。
5. 用户的“所有需求”一般直接根据 query 中明确提出的子任务来判断；如果 query 包含多个子任务，需要检查 final answer 是否覆盖所有子任务。
```

- **Minimality（最小性）**

```plain
你是一名数据质量评估专家。你的任务是评估 assistant 的工具调用轨迹是否满足“最小性”。
“最小性”指的是：assistant 是否只执行了解决用户 query 所必需的工具调用，是否避免了冗余、重复、可合并、无关或没有被最终答案使用的操作。

请判断工具调用轨迹是否存在以下问题：

1. 冗余重复  
指 assistant 调用了不必要的工具，或多次执行了可以合并、可以省略、或与用户需求无关的操作。
2. 无结果引用  
指 assistant 的 final answer 没有引用或使用任何 tool response 信息，说明前面的工具调用可能没有必要。

注意：如果 final answer 使用了至少一个必要 tool response，但没有使用部分冗余 tool response，应优先判断为“冗余重复”，而不是“无结果引用”。“无结果引用”主要用于 final answer 完全没有使用任何 tool response 的情况。

评分标准：

- 2 分：最小  
  工具调用轨迹简洁必要。每个工具调用都服务于用户 query 的核心需求，调用之间没有明显重复、无关或可合并的操作；final answer 合理使用了必要的 tool response。
- 1 分：轻微冗余  
  工具调用轨迹中存在少量冗余、重复、可合并或弱相关操作，但主要工具调用仍然必要，final answer 至少使用了关键 tool response，整体任务完成没有受到明显影响。
- 0 分：不满足最小性  
  工具调用轨迹存在明显大量冗余、无关操作，或 final answer 完全没有引用/使用任何 tool response，说明工具调用与最终回答脱节。

注意事项：
1. 只评估最小性，不评价 query 是否真实、安全等无关标准。
2. 不要因为任务本身需要多个工具就扣分。多个工具调用可以是必要的，只要它们分别服务于不同子任务或合理的工具链步骤。
3. 如果某个工具调用是为了确认前序结果、处理分页、补充缺失信息或满足用户的多个子任务，通常不算冗余。
4. 如果工具调用失败后，assistant 合理换用其他工具或向用户说明失败，不应简单判为冗余。
5. 如果 final answer 使用了部分 tool response，但遗漏使用某些工具结果，应判断这些未使用的工具调用是否必要；若不必要，则属于冗余重复。
6. 对“可合并”和“无关”的判断要谨慎：有些额外操作可能存在隐含关联，例如用于验证、消歧、补充上下文或完成前置依赖。
```
