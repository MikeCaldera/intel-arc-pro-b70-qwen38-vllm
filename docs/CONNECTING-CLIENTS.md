# Connecting Pi / omp / Hermes to the B70 vLLM servers

How to point real agent clients at the B70 inference servers once they are
running. The servers expose an OpenAI-compatible `/v1` API with **tool
calling enabled** (`--enable-auto-tool-choice --tool-call-parser qwen3_coder`),
so Pi, omp, and Hermes-style agents can call functions.

## Ports and model names

| Server | Base URL | Served model names |
|---|---|---|
| Cookbook launcher (any model) | `http://<host>:8000/v1` | `Qwen3.6-35B-A3B-MTP-Preserved-GPTQ-Int4` / `Qwen3.6-27B-MTP-Preserved-GPTQ-Int4` |
| Bridge / agent endpoint (launcher profiles) | `http://<host>:8765/v1` | `active` (alias for the currently loaded model) |

`<host>` is the machine running the server (`localhost` if local, or the LAN
IP). The 8765 bridge also serves the loaded model under its real name, but
agents should use the `active` alias so a model switch does not break the
client config.

## API key

The launcher's bridge uses a fixed local key (in the desktop setup it is a
simple static value chosen by the operator). For your own deployment set one
yourself, e.g. `export INFERENCE_API_KEY='your-own-key'` and start the server
with `--api-key "$INFERENCE_API_KEY"`. Replace `<INFERENCE_API_KEY>` below
with your key.

## Hermes (config.yaml)

Add a provider block (or replace the existing `workstation-loaded` block):

```yaml
providers:
  workstation-loaded:
    api_mode: chat_completions
    base_url: http://<host>:8765/v1          # bridge endpoint
    api_key: <INFERENCE_API_KEY>             # your key
    context_length: 131072
    default_model: active                    # alias → current loaded model
    discover_models: false
    name: WKS Loaded Model [B70 vLLM endpoint]
```

Tool calling works automatically: Hermes sends `tool_choice: "auto"` and the
server's `qwen3_coder` parser handles the model's function-call format.

## omp

Point omp at the same bridge. In the omp model/provider configuration set:

```text
base_url:   http://<host>:8765/v1
api_key:    <INFERENCE_API_KEY>
model:      active
```

If omp complains with
`400: "auto" tool choice requires --enable-auto-tool-choice and
--tool-call-parser to be set`, the server was started without the two tool
flags — restart it with the cookbook launchers (they include them) or add
`--enable-auto-tool-choice --tool-call-parser qwen3_coder` to the serve
command.

## Pi (telegram bridge or Pi client)

Set the endpoint and model in the Pi client config:

```text
base_url: http://<host>:8765/v1
api_key:  <INFERENCE_API_KEY>
model:    active
```

Use `max_tokens >= 512` for reasoning models (128 returns empty content on
the ThinkingCap/Qwen3.6 family). Document-grounded Pi sessions get the
resident-document prefix-cache speedup described in
`REAL-WORLD-PI-BENCHMARKS.md` — a cold document costs one long TTFT, then
follow-ups reuse 89.9–94.7% of tokens.

## Sanity check

```bash
# after the server is up (launcher scripts print the health command):
curl -f http://<host>:8000/health
curl -fsS http://<host>:8000/v1/models
```

A quick tool-call smoke test:

```bash
curl -fsS http://<host>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <INFERENCE_API_KEY>" \
  -d '{
    "model": "active",
    "messages": [{"role": "user", "content": "What is the weather in Berlin? Use the weather tool."}],
    "tools": [{"type": "function", "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }}],
    "tool_choice": "auto",
    "max_tokens": 128
  }'
```

Expected: a `tool_calls` entry with `function.name: get_weather` and
`finish_reason: tool_calls` (verified 2026-08-10 on the dense 27B MTP4
server).
