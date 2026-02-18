# Routing: Local Versus Bigger Model

## Critical Distinction

- Foundation Models framework APIs in app code expose the on-device language model (`SystemLanguageModel`).
- Apple Intelligence as a platform includes both on-device and server-side models (Private Cloud Compute), but those are not the same thing as public Foundation Models app APIs.
- Apple's "Use Model" action in Shortcuts/App Intents is documented as using Apple Intelligence models on device or with Private Cloud Compute.

Treat these as separate routing surfaces unless Apple publishes an explicit server-routing API for `FoundationModels`.

## Routing Strategy

1. Default to on-device Foundation Models in app code.
2. Escalate to a larger model path only when task requirements exceed on-device limits.
3. Preserve user trust with clear privacy/network behavior and graceful fallback UX.

## When to Stay On-Device

- Structured extraction, summarization, rewriting, classification, and tagging.
- Workloads that fit within the session context window.
- Privacy-sensitive data that should remain local.
- Features that need offline support.

## When to Escalate

- Prompt or context size routinely exceeds on-device limits.
- The task needs broad world knowledge or deeper reasoning not reliably met by the on-device model.
- Product requirements explicitly require cloud-scale model behavior.

## Escalation Paths

### Path A: App Intents and Shortcuts

Use this for automation-centric requests that reference App Intents and Shortcuts.

- Integrate app actions and entities with App Intents.
- Design around the "Use Model" action in Shortcuts when users need Apple Intelligence model responses routed on-device or via Private Cloud Compute.

### Path B: In-App Backend Router

Use this for direct in-app experiences that need larger-model fallback.

- Keep a local-first `FoundationModels` path.
- Add a backend model path (your provider choice) for escalation.
- Route based on availability, context size, task type, latency budget, and policy.

## Example Router Shape

```swift
enum GenerationPath {
    case onDevice
    case backend
}

func choosePath(
    localAvailable: Bool,
    estimatedTokens: Int,
    needsBroadWorldKnowledge: Bool
) -> GenerationPath {
    if localAvailable && estimatedTokens <= 4096 && !needsBroadWorldKnowledge {
        return .onDevice
    }
    return .backend
}
```

## Guardrails for Router Design

- Never silently switch to cloud for sensitive content without product and policy review.
- Surface network-dependent states in UX.
- Log routing decisions and failure reasons for observability.
- Keep parity tests so local and backend outputs remain acceptable for your use case.

## Date and Verification Note

As of February 18, 2026, Apple docs position Foundation Models app APIs as on-device access, while on-device/Private Cloud Compute wording appears in Apple Intelligence + Shortcuts/App Intents material. Re-check Apple docs for each OS cycle.

## Sources

- https://developer.apple.com/documentation/foundationmodels
- https://developer.apple.com/apple-intelligence/
- https://developer.apple.com/apple-intelligence/whats-new/
- https://machinelearning.apple.com/research/introducing-apple-foundation-models
- https://machinelearning.apple.com/research/apple-foundation-models-2025-updates
